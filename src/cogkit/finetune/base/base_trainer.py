# -*- coding: utf-8 -*-
import json
import math
import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cogkit.finetune.base import BaseArgs, BaseComponents, BaseState
from cogkit.finetune.logger import get_logger
from cogkit.utils import inject_lora, set_global_seed, save_lora

from ..utils import (
    AppState,
    WandbTracker,
    cast_training_params,
    check_distributed,
    delete_files,
    free_memory,
    get_device,
    get_global_rank,
    get_global_step,
    get_local_rank,
    get_memory_statistics,
    get_world_size,
    is_main_process,
    list_files,
    mkdir,
)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,  # FP16 is Only Support for CogVideoX-2B
    "bf16": torch.bfloat16,
}


class BaseTrainer(ABC):
    """
    Base class for all finetuning trainers.

    Note: This class assumes that only `transformer` module is needed to be trained.
    """

    # If set, should be a list of components to unload (refer to `Components``)
    UNLOAD_LIST: list[str] | None = None

    MODEL_STATE_DICT_FNAME = "model_state_dict.safetensors"
    OPTIM_STATE_DICT_FNAME = "optim_state_dict.safetensors"

    def __init__(self, uargs_fpath: str | Path) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if isinstance(uargs_fpath, str):
            uargs_fpath = Path(uargs_fpath)

        self.uargs = self._init_args(uargs_fpath)

        self._init_distributed()
        self._init_directories()

        self.logger = get_logger(
            name=self.uargs.name4train,
            log_file=self.uargs.output_dir / f"{self.uargs.name4train}.log",
            level=self.uargs.log_level,
        )

        if self.uargs.seed is not None:
            set_global_seed(self.uargs.seed)

        self.train_dataset: Dataset = None
        self.test_dataset: Dataset = None
        self.train_data_loader: DataLoader = None
        self.test_data_loader: DataLoader = None
        self.optimizer = None
        self.lr_scheduler = None

        self.state = self._init_state()
        self.components = self.load_components()
        self.tracker = None
        if self.uargs.report_to is not None:
            self.tracker = WandbTracker(
                name=self.uargs.name4train,
                config=self.uargs.model_dump(),
            )
        self.check_setting()

    def _init_distributed(self) -> None:
        dist.init_process_group(backend="nccl", timeout=self.uargs.nccl_timeout)
        torch.cuda.set_device(get_local_rank())

    def _init_directories(self) -> None:
        mkdir(self.uargs.output_dir)

    def _init_args(self, uargs_fpath: Path) -> BaseArgs:
        return BaseArgs.parse_from_yaml(uargs_fpath)

    def _init_state(self) -> BaseState:
        return BaseState(
            world_size=get_world_size(),
            local_rank=get_local_rank(),
            global_rank=get_global_rank(),
            device=get_device(),
            weight_dtype=_DTYPE_MAP[self.uargs.mixed_precision],
        )

    def fit(self) -> None:
        self.logger.info("Checking settings...")
        self.check_setting()

        self.logger.info("Initializing models...")
        self.prepare_models()

        self.logger.info("Initializing dataset and dataloader...")
        self.prepare_dataset()

        self.logger.info("Initializing trainable parameters...")
        self.prepare_trainable_parameters()

        self.logger.info("Preparing model...")
        self.prepare_model()

        self.logger.info("Initializing optimizer and lr scheduler...")
        self.prepare_optimizer()

        self.logger.info("Starting training...")
        self.train()

        self.logger.info("Cleaning up...")
        self.cleanup()

    def check_setting(self) -> None:
        check_distributed()
        # Check for `UNLOAD_LIST`
        if self.UNLOAD_LIST is None:
            self.logger.warning(
                "\033[91mNo unload_list specified for this Trainer. All components will be loaded to GPU during training.\033[0m"
            )
        else:
            for name in self.UNLOAD_LIST:
                if name not in self.components.model_fields:
                    raise ValueError(f"Invalid component name in unload_list: {name}")

    def prepare_trainable_parameters(self) -> None:
        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.uargs.training_type == "sft" and attr_name == "transformer":
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)

        if self.uargs.training_type == "lora":
            # Initialize LoRA weights
            inject_lora(self.components.transformer, lora_dir_or_state_dict=None)

        if self.uargs.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

        # cast all trainable params to the specified data type (bf16)
        cast_training_params(self.components.transformer, dtype=self.state.weight_dtype)

    def prepare_model(self) -> None:
        match self.uargs.strategy:
            case "NO_SHARD":
                sharding_strategy = ShardingStrategy.NO_SHARD
            case "SHARD_GRAD_OP":
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
            case "FULL_SHARD":
                sharding_strategy = ShardingStrategy.FULL_SHARD
            case "HYBRID_SHARD":
                sharding_strategy = ShardingStrategy.HYBRID_SHARD

        if self.uargs.strategy != "DDP":
            warp_policy = partial(
                size_based_auto_wrap_policy,
                min_num_params=int(1e8),
            )

            self.components.transformer = FSDP(
                module=self.components.transformer,
                device_id=self.state.local_rank,
                sharding_strategy=sharding_strategy,
                auto_wrap_policy=warp_policy,
                cpu_offload=CPUOffload(offload_params=self.uargs.offload_params_grads),
                mixed_precision=MixedPrecision(
                    param_dtype=self.state.weight_dtype,
                    reduce_dtype=self.state.weight_dtype,
                ),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                use_orig_params=True if self.uargs.training_type == "lora" else False,
            )
        else:
            # use qlora means we have already moved the model to the device
            if not self.uargs.low_vram:
                self.components.transformer = self.components.transformer.to(self.state.device)

            self.components.transformer = DDP(
                module=self.components.transformer,
                device_ids=[self.state.local_rank],
            )

        # Load components needed for training to GPU, and cast them to the specified data type
        ignore_list = self.UNLOAD_LIST
        self.move_components_to_device(
            dtype=self.state.weight_dtype,
            device=self.state.device,
            ignore_list=ignore_list + ["transformer"],
        )

    def prepare_optimizer(self) -> None:
        # For LoRA, we only want to train the LoRA weights
        # For SFT, we want to train all the parameters
        trainable_parameters = list(
            filter(
                lambda p: p.requires_grad,
                self.components.transformer.parameters(),
            )
        )
        transformer_parameters_with_lr = {
            "params": trainable_parameters,
            "lr": self.uargs.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)

        optimizer = torch.optim.AdamW(
            params=params_to_optimize,
            lr=self.uargs.learning_rate,
            betas=(self.uargs.beta1, self.uargs.beta2),
            eps=self.uargs.epsilon,
            weight_decay=self.uargs.weight_decay,
        )

        num_update_steps_per_epoch = math.ceil(
            len(self.train_data_loader) / self.uargs.gradient_accumulation_steps
        )
        total_train_steps = self.uargs.train_epochs * num_update_steps_per_epoch

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_train_steps,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train(self) -> None:
        # We need to recalculate our total training steps as the size of the training dataloader may have changed in distributed training
        num_update_steps_per_epoch = math.ceil(
            len(self.train_data_loader) / self.uargs.gradient_accumulation_steps
        )
        self.state.train_steps = self.uargs.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.state.train_epochs = math.ceil(self.state.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

        memory_statistics = get_memory_statistics(self.logger)
        self.logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.uargs.batch_size * self.state.world_size * self.uargs.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.train_dataset),
            "train epochs": self.state.train_epochs,
            "train steps": self.state.train_steps,
            "batches per device": self.uargs.batch_size,
            "total batches observed per epoch": len(self.train_data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.uargs.gradient_accumulation_steps,
        }
        self.logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        initial_epoch = 0
        # Potentially load in the weights and states from a previous save
        if self.uargs.resume_from_checkpoint is not None:
            self.logger.info(f"Resuming from checkpoint {self.uargs.resume_from_checkpoint}")
            global_step = get_global_step(self.uargs.resume_from_checkpoint)
            for _ in range(global_step):
                self.lr_scheduler.step()
            self.resume_from_checkpoint(self.uargs.resume_from_checkpoint)
            initial_epoch = global_step // num_update_steps_per_epoch
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr_scheduler.get_last_lr()[0]

        progress_bar = tqdm(
            range(self.state.train_steps),
            initial=global_step,
            desc="Training steps",
            disable=not is_main_process(),
        )

        generator = torch.Generator(device=self.state.device)
        if self.uargs.seed is not None:
            generator = generator.manual_seed(self.uargs.seed)
        self.state.generator = generator

        free_memory()
        ckpt_path = None
        for epoch in range(initial_epoch, self.uargs.train_epochs):
            self.logger.debug(f"Starting epoch ({epoch + 1}/{self.uargs.train_epochs})")

            self.components.transformer.train()

            for step, batch in enumerate(self.train_data_loader):
                self.logger.debug(f"Starting step {step + 1}, global step: {global_step}")

                is_sync_step = (step + 1) % self.uargs.gradient_accumulation_steps == 0
                is_last_step = (step + 1) == len(self.train_data_loader)
                sync_grad = is_sync_step or is_last_step

                logs = self.train_step(batch, sync_grad=sync_grad)

                if sync_grad:
                    global_step += 1
                    progress_bar.update(1)

                    ckpt_path = self.maybe_save_checkpoint(global_step)

                    progress_bar.set_postfix(logs)

                    if self.tracker is not None:
                        self.tracker.log(logs, step=global_step)

                    if self.uargs.do_validation and global_step % self.uargs.validation_steps == 0:
                        free_memory()
                        self.validate(global_step, ckpt_path=ckpt_path)

            memory_statistics = get_memory_statistics(self.state.device)
            self.logger.info(
                f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}"
            )

    def train_step(self, batch: dict[str, Any], sync_grad: bool) -> dict[str, Any]:
        logs = {}

        sync_context = self.components.transformer.no_sync() if not sync_grad else nullcontext()

        with sync_context:
            loss = self.compute_loss(batch)
            loss = loss / self.uargs.gradient_accumulation_steps
            loss.backward()

        if sync_grad:
            if self.uargs.strategy != "DDP":
                grad_norm = self.components.transformer.clip_grad_norm_(
                    max_norm=self.uargs.max_grad_norm
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.components.transformer.parameters(),
                    max_norm=self.uargs.max_grad_norm,
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            loss = loss.detach()
            dist.all_reduce(grad_norm.to(self.state.device), op=dist.ReduceOp.AVG)
            dist.all_reduce(loss.to(self.state.device), op=dist.ReduceOp.AVG)

            logs["grad_norm"] = grad_norm.item()
            logs["loss"] = loss.item()
            logs["lr"] = self.lr_scheduler.get_last_lr()[0]
            del loss  # release graph

        return logs

    def move_components_to_device(self, dtype, device, ignore_list: list[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if (
                not isinstance(component, type)
                and hasattr(component, "to")
                and name not in ignore_list
            ):
                setattr(
                    self.components,
                    name,
                    component.to(device, dtype=dtype),
                )

    def maybe_save_checkpoint(self, global_step: int, must_save: bool = False) -> str | None:
        if not (must_save or global_step % self.uargs.checkpointing_steps == 0):
            return None

        checkpointing_limit = self.uargs.checkpointing_limit
        output_dir = Path(self.uargs.output_dir)
        logger = self.logger

        if checkpointing_limit is not None:
            checkpoints = list_files(output_dir, prefix="checkpoint")

            def get_checkpoint_number(path):
                try:
                    return int(Path(path).name.split("-")[1])
                except (IndexError, ValueError):
                    raise ValueError(f"Invalid checkpoint path: {path}")

            checkpoints.sort(key=get_checkpoint_number)

            # before we save the new checkpoint, we need to have at_most `checkpoints_total_limit - 1` checkpoints
            if len(checkpoints) >= checkpointing_limit:
                num_to_remove = len(checkpoints) - checkpointing_limit + 1
                checkpoints_to_remove = checkpoints[0:num_to_remove]
                delete_files(checkpoints_to_remove)

        save_dir = output_dir / f"checkpoint-{global_step}"
        mkdir(save_dir)
        logger.info(f"Checkpointing at step {global_step}, saving state to {save_dir} ...")

        saved_model = self.unwrap_model(self.components.transformer)

        state_dict = {
            "app": AppState(saved_model, self.optimizer, lora=self.uargs.training_type == "lora")
        }
        if not self.uargs.low_vram:
            dcp.save(state_dict, checkpoint_id=str(save_dir))
        else:
            if is_main_process():
                save_lora(saved_model, save_dir)

        return save_dir

    def resume_from_checkpoint(self, ckpt_dir: str | Path) -> None:
        transformer = self.unwrap_model(self.components.transformer)
        state_dict = {
            "app": AppState(transformer, self.optimizer, lora=self.uargs.training_type == "lora")
        }
        dcp.load(state_dict, checkpoint_id=str(ckpt_dir))

    def cleanup(self) -> None:
        dist.destroy_process_group()
        if self.tracker is not None:
            self.tracker.finish()

    def unwrap_model(self, model: Any) -> Any:
        if self.uargs.strategy == "DDP":
            return model.module
        else:
            return model

    @abstractmethod
    def load_components(self) -> BaseComponents:
        # note: `self.components.transformer`(model needs to be trained)
        #       and `self.components.pipeline_cls` must be defined
        raise NotImplementedError

    @abstractmethod
    def prepare_models(self) -> None:
        # Doing something like `self.components.vae.enable_slicing()`
        raise NotImplementedError

    @abstractmethod
    def prepare_dataset(self) -> None:
        # initialize `self.train_dataset` and `self.train_data_loader`
        # initialize `self.test_dataset` and `self.test_data_loader` if `self.uargs.do_validation` is True
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def validate(self, step: int, ckpt_path: str | None = None) -> None:
        # validation logic defined here
        # during validation, additional modules in the pipeline may need to be moved to GPU memory
        raise NotImplementedError
