# -*- coding: utf-8 -*-


import json
import logging
import math
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path

import diffusers
import torch
import transformers
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cogkit.finetune.base import BaseArgs, BaseComponents, BaseState
from cogkit.utils.lora import inject_lora, save_lora

from ..utils import (
    cast_training_params,
    free_memory,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    unwrap_model,
    find_files,
    delete_files,
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

    LOG_NAME: str = "BaseTrainer"
    LOG_LEVEL: str = "INFO"

    # If set, should be a list of components to unload (refer to `Components``)
    #    `transformer` is always in UNLOAD_LIST
    UNLOAD_LIST: list[str] | None = None

    def __init__(self) -> None:
        self.logger = get_logger(self.LOG_NAME, self.LOG_LEVEL)
        self.accelerator: Accelerator = None
        self.train_dataset: Dataset = None
        self.test_dataset: Dataset = None
        self.train_data_loader: DataLoader = None
        self.test_data_loader: DataLoader = None
        self.optimizer = None
        self.lr_scheduler = None

        self.args = self._init_args()
        self.state = self._init_state()

        self._init_distributed()
        self._init_logging()
        self._init_directories()

        self.components = self.load_components()

        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, "logs")
        project_config = ProjectConfiguration(
            project_dir=self.args.output_dir, logging_dir=logging_dir
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.accelerator = accelerator

        tracker_name = self.args.tracker_name
        self.accelerator.init_trackers(
            project_name=tracker_name,
            init_kwargs={"wandb": {"name": self.args.output_dir.name}},
        )

        if self.args.seed is not None:
            set_seed(self.args.seed)

    def _init_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=self.LOG_LEVEL,
        )
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        self.logger.info("Initialized Trainer")
        self.logger.info(
            f"Accelerator state: \n{self.accelerator.state}",
            main_process_only=False,
        )

    def _init_directories(self) -> None:
        if self.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)

    def check_setting(self) -> None:
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
        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and attr_name == "transformer":
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)

        if self.args.training_type == "lora":
            # Initialize LoRA weights
            inject_lora(self.components.transformer, lora_dir_or_state_dict=None)
            self.prepare_saving_loading_hooks()

        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

    def prepare_optimizer(self) -> None:
        # Make sure the trainable params are in float32
        # cast_training_params([self.components.transformer], dtype=torch.float32)

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
            "lr": self.args.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)

        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            logger=self.logger,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        # Do not need to divide by num_gpus since acclerate will handle this after prepare lr_scheduler
        num_update_steps_per_epoch = math.ceil(
            len(self.train_data_loader) / self.args.gradient_accumulation_steps
        )
        total_train_steps = self.args.train_epochs * num_update_steps_per_epoch
        total_num_warmup_steps = max(int(total_train_steps * self.args.lr_warmup_ratio), 0)

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_train_steps,
                num_warmup_steps=total_num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=total_num_warmup_steps,
                num_training_steps=total_train_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_for_training(self) -> None:
        # cast training params to the specified data type (bf16)
        cast_training_params(self.components.transformer, dtype=self.state.weight_dtype)

        (
            self.components.transformer,
            self.optimizer,
            self.train_data_loader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.components.transformer,
            self.optimizer,
            self.train_data_loader,
            self.lr_scheduler,
        )

        # Load components needed for training to GPU (except transformer), and cast them to the specified data type
        ignore_list = self.UNLOAD_LIST
        self.move_components_to_device(
            dtype=self.state.weight_dtype, device=self.accelerator.device, ignore_list=ignore_list
        )

        if self.args.do_validation:
            assert self.test_data_loader is not None
            self.test_data_loader = self.accelerator.prepare_data_loader(self.test_data_loader)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed in distributed training
        num_update_steps_per_epoch = math.ceil(
            len(self.train_data_loader) / self.args.gradient_accumulation_steps
        )
        self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    def train(self) -> None:
        memory_statistics = get_memory_statistics(self.logger)
        self.logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.train_dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.train_data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        self.logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
            logger=self.logger,
        )
        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)

        progress_bar = tqdm(
            range(self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        free_memory()
        ckpt_path = None
        for epoch in range(first_epoch, self.args.train_epochs):
            self.logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]

            for step, batch in enumerate(self.train_data_loader):
                self.logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    loss = self.compute_loss(batch)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.components.transformer.parameters(),
                                self.args.max_grad_norm,
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    ckpt_path = self.maybe_save_checkpoint(global_step)

                    logs["loss"] = loss.detach().item()
                    logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                    progress_bar.set_postfix(logs)

                    # Maybe run validation
                    should_run_validation = (
                        self.args.do_validation
                        and global_step % self.args.validation_steps == 0
                        and accelerator.sync_gradients
                    )
                    if should_run_validation:
                        del loss
                        free_memory()
                        self.validate(global_step, ckpt_path=ckpt_path)

                    accelerator.log(logs, step=global_step)

                    if global_step >= self.args.train_steps:
                        break

            memory_statistics = get_memory_statistics(self.logger)
            self.logger.info(
                f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}"
            )

        accelerator.wait_for_everyone()
        ckpt_path = self.maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step, ckpt_path=ckpt_path)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics(self.logger)
        self.logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    def fit(self) -> None:
        self.logger.info("Checking settings...")
        self.check_setting()

        self.logger.info("Initializing models...")
        self.prepare_models()

        self.logger.info("Initializing dataset and dataloader...")
        self.prepare_dataset()

        self.logger.info("Initializing trainable parameters...")
        self.prepare_trainable_parameters()

        self.logger.info("Initializing optimizer and lr scheduler...")
        self.prepare_optimizer()

        self.logger.info("Preparing for training...")
        self.prepare_for_training()

        self.logger.info("Starting training...")
        self.train()

    @abstractmethod
    def _init_args(self) -> BaseArgs:
        raise NotImplementedError

    @abstractmethod
    def _init_state(self) -> BaseState:
        raise NotImplementedError

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
        # initialize `self.test_dataset` and `self.test_data_loader` if `self.args.do_validation` is True
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def validate(self, step: int, ckpt_path: str | None = None) -> None:
        # validation logic defined here
        # during validation, additional modules in the pipeline may need to be moved to GPU memory
        raise NotImplementedError

    def get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")

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

    def prepare_saving_loading_hooks(self):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            assert self.accelerator.distributed_type != DistributedType.DEEPSPEED

            for model in models:
                original_model = unwrap_model(self.accelerator, model)
                original_transformer = unwrap_model(self.accelerator, self.components.transformer)
                if isinstance(original_model, type(original_transformer)):
                    if self.accelerator.is_main_process:
                        save_lora(model, output_dir)
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

        def load_model_hook(models, input_dir):
            assert self.accelerator.distributed_type != DistributedType.DEEPSPEED

            for model in models:
                original_model = unwrap_model(self.accelerator, model)
                original_transformer = unwrap_model(self.accelerator, self.components.transformer)
                if isinstance(original_model, type(original_transformer)):
                    inject_lora(model, input_dir)
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def maybe_save_checkpoint(self, global_step: int, must_save: bool = False) -> str | None:
        if not (must_save or global_step % self.args.checkpointing_steps == 0):
            return None

        checkpointing_limit = self.args.checkpointing_limit
        output_dir = Path(self.args.output_dir)
        logger = self.logger

        if checkpointing_limit is not None:
            checkpoints = find_files(output_dir, prefix="checkpoint")

            # before we save the new checkpoint, we need to have at_most `checkpoints_total_limit - 1` checkpoints
            if len(checkpoints) >= checkpointing_limit:
                num_to_remove = len(checkpoints) - checkpointing_limit + 1
                checkpoints_to_remove = checkpoints[0:num_to_remove]
                if self.accelerator.is_main_process:
                    delete_files(checkpoints_to_remove, logger)

        logger.info(f"Checkpointing at step {global_step}")
        save_path = output_dir / f"checkpoint-{global_step}"
        logger.info(f"Saving state to {save_path}")

        self.accelerator.save_state(save_path, safe_serialization=True)

        self.accelerator.wait_for_everyone()
        return save_path
