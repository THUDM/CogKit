import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import wandb
from accelerate import cpu_offload
from torch.utils.data import DistributedSampler
from PIL import Image
from typing_extensions import override

from cogkit.finetune.base import BaseTrainer
from cogkit.finetune.samplers import DistPackingSampler
from cogkit.utils import expand_list, guess_generation_mode
from cogkit.types import GenerationMode
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video

from ..utils import (
    free_memory,
    get_memory_statistics,
    gather_object,
    mkdir,
)
from .schemas import DiffusionArgs, DiffusionComponents, DiffusionState


class DiffusionTrainer(BaseTrainer):
    @override
    def __init__(self, uargs_fpath: str | Path) -> None:
        super().__init__(uargs_fpath)
        self.uargs: DiffusionArgs
        self.state: DiffusionState
        self.components: DiffusionComponents

    @override
    def _init_args(self, uargs_fpath: Path) -> DiffusionArgs:
        return DiffusionArgs.parse_from_yaml(uargs_fpath)

    @override
    def _init_state(self) -> DiffusionState:
        state = DiffusionState(**super()._init_state().model_dump())
        state.train_resolution = self.uargs.train_resolution
        return state

    @override
    def prepare_models(self) -> None:
        if self.components.vae is not None:
            if self.uargs.enable_slicing:
                self.components.vae.enable_slicing()
            if self.uargs.enable_tiling:
                self.components.vae.enable_tiling()

        self.state.transformer_config = self.components.transformer.config

    @override
    def prepare_dataset(self) -> None:
        generation_mode = guess_generation_mode(self.components.pipeline_cls)
        match generation_mode:
            case GenerationMode.TextToImage:
                from cogkit.finetune.datasets import (
                    T2IDatasetWithFactorResize,
                    T2IDatasetWithPacking,
                    T2IDatasetWithResize,
                )

                dataset_cls = T2IDatasetWithResize
                if self.uargs.enable_packing:
                    dataset_cls = T2IDatasetWithFactorResize
                    dataset_cls_packing = T2IDatasetWithPacking

            case GenerationMode.TextToVideo:
                from cogkit.finetune.datasets import BaseT2VDataset, T2VDatasetWithResize

                dataset_cls = T2VDatasetWithResize
                if self.uargs.enable_packing:
                    dataset_cls = BaseT2VDataset
                    raise NotImplementedError("Packing for T2V is not implemented")

            case GenerationMode.ImageToVideo:
                from cogkit.finetune.datasets import BaseI2VDataset, I2VDatasetWithResize

                dataset_cls = I2VDatasetWithResize
                if self.uargs.enable_packing:
                    dataset_cls = BaseI2VDataset
                    raise NotImplementedError("Packing for I2V is not implemented")

            case _:
                raise ValueError(f"Invalid generation mode: {generation_mode}")

        additional_args = {
            "device": self.state.device,
            "trainer": self,
        }

        self.train_dataset = dataset_cls(
            **(self.uargs.model_dump()),
            **additional_args,
            using_train=True,
        )
        if self.uargs.do_validation:
            self.test_dataset = dataset_cls(
                **(self.uargs.model_dump()),
                **additional_args,
                using_train=False,
            )

        ### Prepare VAE and text encoder for encoding
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae.to(self.state.device, dtype=self.state.weight_dtype)
        if self.uargs.low_vram:  # offload text encoder to CPU
            cpu_offload(self.components.text_encoder, self.state.device)
        else:
            self.components.text_encoder.to(self.state.device, dtype=self.state.weight_dtype)

        ### Precompute embedding
        self.logger.info("Precomputing embedding ...")
        self.state.negative_prompt_embeds = self.get_negtive_prompt_embeds().to(self.state.device)

        for dataset in [self.train_dataset, self.test_dataset]:
            if dataset is None:
                continue
            tmp_data_loader = torch.utils.data.DataLoader(
                dataset,
                collate_fn=self.collate_fn,
                batch_size=1,
                num_workers=0,
                pin_memory=self.uargs.pin_memory,
                sampler=DistributedSampler(
                    dataset,
                    num_replicas=self.state.world_size,
                    rank=self.state.global_rank,
                    shuffle=False,
                ),
            )
            for _ in tmp_data_loader:
                ...

        self.logger.info("Precomputing embedding ... Done")
        dist.barrier()

        self.components.vae = self.components.vae.to("cpu")
        if not self.uargs.low_vram:
            self.components.text_encoder = self.components.text_encoder.to("cpu")
        free_memory()

        if not self.uargs.enable_packing:
            self.train_data_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                collate_fn=self.collate_fn,
                batch_size=self.uargs.batch_size,
                num_workers=self.uargs.num_workers,
                pin_memory=self.uargs.pin_memory,
                sampler=DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.state.world_size,
                    rank=self.state.global_rank,
                    shuffle=True,
                ),
            )
        else:
            length_list = [self.sample_to_length(sample) for sample in self.train_dataset]
            self.train_dataset = dataset_cls_packing(self.train_dataset)
            self.train_data_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                collate_fn=self.collate_fn_packing,
                batch_size=self.uargs.batch_size,
                num_workers=self.uargs.num_workers,
                pin_memory=self.uargs.pin_memory,
                sampler=DistPackingSampler(
                    length_list,
                    self.state.training_seq_length,
                    shuffle=True,
                    world_size=self.state.world_size,
                    global_rank=self.state.global_rank,
                ),
            )

        if self.uargs.do_validation:
            self.test_data_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                collate_fn=self.collate_fn,
                batch_size=1,
                num_workers=self.uargs.num_workers,
                pin_memory=self.uargs.pin_memory,
                sampler=DistributedSampler(
                    self.test_dataset,
                    num_replicas=self.state.world_size,
                    rank=self.state.global_rank,
                    shuffle=False,
                ),
            )

    @override
    def validate(self, step: int, ckpt_path: str | None = None) -> None:
        self.logger.info("Starting validation")

        num_validation_samples = len(self.test_data_loader)
        if num_validation_samples == 0:
            self.logger.warning("No validation samples found. Skipping validation.")
            return

        # self.components.transformer.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics(self.logger)
        self.logger.info(
            f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}"
        )

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline(ckpt_path=ckpt_path)

        # if not using deepspeed, use model_cpu_offload to further reduce memory usage
        # Or use pipe.enable_sequential_cpu_offload() to further reduce memory usage
        if self.uargs.low_vram:
            pipe.enable_sequential_cpu_offload(device=self.state.device)
        else:
            pipe.enable_model_cpu_offload(device=self.state.device)

        # Convert all model weights to training dtype
        # Note, this will change LoRA weights in self.components.transformer to training dtype, rather than keep them in fp32
        pipe = pipe.to(dtype=self.state.weight_dtype)

        #################################

        all_processes_artifacts = []
        for i, batch in enumerate(self.test_data_loader):
            # only batch size = 1 is currently supported
            prompt = batch.get("prompt", None)
            prompt = prompt[0] if prompt else None
            prompt_embedding = batch.get("prompt_embedding", None)

            image = batch.get("image", None)
            image = image[0] if image else None
            encoded_image = batch.get("encoded_image", None)

            video = batch.get("video", None)
            video = video[0] if video else None
            encoded_video = batch.get("encoded_video", None)

            self.logger.debug(
                f"Validating sample {i + 1}/{num_validation_samples} on process {self.state.global_rank}. Prompt: {prompt}",
            )
            val_res = self.validation_step(
                pipe=pipe,
                eval_data={
                    "prompt": prompt,
                    "prompt_embedding": prompt_embedding,
                    "image": image,
                    "encoded_image": encoded_image,
                    "video": video,
                    "encoded_video": encoded_video,
                },
            )

            artifacts = {}
            val_path = self.uargs.output_dir / "validation_res" / f"validation-{step}"
            mkdir(val_path)
            filename = f"artifact-process{self.state.global_rank}-batch{i}"

            image = val_res.get("image", None)
            video = val_res.get("video", None)
            with open(val_path / f"{filename}.txt", "w") as f:
                f.write(prompt)
            if image:
                fpath = str(val_path / f"{filename}.png")
                image.save(fpath)
                artifacts["image"] = wandb.Image(fpath, caption=prompt)
            if video:
                fpath = str(val_path / f"{filename}.mp4")
                export_to_video(video, fpath, fps=self.uargs.gen_fps)
                artifacts["video"] = wandb.Video(fpath, caption=prompt)

            all_processes_artifacts.append(artifacts)

        if self.tracker is not None:
            all_artifacts = gather_object(all_processes_artifacts)
            all_artifacts = [item for sublist in all_artifacts for item in sublist]
            all_artifacts = expand_list(all_artifacts)
            self.tracker.log({"validation": all_artifacts}, step=step)

        # =============  Clean up  =============
        pipe.remove_all_hooks()
        del pipe
        # Load models except those not needed for training
        self.move_components_to_device(
            dtype=self.state.weight_dtype,
            device=self.state.device,
            ignore_list=self.UNLOAD_LIST,
        )
        # self.components.transformer.to(self.state.device, dtype=self.state.weight_dtype)

        # Change trainable weights back to fp32 to keep with dtype after prepare the model
        # cast_training_params([self.components.transformer], dtype=torch.float32)

        free_memory()
        dist.barrier()
        # =======================================

        memory_statistics = get_memory_statistics(self.logger)
        self.logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(self.state.device)

        torch.set_grad_enabled(True)
        self.components.transformer.train()

    @override
    def load_components(self) -> DiffusionComponents:
        raise NotImplementedError

    @override
    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def collate_fn(self, samples: list[dict[str, Any]]):
        """
        Note: This collate_fn function are used for both training and validation.
        """
        raise NotImplementedError

    def initialize_pipeline(self, ckpt_path: str | None = None) -> DiffusionPipeline:
        raise NotImplementedError

    def encode_text(self, text: str) -> torch.Tensor:
        # shape of output text: [sequence length, embedding dimension]
        raise NotImplementedError

    def get_negtive_prompt_embeds(self) -> torch.Tensor:
        # shape of output text: [sequence length, embedding dimension]
        raise NotImplementedError

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        # shape of input image: [B, C, H, W], where B = 1
        # shape of output image: [B, C', H', W'], where B = 1
        raise NotImplementedError

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W], where B = 1
        # shape of output video: [B, C', F', H', W'], where B = 1
        raise NotImplementedError

    def validation_step(
        self, pipe: DiffusionPipeline, eval_data: dict[str, Any]
    ) -> dict[str, str | Image.Image | list[Image.Image]]:
        """
        Perform a validation step using the provided pipeline and evaluation data.

        Args:
            pipe: The diffusion pipeline instance used for validation.
            eval_data: A dictionary containing data for validation, may include:
                - "prompt": Text prompt for generation (str).
                - "prompt_embedding": Pre-computed text embeddings.
                - "image": Input image for image-to-image tasks.
                - "encoded_image": Pre-computed image embeddings.
                - "video": Input video for video tasks.
                - "encoded_video": Pre-computed video embeddings.

        Returns:
            A dictionary containing generated artifacts with keys:
                - "text": Text data (str).
                - "image": Generated image (PIL.Image.Image).
                - "video": Generated video (list[PIL.Image.Image]).
        """
        raise NotImplementedError

    # ==========  Packing related functions  ==========
    def sample_to_length(self, sample: dict[str, Any]) -> int:
        """Map sample to length for packing sampler"""
        raise NotImplementedError

    def collate_fn_packing(self, samples: list[dict[str, list[Any]]]) -> dict[str, Any]:
        """Collate function for packing sampler"""
        raise NotImplementedError

    # =================================================
