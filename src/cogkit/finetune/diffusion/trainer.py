import json
from typing import Any

import torch
import wandb
from accelerate import cpu_offload
from accelerate.utils import gather_object
from PIL import Image
from typing_extensions import override

from cogkit.finetune.base import BaseTrainer
from cogkit.samplers import NaivePackingSampler
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video

from ..utils import (
    cast_training_params,
    free_memory,
    get_memory_statistics,
    unload_model,
)
from .constants import LOG_LEVEL, LOG_NAME
from .schemas import DiffusionArgs, DiffusionComponents, DiffusionState


class DiffusionTrainer(BaseTrainer):
    # If set, should be a list of components to unload (refer to `Components``)
    UNLOAD_LIST: list[str] = None
    LOG_NAME: str = LOG_NAME
    LOG_LEVEL: str = LOG_LEVEL

    @override
    def _init_args(self) -> DiffusionArgs:
        return DiffusionArgs.parse_args()

    @override
    def _init_state(self) -> DiffusionState:
        return DiffusionState(
            weight_dtype=self.get_training_dtype(),
            train_resolution=self.args.train_resolution,
        )

    @override
    def prepare_models(self) -> None:
        if self.components.vae is not None:
            if self.args.enable_slicing:
                self.components.vae.enable_slicing()
            if self.args.enable_tiling:
                self.components.vae.enable_tiling()

        self.state.transformer_config = self.components.transformer.config

    @override
    def prepare_dataset(self) -> None:
        # TODO: refactor later
        match self.args.model_type:
            case "i2v":
                from cogkit.datasets import BaseI2VDataset, I2VDatasetWithResize

                dataset_cls = I2VDatasetWithResize
                if self.args.enable_packing:
                    dataset_cls = BaseI2VDataset
                    raise NotImplementedError("Packing for I2V is not implemented")

            case "t2v":
                from cogkit.datasets import BaseT2VDataset, T2VDatasetWithResize

                dataset_cls = T2VDatasetWithResize
                if self.args.enable_packing:
                    dataset_cls = BaseT2VDataset
                    raise NotImplementedError("Packing for T2V is not implemented")

            case "t2i":
                from cogkit.datasets import (
                    BaseT2IDataset,
                    T2IDatasetWithPacking,
                    T2IDatasetWithResize,
                )

                dataset_cls = T2IDatasetWithResize
                if self.args.enable_packing:
                    dataset_cls = BaseT2IDataset
                    dataset_cls_packing = T2IDatasetWithPacking

            case _:
                raise ValueError(f"Invalid model type: {self.args.model_type}")

        additional_args = {
            "device": self.accelerator.device,
            "trainer": self,
        }

        self.train_dataset = dataset_cls(
            **(self.args.model_dump()),
            **additional_args,
            using_train=True,
        )
        if self.args.do_validation:
            self.test_dataset = dataset_cls(
                **(self.args.model_dump()),
                **additional_args,
                using_train=False,
            )

        ### Prepare VAE and text encoder for encoding
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)
        if self.args.low_vram:  # offload text encoder to CPU
            cpu_offload(self.components.text_encoder, self.accelerator.device)
        else:
            self.components.text_encoder.to(self.accelerator.device, dtype=self.state.weight_dtype)

        ### Precompute embedding
        self.logger.info("Precomputing embedding ...")
        self.state.negative_prompt_embeds = self.get_negtive_prompt_embeds()

        for dataset in [self.train_dataset, self.test_dataset]:
            if dataset is None:
                continue
            tmp_data_loader = torch.utils.data.DataLoader(
                dataset,
                collate_fn=self.collate_fn,
                batch_size=1,
                num_workers=0,
                pin_memory=self.args.pin_memory,
            )
            tmp_data_loader = self.accelerator.prepare_data_loader(tmp_data_loader)
            for _ in tmp_data_loader:
                ...

        self.accelerator.wait_for_everyone()
        self.logger.info("Precomputing embedding ... Done")

        unload_model(self.components.vae)
        if not self.args.low_vram:
            unload_model(self.components.text_encoder)
        free_memory()

        if not self.args.enable_packing:
            self.train_data_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                collate_fn=self.collate_fn,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory,
                shuffle=True,
            )
        else:
            length_list = [self.sample_to_length(sample) for sample in self.train_dataset]
            self.train_dataset = dataset_cls_packing(self.train_dataset)
            self.train_data_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                collate_fn=self.collate_fn_packing,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory,
                sampler=NaivePackingSampler(
                    length_list,
                    self.state.training_seq_length,
                    shuffle=True,
                ),
            )

        if self.args.do_validation:
            self.test_data_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                collate_fn=self.collate_fn,
                batch_size=1,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory,
                shuffle=False,
            )

    @override
    def validate(self, step: int, ckpt_path: str | None = None) -> None:
        # TODO: refactor later
        self.logger.info("Starting validation")

        accelerator = self.accelerator
        num_validation_samples = len(self.test_data_loader)

        if num_validation_samples == 0:
            self.logger.warning("No validation samples found. Skipping validation.")
            return

        self.components.transformer.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics(self.logger)
        self.logger.info(
            f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}"
        )

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline(ckpt_path=ckpt_path)

        if self.state.using_deepspeed:
            # Can't using model_cpu_offload in deepspeed,
            # so we need to move all components in pipe to device
            self.move_components_to_device(
                dtype=self.state.weight_dtype,
                device=self.accelerator.device,
                ignore_list=["transformer"],
            )
        else:
            # if not using deepspeed, use model_cpu_offload to further reduce memory usage
            # Or use pipe.enable_sequential_cpu_offload() to further reduce memory usage
            if self.args.low_vram:
                pipe.enable_sequential_cpu_offload(device=self.accelerator.device)
            else:
                pipe.enable_model_cpu_offload(device=self.accelerator.device)

            # Convert all model weights to training dtype
            # Note, this will change LoRA weights in self.components.transformer to training dtype, rather than keep them in fp32
            pipe = pipe.to(dtype=self.state.weight_dtype)

        #################################

        all_processes_artifacts = []
        for i, batch in enumerate(self.test_data_loader):
            # only batch size = 1 is currently supported
            prompt = batch.get("prompt", [])
            prompt = prompt[0] if prompt else prompt
            prompt_embedding = batch.get("prompt_embedding", None)

            image = batch.get("image", [])
            image = image[0] if image else image
            encoded_image = batch.get("encoded_image", None)

            video = batch.get("video", [])
            video = video[0] if video else video
            encoded_video = batch.get("encoded_video", None)

            self.logger.debug(
                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
                main_process_only=False,
            )
            validation_artifacts = self.validation_step(
                {
                    "prompt": prompt,
                    "prompt_embedding": prompt_embedding,
                    "image": image,
                    "encoded_image": encoded_image,
                    "video": video,
                    "encoded_video": encoded_video,
                },
                pipe,
            )

            artifacts = {}
            for ii, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                artifacts.update(
                    {
                        f"artifact_{ii}": {
                            "type": artifact_type,
                            "value": artifact_value,
                        }
                    }
                )
            self.logger.debug(
                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
                main_process_only=False,
            )

            for key, value in list(artifacts.items()):
                artifact_type = value["type"]
                artifact_value = value["value"]
                if artifact_type not in ["text", "image", "video"] or artifact_value is None:
                    continue

                match artifact_type:
                    case "text":
                        extension = "txt"
                    case "image":
                        extension = "png"
                    case "video":
                        extension = "mp4"
                validation_path = self.args.output_dir / "validation_res" / f"validation-{step}"
                validation_path.mkdir(parents=True, exist_ok=True)
                filename = f"artifact-process{accelerator.process_index}-batch{i}.{extension}"
                filename = str(validation_path / filename)

                if artifact_type == "image":
                    self.logger.debug(f"Saving image to {filename}")
                    artifact_value.save(filename)
                    artifact_value = wandb.Image(filename)
                elif artifact_type == "video":
                    self.logger.debug(f"Saving video to {filename}")
                    export_to_video(artifact_value, filename, fps=self.args.gen_fps)
                    artifact_value = wandb.Video(filename)
                elif artifact_type == "text":
                    self.logger.debug(f"Saving text to {filename}")
                    with open(filename, "w") as f:
                        f.write(artifact_value)
                    artifact_value = str(artifact_value)

                all_processes_artifacts.append(artifact_value)

        all_artifacts = gather_object(all_processes_artifacts)

        if accelerator.is_main_process:
            tracker_key = "validation"
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    text_artifacts = [
                        artifact for artifact in all_artifacts if isinstance(artifact, str)
                    ]
                    image_artifacts = [
                        artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)
                    ]
                    video_artifacts = [
                        artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)
                    ]
                    tracker.log(
                        {
                            tracker_key: {
                                "texts": text_artifacts,
                                "images": image_artifacts,
                                "videos": video_artifacts,
                            },
                        },
                        step=step,
                    )

        ##########  Clean up  ##########
        if self.state.using_deepspeed:
            del pipe
            # Unload models except those needed for training
            self.move_components_to_device(
                dtype=self.state.weight_dtype, device="cpu", ignore_list=self.UNLOAD_LIST
            )
        else:
            pipe.remove_all_hooks()
            del pipe
            # Load models except those not needed for training
            self.move_components_to_device(
                dtype=self.state.weight_dtype,
                device=self.accelerator.device,
                ignore_list=self.UNLOAD_LIST,
            )
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)

            # Change trainable weights back to fp32 to keep with dtype after prepare the model
            cast_training_params([self.components.transformer], dtype=torch.float32)

        free_memory()
        accelerator.wait_for_everyone()
        ################################

        memory_statistics = get_memory_statistics(self.logger)
        self.logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        torch.set_grad_enabled(True)
        self.components.transformer.train()

    @override
    def load_components(self) -> DiffusionComponents:
        raise NotImplementedError

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def collate_fn(self, samples: list[dict[str, Any]]):
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
        self,
    ) -> list[tuple[str, Image.Image | list[Image.Image]]]:
        raise NotImplementedError

    # ==========  Packing related functions  ==========
    def sample_to_length(self, sample: dict[str, Any]) -> int:
        """Map sample to length for packing sampler"""
        raise NotImplementedError

    def collate_fn_packing(self, samples: list[dict[str, list[Any]]]) -> dict[str, Any]:
        """Collate function for packing sampler"""
        raise NotImplementedError

    # =================================================
