# -*- coding: utf-8 -*-


from typing import Any, Tuple

import torch
from typing_extensions import override

from cogkit.finetune import register
from cogkit.finetune.utils import (
    process_latent_attention_mask,
    process_prompt_attention_mask,
)
from cogkit.utils import expand_list
from diffusers.models.transformers.transformer_cogview4 import CogView4RotaryPosEmbed

from .lora_trainer import Cogview4Trainer


class Cogview4LoraPackingTrainer(Cogview4Trainer):
    IMAGE_FACTOR = 32  # Size of image (height, width) to be trained should be a multiple of 32

    DOWNSAMPLER_FACTOR = 8
    PATCH_SIZE: int
    ATTN_HEAD: int
    ATTEN_DIM: int
    ROPE_DIM: Tuple[int, int]

    max_vtoken_length: int
    training_seq_length: int
    rope: CogView4RotaryPosEmbed

    @override
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        transformer = self.components.transformer
        self.PATCH_SIZE = transformer.config.patch_size
        self.ATTN_HEAD = transformer.config.num_attention_heads
        self.ATTEN_DIM = transformer.config.attention_head_dim
        self.ROPE_DIM = transformer.config.rope_axes_dim

        patch_size = self.PATCH_SIZE
        height, width = self.args.train_resolution
        sample_height, sample_width = (
            height // self.DOWNSAMPLER_FACTOR,
            width // self.DOWNSAMPLER_FACTOR,
        )
        self.max_vtoken_length = sample_height * sample_width // patch_size**2
        self.training_seq_length = self.max_vtoken_length + self.MAX_TTOKEN_LENGTH
        self.state.training_seq_length = self.training_seq_length

        self.rope = CogView4RotaryPosEmbed(
            dim=self.ATTEN_DIM,
            patch_size=self.PATCH_SIZE,
            rope_axes_dim=self.ROPE_DIM,
        )

    @override
    def sample_to_length(self, sample: dict[str, Any]) -> int:
        # shape of image_latent: [num_channels, height, width]
        image_latent = sample["encoded_image"]
        prompt_embedding = sample["prompt_embedding"]
        assert image_latent.ndim == 3
        assert prompt_embedding.ndim == 2
        num_channels, latent_height, latent_width = image_latent.shape

        patch_size = self.PATCH_SIZE
        paded_width = latent_width + latent_width % patch_size
        paded_height = latent_height + latent_height % patch_size
        latent_length = paded_height * paded_width // patch_size**2
        if latent_length > self.max_vtoken_length:
            raise ValueError(
                f"latent_length {latent_length} is greater than max_vtoken_length {self.max_vtoken_length}"
            )

        assert self.MAX_TTOKEN_LENGTH + self.max_vtoken_length == self.training_seq_length
        assert latent_length + prompt_embedding.shape[0] <= self.training_seq_length

        return latent_length + prompt_embedding.shape[0]

    @override
    def collate_fn_packing(self, samples: list[dict[str, list[Any]]]) -> dict[str, Any]:
        """Collate function for training dataloader with packing support.

        This function processes batches of samples from the `T2IDatasetWithPacking` dataset,
        combining multiple samples into a single batch while maintaining proper attention masks
        and positional embeddings.

        Args:
            samples: List of dictionaries containing packed samples. Each sample contains:
                - prompt: List of text prompts
                - prompt_embedding: List of prompt embeddings
                - encoded_image: List of encoded image latents
                - image: List of original images

        Returns:
            dict: A dictionary containing batched data with the following keys:
                - prompt_embedding: Batched prompt embeddings
                - encoded_image: Batched encoded image latents
                - image_rotary_emb: Rotary embeddings for images
                - attention_kwargs: Dictionary containing:
                    - batch_flag: Indices indicating which sample each item belongs to
                    - text_attn_mask: Attention mask for text embeddings
                    - latent_attn_mask: Attention mask for latent embeddings
                - pixel_mask: Mask for valid pixel regions
                - original_size: Original dimensions of the images

        Note:
            This function is specifically used for the training dataloader.
            For validation, the collate_fn from Cogview4Trainer should be used instead.
        """
        batched_data = {
            "prompt_embedding": None,
            "encoded_image": None,
            "image_rotary_emb": None,
            "attention_kwargs": {
                "batch_flag": None,
                "text_attn_mask": None,
                "latent_attn_mask": None,
            },
            "pixel_mask": None,
            "original_size": None,
        }
        batch_flag = [[idx] * len(slist["prompt"]) for idx, slist in enumerate(samples)]
        batch_flag = sum(batch_flag, [])
        batch_flag = torch.tensor(batch_flag, dtype=torch.int32)
        samples = expand_list(samples)
        assert len(batch_flag) == len(samples["prompt"])

        prompt_embedding, prompt_attention_mask = process_prompt_attention_mask(
            self.components.tokenizer,
            samples["prompt"],
            samples["prompt_embedding"],
            self.MAX_TTOKEN_LENGTH,
            self.TEXT_TOKEN_FACTOR,
        )

        patch_size = self.PATCH_SIZE
        image_rotary_emb = [self.rope(ei.unsqueeze(0)) for ei in samples["encoded_image"]]
        padded_latent, vtoken_attention_mask, pixel_mask = process_latent_attention_mask(
            samples["encoded_image"], patch_size
        )

        # Store in batched_data
        batched_data["prompt_embedding"] = prompt_embedding
        batched_data["attention_kwargs"]["text_attn_mask"] = prompt_attention_mask
        batched_data["encoded_image"] = padded_latent
        batched_data["image_rotary_emb"] = image_rotary_emb
        batched_data["attention_kwargs"]["latent_attn_mask"] = vtoken_attention_mask.reshape(
            len(batch_flag), -1
        )
        batched_data["pixel_mask"] = pixel_mask

        batched_data["attention_kwargs"]["batch_flag"] = batch_flag
        batched_data["original_size"] = torch.tensor(
            [(img.height, img.width) for img in samples["image"]]
        )

        return batched_data

    @override
    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        dtype = self.get_training_dtype()
        prompt_embeds = batch["prompt_embedding"]
        latent = batch["encoded_image"]
        image_rotary_emb = batch["image_rotary_emb"]
        batch_size, text_seqlen, text_embedding_dim = prompt_embeds.shape
        batch_size, num_channels, height, width = latent.shape

        attention_kwargs = batch["attention_kwargs"]
        latent_attention_mask = attention_kwargs["latent_attn_mask"].float()
        assert latent_attention_mask.dim() == 2
        vtoken_seq_len = torch.sum(latent_attention_mask != 0, dim=1)

        original_size = batch["original_size"]

        num_train_timesteps = self.components.scheduler.config.num_train_timesteps
        sigmas = self.get_sigmas(batch_size, vtoken_seq_len)
        timestep = self.get_timestep(batch_size, num_train_timesteps)

        noise = torch.randn_like(latent, dtype=dtype)
        model_input, model_label = self.add_noise(latent, noise, timestep, sigmas)

        original_size = original_size.to(dtype=dtype, device=self.accelerator.device)
        target_size = original_size.clone().to(dtype=dtype, device=self.accelerator.device)
        crop_coords = torch.tensor(
            [[0, 0] for _ in range(batch_size)], dtype=dtype, device=self.accelerator.device
        )

        noise_pred_cond = self.components.transformer(
            hidden_states=model_input.to(dtype=dtype),
            encoder_hidden_states=prompt_embeds.to(dtype=dtype),
            timestep=timestep,
            original_size=original_size,
            target_size=target_size,
            crop_coords=crop_coords,
            return_dict=False,
            image_rotary_emb=image_rotary_emb,
            attention_kwargs=attention_kwargs,
        )[0]

        pixel_mask = batch["pixel_mask"]
        loss = torch.sum(((noise_pred_cond - model_label) ** 2) * pixel_mask, dim=(1, 2, 3))
        loss = loss / torch.sum(pixel_mask, dim=(1, 2, 3))
        loss = loss.mean()

        return loss


register("cogview4-6b", "lora-packing", Cogview4LoraPackingTrainer)
