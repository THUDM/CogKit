# -*- coding: utf-8 -*-


from typing import Any, Tuple

import torch
from PIL import Image
from transformers import AutoTokenizer, BitsAndBytesConfig, GlmForCausalLM
from typing_extensions import override

from cogkit.finetune import register
from cogkit.finetune.diffusion.schemas import DiffusionComponents
from cogkit.finetune.diffusion.trainer import DiffusionTrainer
from cogkit.finetune.utils import process_prompt_attention_mask, unwrap_model
from cogkit.utils import load_lora_checkpoint, unload_lora_checkpoint
from diffusers import (
    AutoencoderKL,
    CogView4Pipeline,
    CogView4Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)


class Cogview4Trainer(DiffusionTrainer):
    UNLOAD_LIST = ["text_encoder", "vae"]
    MAX_TTOKEN_LENGTH = 224
    NEGATIVE_PROMPT = ""
    TEXT_TOKEN_FACTOR = 16

    @override
    def load_components(self) -> DiffusionComponents:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        components = DiffusionComponents()
        model_path = str(self.args.model_path)

        ### pipeline
        components.pipeline_cls = CogView4Pipeline

        ### tokenizer
        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        ### text encoder
        components.text_encoder = GlmForCausalLM.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        )

        ### transformer
        if not self.args.low_vram:
            components.transformer = CogView4Transformer2DModel.from_pretrained(
                model_path, subfolder="transformer", torch_dtype=torch.bfloat16, device="cpu"
            )
        else:
            components.transformer = CogView4Transformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
                quantization_config=nf4_config,
                device_map="auto",
            )

        components.vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.bfloat16, device="cpu"
        )

        components.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        return components

    @override
    def initialize_pipeline(self, ckpt_path: str | None = None) -> CogView4Pipeline:
        if not self.args.low_vram:
            pipe = CogView4Pipeline(
                tokenizer=self.components.tokenizer,
                text_encoder=self.components.text_encoder,
                vae=self.components.vae,
                transformer=unwrap_model(self.accelerator, self.components.transformer),
                scheduler=self.components.scheduler,
            )
        else:
            assert self.args.training_type == "lora"
            # using bf16 model rather than quantized ones
            transformer = CogView4Transformer2DModel.from_pretrained(
                str(self.args.model_path),
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
                device="cpu",
            )
            pipe = CogView4Pipeline(
                tokenizer=self.components.tokenizer,
                text_encoder=self.components.text_encoder,
                vae=self.components.vae,
                transformer=transformer,
                scheduler=self.components.scheduler,
            )
            unload_lora_checkpoint(pipe)
            load_lora_checkpoint(pipe, ckpt_path)

        return pipe

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        """
        Note: For the GLM text encoder, the number of tokens should be a multiple of 16.
        """
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding=True,
            max_length=self.MAX_TTOKEN_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
            pad_to_multiple_of=self.TEXT_TOKEN_FACTOR,
        ).input_ids

        prompt_embedding = self.components.text_encoder(
            prompt_token_ids.to(self.accelerator.device), output_hidden_states=True
        ).hidden_states[-2][0]
        # shape of prompt_embedding: [sequence length, embedding dimension(4096)]
        return prompt_embedding

    @override
    def get_negtive_prompt_embeds(self) -> torch.Tensor:
        return self.encode_text(self.NEGATIVE_PROMPT)

    @override
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        image = image.to(self.accelerator.device, dtype=vae.dtype)
        latent_dist = vae.encode(image).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """
        This function assumes that all the images in samples have the same resolution.
        """
        ret = {
            "prompt": [],
            "prompt_embedding": [],
            "image": [],
            "encoded_image": [],
            "attention_mask": {"text_embedding_attn_mask": None},
        }

        for sample in samples:
            prompt = sample.get("prompt", None)
            prompt_embedding = sample.get("prompt_embedding", None)
            image = sample.get("image", None)
            encoded_image = sample.get("encoded_image", None)

            ret["prompt"].append(prompt)
            ret["prompt_embedding"].append(prompt_embedding)
            # image and encoded_image maybe None during validation
            if image is not None:
                ret["image"].append(image)
            if encoded_image is not None:
                ret["encoded_image"].append(encoded_image)

        prompt_embedding, prompt_attention_mask = process_prompt_attention_mask(
            self.components.tokenizer,
            ret["prompt"],
            ret["prompt_embedding"],
            self.MAX_TTOKEN_LENGTH,
            self.TEXT_TOKEN_FACTOR,
        )

        ret["prompt_embedding"] = prompt_embedding
        ret["attention_mask"]["text_embedding_attn_mask"] = prompt_attention_mask

        ret["encoded_image"] = torch.stack(ret["encoded_image"]) if ret["encoded_image"] else None

        # shape of prompt_embedding: [batch_size, sequence_length, embedding_dim(4096)]
        assert (
            ret["attention_mask"]["text_embedding_attn_mask"].shape
            == ret["prompt_embedding"].shape[:2]
        )

        return ret

    @override
    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        batch_size, text_seqlen, text_embedding_dim = batch["prompt_embedding"].shape
        prompt_embeds = batch["prompt_embedding"]
        latent = batch["encoded_image"]

        batch_size, num_channels, height, width = latent.shape
        image_height, image_width = self.state.train_resolution
        vae_scale_factor = 8
        image_seq_len = (
            (image_height // vae_scale_factor) * (image_width // vae_scale_factor)
        ) // (self.state.transformer_config.patch_size**2)

        attention_mask = batch["attention_mask"]

        # prepare timesteps
        m = (image_seq_len / self.components.scheduler.config.base_image_seq_len) ** 0.5
        mu = (
            m * self.components.scheduler.config.max_shift
            + self.components.scheduler.config.base_shift
        )
        self.components.scheduler.set_timesteps(
            self.components.scheduler.config.num_train_timesteps,
            mu=mu,
            device=self.accelerator.device,
        )
        timestep = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (1,),
            device=self.accelerator.device,
        ).long()

        noise = torch.randn_like(latent)
        model_input, model_label = self.add_noise(latent, noise, timestep[0])
        original_size = torch.tensor(
            [[image_height, image_width] for _ in range(batch_size)],
            dtype=latent.dtype,
            device=self.accelerator.device,
        )
        target_size = torch.tensor(
            [[image_height, image_width] for _ in range(batch_size)],
            dtype=latent.dtype,
            device=self.accelerator.device,
        )
        crop_coords = torch.tensor(
            [[0, 0] for _ in range(batch_size)], dtype=latent.dtype, device=self.accelerator.device
        )

        noise_pred_cond = self.components.transformer(
            hidden_states=model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            original_size=original_size,
            target_size=target_size,
            crop_coords=crop_coords,
            return_dict=False,
            attention_mask=attention_mask,
        )[0]

        loss = torch.mean((noise_pred_cond - model_label) ** 2, dim=(1, 2, 3))
        loss = loss.mean()

        return loss

    def add_noise(
        self, latent: torch.Tensor, noise: torch.Tensor, timestep: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to the latent vector based on the timestep.

        Args:
            latent (torch.Tensor): The latent vector to add noise to.
            noise (torch.Tensor): The noise tensor to add.
            timestep (torch.LongTensor): The current timestep.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The noisy latent vector that will be input to the model and the model label.
        """
        num_train_timesteps = self.components.scheduler.config.num_train_timesteps
        # note: sigmas in scheduler is arranged in reversed order
        scale_factor = self.components.scheduler.sigmas[num_train_timesteps - timestep]
        model_input = latent * (1 - scale_factor) + noise * scale_factor
        model_label = noise - latent
        return model_input, model_label

    @override
    def validation_step(
        self, eval_data: dict[str, Any], pipe: CogView4Pipeline
    ) -> list[tuple[str, Image.Image | list[Image.Image]]]:
        """
        Return the data that needs to be saved. For images, the data format is PIL
        """
        prompt = eval_data["prompt"]
        prompt_embedding = eval_data["prompt_embedding"]

        # FIXME
        image_generate = pipe(
            height=self.state.train_resolution[0],
            width=self.state.train_resolution[1],
            prompt_embeds=prompt_embedding,
            negative_prompt_embeds=self.state.negative_prompt_embeds.unsqueeze(
                0
            ),  # Add batch dimension
            generator=self.state.generator,
        ).images[0]
        return [("text", prompt), ("image", image_generate)]


register("cogview4-6b", "lora", Cogview4Trainer)
