# -*- coding: utf-8 -*-


from typing import Any, Tuple

import torch
from PIL import Image
from transformers import AutoTokenizer, BitsAndBytesConfig, GlmForCausalLM
from typing_extensions import override

from cogkit.finetune import register
from cogkit.finetune.diffusion.schemas import DiffusionComponents
from cogkit.finetune.diffusion.trainer import DiffusionTrainer
from cogkit.finetune.utils import (
    process_prompt_attention_mask,
    replace_attn_processor,
)
from cogkit.utils import load_lora_checkpoint, unload_lora_checkpoint
from diffusers import (
    AutoencoderKL,
    CogView4Pipeline,
    CogView4Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models.transformers.transformer_cogview4 import CogView4TrainingAttnProcessor


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
        dtype = self.state.weight_dtype

        components = DiffusionComponents()
        model_path = str(self.uargs.model_path)

        ### pipeline
        components.pipeline_cls = CogView4Pipeline

        ### tokenizer
        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        ### text encoder
        components.text_encoder = GlmForCausalLM.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
        )

        ### transformer
        if not self.uargs.low_vram:
            components.transformer = CogView4Transformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=dtype,
            )
        else:
            components.transformer = CogView4Transformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=dtype,
                quantization_config=nf4_config,
                device=self.state.device,
            )
        replace_attn_processor(components.transformer, CogView4TrainingAttnProcessor())

        ### vae
        components.vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=dtype
        )

        ### scheduler
        components.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        return components

    @override
    def initialize_pipeline(self, ckpt_path: str | None = None) -> CogView4Pipeline:
        # using bf16 model rather than quantized ones
        if not self.uargs.low_vram:
            pipe = CogView4Pipeline(
                tokenizer=self.components.tokenizer,
                text_encoder=self.components.text_encoder,
                vae=self.components.vae,
                transformer=self.unwrap_model(self.components.transformer),
                scheduler=self.components.scheduler,
            )
        else:
            assert self.uargs.training_type == "lora"
            transformer = CogView4Transformer2DModel.from_pretrained(
                str(self.uargs.model_path),
                subfolder="transformer",
                torch_dtype=self.state.weight_dtype,
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
            prompt_token_ids.to(self.state.device), output_hidden_states=True
        ).hidden_states[-2][0]
        # shape of prompt_embedding: [sequence length, embedding dimension(4096)]
        return prompt_embedding

    @override
    def get_negtive_prompt_embeds(self) -> torch.Tensor:
        return self.encode_text(self.NEGATIVE_PROMPT)

    @override
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        image = image.to(self.state.device, dtype=vae.dtype)
        latent_dist = vae.encode(image).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate function that processes a batch of samples from the `T2IDatasetWithFactorResize` dataset.

        This function combines individual samples into a batch that can be processed by the model.
        It handles prompt embeddings, images, and attention masks, ensuring proper formatting
        for model training.

        This function is shared between training and validation dataloaders:
        - During training: All fields (prompt, prompt_embedding, image, encoded_image) are provided
        - During validation: Only 'prompt' and 'prompt_embedding' are provided, while 'image' and
          'encoded_image' will be None

        Args:
            samples: A list of dictionaries, each representing a sample with keys:
                - 'prompt': Text prompt string
                - 'prompt_embedding': Encoded text prompt tensor
                - 'image': Original image tensor (provided only during training)
                - 'encoded_image': VAE-encoded latent representation (provided only during training)

        Returns:
            A dictionary containing batch-processed data with keys:
                - 'prompt': List of prompt strings
                - 'prompt_embedding': Tensor of shape [batch_size, sequence_length, embedding_dim]
                - 'image': List of image tensors (will be empty during validation)
                - 'encoded_image': Tensor of shape [batch_size, channels, height, width] (None during validation)
                - 'text_attn_mask': Tensor of shape [batch_size, sequence_length] for transformer attention

        Note:
            This function assumes that all images in the batch have the same resolution.
        """
        ret = {
            "prompt": [],
            "prompt_embedding": [],
            "image": [],
            "encoded_image": [],
            "text_attn_mask": None,
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
        ret["text_attn_mask"] = prompt_attention_mask

        ret["encoded_image"] = torch.stack(ret["encoded_image"]) if ret["encoded_image"] else None

        # shape of prompt_embedding: [batch_size, sequence_length, embedding_dim(4096)]
        assert ret["text_attn_mask"].shape == ret["prompt_embedding"].shape[:2]

        return ret

    @override
    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        batch_size, text_seqlen, text_embedding_dim = batch["prompt_embedding"].shape
        device = self.state.device
        prompt_embeds = batch["prompt_embedding"].to(device)
        latent = batch["encoded_image"].to(device)

        batch_size, num_channels, height, width = latent.shape
        image_height, image_width = self.state.train_resolution
        vae_scale_factor = 8
        image_seq_len = (
            (image_height // vae_scale_factor) * (image_width // vae_scale_factor)
        ) // (self.state.transformer_config.patch_size**2)
        image_seq_len = torch.tensor([image_seq_len], device=device)

        text_attn_mask = batch["text_attn_mask"]

        num_train_timesteps = self.components.scheduler.config.num_train_timesteps
        sigmas = self.get_sigmas(batch_size, image_seq_len)
        timestep = self.get_timestep(batch_size, num_train_timesteps)

        noise = torch.randn_like(latent)
        model_input, model_label = self.add_noise(latent, noise, timestep, sigmas)

        original_size = torch.tensor(
            [[image_height, image_width] for _ in range(batch_size)],
            dtype=latent.dtype,
            device=device,
        )
        target_size = torch.tensor(
            [[image_height, image_width] for _ in range(batch_size)],
            dtype=latent.dtype,
            device=device,
        )
        crop_coords = torch.tensor(
            [[0, 0] for _ in range(batch_size)], dtype=latent.dtype, device=device
        )

        noise_pred_cond = self.components.transformer(
            hidden_states=model_input.to(dtype=self.state.weight_dtype),
            encoder_hidden_states=prompt_embeds.to(dtype=self.state.weight_dtype),
            timestep=timestep,
            original_size=original_size,
            target_size=target_size,
            crop_coords=crop_coords,
            return_dict=False,
            attention_kwargs={"text_attn_mask": text_attn_mask},
        )[0]

        loss = torch.mean((noise_pred_cond - model_label) ** 2, dim=(1, 2, 3))
        loss = loss.mean()

        return loss

    def get_sigmas(self, batch_size: int, vtoken_seq_len: torch.Tensor) -> torch.Tensor:
        assert vtoken_seq_len.ndim == 1
        if vtoken_seq_len.size(0) == 1:
            vtoken_seq_len = vtoken_seq_len.repeat(batch_size)
        else:
            assert vtoken_seq_len.size(0) == batch_size

        scheduler = self.components.scheduler
        scheduler = self.components.scheduler
        sigmas = torch.linspace(
            scheduler.sigma_min,
            scheduler.sigma_max,
            scheduler.config.num_train_timesteps,
            device=self.state.device,
        )
        m = (vtoken_seq_len / scheduler.config.base_image_seq_len) ** 0.5
        mu = m * scheduler.config.max_shift + scheduler.config.base_shift
        mu = mu.unsqueeze(1).to(sigmas.device)
        sigmas = mu / (mu + (1 / sigmas - 1))
        sigmas = torch.cat([torch.zeros((batch_size, 1), device=sigmas.device), sigmas], dim=1)
        return sigmas

    def get_timestep(self, batch_size: int, num_train_timesteps: int) -> torch.LongTensor:
        return torch.randint(
            0,
            num_train_timesteps,
            (batch_size,),
            device=self.state.device,
        )

    def add_noise(
        self,
        latent: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert latent.shape[0] == noise.shape[0] == timestep.shape[0] == sigmas.shape[0]
        index = timestep
        scale_factor = (
            torch.gather(sigmas, dim=1, index=index.unsqueeze(1))
            .squeeze(1)
            .view(-1, 1, 1, 1)
            .to(latent.device)
        )

        model_input = latent * (1 - scale_factor) + noise * scale_factor
        model_label = noise - latent
        return model_input, model_label

    @override
    def validation_step(
        self, pipe: CogView4Pipeline, eval_data: dict[str, Any]
    ) -> dict[str, str | Image.Image]:
        prompt = eval_data["prompt"]
        prompt_embedding = eval_data["prompt_embedding"]

        image_generate = pipe(
            height=self.state.train_resolution[0],
            width=self.state.train_resolution[1],
            prompt_embeds=prompt_embedding.to(self.state.device),
            negative_prompt_embeds=self.state.negative_prompt_embeds.unsqueeze(
                0
            ),  # Add batch dimension
            generator=self.state.generator,
        ).images[0]
        return {"text": prompt, "image": image_generate}


register("cogview4-6b", "lora", Cogview4Trainer)
