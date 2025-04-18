# -*- coding: utf-8 -*-


from typing import Any

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel, BitsAndBytesConfig
from typing_extensions import override

from cogkit.finetune import register
from cogkit.finetune.diffusion.schemas import DiffusionComponents
from cogkit.finetune.diffusion.trainer import DiffusionTrainer
from cogkit.finetune.utils import unwrap_model
from cogkit.utils import load_lora_checkpoint, unload_lora_checkpoint


class CogVideoXT2VLoraTrainer(DiffusionTrainer):
    UNLOAD_LIST = ["text_encoder", "vae"]
    NEGATIVE_PROMPT = ""

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
        model_path = str(self.args.model_path)

        ### pipeline
        components.pipeline_cls = CogVideoXPipeline

        ### tokenizer
        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        ### text encoder
        components.text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=dtype
        )

        ### transformer
        if not self.args.low_vram:
            components.transformer = CogVideoXTransformer3DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=dtype,
            )
        else:
            components.transformer = CogVideoXTransformer3DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                quantization_config=nf4_config,
                device=self.accelerator.device,
                torch_dtype=dtype,
            )

        ### vae
        components.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_path, subfolder="vae", torch_dtype=dtype
        )

        ### scheduler
        components.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        return components

    @override
    def initialize_pipeline(self, ckpt_path: str | None = None) -> CogVideoXPipeline:
        if not self.args.low_vram:
            pipe = CogVideoXPipeline(
                tokenizer=self.components.tokenizer,
                text_encoder=self.components.text_encoder,
                vae=self.components.vae,
                transformer=unwrap_model(self.accelerator, self.components.transformer),
                scheduler=self.components.scheduler,
            )
        else:
            assert self.args.training_type == "lora"
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                str(self.args.model_path),
                subfolder="transformer",
                torch_dtype=self.state.weight_dtype,
            )
            pipe = CogVideoXPipeline(
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
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(
            prompt_token_ids.to(self.accelerator.device)
        ).last_hidden_state[0]

        # shape of prompt_embedding: [seq_len, hidden_size]
        assert prompt_embedding.ndim == 2
        return prompt_embedding

    @override
    def get_negtive_prompt_embeds(self) -> torch.Tensor:
        return self.encode_text(self.NEGATIVE_PROMPT)

    @override
    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        ret = {"prompt": [], "prompt_embedding": [], "encoded_videos": []}

        for sample in samples:
            prompt = sample.get("prompt", None)
            prompt_embedding = sample.get("prompt_embedding", None)
            encoded_video = sample.get("encoded_video", None)

            ret["prompt"].append(prompt)
            ret["prompt_embedding"].append(prompt_embedding)
            if encoded_video is not None:
                ret["encoded_videos"].append(encoded_video)

        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["encoded_videos"] = (
            torch.stack(ret["encoded_videos"]) if ret["encoded_videos"] else None
        )

        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]

        assert latent is not None and prompt_embedding is not None

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        )
        timesteps = timesteps.long()

        # Add noise to latent
        latent = latent.permute(0, 2, 1, 3, 4)  # from [B, C, F, H, W] to [B, F, C, H, W]
        noise = torch.randn_like(latent)
        latent_added_noise = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise
        predicted_noise = self.components.transformer(
            hidden_states=latent_added_noise,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        # Denoise
        latent_pred = self.components.scheduler.get_velocity(
            predicted_noise, latent_added_noise, timesteps
        )

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean(
            (weights * (latent_pred - latent) ** 2).reshape(batch_size, -1),
            dim=1,
        )
        loss = loss.mean()

        return loss

    @override
    def validation_step(
        self, eval_data: dict[str, Any], pipe: CogVideoXPipeline
    ) -> dict[str, str | list[Image.Image]]:
        prompt = eval_data["prompt"]
        prompt_embedding = eval_data["prompt_embedding"]

        video_generate = pipe(
            num_frames=self.state.train_resolution[0],
            height=self.state.train_resolution[1],
            width=self.state.train_resolution[2],
            prompt_embeds=prompt_embedding,
            negative_prompt_embeds=self.get_negtive_prompt_embeds().unsqueeze(0),
            generator=self.state.generator,
        ).frames[0]
        return {"text": prompt, "video": video_generate}

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (
                num_frames + transformer_config.patch_size_t - 1
            ) // transformer_config.patch_size_t
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin


register("cogvideox-t2v", "lora", CogVideoXT2VLoraTrainer)
