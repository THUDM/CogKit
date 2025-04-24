# -*- coding: utf-8 -*-


import os
import numpy as np
import torch

from cogkit.api.logging import get_logger
from cogkit.api.settings import APISettings
from cogkit.python import generate_image
from cogkit.utils import load_lora_checkpoint, load_pipeline, unload_lora_checkpoint

_logger = get_logger(__name__)


class ImageGenerationService(object):
    def __init__(self, settings: APISettings) -> None:
        self._models = {}

        # TODO: Refactor this to switch by LoRA endpoint API
        self._current_lora = {}  # Track currently loaded LORA for each model

        if settings.cogview4_path is not None:
            torch_dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float32
            cogview4_pl = load_pipeline(
                model_id_or_path=settings.cogview4_path,
                lora_model_id_or_path=settings.lora_dir,
                transformer_path=settings.cogview4_transformer_path,
                dtype=torch_dtype,
            )
            self._models["cogview-4"] = cogview4_pl
            self._current_lora["cogview-4"] = None  # Initialize with no LORA loaded

        for model in self._models.keys():
            if model not in settings._supported_models:
                raise ValueError(
                    f"Registered model {model} not in supported list: {settings._supported_models}"
                )
        for model in settings._supported_models:
            if model not in self._models:
                _logger.warning(f"Model {model} not loaded")

    @property
    def supported_models(self) -> list[str]:
        return list(self._models.keys())

    def generate(
        self,
        model: str,
        prompt: str,
        size: str,
        num_images: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> list[np.ndarray]:
        if model not in self._models:
            raise ValueError(f"Model {model} not loaded")
        width, height = list(map(int, size.split("x")))
        # TODO: Refactor this to switch by LoRA endpoint API
        if lora_path != self._current_lora[model]:
            if lora_path is not None:
                adapter_name = os.path.basename(lora_path)
                _logger.info(
                    f"Loading LORA weights from {adapter_name} and unload previous weights {self._current_lora[model]}"
                )
                unload_lora_checkpoint(self._models[model])
                load_lora_checkpoint(self._models[model], lora_path)
            else:
                _logger.info(f"Unloading LORA weights {self._current_lora[model]}")
                unload_lora_checkpoint(self._models[model])

            self._current_lora[model] = lora_path
        output = generate_image(
            prompt=prompt,
            height=height,
            pipeline=self._models[model],
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            output_type="np",
        )

        image_lst = self.postprocess(output)
        return image_lst

    def is_valid_model(self, model: str) -> bool:
        return model in self._models

    def postprocess(self, image_np: np.ndarray) -> list[np.ndarray]:
        image_lst = np.split(image_np, image_np.shape[0], axis=0)
        image_lst = [img.squeeze(0) for img in image_lst]
        return image_lst
