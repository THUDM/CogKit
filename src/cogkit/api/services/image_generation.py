# -*- coding: utf-8 -*-


import numpy as np
from diffusers import CogView4Pipeline

from cogkit.api.logging import get_logger
from cogkit.api.settings import APISettings

_logger = get_logger(__name__)


class ImageGenerationService(object):
    def __init__(self, settings: APISettings) -> None:
        self._models = {}
        if settings.cogview4_path is not None:
            cogview4_pl = CogView4Pipeline.from_pretrained(settings.cogview4_path)
            cogview4_pl.enable_model_cpu_offload()
            cogview4_pl.vae.enable_slicing()
            cogview4_pl.vae.enable_tiling()
            self._models["cogview-4"] = cogview4_pl

        ### Check if loaded models are supported
        for model in self._models.keys():
            if model not in settings._supported_models:
                raise ValueError(
                    f"Registered model {model} not in supported list: {settings._supported_models}"
                )

        ### Check if all supported models are loaded
        for model in settings._supported_models:
            if model not in self._models:
                _logger.warning(f"Model {model} not loaded")

    def generate(self, model: str, prompt: str, size: str, num_images: int) -> list[np.ndarray]:
        if model not in self._models:
            raise ValueError(f"Model {model} not loaded")
        width, height = list(map(int, size.split("x")))

        # shape of image_np: (n, h, w, c)
        image_np = self._models[model](
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=50,
            guidance_scale=3.5,
            num_images_per_prompt=num_images,
            output_type="np",
        ).images
        assert image_np.ndim == 4, f"Expected 4D array, got {image_np.ndim}D array"

        image_lst = self.postprocess(image_np)
        return image_lst

    def is_valid_model(self, model: str) -> bool:
        return model in self._models

    def postprocess(self, image_np: np.ndarray) -> list[np.ndarray]:
        image_np = (image_np * 255).round().astype("uint8")
        image_lst = np.split(image_np, image_np.shape[0], axis=0)
        image_lst = [img.squeeze(0) for img in image_lst]
        return image_lst
