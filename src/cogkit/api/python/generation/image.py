# -*- coding: utf-8 -*-


from typing import Literal

import numpy as np
import torch
from PIL import Image

from cogkit.logging import get_logger
from cogkit.utils import (
    rand_generator,
)
from diffusers import DiffusionPipeline

from .util import before_generation, guess_resolution

_logger = get_logger(__name__)


def generate_image(
    prompt: str,
    pipeline: DiffusionPipeline,
    num_images_per_prompt: int = 1,
    output_type: Literal["pil", "pt", "np"] = "pil",
    load_type: Literal["cuda", "cpu_model_offload", "sequential_cpu_offload"] = "cpu_model_offload",
    height: int | None = None,
    width: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.5,
    seed: int | None = None,
) -> list[Image.Image] | torch.Tensor | np.ndarray:
    """Generates images from a text prompt using a diffusion model pipeline.

    This function leverages a diffusion pipeline to create images based on a given text prompt. It supports
    customization of image dimensions, inference steps, and guidance scale, as well as optional LoRA (Low-Rank
    Adaptation) fine-tuning. The output can be returned in different formats: PIL images, PyTorch tensors, or
    NumPy arrays.

    Args:
        - prompt: The text description used to guide the image generation process.
        - pipeline: Preloaded DiffusionPipeline instance.
        - num_images_per_prompt: Number of images to generate per prompt. Defaults to 1.
        - output_type: Format of the output images. Options are "pil" (PIL.Image), "pt" (PyTorch tensor), or
            "np" (NumPy array). Defaults to "pil".
        - load_type: Type of offloading to use for the model, use "cuda" if you have enough GPU memory. Defaults to "cpu_model_offload".
        - height: Desired height of the output images in pixels. If None, inferred from the pipeline.
        - width: Desired width of the output images in pixels. If None, inferred from the pipeline.
        - num_inference_steps: Number of denoising steps during generation. Defaults to 50.
        - guidance_scale: Strength of the prompt guidance (classifier-free guidance scale). Defaults to 3.5.
        - seed: Optional random seed for reproducible results. Defaults to None.

    Returns:
        A list of generated images in the specified format:
        - If output_type is "pil": List of PIL.Image.Image objects.
        - If output_type is "pt": PyTorch tensor of shape (num_images, 3, height, width) with dtype torch.uint8.
        - If output_type is "np": NumPy array of shape (num_images, height, width, 3) with dtype uint8.
    """

    height, width = guess_resolution(pipeline, height, width)

    _logger.info(f"Generation config: height {height}, width {width}.")

    before_generation(pipeline, load_type)

    output = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=rand_generator(seed),
        output_type=output_type,
    ).images

    if output_type != "pil":
        output = (output * 255).round()
        if output_type == "pt":
            assert output.ndim == 4, f"Expected 4D numpy array, got {output.ndim}D array"
            assert output.shape[1] == 3, f"Expected 3 channels, got {output.shape[3]} channels"
            output = output.to(torch.uint8)
        elif output_type == "np":
            assert output.ndim == 4, f"Expected 4D torch tensor, got {output.ndim}D torch tensor"
            # Dim of image_np: (num_images, height, width, 3)
            assert output.shape[3] == 3, f"Expected 3 channels, got {output.shape[3]} channels"
            output = output.astype("uint8")

    return output
