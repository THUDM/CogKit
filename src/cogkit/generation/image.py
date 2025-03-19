# -*- coding: utf-8 -*-


import os
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

from cogkit.generation.util import before_generation, guess_resolution
from cogkit.logging import get_logger
from cogkit.utils import mkdir, rand_generator, resolve_path

_logger = get_logger(__name__)


def generate_image(
    prompt: str,
    model_id_or_path: str,
    output_file: str | Path,
    # * params for model loading
    dtype: torch.dtype = torch.bfloat16,
    transformer_path: str | None = None,
    # * params for generated images
    height: int | None = None,
    width: int | None = None,
    # * params for the generation process
    num_inference_steps: int = 50,
    guidance_scale: float = 3.5,
    seed: int | None = 42,
):
    pipeline = DiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=dtype)

    if transformer_path is not None:
        pipeline.transformer.save_config(transformer_path)
        pipeline.transformer = pipeline.transformer.from_pretrained(transformer_path)

    height, width = guess_resolution(pipeline, height, width)

    _logger.info(f"Generation config: height {height}, width {width}.")

    before_generation(pipeline)

    batch_image = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        generator=rand_generator(seed),
    ).images

    output_file = resolve_path(output_file)
    mkdir(output_file.parent)
    _logger.info("Saving the generated image to path '%s'.", os.fspath(output_file))
    batch_image[0].save(output_file)
