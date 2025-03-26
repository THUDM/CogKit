# -*- coding: utf-8 -*-


import os
from pathlib import Path
from typing import Literal

import click
from diffusers.utils import export_to_video
from PIL import Image

from cogkit.api.python import generate_image, generate_video
from cogkit.logging import get_logger
from cogkit.types import GenerationMode
from cogkit.utils import (
    cast_to_torch_dtype,
    guess_generation_mode,
    load_lora_checkpoint,
    load_pipeline,
    mkdir,
    resolve_path,
    unload_lora_checkpoint,
)

_logger = get_logger(__name__)


@click.command()
@click.option(
    "--output_file",
    type=click.Path(dir_okay=False, writable=True),
    help="the path to save the generated image/video. If not provided, the generated image/video will be saved to 'output.png/mp4'.",
)
@click.option(
    "--image_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="the image to guide the image/video generation (for i2i/i2v generation task)",
)
@click.option(
    "--dtype",
    type=click.Choice(choices=["bfloat16", "float16"], case_sensitive=False),
    default="bfloat16",
    help="the data type used in the computation, default is 'bfloat16'",
)
@click.option(
    "--transformer_path",
    type=click.Path(file_okay=False, exists=True),
    default=None,
    help="the path to load the transformer model, default is None",
)
@click.option("--lora_model_id_or_path", help="the id or the path of the LoRA weights")
@click.option(
    "--load_type",
    type=click.Choice(
        choices=["cuda", "cpu_model_offload", "sequential_cpu_offload"], case_sensitive=False
    ),
    default="cpu_model_offload",
    help="the type of offloading to use for the model, from fastest to slowest and lowest to highest GPU memory usage. default is 'cpu_model_offload'.",
)
@click.option(
    "--height",
    type=click.IntRange(min=1),
    default=None,
    help="the height of the generated image/video, will be inferred from the model if not provided",
)
@click.option(
    "--width",
    type=click.IntRange(min=1),
    default=None,
    help="the width of the generated image/video, will be inferred from the model if not provided",
)
@click.option(
    "--num_inference_steps",
    type=click.IntRange(min=1),
    default=50,
    help="the number of inference steps, default is 50",
)
@click.option("--seed", type=int, help="the seed for reproducibility")
@click.argument("prompt")
@click.argument("model_id_or_path")
def inference(
    prompt: str,
    model_id_or_path: str,
    output_file: str | Path | None = None,
    # * additional input
    image_file: str | Path | None = None,
    # * params for model loading
    dtype: Literal["bfloat16", "float16"] = "bfloat16",
    transformer_path: str | None = None,
    lora_model_id_or_path: str | None = None,
    load_type: Literal["cuda", "cpu_model_offload", "sequential_cpu_offload"] = "cpu_model_offload",
    # * params for output
    height: int | None = None,
    width: int | None = None,
    num_inference_steps: int = 50,
    seed: int = 42,
) -> None:
    """
    Generates an image (or video) based on the given prompt.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_id_or_path (str): The path of the pre-trained model to be used.
    - output_file (str | Path): The path where the generated image(.png) or video(.mp4) will be saved.
    - image_file (str | Path | None): The path of the image (for i2v generation task).
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - transformer_path (str | None): The path to load the transformer model.
    - lora_model_id_or_path (str | None): The path of the LoRA weights to be used.
    - load_type (str): The type of offloading to use for the model.
    - height (int | None): The height of the generated image/video.
    - width (int | None): The width of the generated image/video.
    - num_inference_steps (int): The number of inference steps.
    - seed (int): The seed for reproducibility.
    """

    dtype = cast_to_torch_dtype(dtype)
    pipeline = load_pipeline(model_id_or_path, transformer_path, dtype)
    image = None
    # TODO: No need to load the image every time. Some generation tasks cannot handle images.
    if image_file is not None:
        image = Image.open(image_file)
    task = guess_generation_mode(pipeline=pipeline, image=image)

    if lora_model_id_or_path is not None:
        load_lora_checkpoint(pipeline, lora_model_id_or_path)
    else:
        unload_lora_checkpoint(pipeline)

    if task in (
        GenerationMode.TextToVideo,
        GenerationMode.ImageToVideo,
    ):
        output, fps = generate_video(
            prompt=prompt,
            pipeline=pipeline,
            num_videos_per_prompt=1,
            output_type="pil",
            input_image=image,
            load_type=load_type,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        assert len(output) == 1

        output_file = output_file or "output.mp4"
        output_file = resolve_path(output_file)
        mkdir(output_file.parent)
        _logger.info("Saving the generated video to path '%s'.", os.fspath(output_file))
        export_to_video(output[0], output_file, fps=fps)

    elif task in (GenerationMode.TextToImage,):
        batched_images = generate_image(
            prompt=prompt,
            pipeline=pipeline,
            num_images_per_prompt=1,
            output_type="pil",
            load_type=load_type,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        assert len(batched_images) == 1

        output_file = output_file or "output.png"
        output_file = resolve_path(output_file)
        mkdir(output_file.parent)
        _logger.info("Saving the generated image to path '%s'.", os.fspath(output_file))
        batched_images[0].save(output_file)
    else:
        err_msg = f"Unknown generation task: {task.value}"
        raise ValueError(err_msg)
