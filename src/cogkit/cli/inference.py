# -*- coding: utf-8 -*-


from pathlib import Path
from typing import Literal

import click

from cogkit.generation import generate_image, generate_video
from cogkit.types import GenerationMode
from cogkit.utils import cast_to_torch_dtype, guess_generation_mode


@click.command()
@click.option(
    "--output_file",
    type=click.Path(dir_okay=False, writable=True),
    help="the path to save the generated image (or video). If not provided, the generated image (or video) will be saved to 'output.png' (or 'output.mp4').",
)
@click.option(
    "--task",
    type=click.Choice(
        choices=[mode.value for mode in GenerationMode],
        case_sensitive=False,
    ),
    help="the generation task",
)
@click.option(
    "--image_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="the image to guide the video generation (NOT EFFECTIVE in the image generation task)",
)
@click.option(
    "--video_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="the video to guide the video generation (NOT EFFECTIVE in the image generation task)",
)
@click.option(
    "--dtype",
    type=click.Choice(choices=["bfloat16", "float16"], case_sensitive=False),
    default="bfloat16",
    help="the data type used in the computation",
)
# FIXME: support model_id?
@click.option(
    "--transformer_path",
    type=click.Path(file_okay=False, exists=True),
    default=None,
    help="the path to load the transformer model",
)
@click.option("--lora_model_id_or_path", help="the id or the path of the LoRA weights")
@click.option(
    "--lora_rank",
    type=click.IntRange(min=1),
    default=128,
    help="the rank of the LoRA weights",
)
@click.option(
    "--height",
    type=click.IntRange(min=1),
    help="the height of the generated image/video",
)
@click.option(
    "--width",
    type=click.IntRange(min=1),
    help="the width of the generated image/video",
)
@click.option(
    "--num_frames",
    type=click.IntRange(min=1),
    default=81,
    help="the number of the frames in the generated video (NOT EFFECTIVE in the image generation task)",
)
@click.option(
    "--fps",
    type=click.IntRange(min=1),
    default=16,
    help="the frames per second of the generated video (NOT EFFECTIVE in the image generation task)",
)
@click.option("--seed", type=int, help="the seed for reproducibility")
@click.argument("prompt")
@click.argument("model_id_or_path")
def inference(
    prompt: str,
    model_id_or_path: str,
    output_file: str | Path | None = None,
    task: GenerationMode | None = None,
    # * additional input
    image_file: str | Path | None = None,
    video_file: str | Path | None = None,
    # * params for model loading
    dtype: Literal["bfloat16", "float16"] = "bfloat16",
    transformer_path: str | None = None,
    lora_model_id_or_path: str | None = None,
    lora_rank: int = 128,
    # * params for output
    height: int | None = None,
    width: int | None = None,
    num_frames: int = 81,
    fps: int = 16,
    seed: int = 42,
) -> None:
    """
    Generates an image (or video) based on the given prompt.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_id_or_path (str): The path of the pre-trained model to be used.
    - task (GenerationMode): The type of generation task to be performed (e.g., 't2v', 'i2v', 'v2v', 't2i').
    - output_file (str | Path): The path where the generated image or video will be saved.
    - image_file (str | Path | None): The path of the image to be used as the background of the video (if applicable).
    - video_file (str | Path | None): The path of the video to be used as the background of the video (if applicable).
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - lora_model_id_or_path (str | None): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - fps (int): The frames per second for the generated video.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - seed (int): The seed for reproducibility.
    """

    task = guess_generation_mode(model_id_or_path, task, image_file, video_file)
    dtype = cast_to_torch_dtype(dtype)

    if task in (
        GenerationMode.TextToVideo,
        GenerationMode.ImageToVideo,
    ):
        generate_video(
            task,
            prompt,
            model_id_or_path,
            output_file or "output.mp4",
            image_file,
            video_file,
            dtype=dtype,
            transformer_path=transformer_path,
            lora_model_id_or_path=lora_model_id_or_path,
            lora_rank=lora_rank,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
        )
    elif task in (GenerationMode.TextToImage,):
        generate_image(
            prompt,
            model_id_or_path,
            output_file or "output.png",
            dtype=dtype,
            transformer_path=transformer_path,
            height=height,
            width=width,
            seed=seed,
        )
    else:
        err_msg = f"Unknown generation task: {task.value}"
        raise ValueError(err_msg)
