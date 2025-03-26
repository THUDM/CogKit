# -*- coding: utf-8 -*-


from functools import partial
from typing import Any, List, Literal

import numpy as np
import torch
from PIL import Image

from cogkit.logging import get_logger
from cogkit.types import GenerationMode
from cogkit.utils import (
    guess_generation_mode,
    rand_generator,
)
from diffusers import DiffusionPipeline
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput

from .util import before_generation, guess_frames, guess_resolution

_logger = get_logger(__name__)


def _cast_to_pipeline_output(output: Any) -> CogVideoXPipelineOutput:
    if isinstance(output, CogVideoXPipelineOutput):
        return output
    if isinstance(output, tuple):
        return CogVideoXPipelineOutput(frames=output[0])

    err_msg = f"Cannot cast a `{output.__class__.__name__}` to a `CogVideoXPipelineOutput`."
    raise ValueError(err_msg)


def generate_video(
    prompt: str,
    pipeline: DiffusionPipeline,
    num_videos_per_prompt: int = 1,
    output_type: Literal["pil", "pt", "np"] = "pil",
    input_image: Image.Image | None = None,
    # * params for model loading
    load_type: Literal["cuda", "cpu_model_offload", "sequential_cpu_offload"] = "cpu_model_offload",
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int | None = 42,
) -> tuple[List[Image.Image] | torch.Tensor | np.ndarray, int]:
    """Main function for video generation, supporting both text-to-video and image-to-video generation modes.

    Args:
        - prompt (str): Text prompt describing the desired video content.
        - pipeline (DiffusionPipeline): Pre-loaded diffusion model pipeline.
        - num_videos_per_prompt (int, optional): Number of videos to generate per prompt. Defaults to 1.
        - output_type (Literal, optional): Output type, one of "pil", "pt", or "np". Defaults to "pil".
        - input_image (Image.Image | None, optional): Input image for image-to-video generation. Defaults to None.
        - load_type (Literal, optional): Model loading type, one of "cuda", "cpu_model_offload", or
            "sequential_cpu_offload". Defaults to "cpu_model_offload".
        - height (int | None, optional): Height of output video. If None, will be inferred. Defaults to None.
        - width (int | None, optional): Width of output video. If None, will be inferred. Defaults to None.
        - num_frames (int | None, optional): Number of frames in generated video. If None, will be inferred.
            Defaults to None.
        - num_inference_steps (int, optional): Number of inference steps. Defaults to 50.
        - guidance_scale (float, optional): Classifier guidance scale. Defaults to 6.0.
        - seed (int | None, optional): Random seed for generation. Defaults to 42.

    Returns:
        tuple[torch.Tensor, int]: Returns a tuple containing:
            - Generated video tensor with shape (num_videos, num_frames, height, width, 3)
            - Video frame rate (fps)

    Raises:
        ValueError: When provided generation mode is unknown or output cannot be cast to CogVideoXPipelineOutput.
        AssertionError: When both pipeline and model_id_or_path are None or both are provided.

    Note:
        - Either pipeline or model_id_or_path must be provided, but not both.
        - If lora_model_id_or_path is provided, LoRA weights will be loaded and applied.
        - Height, width, number of frames, and fps will be automatically inferred if not specified.
    """

    task = guess_generation_mode(
        pipeline=pipeline,
        generation_mode=None,
        image_file=input_image,
    )

    height, width = guess_resolution(pipeline, height, width)
    num_frames, fps = guess_frames(pipeline, num_frames)

    _logger.info(
        f"Generation config: height {height}, width {width}, num_frames {num_frames}, fps {fps}."
    )

    before_generation(pipeline, load_type)

    pipeline_fn = partial(
        pipeline,
        height=height,
        width=width,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        output_type=output_type,
        generator=rand_generator(seed),
    )
    if task == GenerationMode.TextToVideo:
        pipeline_out = pipeline_fn()
    elif task == GenerationMode.ImageToVideo:
        # pipeline_out = pipeline_fn(image=Image.open(input_image))
        pipeline_out = pipeline_fn(image=input_image)
    else:
        err_msg = f"Unknown generation mode: {task.value}"
        raise ValueError(err_msg)

    batch_video = _cast_to_pipeline_output(pipeline_out).frames

    if output_type in ("pt", "np"):
        # Dim of a video: (num_videos, num_frames, 3, height, width)
        assert batch_video.ndim == 5, f"Expected 5D array, got {batch_video[0].ndim}D array"
        assert batch_video.shape[2] == 3, (
            f"Expected 3 channels, got {batch_video[0].shape[2]} channels"
        )
    return batch_video, fps
