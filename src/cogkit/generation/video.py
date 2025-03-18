# -*- coding: utf-8 -*-


import os
from functools import partial
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.utils import export_to_video
from PIL import Image

from cogkit.generation.util import before_generation, guess_resolution
from cogkit.logging import get_logger
from cogkit.types import GenerationMode
from cogkit.utils import (
    load_lora_checkpoint,
    mkdir,
    rand_generator,
    resolve_path,
)

_logger = get_logger(__name__)


def _cast_to_pipeline_output(output: Any) -> CogVideoXPipelineOutput:
    if isinstance(output, CogVideoXPipelineOutput):
        return output
    if isinstance(output, tuple):
        return CogVideoXPipelineOutput(frames=output[0])

    err_msg = f"Cannot cast a `{output.__class__.__name__}` to a `CogVideoXPipelineOutput`."
    raise ValueError(err_msg)


def generate_video(
    task: GenerationMode,
    prompt: str,
    model_id_or_path: str,
    output_file: str | Path,
    image_file: str | Path | None = None,
    video_file: str | Path | None = None,
    # * params for model loading
    dtype: torch.dtype = torch.bfloat16,
    transformer_path: str | None = None,
    lora_model_id_or_path: str | None = None,
    lora_rank: int = 128,
    # * params for generated videos
    height: int | None = None,
    width: int | None = None,
    num_frames: int = 1,
    fps: int = 16,
    # * params for the generation process
    num_videos_per_prompt: int = 1,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int | None = 42,
) -> None:
    # FIXME: check the value of num_frames as documented in the model card
    # FIXME: check if this line can correctly load t2v pipeline?
    pipeline = DiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=dtype)

    if transformer_path is not None:
        pipeline.transformer.save_config(transformer_path)
        pipeline.transformer = pipeline.transformer.from_pretrained(transformer_path)
    if lora_model_id_or_path is not None:
        load_lora_checkpoint(lora_model_id_or_path, pipeline, lora_rank)
    height, width = guess_resolution(pipeline, height, width)

    before_generation(pipeline)

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
        generator=rand_generator(seed),
    )
    if task == GenerationMode.TextToVideo:
        pipeline_out = pipeline_fn()
    elif task == GenerationMode.ImageToVideo:
        pipeline_out = pipeline_fn(image=Image.open(image_file))
    elif task == GenerationMode.VideoToVideo:
        # FIXME: reads video file
        pipeline_out = pipeline_fn(video=video_file)
    else:
        err_msg = f"Unknown generation mode: {task.value}"
        raise ValueError(err_msg)

    output_file = resolve_path(output_file)
    mkdir(output_file.parent)
    _logger.info("Saving the generated video to path '%s'.", os.fspath(output_file))
    batch_video = _cast_to_pipeline_output(pipeline_out).frames
    export_to_video(batch_video[0], output_file, fps=fps)