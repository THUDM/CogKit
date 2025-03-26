# -*- coding: utf-8 -*-


from PIL import Image

from cogkit.logging import get_logger
from cogkit.types import GenerationMode
from cogkit.utils.diffusion_pipeline import get_pipeline_meta
from diffusers import DiffusionPipeline

_logger = get_logger(__name__)

_SUPPORTED_PIPELINE = (
    "CogView3PlusPipeline",
    "CogView4Pipeline",
    "CogVideoXPipeline",
    "CogVideoXImageToVideoPipeline",
)


def _check_text_to_image_params(
    pl_cls_name: str,
    generation_mode: GenerationMode | None,
    image_file: Image.Image | None,
) -> None:
    if generation_mode is not None and generation_mode != GenerationMode.TextToImage:
        _logger.warning(
            "The pipeline `%s` does not support `%s` task. Will try the `%s` task.",
            pl_cls_name,
            generation_mode.value,
            GenerationMode.TextToImage,
        )
    if image_file is not None:
        _logger.warning(
            "The pipeline `%s` does not support image input. The image file will be ignored.",
            pl_cls_name,
        )


def _check_image_to_video_params(
    pl_cls_name: str,
    generation_mode: GenerationMode | None,
    image_file: Image.Image | None,
) -> None:
    if generation_mode is not None and generation_mode != GenerationMode.ImageToVideo:
        _logger.warning(
            "The pipeline `%s` does not support `%s` task. Will try the `%s` task.",
            pl_cls_name,
            generation_mode.value,
            GenerationMode.ImageToVideo,
        )
    if image_file is not None:
        err_msg = f"Image input is required in the image2video pipeline. Please provide a regular image file (image_file = {image_file})."
        raise ValueError(err_msg)


def guess_generation_mode(
    pipeline: DiffusionPipeline,
    generation_mode: str | GenerationMode | None = None,
    image_file: Image.Image | None = None,
) -> GenerationMode:
    pipeline_meta = get_pipeline_meta(pipeline)
    pl_cls_name = pipeline_meta.get("cls_name", None)
    if pl_cls_name is None:
        err_msg = f"Failed to parse the pipeline configuration (pipeline_cls = {pl_cls_name})."
        raise ValueError(err_msg)

    if pl_cls_name not in _SUPPORTED_PIPELINE:
        err_msg = f"The pipeline '{pl_cls_name}' is not supported."
        raise ValueError(err_msg)
    if generation_mode is not None:
        generation_mode = GenerationMode(generation_mode)

    if pl_cls_name.startswith("CogView"):
        # TextToImage
        _check_text_to_image_params(pl_cls_name, generation_mode, image_file)
        return GenerationMode.TextToImage

    if pl_cls_name == "CogVideoXImageToVideoPipeline":
        _check_image_to_video_params(pl_cls_name, generation_mode, image_file)
        return GenerationMode.ImageToVideo

    if image_file is not None:
        _logger.warning(
            "Pipeline `%s` does not support image input. Will ignore the image file.",
            pl_cls_name,
        )

    return GenerationMode.TextToVideo
