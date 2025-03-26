# -*- coding: utf-8 -*-


from diffusers import DiffusionPipeline
from PIL import Image

from cogkit.logging import get_logger
from cogkit.types import GenerationMode

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
    image: Image.Image | None,
) -> None:
    if generation_mode is not None and generation_mode != GenerationMode.TextToImage:
        _logger.warning(
            "The pipeline `%s` does not support `%s` task. Will try the `%s` task.",
            pl_cls_name,
            generation_mode.value,
            GenerationMode.TextToImage,
        )
    if image is not None:
        _logger.warning(
            "The pipeline `%s` does not support image input. The input image will be ignored.",
            pl_cls_name,
        )


def _check_image_to_video_params(
    pl_cls_name: str,
    generation_mode: GenerationMode | None,
    image: Image.Image | None,
) -> None:
    if generation_mode is not None and generation_mode != GenerationMode.ImageToVideo:
        _logger.warning(
            "The pipeline `%s` does not support `%s` task. Will try the `%s` task.",
            pl_cls_name,
            generation_mode.value,
            GenerationMode.ImageToVideo,
        )
    if image is not None:
        err_msg = f"Image input is required in the image2video pipeline. Please provide a regular image file (image_file = {image})."
        raise ValueError(err_msg)


def guess_generation_mode(
    pipeline: DiffusionPipeline,
    generation_mode: str | GenerationMode | None = None,
    image: Image.Image | None = None,
) -> GenerationMode:
    pl_cls_name = pipeline.__class__.__name__

    if pl_cls_name not in _SUPPORTED_PIPELINE:
        err_msg = f"The pipeline '{pl_cls_name}' is not supported."
        raise ValueError(err_msg)
    if generation_mode is not None:
        generation_mode = GenerationMode(generation_mode)

    if pl_cls_name.startswith("CogView"):
        # TextToImage
        _check_text_to_image_params(pl_cls_name, generation_mode, image)
        return GenerationMode.TextToImage

    if pl_cls_name == "CogVideoXImageToVideoPipeline":
        _check_image_to_video_params(pl_cls_name, generation_mode, image)
        return GenerationMode.ImageToVideo

    if image is not None:
        _logger.warning(
            "Pipeline `%s` does not support image input. Will ignore the image file.",
            pl_cls_name,
        )

    return GenerationMode.TextToVideo
