# -*- coding: utf-8 -*-


from pathlib import Path
from typing import Any

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
    "CogView4ControlPipeline",
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


def _check_control_text_to_image_params(
    pl_cls_name: str,
    generation_mode: GenerationMode | None,
    image: str | Path | None,
) -> None:
    if generation_mode is not None and generation_mode != GenerationMode.CtrlTextToImage:
        _logger.warning(
            "The pipeline `%s` does not support `%s` task. Will try the `%s` task.",
            pl_cls_name,
            generation_mode.value,
            GenerationMode.CtrlTextToImage,
        )
    if image is not None:
        err_msg = f"Image input is required in the image2video pipeline. Please provide a regular image file (image_file = {image})."
        raise ValueError(err_msg)


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
    if image is None:
        err_msg = f"Image input is required in the image2video pipeline. Please provide a regular image file (image_file = {image})."
        raise ValueError(err_msg)


def guess_generation_mode(
    pipeline_or_path: DiffusionPipeline | str,
    generation_mode: str | GenerationMode | None = None,
    image: Image.Image | None = None,
) -> GenerationMode:
    if isinstance(pipeline_or_path, str):
        pl_cls_name = get_pipeline_meta(pipeline_or_path)["cls_name"]
    else:
        pl_cls_name = pipeline_or_path.__class__.__name__

    if pl_cls_name not in _SUPPORTED_PIPELINE:
        err_msg = f"The pipeline '{pl_cls_name}' is not supported."
        raise ValueError(err_msg)
    if generation_mode is not None:
        generation_mode = GenerationMode(generation_mode)

    if pl_cls_name == "CogView4Pipeline":
        # TextToImage
        _check_text_to_image_params(pl_cls_name, generation_mode, image)
        return GenerationMode.TextToImage

    if pl_cls_name == "CogVideoXImageToVideoPipeline":
        _check_image_to_video_params(pl_cls_name, generation_mode, image)
        return GenerationMode.ImageToVideo

    if pl_cls_name == "CogView4ControlPipeline":
        # Control TextToImage
        _check_control_text_to_image_params(pl_cls_name, generation_mode, image)
        return GenerationMode.CtrlTextToImage

    if image is not None:
        _logger.warning(
            "Pipeline `%s` does not support image input. Will ignore the image file.",
            pl_cls_name,
        )

    return GenerationMode.TextToVideo


def flatten_dict(d: dict[str, Any], ignore_none: bool = False) -> dict[str, Any]:
    """
    Flattens a nested dictionary into a single layer dictionary.

    Args:
        d: The dictionary to flatten
        ignore_none: If True, keys with None values will be omitted

    Returns:
        A flattened dictionary

    Raises:
        ValueError: If there are duplicate keys across nested dictionaries

    Examples:
        >>> flatten_dict({"a": 1, "b": {"c": 2, "d": {"e": 3}}, "f": None})
        {"a": 1, "c": 2, "e": 3, "f": None}

        >>> flatten_dict({"a": 1, "b": {"c": 2, "d": {"e": 3}}, "f": None}, ignore_none=True)
        {"a": 1, "c": 2, "e": 3}

        >>> flatten_dict({"a": 1, "b": {"a": 2}})
        ValueError: Duplicate key 'a' found in nested dictionary
    """
    result = {}

    def _flatten(current_dict, result_dict):
        for key, value in current_dict.items():
            if value is None and ignore_none:
                continue

            if isinstance(value, dict):
                _flatten(value, result_dict)
            else:
                if key in result_dict:
                    raise ValueError(f"Duplicate key '{key}' found in nested dictionary")
                result_dict[key] = value

    _flatten(d, result)
    return result


def expand_list(dicts: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    Converts a list of dictionaries into a dictionary of lists.

    For each key in the dictionaries, collects all values corresponding to that key
    into a list.

    Args:
        dicts: A list of dictionaries

    Returns:
        A dictionary where each key maps to a list of values from the input dictionaries

    Examples:
        >>> expand_list([{"a": 1, "b": 2}, {"a": 3, "b": 4, "c": 5}])
        {"a": [1, 3], "b": [2, 4], "c": [5]}

        >>> expand_list([{"x": "value1"}, {"y": "value2"}, {"x": "value3"}])
        {"x": ["value1", "value3"], "y": ["value2"]}

        >>> expand_list([{"x": ["value1", "value2"]}, {"y": "value3"}, {"x": ["value4"]}])
        {"x": ["value1", "value2", "value4"], "y": ["value3"]}
    """
    result = {}

    for d in dicts:
        for key, value in d.items():
            if key not in result:
                result[key] = []

            if isinstance(value, list):
                result[key].extend(value)
            else:
                result[key].append(value)

    return result
