# -*- coding: utf-8 -*-


from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
    CogView4Pipeline,
)

from cogkit.logging import get_logger

_logger = get_logger(__name__)

TVideoPipeline = CogVideoXPipeline | CogVideoXImageToVideoPipeline | CogVideoXVideoToVideoPipeline
TPipeline = CogView4Pipeline | TVideoPipeline

def _validate_dimensions(min_scale: int,
                         max_scale: int,
                         mod: int,
                         width: int,
                         height: int,
                         power: int | None = None,
                         ) -> bool:
    if not (min_scale <= width <= max_scale and min_scale <= height <= max_scale):
        _logger.info("width or height out of range: %s <= height, width <= %s", min_scale, max_scale)
        return False
    if power:
        if width * height > 2**power:
            _logger.info("width * height exceeds the limit: width * height <= 2^%s", power)
            return False
    
    if width % mod != 0 or height % mod != 0:
        _logger.info("width or height is not a multiple of %s: height, width \mod %s = 0", mod, mod)
        return False
    
    return True

def _guess_cogview_resolution(
    pipeline: CogView4Pipeline, height: int | None = None, width: int | None = None
) -> tuple[int, int]:
    default_height = pipeline.transformer.config.sample_size * pipeline.vae_scale_factor
    default_width = pipeline.transformer.config.sample_size * pipeline.vae_scale_factor
    if height is None and width is None:
        return default_height, default_width

    if height is None:
        height = int(width * default_height / default_width)

    if width is None:
        width = int(height * default_width / default_height)

    # FIXME: checks if `(height, width)` is reasonable. If not, warn users and return the default/recommend resolution when required.
    if width and height:
        if not _validate_dimensions(
            min_scale=512,
            max_scale=2048,
            mod=32,
            width=width,
            height=height,
            power=21,
        ):
            return default_height, default_width
    return height, width


def _guess_cogvideox_resolution(
    pipeline: TVideoPipeline, height: int | None, width: int | None = None
) -> tuple[int, int]:
    default_height = pipeline.transformer.config.sample_height * pipeline.vae_scale_factor_spatial
    default_width = pipeline.transformer.config.sample_width * pipeline.vae_scale_factor_spatial

    if height is None and width is None:
        return default_height, default_width

    if height is None:
        height = int(width * default_height / default_width)

    if width is None:
        width = int(height * default_width / default_height)

    # FIXME: checks if `(height, width)` is reasonable. If not, warn users and return the default/recommend resolution when required.
    if width and height:
        if not _validate_dimensions(
                min_scale=768,
                max_scale=1360,
                mod=16,
                height=height,
                width=width,
            ):
            return default_height, default_width
    return height, width


def guess_resolution(
    pipeline: TPipeline,
    height: int | None = None,
    width: int | None = None,
) -> tuple[int, int]:
    if isinstance(pipeline, CogView4Pipeline):
        return _guess_cogview_resolution(pipeline, height=height, width=width)
    if isinstance(pipeline, TVideoPipeline):
        return _guess_cogvideox_resolution(pipeline, height=height, width=width)

    err_msg = f"The pipeline '{pipeline.__class__.__name__}' is not supported."
    raise ValueError(err_msg)


def before_generation(pipeline: TPipeline) -> None:
    if isinstance(pipeline, TVideoPipeline):
        pipeline.scheduler = CogVideoXDPMScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )

    # * enables CPU offload for the model.
    # turns off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")
    # pipe.to("cuda")

    # pipeline.to("cuda")
    pipeline.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    if hasattr(pipeline, "vae"):
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
