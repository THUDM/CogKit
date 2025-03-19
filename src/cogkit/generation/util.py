# -*- coding: utf-8 -*-


from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogView4Pipeline,
)

TVideoPipeline = CogVideoXPipeline | CogVideoXImageToVideoPipeline
TPipeline = CogView4Pipeline | TVideoPipeline


def _is_cogvideox1_0(pipeline: TVideoPipeline) -> bool:
    if isinstance(pipeline, CogVideoXPipeline) or isinstance(
        pipeline, CogVideoXImageToVideoPipeline
    ):
        if (
            not hasattr(pipeline.transformer.config, "patch_size_t")
            or pipeline.transformer.config.patch_size_t is None
        ):
            return True
        else:
            return False
    else:
        raise ValueError(
            f"Unsupported pipeline type in `_is_cogvideox1_0`, pipeline type: {type(pipeline)}"
        )


def _is_cogvideox1_5(pipeline: TVideoPipeline) -> bool:
    if isinstance(pipeline, CogVideoXPipeline) or isinstance(
        pipeline, CogVideoXImageToVideoPipeline
    ):
        return (
            hasattr(pipeline.transformer.config, "patch_size_t")
            and pipeline.transformer.config.patch_size_t == 2
        )
    else:
        raise ValueError(
            f"Unsupported pipeline type in `_is_cogvideox1_5`, pipeline type: {type(pipeline)}"
        )


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

    ### Check resolution according to the model card
    assert height is not None and width is not None
    if isinstance(pipeline, CogView4Pipeline):
        assert height % 32 == 0 and width % 32 == 0, "height and width must be divisible by 32"
    else:
        raise ValueError(
            f"Unsupported pipeline type in `_guess_cogview_resolution`, pipeline type: {type(pipeline)}"
        )

    return height, width


def _guess_cogvideox_resolution(
    pipeline: TVideoPipeline, height: int | None, width: int | None = None
) -> tuple[int, int]:
    default_height = pipeline.transformer.config.sample_height * pipeline.vae_scale_factor_spatial
    default_width = pipeline.transformer.config.sample_width * pipeline.vae_scale_factor_spatial

    if height is None and width is None:
        height, width = default_height, default_width
    elif height is None:
        height = int(width * default_height / default_width)
    elif width is None:
        width = int(height * default_width / default_height)

    ##### Check resolution according to the model card
    if _is_cogvideox1_0(pipeline):
        assert height == 480 and width == 720, "height and width must be 480 and 720"
    elif _is_cogvideox1_5(pipeline):
        if isinstance(pipeline, CogVideoXPipeline):
            assert height == 768 and width == 1360, "height and width must be 768 and 1360"
        elif isinstance(pipeline, CogVideoXImageToVideoPipeline):
            minv = min(height, width)
            maxv = max(height, width)
            assert minv == 768, "minimum value in (height, width) must be 768"
            assert (
                768 <= maxv <= 1360
            ), "maximum value in (height, width) must range from 768 to 1360"
            assert maxv % 16 == 0, "maximum value in (height, width) must be divisible by 16"
    else:
        raise ValueError(
            f"Unsupported pipeline type in `_guess_cogvideox_resolution`, pipeline type: {type(pipeline)}"
        )

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


def guess_frames(pipeline: TVideoPipeline, frames: int | None = None) -> tuple[int, int]:
    if frames is None:
        frames = pipeline.transformer.config.sample_frames

    ##### Check frames according to model card
    if _is_cogvideox1_0(pipeline):
        assert frames <= 49, "frames must <=49"
        assert (frames - 1) % 8 == 0, "frames must be 8N+1"
        fps = 8
    elif _is_cogvideox1_5(pipeline):
        assert frames <= 81, "frames must <=81"
        assert (frames - 1) % 16 == 0, "frames must be 16N+1"
        fps = 16
    else:
        raise ValueError(
            f"Unsupported pipeline type in `guess_frames`, pipeline type: {type(pipeline)}"
        )

    return frames, fps


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
