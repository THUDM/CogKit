from pathlib import Path

from pydantic import ValidationInfo, field_validator
from typing_extensions import override

from ...base.base_args import BaseArgs


class DiffusionArgs(BaseArgs):
    ########## Training #########
    # For cogview models, train_resolution is a list of (height, width)
    # For cogvideo models, train_resolution is a list of (frames, height, width)
    train_resolution: list[int, int] | list[int, int, int]

    enable_slicing: bool = True
    enable_tiling: bool = True

    ########## Packing #########
    enable_packing: bool = False

    ########## Validation ##########
    gen_fps: int | None = None

    @field_validator("train_resolution")
    def validate_train_resolution(
        cls, v: list[int, int] | list[int, int, int], info: ValidationInfo
    ) -> str:
        if len(v) == 2:  # cogview models
            height, width = v

        elif len(v) == 3:  # cogvideo models
            frames, height, width = v
            # Check if (frames - 1) is multiple of 8
            if (frames - 1) % 8 != 0:
                raise ValueError("Number of frames - 1 must be a multiple of 8")

            # Check resolution for cogvideox-5b models
            model_name = info.data.get("model_name", "")
            if model_name in ["cogvideox-5b-i2v", "cogvideox-5b-t2v"]:
                if (height, width) != (480, 720):
                    raise ValueError(
                        "For cogvideox-5b models, height must be 480 and width must be 720"
                    )
        else:
            raise ValueError(
                "train_resolution must be a list of (height, width) for cogview models or (frames, height, width) for cogvideo models"
            )

        return v

    @override
    @classmethod
    def parse_from_yaml(cls, fpath: str | Path) -> "DiffusionArgs":
        return super().parse_from_yaml(fpath)
