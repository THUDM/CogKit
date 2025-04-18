from typing import Literal

from pydantic import ValidationInfo, field_validator
from typing_extensions import override

from ...base.base_args import BaseArgs


class DiffusionArgs(BaseArgs):
    ########## Model ##########
    model_type: Literal["i2v", "t2v", "t2i"]

    ########## Output ##########
    tracker_name: str = "diffusion-tracker"

    ########## Training #########
    # For cogview models, train_resolution is a tuple of (height, width)
    # For cogvideo models, train_resolution is a tuple of (frames, height, width)
    train_resolution: tuple[int, int] | tuple[int, int, int]

    enable_slicing: bool = True
    enable_tiling: bool = True

    ########## Packing #########
    enable_packing: bool = False

    ########## Validation ##########
    gen_fps: int = 15

    @field_validator("train_resolution")
    def validate_train_resolution(
        cls, v: tuple[int, int] | tuple[int, int, int], info: ValidationInfo
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
                "train_resolution must be a tuple of (height, width) for cogview models or (frames, height, width) for cogvideo models"
            )

        return v

    @override
    @classmethod
    def parse_args(cls):
        parser = cls.get_base_parser()

        # Required arguments
        parser.add_argument("--model_type", type=str, required=True)
        parser.add_argument("--train_resolution", type=str, required=True)

        # Model configuration
        parser.add_argument("--enable_slicing", action="store_true")
        parser.add_argument("--enable_tiling", action="store_true")

        # Packing
        parser.add_argument("--enable_packing", type=lambda x: x.lower() == "true", default=False)

        # Validation
        parser.add_argument("--gen_fps", type=int, default=15)

        args = parser.parse_args()

        # Convert train_resolution string to tuple
        parts = args.train_resolution.split("x")
        if len(parts) == 2:
            height, width = parts
            args.train_resolution = (int(height), int(width))
        else:
            frames, height, width = parts
            args.train_resolution = (int(frames), int(height), int(width))

        return cls(**vars(args))
