# -*- coding: utf-8 -*-


from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore", validate_default=True, validate_assignment=True, env_file=".env"
    )
    _supported_models: tuple[str, ...] = ("cogview-4",)

    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    offload_type: Literal["cuda", "cpu_model_offload", "sequential_cpu_offload"] = (
        "cpu_model_offload"
    )

    # cogview-4 related settings
    cogview4_path: str | None = None
    cogview4_transformer_path: str | None = None
