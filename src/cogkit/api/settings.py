# -*- coding: utf-8 -*-


from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore", validate_default=True, validate_assignment=True, env_file=".env"
    )
    _supported_models: tuple[str, ...] = ("cogview-4",)
    cogview4_path: str | None = None
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    offload_type: Literal["cpu_model_offolad", "no_offload"] = "no_offload"
    openai_api_key: str | None = None
