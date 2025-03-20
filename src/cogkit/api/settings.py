# -*- coding: utf-8 -*-


from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore", validate_default=True, validate_assignment=True
    )
    _supported_models: tuple[str] = ("cogview-4",)
    cogview4_path: str | None = None
