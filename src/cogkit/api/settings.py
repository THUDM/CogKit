# -*- coding: utf-8 -*-


from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore", validate_default=True, validate_assignment=True
    )
    cogview4_path: str = "THUDM/CogView4-6B"
