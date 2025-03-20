# -*- coding: utf-8 -*-


from typing import Literal

from pydantic import field_validator

from cogkit.api.models.request import RequestParams
from cogkit.api.settings import APISettings


settings = APISettings()


class ImageGenerationParams(RequestParams):
    prompt: str
    model: str
    n: int = 1
    size: Literal[
        "1024x1024", "768x1344", "864x1152", "1344x768", "1152x864", "1440x720", "720x1440"
    ] = "1024x1024"
    user: str | None = None
    # ! unsupported parameters
    # quality: Literal["standard", "hd"] = "standard"
    # response_format: Literal["url", "b64_json"] = "url"
    # style: Literal["vivid", "natural"] = "vivid"

    @field_validator("model")
    def validate_model(cls, v):
        if v not in settings.supported_models:
            raise ValueError(
                f"Model {v} not supported, supported list: {settings.supported_models}"
            )
        return v
