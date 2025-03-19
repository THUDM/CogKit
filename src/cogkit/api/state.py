# -*- coding: utf-8 -*-


from typing import TypedDict

from cogkit.api.services import ImageGenerationService


class RequestState(TypedDict):
    image_generation: ImageGenerationService
