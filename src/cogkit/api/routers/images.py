# -*- coding: utf-8 -*-


import time
from typing import Annotated

from fastapi import APIRouter, Depends

from cogkit.api.dependencies import get_image_generation_service
from cogkit.api.models.images import ImageGenerationParams, ImagesResponse
from cogkit.api.services import ImageGenerationService

router = APIRouter()


@router.post("/generations", response_class=ImagesResponse)
def generations(
    image_generation: Annotated[ImageGenerationService, Depends(get_image_generation_service)],
    params: ImageGenerationParams,
) -> ImagesResponse:
    # TODO: completes this function
    images = image_generation.generate(
        model=params.model, prompt=params.prompt, size=params.size, num_images=params.n
    )
    return ImagesResponse(created=int(time.time()), data=[])
