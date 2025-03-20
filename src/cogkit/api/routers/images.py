# -*- coding: utf-8 -*-


import base64
import time
from http import HTTPStatus
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from cogkit.api.dependencies import get_image_generation_service
from cogkit.api.models.images import ImageGenerationParams, ImageInResponse, ImagesResponse
from cogkit.api.services import ImageGenerationService

router = APIRouter()


def np_to_base64(image_array: np.ndarray) -> str:
    byte_stream = image_array.tobytes()
    base64_str = base64.b64encode(byte_stream).decode("utf-8")
    return base64_str


@router.post("/generations", response_model=ImagesResponse)
def generations(
    image_generation: Annotated[ImageGenerationService, Depends(get_image_generation_service)],
    params: ImageGenerationParams,
) -> ImagesResponse:
    if not image_generation.is_valid_model(params.model):
        return HTTPException(
            status_code=HTTPStatus.NOT_FOUND.value,
            detail=f"The model `{params.model}` does not exist.",
        )
    image_lst = image_generation.generate(
        model=params.model, prompt=params.prompt, size=params.size, num_images=params.n
    )
    image_b64_lst = [ImageInResponse(b64_json=np_to_base64(image)) for image in image_lst]
    return ImagesResponse(created=int(time.time()), data=image_b64_lst)
