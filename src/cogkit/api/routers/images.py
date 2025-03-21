# -*- coding: utf-8 -*-


import base64
import io
import time
from http import HTTPStatus
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from PIL import Image

from cogkit.api.dependencies import get_image_generation_service
from cogkit.api.models.images import ImageGenerationParams, ImageInResponse, ImagesResponse
from cogkit.api.services import ImageGenerationService

router = APIRouter()


def np_to_base64(image_array: np.ndarray) -> str:
    image = Image.fromarray(image_array)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@router.post("/generations", response_model=ImagesResponse)
def generations(
    image_generation: Annotated[ImageGenerationService, Depends(get_image_generation_service)],
    params: ImageGenerationParams,
) -> ImagesResponse:
    if not image_generation.is_valid_model(params.model):
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"The model `{params.model}` does not exist. Supported models: {image_generation.supported_models}",
        )
    # TODO: add exception handling
    image_lst = image_generation.generate(
        model=params.model, prompt=params.prompt, size=params.size, num_images=params.n
    )
    image_b64_lst = [ImageInResponse(b64_json=np_to_base64(image)) for image in image_lst]
    return ImagesResponse(created=int(time.time()), data=image_b64_lst)
