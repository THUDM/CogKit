# -*- coding: utf-8 -*-


from fastapi import Request

from cogkit.api.services import ImageGenerationService


async def get_image_generation_service(request: Request) -> ImageGenerationService:
    if "image_generation" not in request.state:
        err_msg = "The image service is unavailable."
        raise AttributeError(err_msg)
    return request.state["image_generation"]
