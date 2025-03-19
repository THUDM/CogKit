# -*- coding: utf-8 -*-


from cogkit.api.models.response import ResponseBody


class ImageInResponse(ResponseBody):
    b64_json: str | None = None

    revised_prompt: str | None = None

    url: str | None = None


class ImagesResponse(ResponseBody):
    created: int
    data: list[ImageInResponse]
