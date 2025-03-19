# -*- coding: utf-8 -*-


from diffusers import CogView4Pipeline


class ImageGenerationService(object):
    def __init__(self, cogview4_path: str | None) -> None:
        self._models = {}
        if cogview4_path is not None:
            self._models["cogview-4"] = CogView4Pipeline.from_pretrained(cogview4_path)
