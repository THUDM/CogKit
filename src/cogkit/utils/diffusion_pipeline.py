# -*- coding: utf-8 -*-


from typing import TypedDict

from diffusers import DiffusionPipeline


class PipelineMeta(TypedDict):
    cls_name: str | None


def get_pipeline_meta(pipeline: DiffusionPipeline) -> PipelineMeta:
    cls_name = pipeline.__class__.__name__
    return PipelineMeta(cls_name=cls_name)
