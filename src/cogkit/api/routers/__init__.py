# -*- coding: utf-8 -*-


from fastapi import APIRouter

from cogkit.api.routers import images

api_router = APIRouter()
api_router.include_router(images.router, prefix="/images")
