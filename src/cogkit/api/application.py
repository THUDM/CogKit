# -*- coding: utf-8 -*-


from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from cogkit.api.logging import get_logger
from cogkit.api.routers import api_router
from cogkit.api.services import ImageGenerationService
from cogkit.api.settings import APISettings
from cogkit.api.state import RequestState

_logger = get_logger(__name__)


def get_application(settings: APISettings | None = None) -> FastAPI:
    settings = settings or APISettings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[RequestState]:
        yield {"image_generation": ImageGenerationService(settings.cogview4_path)}

    app = FastAPI(lifespan=lifespan)

    app.include_router(api_router, prefix="/v1")

    @app.exception_handler(Exception)
    async def handle_uncaught_exception(_: Request, exc: Exception) -> JSONResponse:
        # ! handles uncaught exceptions
        # ! Normally this function shouldn't be reached.
        # ! Exceptions are supposed to be caught in the controllers.
        _logger.exception(
            "Uncaught exception: %s",
            exc.__class__.__name__,
        )
        return JSONResponse(
            content={
                "error": {
                    "type": exc.__class__.__name__,
                }
            },
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    return app
