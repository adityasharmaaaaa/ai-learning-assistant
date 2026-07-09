"""
Custom exception hierarchy + FastAPI exception handlers.

Every domain-level failure raises one of these instead of a bare Exception,
so the API always returns a consistent, documented error envelope instead of
leaking a stack trace or an opaque 500.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for all handled application errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class RoadmapNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "roadmap_not_found"


class InvalidRequestError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "invalid_request"


class LLMGenerationError(AppError):
    """Raised when the LLM fails to produce valid structured output after all retries."""

    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "llm_generation_failed"


class VectorStoreError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "vector_store_error"


def _error_payload(error_code: str, message: str, details: dict | None = None) -> dict:
    return {"error": {"code": error_code, "message": message, "details": details or {}}}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("%s: %s | details=%s", exc.error_code, exc.message, exc.details)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.error_code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.info("request validation failed: %s", exc.errors())
        safe_errors = jsonable_encoder(exc.errors(), custom_encoder={Exception: str})
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_payload(
                "request_validation_error",
                "The request body failed validation.",
                {"errors": safe_errors},
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled exception while processing %s", request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload("internal_error", "An unexpected error occurred."),
        )
