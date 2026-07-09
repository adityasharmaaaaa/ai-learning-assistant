from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, health, project, roadmap
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging_config import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware
from app.storage.db import init_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db(settings.database_url)
    if not settings.groq_api_key:
        logger.warning(
            "GROQ_API_KEY is not set -- /roadmap, /project, and /chat will fail until it is "
            "configured in your .env file (see .env.example)."
        )
    logger.info("Starting %s in '%s' mode (model=%s)", settings.app_name, settings.environment, settings.groq_model)
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Backend service that generates personalized learning roadmaps, "
        "recommends projects, and answers roadmap questions via a RAG chat assistant.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(roadmap.router)
    app.include_router(project.router)
    app.include_router(chat.router)

    return app


app = create_app()
