"""
Dependency wiring.

Singletons (LLM client, embedder, vector store) are created once via
lru_cache and reused across requests -- each takes no parameters and pulls
settings via the cached get_settings() internally, so FastAPI never tries to
interpret them as request-body parameters when used with Depends(...).
Per-request objects (DB session, repositories) are created fresh via
FastAPI's Depends chain.

Tests override `get_llm_client` and `get_embedder` (see tests/conftest.py)
via `app.dependency_overrides` to inject FakeLLMClient / HashEmbedder
instead of hitting Groq / downloading model weights.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from app.config import get_settings
from app.services.llm_client import GroqLLMClient, LLMClient
from app.services.rag.embeddings import Embedder, build_embedder
from app.services.rag.retriever import RoadmapRetriever
from app.services.rag.vector_store import VectorStore
from app.storage.db import get_session
from app.storage.repository import ChatHistoryRepository, RoadmapRepository


@lru_cache
def get_llm_client() -> LLMClient:
    settings = get_settings()
    return GroqLLMClient(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        timeout_s=settings.llm_request_timeout_s,
    )


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return build_embedder(settings.embedding_provider, settings.embedding_model)


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(settings.vector_index_dir)


def get_retriever(
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
) -> RoadmapRetriever:
    settings = get_settings()
    return RoadmapRetriever(vector_store, embedder, settings.retrieval_top_k)


def get_roadmap_repository(session: Session = Depends(get_session)) -> RoadmapRepository:
    return RoadmapRepository(session)


def get_chat_history_repository(session: Session = Depends(get_session)) -> ChatHistoryRepository:
    return ChatHistoryRepository(session)
