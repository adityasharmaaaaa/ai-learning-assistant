from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import (
    get_chat_history_repository,
    get_llm_client,
    get_retriever,
    get_roadmap_repository,
)
from app.schemas.chat import ChatRequest, ChatResponse
from app.services import chat_service
from app.services.llm_client import LLMClient
from app.services.rag.retriever import RoadmapRetriever
from app.storage.repository import ChatHistoryRepository, RoadmapRepository

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, status_code=200)
def post_chat(
    payload: ChatRequest,
    llm_client: LLMClient = Depends(get_llm_client),
    roadmap_repository: RoadmapRepository = Depends(get_roadmap_repository),
    retriever: RoadmapRetriever = Depends(get_retriever),
    history_repository: ChatHistoryRepository = Depends(get_chat_history_repository),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    return chat_service.chat(
        request=payload,
        llm_client=llm_client,
        roadmap_repository=roadmap_repository,
        retriever=retriever,
        history_repository=history_repository,
        max_history_turns=settings.max_history_turns,
        max_retries=settings.max_llm_retries,
    )
