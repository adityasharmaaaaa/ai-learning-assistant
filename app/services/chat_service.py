from __future__ import annotations

import logging

from app.core.exceptions import LLMGenerationError, RoadmapNotFoundError
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.graphs.chat_graph import build_chat_graph
from app.services.llm_client import LLMClient
from app.services.rag.retriever import RoadmapRetriever
from app.storage.repository import ChatHistoryRepository, RoadmapRepository

logger = logging.getLogger(__name__)


def chat(
    request: ChatRequest,
    llm_client: LLMClient,
    roadmap_repository: RoadmapRepository,
    retriever: RoadmapRetriever,
    history_repository: ChatHistoryRepository,
    max_history_turns: int,
    max_retries: int,
) -> ChatResponse:
    roadmap = roadmap_repository.get(request.roadmap_id)
    if roadmap is None:
        raise RoadmapNotFoundError(f"No roadmap found with id '{request.roadmap_id}'.")

    history = history_repository.recent(request.roadmap_id, max_history_turns)

    graph = build_chat_graph(llm_client, retriever, history_repository, max_retries)
    final_state = graph.invoke(
        {
            "roadmap": roadmap,
            "message": request.message,
            "history": history,
            "retrieved_context": [],
            "result": None,
        }
    )

    if final_state["result"] is None:
        raise LLMGenerationError(
            "Failed to generate a valid chat response from the LLM after multiple attempts."
        )

    return ChatResponse(**final_state["result"].model_dump())
