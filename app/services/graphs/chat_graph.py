"""
RAG chat pipeline as an explicit LangGraph.

    retrieve_context --> generate_answer --> persist_history --> END

`generate_answer` delegates to the shared generate->validate->repair
subgraph (structured_output_graph) for reliable structured output, so the
retry/repair policy is defined once and reused everywhere in the app.
"""
from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.schemas.chat import ChatLLMOutput, ChatTurn
from app.schemas.roadmap import RoadmapResponse
from app.services.graphs.structured_output_graph import (
    StructuredGenerationError,
    run_structured_generation,
)
from app.services.llm_client import LLMClient
from app.services.rag.retriever import RoadmapRetriever
from app.storage.repository import ChatHistoryRepository
from app.prompts import chat_prompts

logger = logging.getLogger(__name__)


class ChatGraphState(TypedDict):
    roadmap: RoadmapResponse
    message: str
    history: list[ChatTurn]
    retrieved_context: list[str]
    result: ChatLLMOutput | None


def build_chat_graph(
    llm_client: LLMClient,
    retriever: RoadmapRetriever,
    history_repo: ChatHistoryRepository,
    max_retries: int,
):
    def retrieve_context(state: ChatGraphState) -> ChatGraphState:
        state["retrieved_context"] = retriever.retrieve(state["roadmap"], state["message"])
        logger.info("chat_graph: retrieved %d context chunks", len(state["retrieved_context"]))
        return state

    def generate_answer(state: ChatGraphState) -> ChatGraphState:
        user_prompt = chat_prompts.build_user_prompt(
            message=state["message"],
            retrieved_context=state["retrieved_context"],
            history=state["history"],
        )
        try:
            result = run_structured_generation(
                llm_client=llm_client,
                schema=ChatLLMOutput,
                system_prompt=chat_prompts.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_retries=max_retries,
            )
            state["result"] = result
        except StructuredGenerationError:
            state["result"] = None
        return state

    def persist_history(state: ChatGraphState) -> ChatGraphState:
        if state["result"] is not None:
            history_repo.append(state["roadmap"].roadmap_id, "user", state["message"])
            history_repo.append(state["roadmap"].roadmap_id, "assistant", state["result"].response)
        return state

    graph = StateGraph(ChatGraphState)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("persist_history", persist_history)

    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "generate_answer")
    graph.add_edge("generate_answer", "persist_history")
    graph.add_edge("persist_history", END)

    return graph.compile()
