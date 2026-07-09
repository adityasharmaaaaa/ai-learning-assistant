from __future__ import annotations

import logging

from app.core.exceptions import LLMGenerationError
from app.prompts import roadmap_prompts
from app.schemas.roadmap import RoadmapLLMOutput, RoadmapRequest, RoadmapResponse
from app.services.graphs.structured_output_graph import (
    StructuredGenerationError,
    run_structured_generation,
)
from app.services.llm_client import LLMClient
from app.services.rag.retriever import RoadmapRetriever
from app.storage.repository import RoadmapRepository

logger = logging.getLogger(__name__)


def generate_roadmap(
    request: RoadmapRequest,
    llm_client: LLMClient,
    repository: RoadmapRepository,
    retriever: RoadmapRetriever,
    max_retries: int,
) -> RoadmapResponse:
    system_prompt = roadmap_prompts.SYSTEM_PROMPT
    user_prompt = roadmap_prompts.build_user_prompt(request)

    try:
        llm_output: RoadmapLLMOutput = run_structured_generation(
            llm_client=llm_client,
            schema=RoadmapLLMOutput,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_retries=max_retries,
        )
    except StructuredGenerationError as exc:
        logger.error("roadmap generation failed: %s", exc)
        raise LLMGenerationError(
            "Failed to generate a valid roadmap from the LLM after multiple attempts.",
            details={"last_error": exc.last_error},
        ) from exc

    roadmap = RoadmapResponse(goal_title=request.goal_title, **llm_output.model_dump())
    repository.save(roadmap)

    # Eagerly build the RAG index so the first /chat call for this roadmap
    # doesn't pay the embedding-cold-start cost.
    try:
        retriever.index_roadmap(roadmap)
    except Exception:  # noqa: BLE001 - indexing failure shouldn't fail the roadmap response
        logger.exception("failed to eagerly index roadmap %s for RAG; will retry on first chat", roadmap.roadmap_id)

    return roadmap
