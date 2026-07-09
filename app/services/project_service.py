from __future__ import annotations

import logging

from app.core.exceptions import LLMGenerationError, RoadmapNotFoundError
from app.prompts import project_prompts
from app.schemas.project import ProjectLLMOutput, ProjectRequest, ProjectResponse
from app.services.graphs.structured_output_graph import (
    StructuredGenerationError,
    run_structured_generation,
)
from app.services.llm_client import LLMClient
from app.storage.repository import RoadmapRepository

logger = logging.getLogger(__name__)


def recommend_project(
    request: ProjectRequest,
    llm_client: LLMClient,
    roadmap_repository: RoadmapRepository,
    max_retries: int,
) -> ProjectResponse:
    if request.roadmap_id:
        roadmap = roadmap_repository.get(request.roadmap_id)
        if roadmap is None:
            raise RoadmapNotFoundError(f"No roadmap found with id '{request.roadmap_id}'.")
        goal_title = roadmap.goal_title
        skills = roadmap.skills
        roadmap_context = (
            f"Total roadmap time: {roadmap.estimated_hours}h. "
            f"Tasks: {', '.join(t.title for t in roadmap.tasks)}."
        )
    else:
        goal_title = request.goal_title  # validated non-None by ProjectRequest model_validator
        skills = request.skills
        roadmap_context = None

    system_prompt = project_prompts.SYSTEM_PROMPT
    user_prompt = project_prompts.build_user_prompt(goal_title, skills, roadmap_context)

    try:
        llm_output: ProjectLLMOutput = run_structured_generation(
            llm_client=llm_client,
            schema=ProjectLLMOutput,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_retries=max_retries,
        )
    except StructuredGenerationError as exc:
        logger.error("project generation failed: %s", exc)
        raise LLMGenerationError(
            "Failed to generate a valid project recommendation from the LLM after multiple attempts.",
            details={"last_error": exc.last_error},
        ) from exc

    return ProjectResponse(**llm_output.model_dump())
