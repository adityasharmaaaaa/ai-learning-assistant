from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_llm_client, get_roadmap_repository
from app.schemas.project import ProjectRequest, ProjectResponse
from app.services import project_service
from app.services.llm_client import LLMClient
from app.storage.repository import RoadmapRepository

router = APIRouter(tags=["project"])


@router.post("/project", response_model=ProjectResponse, status_code=201)
def create_project(
    payload: ProjectRequest,
    llm_client: LLMClient = Depends(get_llm_client),
    roadmap_repository: RoadmapRepository = Depends(get_roadmap_repository),
    settings: Settings = Depends(get_settings),
) -> ProjectResponse:
    return project_service.recommend_project(
        request=payload,
        llm_client=llm_client,
        roadmap_repository=roadmap_repository,
        max_retries=settings.max_llm_retries,
    )
