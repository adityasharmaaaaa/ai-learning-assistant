from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_llm_client, get_retriever, get_roadmap_repository
from app.schemas.roadmap import RoadmapRequest, RoadmapResponse
from app.services import roadmap_service
from app.services.llm_client import LLMClient
from app.services.rag.retriever import RoadmapRetriever
from app.storage.repository import RoadmapRepository

router = APIRouter(tags=["roadmap"])


@router.post("/roadmap", response_model=RoadmapResponse, status_code=201)
def create_roadmap(
    payload: RoadmapRequest,
    llm_client: LLMClient = Depends(get_llm_client),
    repository: RoadmapRepository = Depends(get_roadmap_repository),
    retriever: RoadmapRetriever = Depends(get_retriever),
    settings: Settings = Depends(get_settings),
) -> RoadmapResponse:
    return roadmap_service.generate_roadmap(
        request=payload,
        llm_client=llm_client,
        repository=repository,
        retriever=retriever,
        max_retries=settings.max_llm_retries,
    )
