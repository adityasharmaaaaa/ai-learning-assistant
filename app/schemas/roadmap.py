from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

ExperienceLevel = Literal[
    "Less than 1 year", "1-2 years", "2-4 years", "4+ years"
]
LearningStyle = Literal["Project Based", "Theory First", "Video Based", "Mixed"]


class RoadmapRequest(BaseModel):
    goal_title: str = Field(..., min_length=2, max_length=100, examples=["Backend Developer"])
    experience: ExperienceLevel = Field(..., examples=["Less than 1 year"])
    known_skills: list[str] = Field(default_factory=list, max_length=50)
    learning_style: LearningStyle = Field(..., examples=["Project Based"])
    weekly_hours: int = Field(..., ge=1, le=80, examples=[15])

    @field_validator("known_skills")
    @classmethod
    def dedupe_skills(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v if s.strip()]
        # de-dupe while preserving order
        seen: set[str] = set()
        result = []
        for skill in cleaned:
            key = skill.lower()
            if key not in seen:
                seen.add(key)
                result.append(skill)
        return result

    @field_validator("goal_title")
    @classmethod
    def strip_goal_title(cls, v: str) -> str:
        return v.strip()


class Subtask(BaseModel):
    title: str = Field(..., min_length=2, max_length=150)


class Task(BaseModel):
    title: str = Field(..., min_length=2, max_length=150)
    estimated_hours: int = Field(..., ge=1, le=200)
    subtasks: list[Subtask] = Field(default_factory=list)


class RoadmapLLMOutput(BaseModel):
    """
    Schema the LLM is asked to produce. Kept separate from RoadmapResponse
    because the LLM should never be trusted to invent the roadmap_id — that
    is assigned server-side after validation succeeds.
    """

    estimated_hours: int = Field(..., ge=1, le=2000)
    skills: list[str] = Field(..., min_length=1, max_length=30)
    tasks: list[Task] = Field(..., min_length=1, max_length=30)


class RoadmapResponse(RoadmapLLMOutput):
    roadmap_id: str = Field(default_factory=lambda: str(uuid4()))
    # Not part of the LLM's output -- stamped on by the service layer from the
    # original request so the roadmap is self-describing once persisted and
    # can be reused as context by /project and /chat. Additive extra field,
    # backward compatible with the spec's example response.
    goal_title: str = ""
