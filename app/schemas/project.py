from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Difficulty = Literal["Beginner", "Intermediate", "Advanced"]


class ProjectRequest(BaseModel):
    """
    Accepts EITHER a roadmap_id (preferred — reuses generated context) OR a
    standalone (goal_title, skills) pair for ad-hoc recommendations without
    first generating a roadmap.
    """

    roadmap_id: str | None = None
    goal_title: str | None = Field(default=None, min_length=2, max_length=100)
    skills: list[str] | None = Field(default=None, max_length=50)

    @model_validator(mode="after")
    def check_one_mode_provided(self) -> "ProjectRequest":
        has_roadmap = bool(self.roadmap_id)
        has_adhoc = bool(self.goal_title) and bool(self.skills)
        if not has_roadmap and not has_adhoc:
            raise ValueError(
                "Provide either 'roadmap_id', or both 'goal_title' and 'skills'."
            )
        return self


class ProjectLLMOutput(BaseModel):
    title: str = Field(..., min_length=2, max_length=150)
    difficulty: Difficulty
    estimated_hours: int = Field(..., ge=1, le=500)
    tech_stack: list[str] = Field(..., min_length=1, max_length=20)
    features: list[str] = Field(..., min_length=1, max_length=20)
    why_this_project: str = Field(..., min_length=10, max_length=500)


class ProjectResponse(ProjectLLMOutput):
    pass
