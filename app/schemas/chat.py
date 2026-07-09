from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    roadmap_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=1000)


class ChatLLMOutput(BaseModel):
    response: str = Field(..., min_length=1, max_length=2000)
    follow_up_questions: list[str] = Field(default_factory=list, max_length=5)


class ChatResponse(ChatLLMOutput):
    pass


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str
