from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.storage.db import Base


class RoadmapRecord(Base):
    __tablename__ = "roadmaps"

    roadmap_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    goal_title: Mapped[str] = mapped_column(String(200))
    payload: Mapped[dict] = mapped_column(JSON)  # full RoadmapResponse, JSON-serialized
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class ChatMessageRecord(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    roadmap_id: Mapped[str] = mapped_column(String(36), index=True)
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
