from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.schemas.chat import ChatTurn
from app.schemas.roadmap import RoadmapResponse
from app.storage.models import ChatMessageRecord, RoadmapRecord


class RoadmapRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def save(self, roadmap: RoadmapResponse) -> None:
        record = RoadmapRecord(
            roadmap_id=roadmap.roadmap_id,
            goal_title=roadmap.goal_title,
            payload=roadmap.model_dump(),
        )
        self.session.merge(record)
        self.session.commit()

    def get(self, roadmap_id: str) -> RoadmapResponse | None:
        record = self.session.get(RoadmapRecord, roadmap_id)
        if record is None:
            return None
        return RoadmapResponse.model_validate(record.payload)

    def exists(self, roadmap_id: str) -> bool:
        return self.session.get(RoadmapRecord, roadmap_id) is not None


class ChatHistoryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def append(self, roadmap_id: str, role: str, content: str) -> None:
        self.session.add(ChatMessageRecord(roadmap_id=roadmap_id, role=role, content=content))
        self.session.commit()

    def recent(self, roadmap_id: str, limit: int) -> list[ChatTurn]:
        stmt = (
            select(ChatMessageRecord)
            .where(ChatMessageRecord.roadmap_id == roadmap_id)
            .order_by(ChatMessageRecord.id.desc())
            .limit(limit)
        )
        rows = list(self.session.execute(stmt).scalars())
        rows.reverse()
        return [ChatTurn(role=r.role, content=r.content) for r in rows]
