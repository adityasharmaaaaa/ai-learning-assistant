"""
Structured (schema-aware) chunking of a roadmap into retrieval units.

We deliberately do NOT run the roadmap JSON through a generic
character/token text splitter (e.g. RecursiveCharacterTextSplitter). The
roadmap is structured data, not prose: naive character splitting would sever
a task from its own subtasks or its hour estimate mid-sentence, degrading
retrieval quality. Instead each chunk is built to be a self-contained,
semantically coherent unit:

  - one "summary" chunk: goal-level info (skills, total hours) so the model
    always has global context available even if no task-level chunk matches,
  - one chunk per task: title + hours + its subtasks, so a query like
    "how long will Docker take" retrieves the whole relevant unit.

Each chunk carries metadata (chunk_type, task_index) so the retriever can
reason about *what* was retrieved, not just raw text.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.roadmap import RoadmapResponse


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_roadmap(roadmap: RoadmapResponse) -> list[Chunk]:
    chunks: list[Chunk] = []

    summary_text = (
        f"Goal: {roadmap.goal_title}. "
        f"Total estimated time: {roadmap.estimated_hours} hours. "
        f"Skills covered: {', '.join(roadmap.skills)}."
    )
    chunks.append(Chunk(text=summary_text, metadata={"chunk_type": "summary"}))

    for idx, task in enumerate(roadmap.tasks):
        subtask_lines = "; ".join(st.title for st in task.subtasks) or "no subtasks listed"
        task_text = (
            f"Task: {task.title}. Estimated time: {task.estimated_hours} hours. "
            f"Subtasks: {subtask_lines}."
        )
        chunks.append(
            Chunk(
                text=task_text,
                metadata={"chunk_type": "task", "task_index": idx, "task_title": task.title},
            )
        )

    return chunks
