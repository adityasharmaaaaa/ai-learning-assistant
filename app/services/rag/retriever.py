"""
High-level retriever: ensures a roadmap's chunks are embedded/indexed, then
answers similarity queries against them.

Retrieval strategy: top-k similarity search (k=RETRIEVAL_TOP_K) over
task-level chunks, with the roadmap's summary chunk always included as extra
context regardless of similarity score. This ensures the model always has
goal-level context (total hours, full skill list) even when the learner asks
a question that only fuzzily matches a specific task chunk.
"""
from __future__ import annotations

import logging

from app.schemas.roadmap import RoadmapResponse
from app.services.rag.chunking import Chunk, chunk_roadmap
from app.services.rag.embeddings import Embedder
from app.services.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RoadmapRetriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder, top_k: int) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def index_roadmap(self, roadmap: RoadmapResponse) -> None:
        chunks = chunk_roadmap(roadmap)
        self.vector_store.build_and_persist(roadmap.roadmap_id, chunks, self.embedder)

    def ensure_indexed(self, roadmap: RoadmapResponse) -> None:
        """Index the roadmap if it hasn't been indexed yet (idempotent, cheap check)."""
        if self.vector_store.load(roadmap.roadmap_id, self.embedder) is None:
            self.index_roadmap(roadmap)

    def retrieve(self, roadmap: RoadmapResponse, query: str) -> list[str]:
        self.ensure_indexed(roadmap)
        results: list[Chunk] = self.vector_store.similarity_search(
            roadmap.roadmap_id, query, self.top_k, self.embedder
        )

        texts = [c.text for c in results]
        summary_text = next(
            (c.text for c in chunk_roadmap(roadmap) if c.metadata.get("chunk_type") == "summary"), None
        )
        if summary_text and summary_text not in texts:
            texts.insert(0, summary_text)

        return texts
