"""
FAISS-backed per-roadmap vector store.

Design notes (documented for the README's "retrieval strategy" write-up):
  - One FAISS index per roadmap_id, persisted to disk under
    VECTOR_INDEX_DIR/{roadmap_id}/ so it survives process restarts.
  - An in-process LRU-ish cache avoids re-embedding + re-loading the index on
    every single /chat call for the same roadmap (this doubles as our
    "Product Thinking" caching angle for the hot path, alongside the
    conversation-history feature).
  - FAISS is appropriate here: each roadmap's knowledge base is small
    (tens of chunks), so an exact in-memory index is both fast and simple.
    At real production scale (many roadmaps, shared infra, need for
    filtering/multi-tenancy) this would move to a managed/persistent vector
    DB such as pgvector or Pinecone -- the `VectorStore` interface below is
    the seam where that swap would happen.
"""
from __future__ import annotations

import logging
import os
import threading

from app.services.rag.chunking import Chunk
from app.services.rag.embeddings import Embedder

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, index_dir: str) -> None:
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self._cache: dict[str, object] = {}
        self._lock = threading.Lock()

    def _path(self, roadmap_id: str) -> str:
        return os.path.join(self.index_dir, roadmap_id)

    def build_and_persist(self, roadmap_id: str, chunks: list[Chunk], embedder: Embedder) -> None:
        from langchain_community.vectorstores import FAISS

        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        store = FAISS.from_texts(texts=texts, embedding=embedder, metadatas=metadatas)

        with self._lock:
            store.save_local(self._path(roadmap_id))
            self._cache[roadmap_id] = store
        logger.info("vector_store: built + persisted index for roadmap_id=%s (%d chunks)", roadmap_id, len(chunks))

    def load(self, roadmap_id: str, embedder: Embedder):
        with self._lock:
            if roadmap_id in self._cache:
                return self._cache[roadmap_id]

        from langchain_community.vectorstores import FAISS

        path = self._path(roadmap_id)
        if not os.path.isdir(path):
            return None

        store = FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)
        with self._lock:
            self._cache[roadmap_id] = store
        logger.info("vector_store: loaded persisted index for roadmap_id=%s from disk", roadmap_id)
        return store

    def similarity_search(self, roadmap_id: str, query: str, k: int, embedder: Embedder) -> list[Chunk]:
        store = self.load(roadmap_id, embedder)
        if store is None:
            return []
        results = store.similarity_search(query, k=k)
        return [Chunk(text=doc.page_content, metadata=doc.metadata) for doc in results]

    def invalidate_cache(self, roadmap_id: str) -> None:
        with self._lock:
            self._cache.pop(roadmap_id, None)
