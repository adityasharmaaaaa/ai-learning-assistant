"""
Embedding provider abstraction.

Default provider is a local sentence-transformers model (all-MiniLM-L6-v2)
via langchain_huggingface -- it runs on CPU, needs no external API key or
per-call cost, and is a reasonable production choice at this scale. A
`HashEmbedder` fallback is provided for fully offline unit tests / CI
environments where downloading model weights isn't possible or desired; it
is NOT semantically meaningful and must never be used in production
(enforced via EMBEDDING_PROVIDER config, defaulting to "huggingface").
"""
from __future__ import annotations

import hashlib
import logging
from typing import Protocol

from langchain_core.embeddings import Embeddings as LangchainEmbeddings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class HuggingFaceEmbedder(LangchainEmbeddings):
    def __init__(self, model_name: str) -> None:
        from langchain_huggingface import HuggingFaceEmbeddings

        self._impl = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._impl.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._impl.embed_query(text)


class HashEmbedder(LangchainEmbeddings):
    """
    Deterministic, dependency-free "embedding" built from token hashing into
    a fixed-size bag-of-words vector. Good enough to exercise the retrieval
    *pipeline* in tests without network access; not semantically meaningful
    and not suitable for production ranking quality.
    """

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for token in text.lower().split():
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dims
            vec[idx] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


def build_embedder(provider: str, model_name: str) -> Embedder:
    if provider == "huggingface":
        return HuggingFaceEmbedder(model_name)
    if provider == "hash":
        logger.warning("Using HashEmbedder -- offline/test mode only, not for production retrieval quality.")
        return HashEmbedder()
    raise ValueError(f"Unknown embedding provider: {provider}")
