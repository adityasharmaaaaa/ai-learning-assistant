from __future__ import annotations

import os
import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
os.environ.setdefault("EMBEDDING_PROVIDER", "hash")  # offline, no model download in CI

from app.config import get_settings  # noqa: E402
from app.dependencies import get_embedder, get_llm_client, get_vector_store  # noqa: E402
from app.services.llm_client import FakeLLMClient  # noqa: E402
from app.services.rag.embeddings import HashEmbedder  # noqa: E402
from app.services.rag.vector_store import VectorStore  # noqa: E402
from app.storage.db import init_db  # noqa: E402


@pytest.fixture()
def temp_dirs():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture()
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture()
def app_client(temp_dirs, fake_llm):
    get_settings.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{temp_dirs}/test.db"
    os.environ["VECTOR_INDEX_DIR"] = f"{temp_dirs}/vector_indexes"

    from app.main import create_app

    init_db(os.environ["DATABASE_URL"])
    app = create_app()

    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()
    app.dependency_overrides[get_vector_store] = lambda: VectorStore(os.environ["VECTOR_INDEX_DIR"])

    with TestClient(app) as client:
        yield client, fake_llm

    get_settings.cache_clear()
