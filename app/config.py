"""
Centralized application configuration.

All environment-driven configuration lives here. Nothing else in the codebase
should call os.environ directly — this keeps configuration auditable and
makes it trivial to override settings in tests (see tests/conftest.py).
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "AI Learning Assistant"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # --- LLM (Groq) ---
    groq_api_key: str = Field(default="", description="Groq API key. Required in production.")
    groq_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Groq production model id. Groq deprecated llama-3.3-70b-versatile in "
        "June 2026; openai/gpt-oss-120b is the current recommended general-purpose flagship.",
    )
    llm_temperature: float = 0.3
    llm_request_timeout_s: int = 30
    max_llm_retries: int = 3  # attempts for the generate->validate->repair loop

    # --- Embeddings / RAG ---
    embedding_provider: Literal["huggingface", "hash"] = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_top_k: int = 4
    vector_index_dir: str = "storage_data/vector_indexes"

    # --- Persistence ---
    database_url: str = "sqlite:///./storage_data/app.db"

    # --- Conversation history (product feature) ---
    max_history_turns: int = 6  # turns of chat history fed back into the prompt

    # --- CORS ---
    cors_allow_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton. Use as a FastAPI dependency via Depends(get_settings)."""
    return Settings()
