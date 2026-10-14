"""Runtime configuration, loaded from environment variables (and an optional .env file)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM: "gemini" (default) or "anthropic". Keys can also be set at runtime from the UI's Settings.
    llm_provider: str = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "low"
    llm_max_tokens: int = 4000
    llm_server_fallbacks: bool = True

    # Embeddings: "fastembed" (local ONNX model, no key needed) or "hashing" (offline, for tests)
    embedding_backend: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path | None = None  # defaults to DATA_DIR/models

    # Chunking
    chunk_size: int = 700
    chunk_overlap: int = 120

    # Retrieval
    top_k: int = 8
    min_relevance: float = 0.5

    # Storage & ingestion
    data_dir: Path = Path("data")
    max_upload_mb: int = 20
    seed_dir: Path | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
