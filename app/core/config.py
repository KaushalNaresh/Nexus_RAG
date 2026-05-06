"""Central application configuration via Pydantic BaseSettings.

All values are read from environment variables or a .env file.
Type-safe, validated at startup — no raw os.getenv() calls anywhere else.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ── Pinecone ──────────────────────────────────────────────────────────────
    pinecone_api_key: str
    pinecone_index_name: str = "nexus-rag"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ── Redis (Semantic Cache) ─────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    enable_semantic_cache: bool = True
    cache_ttl_seconds: int = 3600

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    # alpha: 0.0 = pure sparse (keyword), 1.0 = pure dense (semantic)
    hybrid_alpha: float = 0.5

    # ── Reranker ──────────────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Guardrails ────────────────────────────────────────────────────────────
    enable_nemo_guardrails: bool = True
    enable_output_guardrails: bool = True

    # ── Application ───────────────────────────────────────────────────────────
    environment: Literal["development", "production"] = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
