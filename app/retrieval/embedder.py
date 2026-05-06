"""Dense embedding wrapper using OpenAI text-embedding-3-small.

Thin wrapper around LangChain's OpenAIEmbeddings so the rest of the
codebase doesn't import OpenAI directly — easy to swap to a local model.
"""
from typing import List

import structlog
from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings

logger = structlog.get_logger()


class DenseEmbedder:
    """Produces 1536-dim dense vectors for hybrid Pinecone indexing."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = OpenAIEmbeddings(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            openai_api_key=settings.openai_api_key,
        )
        logger.info(
            "Dense embedder initialised",
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self._model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document texts."""
        return self._model.embed_documents(texts)
