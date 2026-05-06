"""Shared pytest fixtures for unit and integration tests."""
import os
from unittest.mock import MagicMock

import pytest

# Set required env vars before any app imports so Pydantic Settings doesn't raise
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-do-not-use")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture(scope="session")
def settings():
    from app.core.config import get_settings

    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def mock_embedder():
    """Stub embedder that returns deterministic fake vectors."""
    m = MagicMock()
    m.embed_query.return_value = [0.1] * 1536
    m.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
    return m


@pytest.fixture
def sample_documents():
    from langchain_core.documents import Document

    return [
        Document(
            page_content="RAG combines retrieval with language model generation.",
            metadata={"source": "test.pdf", "page": 1},
        ),
        Document(
            page_content="Hybrid search uses both dense and sparse vectors for recall.",
            metadata={"source": "test.pdf", "page": 2},
        ),
        Document(
            page_content="Cross-encoders jointly score query-document pairs for precision.",
            metadata={"source": "test.pdf", "page": 3},
        ),
    ]


@pytest.fixture
def sample_candidates():
    """Typical output from HybridSearcher.search()."""
    return [
        {
            "id": "vec-1",
            "score": 0.92,
            "text": "RAG combines retrieval with language model generation for grounded answers.",
            "metadata": {"source": "test.pdf"},
        },
        {
            "id": "vec-2",
            "score": 0.85,
            "text": "Hybrid search uses both dense and sparse vectors to improve recall.",
            "metadata": {"source": "test.pdf"},
        },
        {
            "id": "vec-3",
            "score": 0.78,
            "text": "Cross-encoders rerank candidates using joint query-document scoring.",
            "metadata": {"source": "test.pdf"},
        },
    ]
