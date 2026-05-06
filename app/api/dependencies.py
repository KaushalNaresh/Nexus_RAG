"""FastAPI dependency injection providers.

Using @lru_cache makes these true singletons — the heavy constructors
(model loading, index connection) run exactly once per process.
"""
from functools import lru_cache

from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.reranker import CrossEncoderReranker, get_reranker as _get_reranker


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline()


@lru_cache
def get_hybrid_searcher() -> HybridSearcher:
    return HybridSearcher()


def get_reranker() -> CrossEncoderReranker:
    # Delegates to the module-level singleton (loads model once)
    return _get_reranker()
