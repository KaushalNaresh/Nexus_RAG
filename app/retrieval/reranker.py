"""Cross-encoder reranker for precise passage scoring.

Why cross-encoders?
───────────────────
Bi-encoders (used for retrieval) embed query and document independently,
which is fast but loses cross-attention between the two. Cross-encoders
jointly encode the (query, passage) pair and output a calibrated relevance
score — much more accurate, but O(n) model forward passes.

Strategy: retrieve top-20 with the fast bi-encoder + BM25, then rerank
to top-5 with the cross-encoder. This balances speed and precision.
"""
from typing import Any, Dict, List, Optional

import structlog
from sentence_transformers import CrossEncoder

from app.core.config import get_settings

logger = structlog.get_logger()

# Module-level singleton — loads the model once per process
_reranker_instance: Optional["CrossEncoderReranker"] = None


class CrossEncoderReranker:
    """Reranks candidates using ms-marco-MiniLM-L-6-v2 cross-encoder."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = CrossEncoder(settings.reranker_model)
        logger.info("Cross-encoder loaded", model=settings.reranker_model)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Score and rerank a list of candidate documents.

        Args:
            query:      Original user query.
            candidates: List of dicts from HybridSearcher (must have 'text' key).
            top_k:      How many to return after reranking (overrides settings).

        Returns:
            Top-k candidates sorted descending by 'rerank_score'.
        """
        if not candidates:
            return []

        settings = get_settings()
        top_k = top_k or settings.rerank_top_k

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        top = reranked[:top_k]

        logger.info(
            "Reranking complete",
            input_candidates=len(candidates),
            output_candidates=len(top),
            best_score=round(top[0]["rerank_score"], 4) if top else None,
        )
        return top


def get_reranker() -> CrossEncoderReranker:
    """Return (or create) the module-level singleton reranker."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance
