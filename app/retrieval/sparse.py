"""BM25 sparse encoder for keyword-based retrieval.

Uses pinecone-text's BM25Encoder to produce sparse vectors compatible
with Pinecone's native hybrid query format.

BM25 statistics are persisted to disk after fitting so they survive
restarts and can be shared between the ingestion and query services.
"""
from pathlib import Path
from typing import Any, Dict, List

import structlog
from pinecone_text.sparse import BM25Encoder

logger = structlog.get_logger()

_PARAMS_PATH = Path("bm25_params.json")


class SparseEncoder:
    """
    Wraps pinecone-text BM25Encoder with fit/persist/load lifecycle.

    Usage:
        # During ingestion
        encoder.fit(all_chunk_texts)

        # During query
        sparse_vec = encoder.encode_query("my question")
    """

    def __init__(self) -> None:
        self._encoder = BM25Encoder()
        self._fitted = False
        self._try_load_params()

    def _try_load_params(self) -> None:
        """Load pre-computed BM25 parameters from disk if available."""
        if _PARAMS_PATH.exists():
            try:
                self._encoder.load(_PARAMS_PATH)
                self._fitted = True
                logger.info("BM25 params loaded from disk", path=str(_PARAMS_PATH))
            except Exception as exc:
                logger.warning("Could not load BM25 params", error=str(exc))

    def fit(self, corpus: List[str]) -> None:
        """
        Fit BM25 on the ingested corpus.

        Must be called with all documents before encoding queries so
        IDF scores reflect the full corpus distribution.
        """
        self._encoder.fit(corpus)
        self._fitted = True
        self._encoder.dump(str(_PARAMS_PATH))
        logger.info("BM25 encoder fitted and saved", corpus_size=len(corpus))

    def encode_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Return sparse vectors (indices + values) for a list of documents."""
        self._require_fitted()
        return self._encoder.encode_documents(texts)

    def encode_query(self, text: str) -> Dict[str, Any]:
        """Return a sparse vector for a single query string."""
        self._require_fitted()
        result = self._encoder.encode_queries(text)
        # encode_queries can return a list or a single dict depending on input type
        if isinstance(result, list):
            return result[0]
        return result

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "BM25 encoder is not fitted. "
                "Run the ingestion pipeline first to fit and persist BM25 parameters."
            )
