"""Hybrid search: dense semantic vectors + BM25 sparse vectors via Pinecone.

How it works
────────────
1. The user query is embedded with OpenAI (dense) and BM25 (sparse).
2. Both vectors are scaled by alpha (dense weight) and (1-alpha) (sparse
   weight) before being sent in a single Pinecone hybrid query.
3. Pinecone's native hybrid scoring merges the two ranked lists internally
   using a weighted dotproduct, equivalent to alpha-scaled RRF.
4. Results include both text and metadata payloads for the reranker.

alpha = 1.0  →  pure semantic search (dense only)
alpha = 0.0  →  pure keyword search  (sparse BM25 only)
alpha = 0.5  →  balanced (recommended default)
"""
from typing import Any, Dict, List, Optional

import structlog
from pinecone import Pinecone

from app.core.config import get_settings
from app.retrieval.embedder import DenseEmbedder
from app.retrieval.sparse import SparseEncoder

logger = structlog.get_logger()


class HybridSearcher:
    """
    Executes alpha-weighted hybrid queries against a Pinecone index.

    Thread-safe — the Pinecone client and both encoders are stateless
    after initialisation, so a single instance can serve all requests.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = DenseEmbedder()
        self.sparse_encoder = SparseEncoder()

        pc = Pinecone(api_key=self.settings.pinecone_api_key)
        self.index = pc.Index(self.settings.pinecone_index_name)
        logger.info("HybridSearcher initialised", index=self.settings.pinecone_index_name)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run a hybrid query and return a list of result dicts.

        Args:
            query:  Natural language query string.
            top_k:  Number of candidates to retrieve (overrides settings).
            alpha:  Dense weight in [0, 1] (overrides settings).
            filter: Pinecone metadata filter, e.g. {"source": {"$eq": "doc.pdf"}}.

        Returns:
            List of dicts: {"id", "score", "text", "metadata"}.
        """
        top_k = top_k or self.settings.retrieval_top_k
        alpha = alpha if alpha is not None else self.settings.hybrid_alpha

        # Encode query
        dense_vec = self.embedder.embed_query(query)
        sparse_vec = self.sparse_encoder.encode_query(query)

        # Apply alpha weighting — Pinecone combines them via dotproduct sum
        scaled_dense = [v * alpha for v in dense_vec]
        scaled_sparse = {
            "indices": sparse_vec["indices"],
            "values": [v * (1.0 - alpha) for v in sparse_vec["values"]],
        }

        response = self.index.query(
            vector=scaled_dense,
            sparse_vector=scaled_sparse,
            top_k=top_k,
            include_metadata=True,
            filter=filter,
        )

        results = [
            {
                "id": match.id,
                "score": float(match.score),
                "text": match.metadata.get("text", ""),
                "metadata": {k: v for k, v in match.metadata.items() if k != "text"},
            }
            for match in response.matches
        ]

        logger.info(
            "Hybrid search complete",
            query_len=len(query),
            top_k=top_k,
            alpha=alpha,
            hits=len(results),
        )
        return results
