"""Ingestion pipeline: load → chunk → embed (dense + sparse) → upsert Pinecone.

The pipeline is intentionally stateless at the class level — each call to
`run()` produces a fresh set of BM25 statistics fitted to the current batch.
For multi-batch incremental indexing, persist and reload BM25 params via
`SparseEncoder.fit()` / `SparseEncoder.load()`.
"""
import uuid
from typing import List, Optional

import structlog
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from app.core.config import get_settings
from app.ingestion.chunker import chunk_documents

logger = structlog.get_logger()


def _create_or_get_index():
    """Initialize Pinecone and return the hybrid-ready index handle."""
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    existing_names = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing_names:
        logger.info("Creating Pinecone index", name=settings.pinecone_index_name)
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimensions,
            metric="dotproduct",  # dotproduct is required for hybrid sparse+dense
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        logger.info("Pinecone index created", name=settings.pinecone_index_name)
    else:
        logger.info("Pinecone index already exists", name=settings.pinecone_index_name)

    return pc.Index(settings.pinecone_index_name)


class IngestionPipeline:
    """
    Orchestrates the full ingestion flow:
        Documents → Chunks → Dense embeddings + BM25 sparse vectors → Pinecone
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._embedder = None
        self._sparse_encoder = None
        self._index = None

    # ── Lazy-loaded dependencies ──────────────────────────────────────────────

    @property
    def embedder(self):
        if self._embedder is None:
            from app.retrieval.embedder import DenseEmbedder
            self._embedder = DenseEmbedder()
        return self._embedder

    @property
    def sparse_encoder(self):
        if self._sparse_encoder is None:
            from app.retrieval.sparse import SparseEncoder
            self._sparse_encoder = SparseEncoder()
        return self._sparse_encoder

    @property
    def index(self):
        if self._index is None:
            self._index = _create_or_get_index()
        return self._index

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Run the full ingestion pipeline.

        Args:
            documents:     Raw LangChain Documents from any loader.
            chunk_size:    Override settings.chunk_size.
            chunk_overlap: Override settings.chunk_overlap.
            batch_size:    How many chunks to upsert per Pinecone API call.

        Returns:
            Total number of vector chunks indexed.
        """
        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size or self.settings.chunk_size,
            chunk_overlap=chunk_overlap or self.settings.chunk_overlap,
        )

        if not chunks:
            logger.warning("No chunks produced — aborting ingestion")
            return 0

        texts = [c.page_content for c in chunks]

        # Fit BM25 on the full corpus before encoding (must see all docs)
        self.sparse_encoder.fit(texts)

        total = 0
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]
            batch_texts = [c.page_content for c in batch_chunks]

            dense_vecs = self.embedder.embed_documents(batch_texts)
            sparse_vecs = self.sparse_encoder.encode_documents(batch_texts)

            records = []
            for chunk, dense, sparse in zip(batch_chunks, dense_vecs, sparse_vecs):
                records.append(
                    {
                        "id": str(uuid.uuid4()),
                        "values": dense,
                        "sparse_values": sparse,
                        "metadata": {
                            "text": chunk.page_content,
                            **{k: str(v) for k, v in chunk.metadata.items()},
                        },
                    }
                )

            self.index.upsert(vectors=records)
            total += len(records)
            logger.info(
                "Upserted batch",
                batch=batch_start // batch_size + 1,
                count=len(records),
            )

        logger.info("Ingestion complete", total_chunks=total)
        return total
