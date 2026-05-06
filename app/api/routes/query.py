"""Query endpoint — the full RAG pipeline in a single route.

Request flow
────────────
  1. NeMo Guardrails  →  block prompt injection / jailbreak / off-topic
  2. Hybrid Search    →  retrieve top-K candidates (dense + BM25)
  3. Cross-Encoder    →  rerank candidates to top-5
  4. LangChain LCEL   →  generate grounded answer (GPT-4o-mini)
  5. Output Guard     →  PII masking, faithfulness check, toxicity filter
  6. Return           →  QueryResponse with sources and latency
"""
import time

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_hybrid_searcher, get_reranker
from app.generation.chain import get_rag_chain
from app.guardrails.nemo.rails import check_input
from app.guardrails.output.guard import get_output_guard
from app.models.request import QueryRequest
from app.models.response import QueryResponse, SourceDocument
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.reranker import CrossEncoderReranker

logger = structlog.get_logger()
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    searcher: HybridSearcher = Depends(get_hybrid_searcher),
    reranker: CrossEncoderReranker = Depends(get_reranker),
) -> QueryResponse:
    """Execute a RAG query with dual-layer guardrails and reranking."""
    t0 = time.perf_counter()

    # ── Step 1: Input guardrails ──────────────────────────────────────────────
    is_safe, block_message = await check_input(body.query)
    if not is_safe:
        return QueryResponse(
            answer=block_message,
            sources=[],
            session_id=body.session_id,
            latency_ms=_ms(t0),
            guardrail_triggered=True,
            guardrail_message="Input guardrail triggered (NeMo).",
        )

    # ── Step 2: Hybrid search ─────────────────────────────────────────────────
    try:
        candidates = searcher.search(
            query=body.query,
            top_k=body.top_k,
            alpha=body.alpha,
        )
    except Exception as exc:
        logger.error("Hybrid search failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Retrieval service unavailable.")

    if not candidates:
        return QueryResponse(
            answer=(
                "I could not find relevant information in the knowledge base for your query. "
                "Please try rephrasing or ingest more documents."
            ),
            sources=[],
            session_id=body.session_id,
            latency_ms=_ms(t0),
        )

    # ── Step 3: Cross-encoder reranking ───────────────────────────────────────
    reranked = reranker.rerank(query=body.query, candidates=candidates)

    # ── Step 4: LLM generation ────────────────────────────────────────────────
    try:
        chain = get_rag_chain()
        raw_answer: str = chain.invoke(
            {"question": body.query, "context": reranked}
        )
    except Exception as exc:
        logger.error("Generation failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Generation service unavailable.")

    # ── Step 5: Output guardrails ─────────────────────────────────────────────
    context_texts = [doc["text"] for doc in reranked]
    guard = get_output_guard()
    result = guard.validate(answer=raw_answer, context_chunks=context_texts)

    # ── Step 6: Build response ────────────────────────────────────────────────
    sources = [
        SourceDocument(
            content=doc["text"][:500],
            source=doc.get("metadata", {}).get("source"),
            score=doc.get("rerank_score"),
            metadata=doc.get("metadata", {}),
        )
        for doc in reranked
    ]

    latency = _ms(t0)
    logger.info(
        "Query pipeline complete",
        latency_ms=latency,
        sources=len(sources),
        guardrail_triggered=not result.is_safe,
    )

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        session_id=body.session_id,
        latency_ms=latency,
        guardrail_triggered=not result.is_safe,
        guardrail_message=result.message,
    )


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
