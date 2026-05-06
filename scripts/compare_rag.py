#!/usr/bin/env python3
"""Side-by-side comparison: Naive RAG vs Production RAG.

Naive RAG
─────────
  • Dense-only retrieval (no BM25 keywords)
  • No reranking — uses raw vector similarity order
  • No guardrails
  • top_k = 5 straight to LLM

Production RAG (Nexus)
──────────────────────
  • Hybrid retrieval (dense + BM25, alpha=0.5)
  • Cross-encoder reranking (top-20 → top-5)
  • Input guardrails (regex + NeMo)
  • Output guardrails (PII + faithfulness + toxicity)

Usage:
  python scripts/compare_rag.py
  python scripts/compare_rag.py --query "How does RAG prevent hallucinations?"
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.generation.chain import get_rag_chain
from app.retrieval.embedder import DenseEmbedder
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.reranker import get_reranker

# ── Default queries that highlight the difference between approaches ──────────
DEFAULT_QUERIES = [
    "What is Retrieval-Augmented Generation and how does it work?",
    "What are the main challenges of RAG systems?",
    "How does hybrid search improve retrieval quality?",
]


# ── Naive RAG ─────────────────────────────────────────────────────────────────

def naive_rag(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Simplest possible RAG:
      dense-only retrieval → take top-K as-is → generate.
    No BM25, no reranking, no guardrails.
    """
    t0 = time.perf_counter()
    embedder = DenseEmbedder()

    from pinecone import Pinecone  # noqa: PLC0415
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    # Pure dense query — no sparse vector
    dense_vec = embedder.embed_query(query)
    response = index.query(vector=dense_vec, top_k=top_k, include_metadata=True)

    candidates = [
        {
            "text": m.metadata.get("text", ""),
            "score": float(m.score),
            "metadata": {k: v for k, v in m.metadata.items() if k != "text"},
        }
        for m in response.matches
    ]

    chain = get_rag_chain()
    answer = chain.invoke({"question": query, "context": candidates})
    latency = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "answer": answer,
        "sources": candidates,
        "latency_ms": latency,
        "chunks_used": len(candidates),
        "retrieval": "dense-only (no BM25)",
        "reranking": "none",
        "guardrails": "none",
    }


# ── Production RAG ────────────────────────────────────────────────────────────

def production_rag(query: str) -> Dict[str, Any]:
    """
    Full Nexus RAG pipeline:
      hybrid search → cross-encoder rerank → generate → output guard.
    """
    t0 = time.perf_counter()
    settings = get_settings()

    # Input guardrail
    from app.guardrails.nemo.rails import _INJECTION_PATTERNS, _INJECTION_REFUSAL  # noqa: PLC0415
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            return {
                "answer": _INJECTION_REFUSAL,
                "sources": [],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "chunks_used": 0,
                "retrieval": "blocked at input",
                "reranking": "skipped",
                "guardrails": "INPUT BLOCKED",
            }

    # Hybrid search
    searcher = HybridSearcher()
    candidates = searcher.search(query, top_k=settings.retrieval_top_k, alpha=settings.hybrid_alpha)

    # Cross-encoder rerank
    reranker = get_reranker()
    reranked = reranker.rerank(query, candidates, top_k=settings.rerank_top_k)

    # Generate
    chain = get_rag_chain()
    raw_answer = chain.invoke({"question": query, "context": reranked})

    # Output guard
    from app.guardrails.output.guard import get_output_guard  # noqa: PLC0415
    guard = get_output_guard()
    result = guard.validate(raw_answer, context_chunks=[d["text"] for d in reranked])

    latency = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "answer": result.answer,
        "sources": reranked,
        "latency_ms": latency,
        "chunks_used": len(reranked),
        "retrieval": f"hybrid (dense + BM25, alpha={settings.hybrid_alpha})",
        "reranking": f"cross-encoder → top-{settings.rerank_top_k}",
        "guardrails": "input + output (PII + faithfulness + toxicity)",
        "guardrail_triggered": not result.is_safe,
        "faithfulness_score": result.faithfulness_score,
    }


# ── Report printer ────────────────────────────────────────────────────────────

def _bar(score: float, width: int = 20) -> str:
    filled = int(round(max(0.0, min(1.0, score)) * width))
    return "█" * filled + "░" * (width - filled)


def print_comparison(query: str, naive: Dict, prod: Dict) -> None:
    W = 70
    SEP = "═" * W

    print(f"\n{SEP}")
    print(f"  QUERY: {query[:W - 10]}")
    print(SEP)

    for label, result in [("NAIVE RAG", naive), ("PRODUCTION RAG (Nexus)", prod)]:
        print(f"\n  ── {label} {'─' * (W - len(label) - 6)}")
        print(f"  Retrieval : {result['retrieval']}")
        print(f"  Reranking : {result['reranking']}")
        print(f"  Guardrails: {result['guardrails']}")
        print(f"  Latency   : {result['latency_ms']} ms")
        print(f"  Chunks    : {result['chunks_used']}")

        if "faithfulness_score" in result:
            fs = result["faithfulness_score"]
            print(f"  Faithful  : {fs:.2f}  |{_bar(fs)}|")

        if result.get("guardrail_triggered"):
            print(f"  ⚠  GUARDRAIL TRIGGERED")

        print(f"\n  Answer:")
        # Word-wrap the answer at W-4 chars
        words = result["answer"].split()
        line, lines = "", []
        for w in words:
            if len(line) + len(w) + 1 > W - 4:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        if line:
            lines.append(line)
        for l in lines:
            print(f"    {l}")

        if result.get("sources"):
            print(f"\n  Top sources (by score):")
            for i, src in enumerate(result["sources"][:3], 1):
                score = src.get("rerank_score") or src.get("score", 0)
                snippet = src["text"][:80].replace("\n", " ")
                print(f"    [{i}] score={score:+.2f}  \"{snippet}...\"")

    # Delta summary
    latency_delta = prod["latency_ms"] - naive["latency_ms"]
    print(f"\n{SEP}")
    print(f"  DELTA SUMMARY")
    print(f"  Latency overhead : +{latency_delta:.0f} ms (reranking + guardrails)")
    print(f"  Retrieval quality: naive uses top-{naive['chunks_used']} raw dense vectors")
    print(f"                     production reranks {get_settings().retrieval_top_k} hybrid candidates → {prod['chunks_used']}")
    print(SEP + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Naive RAG vs Production RAG.")
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Query to compare. If omitted, runs all default queries.",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging("WARNING")  # Suppress info logs for clean output

    queries = [args.query] if args.query else DEFAULT_QUERIES

    print("\n" + "═" * 70)
    print("  NEXUS RAG — NAIVE vs PRODUCTION COMPARISON")
    print("═" * 70)

    for query in queries:
        print(f"\n  Running naive RAG...", end="", flush=True)
        naive = naive_rag(query)
        print(f" done ({naive['latency_ms']}ms)")

        print(f"  Running production RAG...", end="", flush=True)
        prod = production_rag(query)
        print(f" done ({prod['latency_ms']}ms)")

        print_comparison(query, naive, prod)

    if len(queries) > 1:
        print("Run with --query \"your question\" to test a specific query.\n")


if __name__ == "__main__":
    main()
