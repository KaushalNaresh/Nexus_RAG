#!/usr/bin/env python3
"""CLI: Run Ragas evaluation against the live Nexus RAG pipeline.

This script:
  1. Loads the golden QA dataset from app/evaluation/test_dataset.py
  2. Runs each question through the full retrieval + generation pipeline
  3. Passes all (question, answer, context, ground_truth) tuples to Ragas
  4. Prints a formatted report and saves results to eval_results/

Prerequisites:
  • .env with valid OPENAI_API_KEY and PINECONE_API_KEY
  • Documents must already be ingested (run scripts/ingest_docs.py first)
  • pip install ragas datasets

Usage:
  python scripts/run_evaluation.py
  python scripts/run_evaluation.py --output-dir ./my_results
"""
import argparse
import asyncio
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.evaluation.ragas_eval import print_evaluation_report, run_ragas_evaluation
from app.evaluation.test_dataset import GOLDEN_DATASET
from app.generation.chain import get_rag_chain
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.reranker import get_reranker

logger = structlog.get_logger()


async def _evaluate(output_dir: str) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    # Ragas resolves OpenAI credentials from OS env vars, not pydantic-settings.
    import os
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

    logger.info("Initialising pipeline components...")
    searcher = HybridSearcher()
    reranker = get_reranker()
    chain = get_rag_chain()

    questions: List[str] = []
    answers: List[str] = []
    contexts: List[List[str]] = []
    ground_truths: List[str] = []

    for i, sample in enumerate(GOLDEN_DATASET, 1):
        q: str = sample["question"]
        gt: str = sample.get("ground_truth", "")
        logger.info("Evaluating sample", index=i, total=len(GOLDEN_DATASET))

        candidates = searcher.search(q)
        reranked = reranker.rerank(q, candidates)
        context_texts = [doc["text"] for doc in reranked]
        answer: str = chain.invoke({"question": q, "context": reranked})

        questions.append(q)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(gt)

    scores = run_ragas_evaluation(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths if any(ground_truths) else None,
        output_dir=output_dir,
    )
    print_evaluation_report(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ragas evaluation on Nexus RAG.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save the JSON results file (default: eval_results/).",
    )
    args = parser.parse_args()
    asyncio.run(_evaluate(args.output_dir))


if __name__ == "__main__":
    main()
