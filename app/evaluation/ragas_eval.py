"""Ragas evaluation pipeline.

Measures four core RAG quality metrics on a golden QA dataset:

  faithfulness       — Is every claim in the answer supported by the context?
  answer_relevancy   — Does the answer actually address the question?
  context_precision  — Are the top retrieved chunks relevant to the question?
  context_recall     — Does the context cover the ground truth answer?
                       (requires ground_truth, skipped if not provided)

Results are saved as a timestamped JSON file in eval_results/ and
printed as a formatted ASCII table to stdout.

Usage:
    python scripts/run_evaluation.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()

_EVAL_DIR = Path("eval_results")


def run_ragas_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: Optional[List[str]] = None,
    output_dir: str = "eval_results",
) -> Dict[str, Any]:
    """
    Run Ragas evaluation and return a dict of metric scores.

    Args:
        questions:     List of user questions.
        answers:       Generated answers (one per question).
        contexts:      Retrieved context chunks (list of lists).
        ground_truths: Optional ground truth answers for context_recall.
        output_dir:    Directory to write the JSON results file.

    Returns:
        Dict with metric names as keys and float scores as values,
        plus 'num_samples' and 'timestamp'.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        raise ImportError(
            "Ragas and/or datasets are not installed. "
            "Run: pip install ragas datasets"
        ) from exc

    data: Dict[str, List] = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }

    metrics = [faithfulness, answer_relevancy, context_precision]

    if ground_truths:
        from ragas.metrics import context_recall  # noqa: PLC0415

        data["ground_truth"] = ground_truths
        metrics.append(context_recall)

    dataset = Dataset.from_dict(data)

    logger.info(
        "Starting Ragas evaluation",
        num_samples=len(questions),
        metrics=[m.name for m in metrics],
    )

    result = evaluate(dataset=dataset, metrics=metrics)
    df = result.to_pandas()
    scores: Dict[str, Any] = df.mean(numeric_only=True).round(4).to_dict()
    scores["num_samples"] = len(questions)
    scores["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

    # Persist results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_path = out / f"ragas_{ts}.json"
    result_path.write_text(json.dumps(scores, indent=2))
    logger.info("Evaluation results saved", path=str(result_path))

    return scores


def print_evaluation_report(scores: Dict[str, Any]) -> None:
    """Print a formatted evaluation report to stdout."""
    _SEP = "=" * 62
    _metric_labels = {
        "faithfulness": "Faithfulness        (grounded in context?)",
        "answer_relevancy": "Answer Relevancy    (answers the question?)",
        "context_precision": "Context Precision   (retrieved chunks useful?)",
        "context_recall": "Context Recall      (context covers ground truth?)",
    }

    print(f"\n{_SEP}")
    print("  NEXUS RAG — RAGAS EVALUATION REPORT")
    print(_SEP)

    for key, label in _metric_labels.items():
        if key not in scores:
            continue
        score: float = scores[key]
        filled = int(round(score * 20))
        bar = "█" * filled + "░" * (20 - filled)
        status = "✓" if score >= 0.7 else "~" if score >= 0.5 else "✗"
        print(f"  [{status}] {label}")
        print(f"       Score: {score:.4f}  |{bar}|")
        print()

    print(f"  Samples evaluated : {scores.get('num_samples', 'N/A')}")
    print(f"  Timestamp         : {scores.get('timestamp', 'N/A')}")
    print(f"{_SEP}\n")
