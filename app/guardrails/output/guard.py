"""Output guard pipeline — composes all output validators into one pass.

The guard runs three checks in order:
  1. Toxicity     — hard block; no masking, no second chances.
  2. PII          — mask by default; optionally block.
  3. Faithfulness — block if the answer drifts too far from the context.

Each check is independently configurable via constructor args so you can
tune sensitivity for your specific domain without touching validator logic.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import structlog

from app.guardrails.output.validators import (
    check_faithfulness,
    check_toxicity,
    detect_pii,
    mask_pii,
)

logger = structlog.get_logger()


@dataclass
class GuardResult:
    """Structured result from the output guard pipeline."""

    answer: str
    is_safe: bool
    pii_detected: Dict[str, List[str]] = field(default_factory=dict)
    is_faithful: bool = True
    faithfulness_score: float = 1.0
    is_toxic: bool = False
    message: Optional[str] = None


class OutputGuard:
    """
    Validates LLM-generated answers before returning them to clients.

    Usage:
        guard = OutputGuard()
        result = guard.validate(answer, context_chunks)
        if result.is_safe:
            return result.answer
    """

    def __init__(
        self,
        enable_pii_masking: bool = True,
        enable_faithfulness: bool = True,
        enable_toxicity: bool = True,
        block_on_pii: bool = False,
        faithfulness_threshold: float = 0.20,
    ) -> None:
        self.enable_pii_masking = enable_pii_masking
        self.enable_faithfulness = enable_faithfulness
        self.enable_toxicity = enable_toxicity
        self.block_on_pii = block_on_pii
        self.faithfulness_threshold = faithfulness_threshold

    def validate(
        self,
        answer: str,
        context_chunks: Optional[List[str]] = None,
    ) -> GuardResult:
        """
        Run all validators sequentially on the generated answer.

        Args:
            answer:         Raw LLM output string.
            context_chunks: Retrieved context used for generation (for faithfulness).

        Returns:
            GuardResult — check .is_safe and use .answer (may be masked/replaced).
        """
        from app.core.config import get_settings  # noqa: PLC0415

        settings = get_settings()
        if not settings.enable_output_guardrails:
            return GuardResult(answer=answer, is_safe=True)

        processed = answer
        pii_found: Dict[str, List[str]] = {}

        # ── 1. Toxicity ───────────────────────────────────────────────────────
        if self.enable_toxicity:
            if not check_toxicity(processed):
                logger.warning("Output guard: toxic content blocked")
                return GuardResult(
                    answer=(
                        "I'm unable to provide that response as it contains "
                        "inappropriate content."
                    ),
                    is_safe=False,
                    is_toxic=True,
                    message="Response blocked: toxic content detected.",
                )

        # ── 2. PII detection / masking ────────────────────────────────────────
        if self.enable_pii_masking:
            pii_found = detect_pii(processed)
            if pii_found:
                pii_types = list(pii_found.keys())
                logger.warning("Output guard: PII detected", types=pii_types)
                if self.block_on_pii:
                    return GuardResult(
                        answer=(
                            "The response contained sensitive personal information "
                            "and was blocked for privacy protection."
                        ),
                        is_safe=False,
                        pii_detected=pii_found,
                        message=f"Response blocked: PII types detected: {pii_types}.",
                    )
                # Default: mask and continue
                processed = mask_pii(processed)

        # ── 3. Faithfulness ───────────────────────────────────────────────────
        is_faithful = True
        faithfulness_score = 1.0

        if self.enable_faithfulness and context_chunks:
            is_faithful, faithfulness_score = check_faithfulness(
                processed,
                context_chunks,
                threshold=self.faithfulness_threshold,
            )
            if not is_faithful:
                logger.warning(
                    "Output guard: faithfulness check failed",
                    score=faithfulness_score,
                    threshold=self.faithfulness_threshold,
                )
                return GuardResult(
                    answer=(
                        "I could not verify this response against the knowledge base. "
                        "Please try rephrasing your question."
                    ),
                    is_safe=False,
                    pii_detected=pii_found,
                    is_faithful=False,
                    faithfulness_score=faithfulness_score,
                    message=(
                        f"Response blocked: faithfulness score {faithfulness_score:.2f} "
                        f"below threshold {self.faithfulness_threshold}."
                    ),
                )

        return GuardResult(
            answer=processed,
            is_safe=True,
            pii_detected=pii_found,
            is_faithful=is_faithful,
            faithfulness_score=faithfulness_score,
        )


_guard_instance: Optional[OutputGuard] = None


def get_output_guard() -> OutputGuard:
    """Return (or create) the module-level singleton OutputGuard."""
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = OutputGuard()
    return _guard_instance
