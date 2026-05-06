"""Output validators: PII detection, faithfulness scoring, toxicity filter.

Three independent, composable validators — each returns a structured result
so the OutputGuard can decide whether to mask, block, or pass through.

PII detection uses regex patterns as a fast, dependency-free layer.
Presidio (when installed) can be added as a second, higher-recall layer
by calling _presidio_detect() alongside the regex approach.

Faithfulness is a lightweight heuristic — significant content words in the
answer are checked against the retrieved context. The Ragas pipeline
(ragas_eval.py) provides rigorous offline faithfulness measurement.
"""
import re
from typing import Dict, List, Tuple

import structlog

logger = structlog.get_logger()


# ── PII Patterns ──────────────────────────────────────────────────────────────

_PII_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
    "phone": re.compile(
        r"\b(\+\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"
    ),
    "ip_address": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
}


def detect_pii(text: str) -> Dict[str, List[str]]:
    """
    Scan text for PII using compiled regex patterns.

    Returns:
        Dict mapping PII type → list of matched strings.
        Empty dict if no PII detected.
    """
    found: Dict[str, List[str]] = {}
    for pii_type, pattern in _PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            # findall returns tuples for patterns with groups — flatten
            found[pii_type] = [m if isinstance(m, str) else m[0] for m in matches]
    return found


def mask_pii(text: str) -> str:
    """Replace detected PII with labelled placeholders (non-reversible)."""
    for pii_type, pattern in _PII_PATTERNS.items():
        placeholder = f"[{pii_type.upper()}_REDACTED]"
        text = pattern.sub(placeholder, text)
    return text


# ── Faithfulness ──────────────────────────────────────────────────────────────

# Common English stop-words to exclude from the overlap calculation
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might must shall can need dare used "
    "to of in on at by for with as from into through during before "
    "after above below between among this that these those it its "
    "i me my we our you your he him his she her they them their what "
    "which who whom how when where why all any both each few more most "
    "other some such no nor not only own same so than too very just".split()
)


def check_faithfulness(
    answer: str,
    context_chunks: List[str],
    threshold: float = 0.35,
) -> Tuple[bool, float]:
    """
    Heuristic faithfulness check: measures content word overlap between
    the answer and the retrieved context.

    Args:
        answer:         Generated LLM answer.
        context_chunks: Retrieved text chunks used as context.
        threshold:      Minimum overlap ratio to be considered faithful.

    Returns:
        (is_faithful, overlap_score) where overlap_score ∈ [0, 1].
    """
    if not context_chunks or not answer:
        return True, 1.0

    combined = " ".join(context_chunks).lower()
    answer_lower = answer.lower()

    # Extract content words (len > 3, not stop-words)
    content_words = [
        w for w in re.findall(r"\b[a-z]{4,}\b", answer_lower)
        if w not in _STOP_WORDS
    ]

    if not content_words:
        return True, 1.0

    found = sum(1 for w in content_words if w in combined)
    score = found / len(content_words)
    is_faithful = score >= threshold

    logger.debug(
        "Faithfulness check",
        content_words=len(content_words),
        found=found,
        score=round(score, 3),
        is_faithful=is_faithful,
    )
    return is_faithful, round(score, 4)


# ── Toxicity ──────────────────────────────────────────────────────────────────

_TOXIC_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(how to (make|build|create|synthesize) (a |an )?(bomb|weapon|explosive|poison|drug))\b", re.I),
    re.compile(r"\b(kill|murder|assassinate|harm|hurt)\s+(yourself|himself|herself|themselves|people|someone)\b", re.I),
    re.compile(r"\b(child (porn|abuse|exploitation)|csam)\b", re.I),
]


def check_toxicity(text: str) -> bool:
    """
    Check for toxic content using pattern matching.

    Returns:
        True if text is safe, False if toxic content is detected.
    """
    for pattern in _TOXIC_PATTERNS:
        if pattern.search(text):
            logger.warning("Toxic content pattern matched")
            return False
    return True
