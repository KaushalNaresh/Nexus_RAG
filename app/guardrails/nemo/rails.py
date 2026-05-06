"""NeMo Guardrails — input-layer wrapper.

NeMo Guardrails intercepts the raw user query before retrieval starts.
It enforces conversational rails defined in rails.co (Colang):
  • Jailbreak deflection
  • Prompt injection blocking
  • Off-topic query refusal

The LLMRails instance is loaded lazily and cached to avoid the ~2s model
loading cost on every request. If nemoguardrails is not installed or the
config directory is malformed, the wrapper degrades gracefully (fail-open)
so the rest of the pipeline continues unaffected.

Fail-open rationale: it is better to serve an unguarded answer than to
take the entire service down due to a guardrail misconfiguration. Alerts
should be set on the "NeMo Guardrails disabled" log line in production.
"""
import re
from pathlib import Path
from typing import Optional, Tuple

import structlog

logger = structlog.get_logger()

_RAILS_CONFIG_DIR = Path(__file__).parent
_rails_instance = None

# ── Fast regex pre-check (no LLM call required) ───────────────────────────────
# Catches common prompt injection / jailbreak patterns before NeMo runs.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"you\s+are\s+now\s+DAN", re.I),
    re.compile(r"forget\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"(disregard|bypass)\s+(all\s+)?(previous\s+)?instructions", re.I),
    re.compile(r"###\s*instruction", re.I),
    re.compile(r"\[INST\]\s*ignore", re.I),
    re.compile(r"<\|im_start\|>\s*system", re.I),
    re.compile(r"pretend\s+you\s+are\s+(an?\s+)?(evil|unrestricted|unfiltered)", re.I),
    re.compile(r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|internal\s+instructions)", re.I),
    re.compile(r"override\s+(your\s+)?programming", re.I),
]

_INJECTION_REFUSAL = (
    "I detected an attempt to inject instructions or bypass my guidelines. "
    "I can only answer questions about the knowledge base."
)


def _get_rails():
    """Lazy-load and cache the NeMo LLMRails instance."""
    global _rails_instance
    if _rails_instance is not None:
        return _rails_instance

    try:
        from nemoguardrails import LLMRails, RailsConfig  # noqa: PLC0415

        # NeMo reads API keys from OS env vars, not from pydantic-settings.
        # Ensure the key is exported before initialising LLMRails.
        from app.core.config import get_settings as _get_settings  # noqa: PLC0415
        import os  # noqa: PLC0415
        _s = _get_settings()
        os.environ.setdefault("OPENAI_API_KEY", _s.openai_api_key)

        config = RailsConfig.from_path(str(_RAILS_CONFIG_DIR))
        _rails_instance = LLMRails(config)
        logger.info("NeMo Guardrails initialised", config_dir=str(_RAILS_CONFIG_DIR))
    except ImportError:
        logger.warning(
            "nemoguardrails not installed — input guardrails disabled. "
            "Install with: pip install nemoguardrails"
        )
    except Exception as exc:
        logger.error("Failed to initialise NeMo Guardrails", error=str(exc))

    return _rails_instance


# Phrases that appear in NeMo's colang refusal responses
_REFUSAL_MARKERS = (
    "unable to comply",
    "cannot comply",
    "i'm sorry",
    "i am sorry",
    "i'm specialised",
    "i detected",
    "not able to",
)


async def check_input(query: str) -> Tuple[bool, str]:
    """
    Run NeMo input guardrails on the user query.

    Returns:
        (is_safe, refusal_message)
        • is_safe=True  → query passed all rails; proceed with retrieval.
        • is_safe=False → a rail fired; return refusal_message directly.
    """
    from app.core.config import get_settings  # noqa: PLC0415

    settings = get_settings()

    # ── Layer 1: fast regex check (always runs, no LLM cost) ─────────────────
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            logger.warning(
                "Input guardrail: regex injection pattern matched",
                query_snippet=query[:80],
                pattern=pattern.pattern,
            )
            return False, _INJECTION_REFUSAL

    if not settings.enable_nemo_guardrails:
        return True, ""

    rails = _get_rails()
    if rails is None:
        return True, ""  # Fail-open

    try:
        messages = [{"role": "user", "content": query}]
        response = await rails.generate_async(messages=messages)

        # NeMo returns a dict {"role": "assistant", "content": "..."}
        if isinstance(response, dict):
            bot_text: str = response.get("content", "")
        else:
            bot_text = str(response)

        is_blocked = any(marker in bot_text.lower() for marker in _REFUSAL_MARKERS)
        if is_blocked:
            logger.warning(
                "NeMo guardrail triggered",
                query_snippet=query[:80],
                response=bot_text[:120],
            )
            return False, bot_text

        return True, ""

    except Exception as exc:
        logger.error("NeMo guardrail error — failing open", error=str(exc))
        return True, ""
