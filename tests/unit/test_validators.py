"""Unit tests for output validators (no external services required)."""
import pytest

from app.guardrails.output.validators import (
    check_faithfulness,
    check_toxicity,
    detect_pii,
    mask_pii,
)


# ── PII Detection ─────────────────────────────────────────────────────────────

class TestDetectPII:
    def test_detects_email(self):
        pii = detect_pii("Contact john.doe@example.com for support.")
        assert "email" in pii
        assert "john.doe@example.com" in pii["email"]

    def test_detects_ssn(self):
        pii = detect_pii("The SSN on record is 123-45-6789.")
        assert "ssn" in pii

    def test_detects_phone(self):
        pii = detect_pii("Call us at 555-123-4567 anytime.")
        assert "phone" in pii

    def test_no_pii_clean_text(self):
        pii = detect_pii("The weather is sunny today in San Francisco.")
        assert pii == {}

    def test_multiple_pii_types(self):
        pii = detect_pii("Email: test@test.com, SSN: 987-65-4321")
        assert "email" in pii
        assert "ssn" in pii


class TestMaskPII:
    def test_masks_email(self):
        result = mask_pii("Send mail to alice@corp.org please.")
        assert "@" not in result
        assert "[EMAIL_REDACTED]" in result

    def test_masks_ssn(self):
        result = mask_pii("SSN: 123-45-6789")
        assert "123-45-6789" not in result
        assert "[SSN_REDACTED]" in result

    def test_clean_text_unchanged(self):
        text = "This is a perfectly safe sentence."
        assert mask_pii(text) == text


# ── Faithfulness ──────────────────────────────────────────────────────────────

class TestFaithfulness:
    def test_faithful_answer(self):
        answer = "RAG uses retrieval augmented generation for grounded responses."
        context = ["Retrieval augmented generation combines search with language models."]
        is_faithful, score = check_faithfulness(answer, context)
        assert score > 0.35

    def test_unfaithful_answer(self):
        answer = "The stock market crashed dramatically in 1929."
        context = ["Hybrid search retrieval uses dense and sparse vectors."]
        _, score = check_faithfulness(answer, context)
        assert score < 0.35

    def test_empty_context_returns_faithful(self):
        is_faithful, score = check_faithfulness("Any answer", [])
        assert is_faithful is True
        assert score == 1.0

    def test_empty_answer_returns_faithful(self):
        is_faithful, score = check_faithfulness("", ["some context"])
        assert is_faithful is True

    def test_custom_threshold(self):
        answer = "RAG uses retrieval for grounded generation."
        context = ["RAG combines retrieval with language model generation."]
        is_faithful_strict, _ = check_faithfulness(answer, context, threshold=0.99)
        is_faithful_lenient, _ = check_faithfulness(answer, context, threshold=0.1)
        assert not is_faithful_strict
        assert is_faithful_lenient


# ── Toxicity ──────────────────────────────────────────────────────────────────

class TestToxicity:
    def test_safe_text(self):
        assert check_toxicity("This is a helpful AI assistant response.") is True

    def test_harmful_instruction(self):
        assert check_toxicity("How to make a bomb at home step by step.") is False

    def test_self_harm(self):
        assert check_toxicity("Instructions to kill yourself.") is False

    def test_neutral_sentence(self):
        assert check_toxicity("The retrieval pipeline uses BM25 and dense vectors.") is True
