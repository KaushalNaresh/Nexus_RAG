"""Unit tests for the OutputGuard pipeline."""
import pytest

from app.guardrails.output.guard import GuardResult, OutputGuard


@pytest.fixture
def guard():
    return OutputGuard(
        enable_pii_masking=True,
        enable_faithfulness=True,
        enable_toxicity=True,
        block_on_pii=False,
        faithfulness_threshold=0.35,
    )


@pytest.fixture
def context():
    return [
        "RAG uses retrieval augmented generation combining search and language models.",
        "Hybrid search merges dense embeddings with BM25 sparse keyword vectors.",
    ]


class TestOutputGuard:
    def test_safe_answer_passes(self, guard, context):
        answer = "RAG combines retrieval with language model generation."
        result = guard.validate(answer=answer, context_chunks=context)
        assert result.is_safe is True
        assert result.answer == answer

    def test_pii_is_masked_not_blocked(self, guard, context):
        # Answer deliberately references context vocabulary (retrieval, augmented,
        # generation, search, language) to pass faithfulness AND contains PII.
        answer = (
            "Email support@example.com to learn about retrieval augmented generation, "
            "hybrid search, and language models."
        )
        result = guard.validate(answer=answer, context_chunks=context)
        assert result.is_safe is True
        assert "support@example.com" not in result.answer
        assert "[EMAIL_REDACTED]" in result.answer
        assert "email" in result.pii_detected

    def test_pii_blocked_when_configured(self, context):
        guard = OutputGuard(block_on_pii=True)
        answer = "User SSN is 123-45-6789."
        result = guard.validate(answer=answer, context_chunks=context)
        assert result.is_safe is False
        assert "ssn" in result.pii_detected

    def test_toxic_content_blocked(self, guard, context):
        answer = "Instructions on how to make a bomb at home."
        result = guard.validate(answer=answer, context_chunks=context)
        assert result.is_safe is False
        assert result.is_toxic is True

    def test_unfaithful_answer_blocked(self, guard):
        answer = "The Roman Empire fell in 476 AD due to barbarian invasions."
        context = ["RAG hybrid search uses BM25 and dense embeddings for retrieval."]
        result = guard.validate(answer=answer, context_chunks=context)
        assert result.is_safe is False
        assert result.is_faithful is False

    def test_guard_disabled_by_settings(self, monkeypatch):
        from app.core.config import Settings

        mock_settings = Settings(
            openai_api_key="sk-test",
            pinecone_api_key="test",
            enable_output_guardrails=False,
        )
        from app.core import config
        monkeypatch.setattr(config, "get_settings", lambda: mock_settings)

        guard = OutputGuard()
        result = guard.validate(
            answer="Completely unrelated dangerous answer.",
            context_chunks=["safe context"],
        )
        # Guard disabled → pass-through
        assert result.is_safe is True
