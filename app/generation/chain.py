"""LangChain LCEL RAG chain.

Architecture
────────────
  Input dict: {"question": str, "context": List[dict]}
      │
      ├─ format_context()   Converts retrieval result dicts → numbered text block
      │
      ├─ RAG_PROMPT         Injects context + question into the system/human template
      │
      ├─ ChatOpenAI         GPT-4o-mini with temperature=0.1 for determinism
      │
      └─ StrOutputParser    Strips the AIMessage wrapper → plain string

The chain is built once (singleton) and reused across requests.
"""
from operator import itemgetter
from typing import Any, Dict, List, Optional

import structlog
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.generation.prompts import RAG_PROMPT

logger = structlog.get_logger()

_chain_instance = None


def format_context(docs: List[Dict[str, Any]]) -> str:
    """
    Convert a list of retrieval result dicts into a numbered context block.

    Each entry is labelled with its source so the LLM can cite it naturally.
    """
    if not docs:
        return "No relevant context found."

    parts: List[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("metadata", {}).get("source", f"Document {i}")
        score = doc.get("rerank_score")
        header = f"[{i}] Source: {source}"
        if score is not None:
            header += f" (relevance: {score:.3f})"
        parts.append(f"{header}\n{doc['text']}")

    return "\n\n---\n\n".join(parts)


def build_rag_chain():
    """Construct the LCEL chain. Called once; result cached in module scope."""
    settings = get_settings()

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.1,
        openai_api_key=settings.openai_api_key,
        max_retries=3,
    )

    chain = (
        {
            "context": itemgetter("context") | RunnableLambda(format_context),
            "question": itemgetter("question"),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    logger.info("RAG chain built", model=settings.openai_model)
    return chain


def get_rag_chain():
    """Return the singleton RAG chain, building it on first call."""
    global _chain_instance
    if _chain_instance is None:
        _chain_instance = build_rag_chain()
    return _chain_instance
