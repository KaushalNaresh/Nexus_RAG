"""Pydantic v2 response models for all API endpoints."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class SourceDocument(BaseModel):
    content: str = Field(..., description="Chunk text snippet (first 500 chars).")
    source: Optional[str] = Field(None, description="Origin file or URL.")
    score: Optional[float] = Field(None, description="Cross-encoder relevance score.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] = Field(default_factory=list)
    session_id: Optional[str] = None
    latency_ms: Optional[float] = None
    guardrail_triggered: bool = False
    guardrail_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=_utcnow)


class IngestResponse(BaseModel):
    success: bool
    chunks_indexed: int
    source: str
    message: str
    timestamp: datetime = Field(default_factory=_utcnow)


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=_utcnow)


class CompareResponse(BaseModel):
    naive: QueryResponse
    production: QueryResponse


class EvalResult(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: Optional[float] = None
    num_samples: int
    timestamp: datetime = Field(default_factory=_utcnow)
    details: Optional[List[Dict[str, Any]]] = None
