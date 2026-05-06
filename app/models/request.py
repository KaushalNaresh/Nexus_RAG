"""Pydantic v2 request models for all API endpoints."""
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language question to answer from the knowledge base.",
    )
    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Override the number of chunks to retrieve. Leave null to use the server default (20).",
    )
    alpha: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Hybrid search weight. 0.0 = pure keyword (BM25), 1.0 = pure semantic. Leave null for balanced (0.5).",
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session identifier for conversation tracking.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is Retrieval-Augmented Generation and how does it work?"
                }
            ]
        }
    }


class IngestURLRequest(BaseModel):
    url: HttpUrl = Field(
        ...,
        description="Publicly accessible URL to fetch and ingest.",
        examples=["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"],
    )
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata to attach to every chunk.",
    )


class IngestTextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        description="Raw text content to ingest.",
    )
    source: str = Field(
        "manual",
        description="Human-readable label for this content's origin.",
    )
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata.",
    )
