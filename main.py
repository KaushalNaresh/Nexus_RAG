"""Nexus RAG — FastAPI application entrypoint.

Run locally:
    uvicorn main:app --reload

Run via Docker:
    docker compose -f docker/docker-compose.yml up
"""
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.api.routes import health, ingest, query

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(
        "Nexus RAG starting",
        environment=settings.environment,
        llm=settings.openai_model,
        embedding=settings.embedding_model,
        pinecone_index=settings.pinecone_index_name,
        nemo_guardrails=settings.enable_nemo_guardrails,
        output_guardrails=settings.enable_output_guardrails,
    )
    yield
    logger.info("Nexus RAG shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Nexus RAG",
        description=(
            "Production-Grade Retrieval-Augmented Generation API.\n\n"
            "Features: Hybrid Search (dense + BM25), Cross-Encoder Reranking, "
            "Dual-Layer Guardrails (NeMo + custom output validators), "
            "and a Ragas evaluation pipeline."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    _settings = get_settings()
    # In production, restrict CORS to the deployed Vercel frontend.
    # Set ALLOWED_ORIGINS env var on Render, e.g.:
    #   https://nexus-rag.vercel.app,https://nexus-rag-git-main.vercel.app
    # Falls back to wildcard for local dev.
    _raw_origins = _settings.allowed_origins or "*"
    _origins = (
        [o.strip() for o in _raw_origins.split(",")]
        if _raw_origins != "*"
        else ["*"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(ingest.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    return app


app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )
