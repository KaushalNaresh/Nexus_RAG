"""Ingestion API routes.

Three ingestion surfaces:
  POST /api/v1/ingest/file   — multipart file upload (PDF / MD / TXT)
  POST /api/v1/ingest/url    — fetch & ingest a public URL
  POST /api/v1/ingest/text   — post raw text directly
"""
import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.dependencies import get_ingestion_pipeline
from app.ingestion.loaders import load_from_bytes, load_text, load_url
from app.ingestion.pipeline import IngestionPipeline
from app.models.request import IngestTextRequest, IngestURLRequest
from app.models.response import IngestResponse

logger = structlog.get_logger()
router = APIRouter()


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Upload and ingest a PDF, Markdown, or plain-text file."""
    try:
        content = await file.read()
        docs = load_from_bytes(content, file.filename or "upload", file.content_type or "")
        count = pipeline.run(docs)
        return IngestResponse(
            success=True,
            chunks_indexed=count,
            source=file.filename or "upload",
            message=f"Indexed {count} chunks from '{file.filename}'.",
        )
    except Exception as exc:
        logger.error("File ingestion failed", filename=file.filename, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(
    body: IngestURLRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Fetch and ingest content from a publicly accessible URL."""
    url_str = str(body.url)
    try:
        docs = load_url(url_str)
        for doc in docs:
            doc.metadata.update(body.metadata or {})
        count = pipeline.run(docs)
        return IngestResponse(
            success=True,
            chunks_indexed=count,
            source=url_str,
            message=f"Indexed {count} chunks from URL.",
        )
    except Exception as exc:
        logger.error("URL ingestion failed", url=url_str, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(
    body: IngestTextRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Ingest raw text content directly."""
    try:
        docs = load_text(body.text, source=body.source, metadata=body.metadata)
        count = pipeline.run(docs)
        return IngestResponse(
            success=True,
            chunks_indexed=count,
            source=body.source,
            message=f"Indexed {count} chunks from text input.",
        )
    except Exception as exc:
        logger.error("Text ingestion failed", source=body.source, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
