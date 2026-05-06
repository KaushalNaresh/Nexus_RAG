"""Health check endpoint — used as a liveness probe by Railway/Render."""
from fastapi import APIRouter

from app.core.config import get_settings
from app.models.response import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Return service health status. Always 200 if the process is alive."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        environment=settings.environment,
    )
