"""Structured JSON logging configuration using structlog.

In development: colourised console output.
In production: JSON lines for log aggregation (Datadog, CloudWatch, etc.).
"""
import logging
import os
import sys

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog for the application."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    if log_level.upper() in ("DEBUG", "INFO") and os.environ.get("ENVIRONMENT", "development") != "production":
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging through structlog so third-party libs appear in logs
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    for noisy_lib in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)
