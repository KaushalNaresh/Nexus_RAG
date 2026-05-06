#!/usr/bin/env python3
"""CLI: Ingest documents into the Nexus RAG knowledge base.

Examples
────────
# Ingest a single PDF
python scripts/ingest_docs.py --path ./data/report.pdf

# Ingest all PDFs and Markdown files in a directory
python scripts/ingest_docs.py --path ./data/

# Ingest a web page
python scripts/ingest_docs.py --url https://example.com/article

# Ingest a raw text string
python scripts/ingest_docs.py --text "Direct text to index." --source "manual_entry"
"""
import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.ingestion.loaders import load_markdown, load_pdf, load_text, load_url
from app.ingestion.pipeline import IngestionPipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest documents into the Nexus RAG Pinecone index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", type=str, help="Path to a file or directory.")
    source.add_argument("--url", type=str, help="URL to fetch and ingest.")
    source.add_argument("--text", type=str, help="Raw text string to ingest.")
    p.add_argument("--source", type=str, default="manual", help="Source label for raw text.")
    p.add_argument("--chunk-size", type=int, default=None, help="Override chunk size.")
    p.add_argument("--chunk-overlap", type=int, default=None, help="Override chunk overlap.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    docs = []

    if args.path:
        p = Path(args.path)
        if not p.exists():
            print(f"ERROR: Path does not exist: {p}", file=sys.stderr)
            sys.exit(1)

        targets = list(p.rglob("*")) if p.is_dir() else [p]
        for f in targets:
            if f.is_file():
                ext = f.suffix.lower()
                if ext == ".pdf":
                    docs.extend(load_pdf(f))
                elif ext in (".md", ".markdown"):
                    docs.extend(load_markdown(f))
                elif ext in (".txt", ".rst"):
                    docs.extend(load_text(f.read_text(encoding="utf-8"), source=str(f)))

        if not docs:
            print("No supported files found (PDF, MD, TXT).", file=sys.stderr)
            sys.exit(1)

    elif args.url:
        docs = load_url(args.url)

    else:
        docs = load_text(args.text, source=args.source)

    pipeline = IngestionPipeline()
    count = pipeline.run(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"\nIngestion complete: {count} chunks indexed into '{settings.pinecone_index_name}'.")


if __name__ == "__main__":
    main()
