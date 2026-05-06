"""Document loaders: PDF, web pages, Markdown, and raw text.

All loaders return a list of LangChain Document objects so the rest of
the pipeline is source-format agnostic.
"""
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

logger = structlog.get_logger()


def load_pdf(file_path: str | Path) -> List[Document]:
    """Load a PDF file via PyMuPDF (fast, no external services)."""
    loader = PyMuPDFLoader(str(file_path))
    docs = loader.load()
    logger.info("Loaded PDF", path=str(file_path), pages=len(docs))
    return docs


def load_url(url: str) -> List[Document]:
    """Fetch and parse an HTML page via BeautifulSoup."""
    import os
    os.environ.setdefault("USER_AGENT", "NexusRAG/1.0 (portfolio project)")
    loader = WebBaseLoader(web_paths=[url])
    docs = loader.load()
    # Attach URL as source metadata
    for doc in docs:
        doc.metadata.setdefault("source", url)
    logger.info("Loaded URL", url=url, docs=len(docs))
    return docs


def load_text(
    text: str,
    source: str = "manual",
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Wrap a raw text string in a Document."""
    doc = Document(
        page_content=text,
        metadata={"source": source, **(metadata or {})},
    )
    logger.info("Loaded raw text", source=source, chars=len(text))
    return [doc]


def load_markdown(file_path: str | Path) -> List[Document]:
    """Load a Markdown file (uses Unstructured under the hood)."""
    loader = UnstructuredMarkdownLoader(str(file_path))
    docs = loader.load()
    logger.info("Loaded Markdown", path=str(file_path), docs=len(docs))
    return docs


def load_from_bytes(
    file_bytes: bytes,
    filename: str,
    content_type: str = "",
) -> List[Document]:
    """
    Load a document from raw bytes — used for the file-upload API endpoint.
    Writes to a temp file and dispatches to the appropriate loader.
    """
    suffix = Path(filename).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    if "pdf" in content_type or suffix == ".pdf":
        return load_pdf(tmp_path)
    elif suffix in (".md", ".markdown"):
        return load_markdown(tmp_path)
    else:
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1", errors="replace")
        return load_text(text, source=filename)
