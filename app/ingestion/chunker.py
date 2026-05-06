"""Text chunking strategies.

Uses LangChain's RecursiveCharacterTextSplitter as the primary splitter.
Separators are ordered from coarse (paragraph) to fine (character) so the
splitter prefers natural semantic boundaries.
"""
from typing import List

import structlog
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = structlog.get_logger()


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Document]:
    """
    Split a list of Documents into overlapping chunks.

    Args:
        documents:     Source documents (output of any loader).
        chunk_size:    Target characters per chunk.
        chunk_overlap: Overlap between consecutive chunks to preserve context.

    Returns:
        Flat list of chunk Documents with inherited metadata plus
        a `start_index` field indicating position within the source.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    logger.info(
        "Chunked documents",
        input_docs=len(documents),
        output_chunks=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return chunks
