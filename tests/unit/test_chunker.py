"""Unit tests for the document chunker."""
import pytest
from langchain_core.documents import Document

from app.ingestion.chunker import chunk_documents


class TestChunkDocuments:
    def test_splits_long_document(self):
        docs = [Document(page_content="word " * 300, metadata={"source": "test"})]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_preserves_source_metadata(self):
        docs = [Document(page_content="Short text.", metadata={"source": "my_file.pdf"})]
        chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=64)
        assert all(c.metadata.get("source") == "my_file.pdf" for c in chunks)

    def test_short_document_stays_one_chunk(self):
        docs = [Document(page_content="Hello world.", metadata={})]
        chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1
        assert chunks[0].page_content == "Hello world."

    def test_start_index_added(self):
        docs = [Document(page_content="A " * 200, metadata={})]
        chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=10)
        assert all("start_index" in c.metadata for c in chunks)

    def test_multiple_documents(self):
        docs = [
            Document(page_content="Doc one content. " * 50, metadata={"source": "a.pdf"}),
            Document(page_content="Doc two content. " * 50, metadata={"source": "b.pdf"}),
        ]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 2
        sources = {c.metadata.get("source") for c in chunks}
        assert "a.pdf" in sources
        assert "b.pdf" in sources

    def test_empty_input(self):
        chunks = chunk_documents([], chunk_size=512, chunk_overlap=64)
        assert chunks == []
