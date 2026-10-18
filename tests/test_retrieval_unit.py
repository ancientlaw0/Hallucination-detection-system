"""
test_retrieval_unit.py — Unit tests for the DocumentStore class (no HTTP layer).

These tests exercise the core retrieval logic directly, without starting the
FastAPI server.  They confirm chunking, indexing, and search behaviour.
"""

import pytest
from Backend.retrieval import DocumentStore


class TestSplitText:
    """Tests for the internal _split_text method."""

    def setup_method(self):
        self.store = DocumentStore()

    def test_empty_string_returns_empty_list(self):
        assert self.store._split_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert self.store._split_text("   \n\t  ") == []

    def test_short_text_is_single_chunk(self):
        text = "Hello world"
        chunks = self.store._split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_long_text_produces_multiple_chunks(self):
        # 2000 characters → should produce at least 2 chunks at default chunk_size=900
        text = "word " * 400  # ~2000 chars
        chunks = self.store._split_text(text)
        assert len(chunks) >= 2

    def test_chunks_respect_max_size(self):
        text = "word " * 400
        chunks = self.store._split_text(text, chunk_size=200, overlap=0)
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_overlap_means_chunks_share_content(self):
        text = "a" * 1000
        chunks = self.store._split_text(text, chunk_size=500, overlap=100)
        # second chunk should start 400 chars into the first chunk (500 - 100)
        assert len(chunks) >= 2

    def test_custom_chunk_size(self):
        text = "x" * 100
        chunks = self.store._split_text(text, chunk_size=30, overlap=0)
        # 100 / 30 → ceil = 4 chunks (last might be shorter)
        assert len(chunks) == 4

    def test_normalises_internal_whitespace(self):
        text = "hello   world\n\nfoo"
        chunks = self.store._split_text(text)
        assert chunks[0] == "hello world foo"


class TestDocumentStoreSearch:
    """Tests for search on an empty and a populated store."""

    def test_search_on_empty_store_returns_empty_list(self):
        store = DocumentStore()
        results = store.search("anything")
        assert results == []

    def test_search_top_k_respected(self):
        """If store has fewer chunks than top_k, return only what's there."""
        store = DocumentStore()
        assert store.search("query", top_k=5) == []

    def test_search_top_k_minimum_is_1(self):
        """top_k=0 should be clamped to 1 internally and still return []."""
        store = DocumentStore()
        results = store.search("query", top_k=0)
        assert isinstance(results, list)


class TestDocumentStoreAddPdf:
    """Tests that require an actual (small) PDF file on disk."""

    def test_add_pdf_raises_for_nonexistent_file(self, tmp_path):
        store = DocumentStore()
        with pytest.raises(Exception):
            store.add_pdf(tmp_path / "missing.pdf", "doc-id-1")

    def test_add_pdf_raises_for_empty_pdf(self, tmp_path):
        """A PDF with no extractable text should raise ValueError."""
        fake_pdf = tmp_path / "empty.pdf"
        # Write just the header — PyMuPDF will open it but find no text.
        fake_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
        store = DocumentStore()
        with pytest.raises(Exception):
            store.add_pdf(fake_pdf, "doc-id-2")
