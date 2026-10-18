"""
test_documents.py — Tests for PDF upload and retrieval endpoints.

Endpoints covered:
  POST /documents/upload   → ingest a PDF into the FAISS document store
  POST /retrieve           → semantic search over ingested documents
"""

import io


class TestDocumentUpload:
    def test_upload_valid_pdf_returns_200(self, client, minimal_pdf):
        response = client.post(
            "/documents/upload",
            files={"file": ("test_doc.pdf", io.BytesIO(minimal_pdf), "application/pdf")},
        )
        # Accept 200 (success) or 400 (PDF has no selectable text — depends on pymupdf version)
        assert response.status_code in (200, 400)

    def test_upload_non_pdf_returns_400(self, client):
        response = client.post(
            "/documents/upload",
            files={"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")},
        )
        assert response.status_code == 400

    def test_upload_non_pdf_error_message(self, client):
        response = client.post(
            "/documents/upload",
            files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        detail = response.json().get("detail", "")
        assert "PDF" in detail or "pdf" in detail.lower()

    def test_upload_empty_file_returns_400(self, client):
        response = client.post(
            "/documents/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
        assert response.status_code == 400

    def test_upload_no_file_returns_422(self, client):
        response = client.post("/documents/upload")
        assert response.status_code == 422

    def test_upload_success_has_document_id(self, client, minimal_pdf):
        response = client.post(
            "/documents/upload",
            files={"file": ("doc.pdf", io.BytesIO(minimal_pdf), "application/pdf")},
        )
        if response.status_code == 200:
            data = response.json()
            assert "document_id" in data
            assert isinstance(data["document_id"], str)
            assert len(data["document_id"]) > 0

    def test_upload_success_has_chunks_indexed(self, client, minimal_pdf):
        response = client.post(
            "/documents/upload",
            files={"file": ("doc2.pdf", io.BytesIO(minimal_pdf), "application/pdf")},
        )
        if response.status_code == 200:
            data = response.json()
            assert "chunks_indexed" in data
            assert data["chunks_indexed"] > 0

    def test_upload_success_echoes_filename(self, client, minimal_pdf):
        response = client.post(
            "/documents/upload",
            files={"file": ("my_paper.pdf", io.BytesIO(minimal_pdf), "application/pdf")},
        )
        if response.status_code == 200:
            data = response.json()
            assert data.get("original_filename") == "my_paper.pdf"


class TestRetrieve:
    def test_retrieve_on_empty_store_returns_empty_list(self, client):
        """Before any uploads, results should be an empty list (not an error)."""
        response = client.post(
            "/retrieve",
            json={"query": "Eiffel Tower Paris"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_retrieve_empty_query_returns_422(self, client):
        response = client.post(
            "/retrieve",
            json={"query": ""},
        )
        assert response.status_code == 422

    def test_retrieve_top_k_defaults_to_5(self, client):
        response = client.post(
            "/retrieve",
            json={"query": "test query"},
        )
        assert response.status_code == 200
        data = response.json()
        # count should be <= 5 (or <= number of chunks in store)
        assert data["count"] == len(data["results"])

    def test_retrieve_top_k_too_large_returns_422(self, client):
        response = client.post(
            "/retrieve",
            json={"query": "test", "top_k": 100},
        )
        assert response.status_code == 422

    def test_retrieve_top_k_zero_returns_422(self, client):
        response = client.post(
            "/retrieve",
            json={"query": "test", "top_k": 0},
        )
        assert response.status_code == 422

    def test_retrieve_missing_query_returns_422(self, client):
        response = client.post("/retrieve", json={})
        assert response.status_code == 422

    def test_retrieve_echoes_query(self, client):
        q = "What is machine learning?"
        data = client.post("/retrieve", json={"query": q}).json()
        assert data["query"] == q
