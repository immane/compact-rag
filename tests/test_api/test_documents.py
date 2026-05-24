"""API integration tests for /v1/documents endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from compact_rag.api.deps import _cached_settings, get_db_session, get_storage_backend
from compact_rag.api.router import create_app


def _make_session():
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.delete = MagicMock()
    return session


@pytest.fixture
def client(test_settings, monkeypatch):
    _cached_settings.cache_clear()

    app = create_app(settings=test_settings)

    async def fake_db_session():
        session = _make_session()
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db_session] = fake_db_session

    with TestClient(app) as c:
        yield c


def _make_document(**overrides):
    defaults = {
        "id": str(uuid4()),
        "collection_id": str(uuid4()),
        "filename": "test-doc.pdf",
        "file_type": "pdf",
        "file_size": 1024,
        "file_hash": "abc123",
        "page_count": 5,
        "chunk_count": 10,
        "table_count": 2,
        "status": "completed",
        "error_message": None,
        "created_at": None,
        "updated_at": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _assert_paginated_contract(data: dict) -> None:
    assert set(data.keys()) == {"data", "pagination"}
    pagination = data["pagination"]
    assert set(pagination.keys()) == {"page", "page_size", "total", "total_pages"}


def _assert_error_format(data: dict) -> None:
    assert "error" in data
    err = data["error"]
    assert set(err.keys()) == {"code", "message", "details", "request_id"}


class _FakeFailingClient:
    """httpx AsyncClient mock that returns a failing response on get()."""
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def get(self, url, **kwargs):
        class FakeResponse:
            status_code = 400
            content = b""

            def raise_for_status(self):
                import httpx
                raise httpx.HTTPStatusError("Bad request", request=object(), response=self)

        return FakeResponse()


# ── POST /v1/documents/ingest ──────────────────────────────────


class TestIngestDocument:
    def test_file_upload_success(self, client, test_settings, monkeypatch):
        ingestion_result = SimpleNamespace(
            doc_id=str(uuid4()),
            filename="test.pdf",
            status="completed",
            chunk_count=5,
            table_count=1,
            error_message=None,
            duration_ms=123.4,
        )

        class FakePipeline:
            def __init__(self, *, settings, session):
                pass

            async def ingest_file(self, *, file_path, collection_name, force=False):
                return ingestion_result

        monkeypatch.setattr(
            "compact_rag.ingestion.pipeline.IngestionPipeline", FakePipeline
        )

        class FakeStorage:
            async def upload_bytes(self, content, key):
                pass

        client.app.dependency_overrides[get_storage_backend] = lambda: FakeStorage()

        with open(__file__, "rb") as f:
            response = client.post(
                "/v1/documents/ingest",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"collection": "default", "force": "false"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == ingestion_result.doc_id
        assert data["filename"] == "test.pdf"
        assert data["status"] == "completed"
        assert data["chunk_count"] == 5
        assert data["table_count"] == 1
        assert data["duration_ms"] == 123.4
        assert data["error_message"] is None

    def test_file_upload_missing_file(self, client):
        response = client.post(
            "/v1/documents/ingest",
            data={"collection": "default"},
        )
        assert response.status_code == 422

    def test_file_upload_no_filename(self, client):
        """Empty filename is rejected by FastAPI's UploadFile validation."""
        response = client.post(
            "/v1/documents/ingest",
            files={"file": ("", b"content", "application/octet-stream")},
            data={"collection": "default"},
        )
        assert response.status_code == 422

    def test_file_upload_unsupported_extension(self, client, test_settings):
        response = client.post(
            "/v1/documents/ingest",
            files={"file": ("video.mp4", b"fake-video", "video/mp4")},
            data={"collection": "default"},
        )
        assert response.status_code == 400
        _assert_error_format(response.json())
        assert response.json()["error"]["code"] == "UnsupportedFormatError"

    def test_file_upload_pipeline_exception_returns_failed_response(self, client, test_settings, monkeypatch):
        class FailingPipeline:
            def __init__(self, *, settings, session):
                pass

            async def ingest_file(self, *, file_path, collection_name, force=False):
                raise RuntimeError("ingestion pipeline crashed")

        monkeypatch.setattr(
            "compact_rag.ingestion.pipeline.IngestionPipeline", FailingPipeline
        )

        class FakeStorage:
            async def upload_bytes(self, content, key):
                pass

        client.app.dependency_overrides[get_storage_backend] = lambda: FakeStorage()

        with open(__file__, "rb") as f:
            response = client.post(
                "/v1/documents/ingest",
                files={"file": ("doc.txt", f, "text/plain")},
                data={"collection": "default"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "ingestion pipeline crashed" in data["error_message"]

    def test_file_upload_storage_upload_bytes_called(self, client, test_settings, monkeypatch):
        ingestion_result = SimpleNamespace(
            doc_id=str(uuid4()),
            filename="report.pdf",
            status="completed",
            chunk_count=3,
            table_count=0,
            error_message=None,
            duration_ms=50.0,
        )

        class FakePipeline:
            def __init__(self, *, settings, session):
                pass

            async def ingest_file(self, *, file_path, collection_name, force=False):
                return ingestion_result

        monkeypatch.setattr(
            "compact_rag.ingestion.pipeline.IngestionPipeline", FakePipeline
        )

        upload_calls = []

        class FakeStorage:
            async def upload_bytes(self, content, key):
                upload_calls.append(key)

        client.app.dependency_overrides[get_storage_backend] = lambda: FakeStorage()

        with open(__file__, "rb") as f:
            response = client.post(
                "/v1/documents/ingest",
                files={"file": ("report.pdf", f, "application/pdf")},
                data={"collection": "default"},
            )

        assert response.status_code == 200
        assert len(upload_calls) >= 2


# ── POST /v1/documents/ingest-url ──────────────────────────────


class TestIngestDocumentUrl:
    def test_ingest_url_success(self, client, test_settings, monkeypatch):
        import httpx

        ingestion_result = SimpleNamespace(
            doc_id=str(uuid4()),
            filename="document.txt",
            status="completed",
            chunk_count=8,
            table_count=0,
            error_message=None,
            duration_ms=200.0,
        )

        class FakePipeline:
            def __init__(self, *, settings, session):
                pass

            async def ingest_file(self, *, file_path, collection_name):
                return ingestion_result

        monkeypatch.setattr(
            "compact_rag.ingestion.pipeline.IngestionPipeline", FakePipeline
        )

        class FakeResponse:
            status_code = 200
            content = b"hello world from url"

            def raise_for_status(self):
                pass

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url, **kwargs):
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())

        response = client.post(
            "/v1/documents/ingest-url",
            json={"url": "https://example.com/doc.txt", "collection": "default"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["chunk_count"] == 8

    def test_ingest_url_invalid_url_missing_scheme(self, client, monkeypatch):
        """URL without scheme causes httpx error, propagating as 500 through TestClient."""
        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _FakeFailingClient())

        with pytest.raises(httpx.HTTPStatusError):
            client.post(
                "/v1/documents/ingest-url",
                json={"url": "not-a-valid-url", "collection": "default"},
            )

    def test_ingest_url_empty_url(self, client, monkeypatch):
        """Empty URL causes httpx error, propagating as 500 through TestClient."""
        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _FakeFailingClient())

        with pytest.raises(httpx.HTTPStatusError):
            client.post(
                "/v1/documents/ingest-url",
                json={"url": "", "collection": "default"},
            )

    def test_ingest_url_default_collection(self, client, test_settings, monkeypatch):
        import httpx

        ingestion_result = SimpleNamespace(
            doc_id=str(uuid4()),
            filename="article.html",
            status="completed",
            chunk_count=12,
            table_count=1,
            error_message=None,
            duration_ms=350.0,
        )

        class FakePipeline:
            def __init__(self, *, settings, session):
                pass

            async def ingest_file(self, *, file_path, collection_name):
                return ingestion_result

        monkeypatch.setattr(
            "compact_rag.ingestion.pipeline.IngestionPipeline", FakePipeline
        )

        class FakeResponse:
            status_code = 200
            content = b"<html>article content</html>"

            def raise_for_status(self):
                pass

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url, **kwargs):
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())

        response = client.post(
            "/v1/documents/ingest-url",
            json={"url": "https://example.com/article.html"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "article.html"


# ── GET /v1/documents ──────────────────────────────────────────


class TestListDocuments:
    def test_list_documents_with_data(self, client, monkeypatch):
        doc1 = _make_document(id="d1", filename="a.pdf", status="completed")
        doc2 = _make_document(id="d2", filename="b.pdf", status="pending")

        class FakeRepo:
            async def list_with_filters(self, session, collection_id=None, status=None, page=1, page_size=20):
                return [doc1, doc2], 2

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.document.DocumentRepository",
            FakeRepo,
        )

        response = client.get("/v1/documents")
        assert response.status_code == 200
        data = response.json()
        _assert_paginated_contract(data)
        assert len(data["data"]) == 2
        assert data["pagination"]["total"] == 2

    def test_list_documents_empty(self, client, monkeypatch):
        class FakeRepo:
            async def list_with_filters(self, session, collection_id=None, status=None, page=1, page_size=20):
                return [], 0

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.document.DocumentRepository",
            FakeRepo,
        )

        response = client.get("/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
        assert data["pagination"]["total"] == 0

    def test_list_documents_with_collection_filter(self, client, monkeypatch):
        doc = _make_document(id="d3", collection_id="coll-x", filename="c.pdf")

        class FakeRepo:
            async def list_with_filters(self, session, collection_id=None, status=None, page=1, page_size=20):
                return [doc], 1

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.document.DocumentRepository",
            FakeRepo,
        )

        response = client.get("/v1/documents", params={"collection": "coll-x"})
        assert response.status_code == 200
        assert len(response.json()["data"]) == 1

    def test_list_documents_with_status_filter(self, client, monkeypatch):
        doc = _make_document(id="d4", status="failed", error_message="something went wrong")

        class FakeRepo:
            async def list_with_filters(self, session, collection_id=None, status=None, page=1, page_size=20):
                return [doc], 1

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.document.DocumentRepository",
            FakeRepo,
        )

        response = client.get("/v1/documents", params={"status": "failed"})
        assert response.status_code == 200
        assert response.json()["data"][0]["status"] == "failed"

    def test_list_documents_with_both_filters(self, client, monkeypatch):
        doc = _make_document(id="d5", collection_id="coll-y", status="completed")

        class FakeRepo:
            async def list_with_filters(self, session, collection_id=None, status=None, page=1, page_size=20):
                return [doc], 1

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.document.DocumentRepository",
            FakeRepo,
        )

        response = client.get("/v1/documents", params={
            "collection": "coll-y", "status": "completed",
        })
        assert response.status_code == 200
        assert len(response.json()["data"]) == 1

    def test_list_documents_pagination_boundaries(self, client, monkeypatch):
        documents = [_make_document(id=f"d{i}", filename=f"doc{i}.pdf") for i in range(50)]

        class FakeRepo:
            async def list_with_filters(self, session, collection_id=None, status=None, page=1, page_size=20):
                start = (page - 1) * page_size
                end = start + page_size
                batch = documents[start:end]
                return batch, len(documents)

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.document.DocumentRepository",
            FakeRepo,
        )

        response = client.get("/v1/documents", params={"page": 1, "page_size": 10})
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 10
        assert data["pagination"]["total"] == 50
        assert data["pagination"]["total_pages"] == 5

        response2 = client.get("/v1/documents", params={"page": 3, "page_size": 10})
        assert response2.status_code == 200
        assert len(response2.json()["data"]) == 10

    def test_list_documents_negative_page_size_422(self, client):
        response = client.get("/v1/documents", params={"page_size": -5})
        assert response.status_code == 422

    def test_list_documents_page_zero_422(self, client):
        response = client.get("/v1/documents", params={"page": 0})
        assert response.status_code == 422


# ── GET /v1/documents/{doc_id} ─────────────────────────────────


class TestGetDocument:
    def test_get_existing_document(self, client, monkeypatch):
        doc = _make_document(id="d-get", filename="existing.pdf")

        class FakeRepo:
            async def get_by_id(self, session, doc_id):
                return doc if doc_id == "d-get" else None

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.document.DocumentRepository",
            FakeRepo,
        )

        response = client.get("/v1/documents/d-get")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "d-get"
        assert data["filename"] == "existing.pdf"
        assert data["file_type"] == "pdf"

    def test_get_nonexistent_document(self, client, monkeypatch):
        class FakeRepo:
            async def get_by_id(self, session, doc_id):
                return None

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.document.DocumentRepository",
            FakeRepo,
        )

        response = client.get("/v1/documents/nonexistent-id")
        assert response.status_code == 404
        _assert_error_format(response.json())
        assert "FileNotFoundError" in response.json()["error"]["code"]

    def test_get_document_with_nullables(self, client, monkeypatch):
        doc = _make_document(id="d-null", page_count=None, error_message=None, created_at=None, updated_at=None)

        class FakeRepo:
            async def get_by_id(self, session, doc_id):
                return doc

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.document.DocumentRepository",
            FakeRepo,
        )

        response = client.get("/v1/documents/d-null")
        assert response.status_code == 200
        data = response.json()
        assert data["page_count"] is None
        assert data["error_message"] is None

    def test_get_document_response_structure(self, client, monkeypatch):
        doc = _make_document(id="d-struct", filename="full.pdf")

        class FakeRepo:
            async def get_by_id(self, session, doc_id):
                return doc

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.document.DocumentRepository",
            FakeRepo,
        )

        response = client.get("/v1/documents/d-struct")
        assert response.status_code == 200
        item = response.json()
        expected_keys = {
            "id", "collection_id", "filename", "file_type", "file_size",
            "file_hash", "page_count", "chunk_count", "table_count",
            "status", "error_message", "created_at", "updated_at",
        }
        assert set(item.keys()) == expected_keys


# ── DELETE /v1/documents/{doc_id} ───────────────────────────────


class TestDeleteDocument:
    def test_delete_existing_document(self, client, monkeypatch):
        doc = _make_document(id="d-del", filename="to-delete.pdf")

        delete_calls = []

        class FakeDocRepo:
            async def get_by_id(self, session, doc_id):
                return doc if doc_id == "d-del" else None

            async def delete(self, session, doc_id):
                delete_calls.append(doc_id)
                return True

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.document.DocumentRepository",
            FakeDocRepo,
        )

        class FakeVectorStore:
            async def delete_by_document(self, doc_id):
                pass

        monkeypatch.setattr(
            "compact_rag.api.deps.get_vector_store",
            lambda: FakeVectorStore(),
        )

        response = client.delete("/v1/documents/d-del")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["doc_id"] == "d-del"
        assert delete_calls == ["d-del"]

    def test_delete_nonexistent_document(self, client, monkeypatch):
        class FakeDocRepo:
            async def get_by_id(self, session, doc_id):
                return None

            async def delete(self, session, doc_id):
                return False

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.document.DocumentRepository",
            FakeDocRepo,
        )

        response = client.delete("/v1/documents/nonexistent-id")
        assert response.status_code == 404
        _assert_error_format(response.json())

    def test_delete_document_vector_store_error_ignored(self, client, monkeypatch):
        doc = _make_document(id="d-vs-err")

        class FakeDocRepo:
            async def get_by_id(self, session, doc_id):
                return doc

            async def delete(self, session, doc_id):
                return True

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.document.DocumentRepository",
            FakeDocRepo,
        )

        class FailingVectorStore:
            async def delete_by_document(self, doc_id):
                raise RuntimeError("chromadb unavailable")

        monkeypatch.setattr(
            "compact_rag.api.deps.get_vector_store",
            lambda: FailingVectorStore(),
        )

        response = client.delete("/v1/documents/d-vs-err")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"
