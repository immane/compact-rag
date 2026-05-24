"""API integration tests for /v1/ingestion-jobs endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from compact_rag.api.deps import _cached_settings, get_db_session
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
def client(test_settings):
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


def _make_job(**overrides):
    defaults = {
        "id": str(uuid4()),
        "collection_id": str(uuid4()),
        "status": "completed",
        "total_files": 5,
        "processed_files": 5,
        "total_chunks": 42,
        "errors": None,
        "started_at": None,
        "completed_at": None,
        "created_at": None,
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


# ── GET /v1/ingestion-jobs ─────────────────────────────────────


class TestListIngestionJobs:
    def test_list_ingestion_jobs_with_data(self, client, monkeypatch):
        job1 = _make_job(id="job-1", status="completed", total_files=3, total_chunks=15)
        job2 = _make_job(id="job-2", status="running", total_files=10, processed_files=5)

        class FakeRepo:
            async def list(self, session, page=1, page_size=20, **filters):
                return [job1, job2], 2

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            FakeRepo,
        )

        response = client.get("/v1/ingestion-jobs")
        assert response.status_code == 200
        data = response.json()
        _assert_paginated_contract(data)
        assert len(data["data"]) == 2
        assert data["pagination"]["total"] == 2
        assert data["data"][0]["status"] == "completed"
        assert data["data"][1]["status"] == "running"

    def test_list_ingestion_jobs_empty(self, client, monkeypatch):
        class FakeRepo:
            async def list(self, session, page=1, page_size=20, **filters):
                return [], 0

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            FakeRepo,
        )

        response = client.get("/v1/ingestion-jobs")
        assert response.status_code == 200
        data = response.json()
        _assert_paginated_contract(data)
        assert data["data"] == []
        assert data["pagination"]["total"] == 0
        assert data["pagination"]["total_pages"] == 0

    def test_list_ingestion_jobs_with_status_filter(self, client, monkeypatch):
        job = _make_job(
            id="job-failed",
            status="failed",
            errors=[{"file": "a.pdf", "error": "parse error"}],
        )

        class FakeRepo:
            async def list(self, session, page=1, page_size=20, **filters):
                return [job], 1

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            FakeRepo,
        )

        response = client.get("/v1/ingestion-jobs", params={"status": "failed"})
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["status"] == "failed"
        assert data["data"][0]["errors"] is not None
        assert len(data["data"][0]["errors"]) == 1

    def test_list_ingestion_jobs_with_collection_filter(self, client, monkeypatch):
        job = _make_job(id="job-coll", collection_id="coll-specific")

        class FakeRepo:
            async def list(self, session, page=1, page_size=20, **filters):
                return [job], 1

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            FakeRepo,
        )

        response = client.get("/v1/ingestion-jobs", params={"collection": "coll-specific"})
        assert response.status_code == 200
        assert len(response.json()["data"]) == 1

    def test_list_ingestion_jobs_with_both_filters(self, client, monkeypatch):
        job = _make_job(
            id="job-both",
            collection_id="coll-x",
            status="running",
            total_files=3,
            processed_files=1,
        )

        class FakeRepo:
            async def list(self, session, page=1, page_size=20, **filters):
                return [job], 1

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            FakeRepo,
        )

        response = client.get("/v1/ingestion-jobs", params={
            "status": "running",
            "collection": "coll-x",
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["total_files"] == 3
        assert data["data"][0]["processed_files"] == 1

    def test_list_ingestion_jobs_pagination(self, client, monkeypatch):
        jobs = [_make_job(id=f"job-{i}") for i in range(45)]

        class FakeRepo:
            async def list(self, session, page=1, page_size=20, **filters):
                start = (page - 1) * page_size
                end = start + page_size
                batch = jobs[start:end]
                return batch, 45

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            FakeRepo,
        )

        response = client.get("/v1/ingestion-jobs", params={"page": 2, "page_size": 20})
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 20
        assert data["pagination"]["page"] == 2
        assert data["pagination"]["total"] == 45
        assert data["pagination"]["total_pages"] == 3

    def test_list_ingestion_jobs_page_size_boundaries(self, client, monkeypatch):
        job = _make_job(id="job-bound")

        class FakeRepo:
            async def list(self, session, page=1, page_size=20, **filters):
                return [job], 1

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            FakeRepo,
        )

        response = client.get("/v1/ingestion-jobs", params={"page_size": 100})
        assert response.status_code == 200

        response = client.get("/v1/ingestion-jobs", params={"page_size": 1})
        assert response.status_code == 200

    def test_list_ingestion_jobs_invalid_page_422(self, client):
        response = client.get("/v1/ingestion-jobs", params={"page": 0})
        assert response.status_code == 422

    def test_list_ingestion_jobs_invalid_page_size_422(self, client):
        response = client.get("/v1/ingestion-jobs", params={"page_size": -1})
        assert response.status_code == 422

    def test_list_ingestion_jobs_page_size_over_max_422(self, client):
        response = client.get("/v1/ingestion-jobs", params={"page_size": 101})
        assert response.status_code == 422


# ── GET /v1/ingestion-jobs/{job_id} ─────────────────────────────


class TestGetIngestionJob:
    def test_get_existing_job_with_details(self, client, monkeypatch):
        job = _make_job(
            id="job-detail",
            collection_id="coll-1",
            status="completed",
            total_files=2,
            processed_files=2,
            total_chunks=20,
            errors=[],
        )

        class FakeRepo:
            async def get_by_id(self, session, job_id):
                return job if job_id == "job-detail" else None

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            FakeRepo,
        )

        response = client.get("/v1/ingestion-jobs/job-detail")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "job-detail"
        assert data["collection_id"] == "coll-1"
        assert data["status"] == "completed"
        assert data["total_files"] == 2
        assert data["processed_files"] == 2
        assert data["total_chunks"] == 20
        assert data["errors"] == []

    def test_get_nonexistent_job(self, client, monkeypatch):
        class FakeRepo:
            async def get_by_id(self, session, job_id):
                return None

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            FakeRepo,
        )

        response = client.get("/v1/ingestion-jobs/nonexistent")
        assert response.status_code == 404
        _assert_error_format(response.json())
        assert "FileNotFoundError" in response.json()["error"]["code"]

    def test_get_job_with_errors(self, client, monkeypatch):
        job = _make_job(
            id="job-err",
            status="completed",
            errors=[
                {"filename": "broken.pdf", "error": "corrupted file"},
                {"filename": "bad.doc", "error": "unsupported format"},
            ],
        )

        class FakeRepo:
            async def get_by_id(self, session, job_id):
                return job

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            FakeRepo,
        )

        response = client.get("/v1/ingestion-jobs/job-err")
        assert response.status_code == 200
        data = response.json()
        assert len(data["errors"]) == 2
        assert data["errors"][0]["filename"] == "broken.pdf"

    def test_get_job_response_structure(self, client, monkeypatch):
        job = _make_job(id="job-struct")

        class FakeRepo:
            async def get_by_id(self, session, job_id):
                return job

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            FakeRepo,
        )

        response = client.get("/v1/ingestion-jobs/job-struct")
        assert response.status_code == 200
        item = response.json()
        expected_keys = {
            "id", "collection_id", "status", "total_files", "processed_files",
            "total_chunks", "errors", "started_at", "completed_at", "created_at",
        }
        assert set(item.keys()) == expected_keys

    def test_get_running_job(self, client, monkeypatch):
        job = _make_job(
            id="job-running",
            status="running",
            total_files=5,
            processed_files=3,
            total_chunks=30,
        )

        class FakeRepo:
            async def get_by_id(self, session, job_id):
                return job

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            FakeRepo,
        )

        response = client.get("/v1/ingestion-jobs/job-running")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["total_files"] == 5
        assert data["processed_files"] == 3

    def test_get_pending_job(self, client, monkeypatch):
        job = _make_job(
            id="job-pending",
            status="pending",
            total_files=1,
            processed_files=0,
            total_chunks=0,
        )

        class FakeRepo:
            async def get_by_id(self, session, job_id):
                return job

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            FakeRepo,
        )

        response = client.get("/v1/ingestion-jobs/job-pending")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["processed_files"] == 0
        assert data["total_chunks"] == 0

    @pytest.mark.integration
    def test_list_ingestion_jobs_filters_propagated_correctly(self, client, monkeypatch):
        captured: dict[str, object] = {}

        class FakeRepo:
            async def list(self, session, page=1, page_size=20, **filters):
                captured["page"] = page
                captured["page_size"] = page_size
                captured["filters"] = filters
                return [], 0

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            FakeRepo,
        )

        response = client.get("/v1/ingestion-jobs", params={
            "status": "completed",
            "collection": "col-abc",
            "page": 3,
            "page_size": 50,
        })

        assert response.status_code == 200
        assert captured["page"] == 3
        assert captured["page_size"] == 50
        assert captured["filters"] == {
            "status": "completed",
            "collection_id": "col-abc",
        }
