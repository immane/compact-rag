from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from compact_rag.api.deps import _cached_settings, get_db_session
from compact_rag.api.router import create_app


@pytest.fixture
def client(test_settings):
    _cached_settings.cache_clear()

    app = create_app(settings=test_settings)

    async def fake_db_session():
        yield object()

    app.dependency_overrides[get_db_session] = fake_db_session

    with TestClient(app) as c:
        yield c


class TestIngestionJobsApi:
    def test_list_ingestion_jobs_applies_filters(self, client, monkeypatch):
        captured: dict[str, object] = {}

        class FakeRepo:
            async def list(self, session, page=1, page_size=20, **filters):
                captured["session"] = session
                captured["page"] = page
                captured["page_size"] = page_size
                captured["filters"] = filters
                return [
                    SimpleNamespace(
                        id="job-1",
                        collection_id="collection-1",
                        status="completed",
                        total_files=1,
                        processed_files=1,
                        total_chunks=3,
                        errors=None,
                        started_at=None,
                        completed_at=None,
                        created_at=None,
                    )
                ], 1

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            FakeRepo,
        )

        response = client.get(
            "/v1/ingestion-jobs",
            params={"status": "completed", "collection": "collection-1", "page": 2, "page_size": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["pagination"]["page"] == 2
        assert data["pagination"]["page_size"] == 5
        assert data["data"][0]["processed_files"] == 1
        assert data["data"][0]["total_chunks"] == 3
        assert captured["filters"] == {
            "status": "completed",
            "collection_id": "collection-1",
        }