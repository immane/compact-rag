"""API integration tests for /v1/collections endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from compact_rag.api.deps import _cached_settings, get_db_session
from compact_rag.api.router import create_app


def _make_session():
    """Build a fake async DB session with async methods."""
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


def _make_collection(**overrides):
    defaults = {
        "id": str(uuid4()),
        "name": "test-collection",
        "description": "A test collection",
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "document_count": 0,
        "created_at": None,
        "updated_at": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _assert_paginated_contract(data: dict) -> None:
    assert set(data.keys()) == {"data", "pagination"}
    pagination = data["pagination"]
    assert set(pagination.keys()) == {"page", "page_size", "total", "total_pages"}
    assert isinstance(data["data"], list)
    assert isinstance(pagination["page"], int)
    assert isinstance(pagination["page_size"], int)
    assert isinstance(pagination["total"], int)
    assert isinstance(pagination["total_pages"], int)


def _assert_error_format(data: dict) -> None:
    assert "error" in data
    err = data["error"]
    assert set(err.keys()) == {"code", "message", "details", "request_id"}
    assert isinstance(err["code"], str)
    assert isinstance(err["message"], str)
    assert isinstance(err["details"], dict)
    assert isinstance(err["request_id"], str)


# ── POST /v1/collections ───────────────────────────────────────


class TestCreateCollection:
    def test_create_collection_success(self, client, monkeypatch):
        collection = _make_collection(
            name="new-collection",
            description="test desc",
            embedding_model="test-model",
            chunk_size=300,
            chunk_overlap=30,
        )

        class FakeRepo:
            async def create(self, session, **kwargs):
                return SimpleNamespace(
                    id=collection.id,
                    name=kwargs.get("name", collection.name),
                    description=kwargs.get("description", collection.description),
                    embedding_model=kwargs.get("embedding_model", collection.embedding_model),
                    chunk_size=kwargs.get("chunk_size", collection.chunk_size),
                    chunk_overlap=kwargs.get("chunk_overlap", collection.chunk_overlap),
                    document_count=0,
                    created_at=None,
                    updated_at=None,
                )

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            FakeRepo,
        )

        payload = {
            "name": "new-collection",
            "description": "test desc",
            "embedding_model": "test-model",
            "chunk_size": 300,
            "chunk_overlap": 30,
        }
        response = client.post("/v1/collections", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "new-collection"
        assert data["description"] == "test desc"
        assert data["embedding_model"] == "test-model"
        assert data["chunk_size"] == 300
        assert data["chunk_overlap"] == 30
        assert data["document_count"] == 0
        assert "id" in data

    def test_create_collection_minimal_payload(self, client, monkeypatch):
        collection = _make_collection(name="minimal")

        class FakeRepo:
            async def create(self, session, **kwargs):
                return SimpleNamespace(
                    id=collection.id,
                    name=kwargs.get("name", "minimal"),
                    description=kwargs.get("description", ""),
                    embedding_model=kwargs.get("embedding_model", ""),
                    chunk_size=kwargs.get("chunk_size", 500),
                    chunk_overlap=kwargs.get("chunk_overlap", 50),
                    document_count=0,
                    created_at=None,
                    updated_at=None,
                )

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            FakeRepo,
        )

        response = client.post("/v1/collections", json={"name": "minimal"})
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "minimal"
        assert data["document_count"] == 0

    def test_create_collection_missing_name_returns_422(self, client):
        response = client.post("/v1/collections", json={})
        assert response.status_code == 422

    def test_create_collection_duplicate_name_raises_error(self, client, monkeypatch):
        class FakeRepo:
            async def create(self, session, **kwargs):
                raise Exception("UNIQUE constraint failed: collections.name")

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            FakeRepo,
        )

        response = client.post("/v1/collections", json={"name": "duplicate"})
        assert response.status_code == 500
        _assert_error_format(response.json())

    def test_create_collection_repo_exception_causes_rollback(self, client, monkeypatch):
        class FakeRepo:
            async def create(self, session, **kwargs):
                raise RuntimeError("database connection lost")

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            FakeRepo,
        )

        response = client.post("/v1/collections", json={"name": "fail"})
        assert response.status_code == 500
        err = response.json()["error"]
        assert err["code"] == "ConfigurationError"
        assert "Failed to create collection" in err["message"]


# ── GET /v1/collections ────────────────────────────────────────


class TestListCollections:
    def test_list_collections_with_data(self, client, monkeypatch):
        coll1 = _make_collection(id="c1", name="alpha", document_count=3)
        coll2 = _make_collection(id="c2", name="beta", document_count=7)

        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                return [coll1, coll2], 2

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            FakeRepo,
        )

        response = client.get("/v1/collections")
        assert response.status_code == 200
        data = response.json()
        _assert_paginated_contract(data)
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["page_size"] == 20
        assert data["pagination"]["total"] == 2
        assert data["pagination"]["total_pages"] == 1
        assert len(data["data"]) == 2
        assert data["data"][0]["name"] == "alpha"
        assert data["data"][1]["name"] == "beta"

    def test_list_collections_empty(self, client, monkeypatch):
        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                return [], 0

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            FakeRepo,
        )

        response = client.get("/v1/collections")
        assert response.status_code == 200
        data = response.json()
        _assert_paginated_contract(data)
        assert data["pagination"]["total"] == 0
        assert data["pagination"]["total_pages"] == 0
        assert data["data"] == []

    def test_list_collections_pagination_page_2(self, client, monkeypatch):
        coll = _make_collection(id="c3", name="gamma")

        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                return [coll], 1

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            FakeRepo,
        )

        response = client.get("/v1/collections", params={"page": 2, "page_size": 50})
        assert response.status_code == 200
        data = response.json()
        assert data["pagination"]["page"] == 2
        assert data["pagination"]["page_size"] == 50
        assert data["pagination"]["total"] == 1

    def test_list_collections_page_size_edge_max(self, client, monkeypatch):
        coll = _make_collection(id="c4", name="delta")

        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                return [coll], 1

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            FakeRepo,
        )

        response = client.get("/v1/collections", params={"page_size": 100})
        assert response.status_code == 200

    def test_list_collections_page_size_edge_1(self, client, monkeypatch):
        coll = _make_collection(id="c5", name="epsilon")

        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                return [coll], 1

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            FakeRepo,
        )

        response = client.get("/v1/collections", params={"page_size": 1})
        assert response.status_code == 200

    def test_list_collections_page_zero_returns_422(self, client):
        response = client.get("/v1/collections", params={"page": 0})
        assert response.status_code == 422

    def test_list_collections_negative_page_size_returns_422(self, client):
        response = client.get("/v1/collections", params={"page_size": -1})
        assert response.status_code == 422

    def test_list_collections_page_size_over_100_returns_422(self, client):
        response = client.get("/v1/collections", params={"page_size": 101})
        assert response.status_code == 422


# ── DELETE /v1/collections/{name} ───────────────────────────────


class TestDeleteCollection:
    def test_delete_existing_collection(self, client, monkeypatch):
        coll = _make_collection(id="c-del", name="to-delete")

        class FakeRepo:
            async def get_by_name(self, session, name):
                return coll

            async def delete(self, session, id_val):
                return True

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            FakeRepo,
        )

        response = client.delete("/v1/collections/to-delete")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["name"] == "to-delete"

    def test_delete_nonexistent_collection(self, client, monkeypatch):
        class FakeRepo:
            async def get_by_name(self, session, name):
                return None

            async def delete(self, session, id_val):
                return False

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            FakeRepo,
        )

        response = client.delete("/v1/collections/nonexistent")
        assert response.status_code == 404
        _assert_error_format(response.json())
        assert response.json()["error"]["code"] == "FileNotFoundError"

    def test_delete_after_creation_simulation(self, client, monkeypatch):
        coll = _make_collection(id="c-rt", name="round-trip")

        class FakeRepo:
            async def get_by_name(self, session, name):
                return coll if name == "round-trip" else None

            async def delete(self, session, id_val):
                return id_val == coll.id

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            FakeRepo,
        )

        response = client.delete("/v1/collections/round-trip")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

    def test_delete_collection_with_special_chars_name(self, client, monkeypatch):
        coll = _make_collection(id="c-enc", name="collection-with-dashes_123")

        class FakeRepo:
            async def get_by_name(self, session, name):
                return coll

            async def delete(self, session, id_val):
                return True

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            FakeRepo,
        )

        response = client.delete("/v1/collections/collection-with-dashes_123")
        assert response.status_code == 200
        assert response.json()["name"] == "collection-with-dashes_123"

    def test_delete_collection_with_unicode_name(self, client, monkeypatch):
        coll = _make_collection(id="c-uni", name="中文集合")

        class FakeRepo:
            async def get_by_name(self, session, name):
                return coll if name == "中文集合" else None

            async def delete(self, session, id_val):
                return True

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            FakeRepo,
        )

        response = client.delete("/v1/collections/%E4%B8%AD%E6%96%87%E9%9B%86%E5%90%88")
        assert response.status_code == 200
        assert response.json()["name"] == "中文集合"

    @pytest.mark.integration
    def test_collection_response_structure(self, client, monkeypatch):
        from datetime import datetime, timezone

        ts = datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc)
        coll = _make_collection(
            id="c-struct",
            name="structure-test",
            description="desc",
            embedding_model="model-x",
            chunk_size=256,
            chunk_overlap=64,
            document_count=10,
            created_at=ts,
            updated_at=ts,
        )

        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                return [coll], 1

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            FakeRepo,
        )

        response = client.get("/v1/collections")
        assert response.status_code == 200
        item = response.json()["data"][0]
        assert set(item.keys()) == {
            "id", "name", "description", "embedding_model",
            "chunk_size", "chunk_overlap", "document_count",
            "created_at", "updated_at",
        }
        assert item["document_count"] == 10
        assert item["chunk_size"] == 256
