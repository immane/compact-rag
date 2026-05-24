"""API integration tests for /v1/api-keys endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
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


def _make_api_key(**overrides):
    defaults = {
        "id": str(uuid4()),
        "name": "test-key",
        "key_hash": "a1b2c3d4e5f6789012345678abcdef01abcdef01abcdef01abcdef01abcdef01",
        "permissions": ["read"],
        "is_active": True,
        "expires_at": None,
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


# ── POST /v1/api-keys ──────────────────────────────────────────


class TestCreateApiKey:
    def test_create_api_key_success(self, client, monkeypatch):
        key = _make_api_key(
            id="ak-1",
            name="my-api-key",
            key_hash="abc123def456abc123def456abc123def456abc123def456abc123def456abc123",
            permissions=["read", "write"],
        )

        class FakeRepo:
            async def create(self, session, **kwargs):
                return SimpleNamespace(
                    id=key.id,
                    name=kwargs.get("name", key.name),
                    key_hash=kwargs.get("key_hash", key.key_hash),
                    permissions=kwargs.get("permissions", key.permissions),
                    is_active=kwargs.get("is_active", True),
                    created_at=None,
                )

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        payload = {"name": "my-api-key", "permissions": ["read", "write"]}
        response = client.post("/v1/api-keys", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "my-api-key"
        assert data["permissions"] == ["read", "write"]
        assert "key" in data
        assert data["key"].startswith("cr-")
        assert len(data["key"]) > 10

    def test_create_api_key_default_permissions(self, client, monkeypatch):
        key = _make_api_key(id="ak-2", name="default-perm-key")

        class FakeRepo:
            async def create(self, session, **kwargs):
                return SimpleNamespace(
                    id=key.id,
                    name=kwargs.get("name", key.name),
                    key_hash=kwargs.get("key_hash", key.key_hash),
                    permissions=kwargs.get("permissions", ["read"]),
                    is_active=True,
                    created_at=None,
                )

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.post("/v1/api-keys", json={"name": "default-perm-key"})
        assert response.status_code == 200
        data = response.json()
        assert data["permissions"] == ["read"]

    def test_create_api_key_empty_name_succeeds(self, client, monkeypatch):
        """Empty string is valid for str field in Pydantic (no min_length constraint)."""
        key = _make_api_key(id="ak-empty", name="")

        class FakeRepo:
            async def create(self, session, **kwargs):
                return SimpleNamespace(
                    id=key.id,
                    name=kwargs.get("name", ""),
                    key_hash=kwargs.get("key_hash", "hash"),
                    permissions=kwargs.get("permissions", ["read"]),
                    is_active=True,
                    created_at=None,
                )

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.post("/v1/api-keys", json={"name": ""})
        assert response.status_code == 200
        assert response.json()["name"] == ""

    def test_create_api_key_missing_name_422(self, client):
        response = client.post("/v1/api-keys", json={})
        assert response.status_code == 422

    def test_create_api_key_repo_exception(self, client, monkeypatch):
        class FakeRepo:
            async def create(self, session, **kwargs):
                raise Exception("database locked")

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.post("/v1/api-keys", json={"name": "fail-key"})
        assert response.status_code == 500
        _assert_error_format(response.json())
        err = response.json()["error"]
        assert err["code"] == "ConfigurationError"
        assert "Failed to create API key" in err["message"]

    def test_create_api_key_raw_key_unique_each_call(self, client, monkeypatch):
        raw_keys = []

        class FakeRepo:
            async def create(self, session, **kwargs):
                return SimpleNamespace(
                    id=str(uuid4()),
                    name=kwargs.get("name", "k"),
                    key_hash=kwargs.get("key_hash", "hash"),
                    permissions=kwargs.get("permissions", ["read"]),
                    is_active=True,
                    created_at=None,
                )

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        for _ in range(3):
            response = client.post("/v1/api-keys", json={"name": "bulk-key"})
            assert response.status_code == 200
            raw_keys.append(response.json()["key"])

        assert len(set(raw_keys)) == 3

    def test_create_api_key_create_response_structure(self, client, monkeypatch):
        key = _make_api_key(id="ak-struct", name="struct-key")

        class FakeRepo:
            async def create(self, session, **kwargs):
                return SimpleNamespace(
                    id=key.id,
                    name=kwargs.get("name", key.name),
                    key_hash=kwargs.get("key_hash", key.key_hash),
                    permissions=kwargs.get("permissions", key.permissions),
                    is_active=True,
                    created_at=None,
                )

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.post("/v1/api-keys", json={"name": "struct-key"})
        assert response.status_code == 200
        item = response.json()
        assert set(item.keys()) == {"id", "name", "key", "permissions", "created_at"}


# ── GET /v1/api-keys ───────────────────────────────────────────


class TestListApiKeys:
    def test_list_api_keys_with_data(self, client, monkeypatch):
        key1 = _make_api_key(id="ak-a", name="alpha-key", key_hash="aaaa1234" + "0" * 56)
        key2 = _make_api_key(id="ak-b", name="beta-key", key_hash="bbbb5678" + "0" * 56)

        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                return [key1, key2], 2

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.get("/v1/api-keys")
        assert response.status_code == 200
        data = response.json()
        _assert_paginated_contract(data)
        assert len(data["data"]) == 2
        assert "..." in data["data"][0]["key_prefix"]
        assert data["data"][0]["name"] == "alpha-key"

    def test_list_api_keys_empty(self, client, monkeypatch):
        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                return [], 0

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.get("/v1/api-keys")
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
        assert data["pagination"]["total"] == 0

    def test_list_api_keys_pagination(self, client, monkeypatch):
        keys = [_make_api_key(id=f"ak-{i}") for i in range(25)]

        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                start = (page - 1) * page_size
                end = start + page_size
                batch = keys[start:end]
                return batch, 25

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.get("/v1/api-keys", params={"page": 1, "page_size": 10})
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 10
        assert data["pagination"]["total"] == 25
        assert data["pagination"]["total_pages"] == 3

    def test_list_api_keys_response_structure(self, client, monkeypatch):
        key = _make_api_key(id="ak-struct")

        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                return [key], 1

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.get("/v1/api-keys")
        assert response.status_code == 200
        item = response.json()["data"][0]
        assert set(item.keys()) == {
            "id", "name", "key_prefix", "permissions", "is_active",
            "expires_at", "created_at",
        }

    def test_list_api_keys_inactive_shown(self, client, monkeypatch):
        key = _make_api_key(id="ak-inactive", is_active=False)

        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                return [key], 1

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.get("/v1/api-keys")
        assert response.status_code == 200
        assert response.json()["data"][0]["is_active"] is False


# ── PATCH /v1/api-keys/{key_id} ────────────────────────────────


class TestUpdateApiKey:
    def test_deactivate_api_key(self, client, monkeypatch):
        key = _make_api_key(id="ak-deact", is_active=True)

        class FakeRepo:
            async def get_by_id(self, session, key_id):
                return key if key_id == "ak-deact" else None

            async def update(self, session, id_val, **kwargs):
                for k, v in kwargs.items():
                    setattr(key, k, v)
                return key

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.patch("/v1/api-keys/ak-deact", json={"is_active": False})
        assert response.status_code == 200
        data = response.json()
        assert data["is_active"] is False

    def test_activate_api_key(self, client, monkeypatch):
        key = _make_api_key(id="ak-act", is_active=False)

        class FakeRepo:
            async def get_by_id(self, session, key_id):
                return key if key_id == "ak-act" else None

            async def update(self, session, id_val, **kwargs):
                for k, v in kwargs.items():
                    setattr(key, k, v)
                return key

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.patch("/v1/api-keys/ak-act", json={"is_active": True})
        assert response.status_code == 200
        assert response.json()["is_active"] is True

    def test_rename_api_key(self, client, monkeypatch):
        key = _make_api_key(id="ak-rename", name="old-name")

        class FakeRepo:
            async def get_by_id(self, session, key_id):
                return key if key_id == "ak-rename" else None

            async def update(self, session, id_val, **kwargs):
                for k, v in kwargs.items():
                    setattr(key, k, v)
                return key

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.patch("/v1/api-keys/ak-rename", json={"name": "new-name"})
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "new-name"

    def test_update_permissions(self, client, monkeypatch):
        key = _make_api_key(id="ak-perm", permissions=["read"])

        class FakeRepo:
            async def get_by_id(self, session, key_id):
                return key if key_id == "ak-perm" else None

            async def update(self, session, id_val, **kwargs):
                for k, v in kwargs.items():
                    setattr(key, k, v)
                return key

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.patch(
            "/v1/api-keys/ak-perm",
            json={"permissions": ["read", "write", "admin"]},
        )
        assert response.status_code == 200
        assert response.json()["permissions"] == ["read", "write", "admin"]

    def test_update_nonexistent_api_key(self, client, monkeypatch):
        class FakeRepo:
            async def get_by_id(self, session, key_id):
                return None

            async def update(self, session, id_val, **kwargs):
                return None

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.patch("/v1/api-keys/nonexistent", json={"is_active": True})
        assert response.status_code == 404
        _assert_error_format(response.json())
        assert response.json()["error"]["code"] == "FileNotFoundError"

    def test_update_empty_body_no_changes(self, client, monkeypatch):
        key = _make_api_key(id="ak-noop")

        class FakeRepo:
            async def get_by_id(self, session, key_id):
                return key

            async def update(self, session, id_val, **kwargs):
                return key

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.patch("/v1/api-keys/ak-noop", json={})
        assert response.status_code == 200


# ── DELETE /v1/api-keys/{key_id} ───────────────────────────────


class TestDeleteApiKey:
    def test_delete_existing_api_key(self, client, monkeypatch):
        key = _make_api_key(id="ak-del", name="delete-me")
        delete_calls = []

        class FakeRepo:
            async def get_by_id(self, session, key_id):
                return key if key_id == "ak-del" else None

            async def delete(self, session, key_id):
                delete_calls.append(key_id)
                return True

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.delete("/v1/api-keys/ak-del")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["key_id"] == "ak-del"
        assert delete_calls == ["ak-del"]

    def test_delete_nonexistent_api_key(self, client, monkeypatch):
        class FakeRepo:
            async def get_by_id(self, session, key_id):
                return None

            async def delete(self, session, key_id):
                return False

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        response = client.delete("/v1/api-keys/nonexistent")
        assert response.status_code == 404
        _assert_error_format(response.json())

    @pytest.mark.integration
    def test_api_key_full_lifecycle(self, client, monkeypatch):
        created_key = _make_api_key(id="ak-lifecycle", name="lifecycle-key")
        stored: dict[str, Any] = {"key": created_key}

        class FakeRepo:
            async def create(self, session, **kwargs):
                k = SimpleNamespace(
                    id=stored["key"].id,
                    name=kwargs.get("name", "k"),
                    key_hash=kwargs.get("key_hash", "hash"),
                    permissions=kwargs.get("permissions", ["read"]),
                    is_active=True,
                    created_at=None,
                )
                return k

            async def list(self, session, page=1, page_size=20):
                return [stored["key"]], 1

            async def get_by_id(self, session, key_id):
                return stored["key"] if key_id == "ak-lifecycle" else None

            async def update(self, session, id_val, **kwargs):
                for k, v in kwargs.items():
                    setattr(stored["key"], k, v)
                return stored["key"]

            async def delete(self, session, id_val):
                if id_val == "ak-lifecycle":
                    stored["key"] = None
                    return True
                return False

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.api_key.ApiKeyRepository",
            FakeRepo,
        )

        resp = client.post("/v1/api-keys", json={"name": "lifecycle-key"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "lifecycle-key"

        resp = client.get("/v1/api-keys")
        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 1

        resp = client.patch("/v1/api-keys/ak-lifecycle", json={"is_active": False})
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False

        resp = client.delete("/v1/api-keys/ak-lifecycle")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
