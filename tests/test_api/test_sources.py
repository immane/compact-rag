"""Tests for dynamic source config and sync endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from compact_rag.api.deps import _cached_settings
from compact_rag.api.router import create_app


def _client(test_settings):
    _cached_settings.cache_clear()
    app = create_app(settings=test_settings)
    with TestClient(app) as c:
        yield c


def _source(name="comments", **overrides):
    base = {
        "name": name,
        "collection": "reviews",
        "base_url": "https://api.example.com",
        "path": "/api/comments",
        "auth_token": "tok",
    }
    base.update(overrides)
    return base


class TestSourceConfigEndpoints:
    def test_get_empty(self, test_settings):
        for c in _client(test_settings):
            assert c.get("/v1/config/sources").json() == {"sources": []}

    def test_put_and_masked_get(self, test_settings):
        for c in _client(test_settings):
            resp = c.put("/v1/config/sources", json={"sources": [_source()]})
            assert resp.status_code == 200
            body = resp.json()["sources"][0]
            assert body["name"] == "comments"
            assert body["auth_token_configured"] is True
            assert "auth_token" not in body

    def test_put_invalid_rejected(self, test_settings):
        for c in _client(test_settings):
            resp = c.put("/v1/config/sources", json={"sources": [_source(name="Bad!")]})
            assert resp.status_code == 400

    def test_put_duplicate_names_rejected(self, test_settings):
        for c in _client(test_settings):
            resp = c.put("/v1/config/sources", json={
                "sources": [_source("a"), _source("a")]
            })
            assert resp.status_code == 400

    def test_put_missing_body_rejected(self, test_settings):
        for c in _client(test_settings):
            assert c.put("/v1/config/sources", json={}).status_code == 400

    def test_token_empty_keeps_stored(self, test_settings):
        for c in _client(test_settings):
            c.put("/v1/config/sources", json={"sources": [_source(auth_token="keep")]})
            entry = dict(_source(auth_token=""))
            c.put("/v1/config/sources", json={"sources": [entry]})
            assert c.get("/v1/config/sources").json()["sources"][0][
                "auth_token_configured"
            ] is True

    def test_delete(self, test_settings):
        for c in _client(test_settings):
            c.put("/v1/config/sources", json={"sources": [_source()]})
            assert c.delete("/v1/config/sources/nope").status_code == 404
            resp = c.delete("/v1/config/sources/comments")
            assert resp.status_code == 200
            assert resp.json() == {"sources": []}


class TestSyncEndpoint:
    def test_sync_uses_service_summary(self, test_settings, monkeypatch):
        import compact_rag.ingestion.sync as sync_module

        monkeypatch.setattr(
            sync_module,
            "sync_sources",
            AsyncMock(return_value=[{"source": "comments", "completed": 2}]),
        )
        for c in _client(test_settings):
            resp = c.post("/v1/ingestion/sources/sync")
            assert resp.status_code == 200
            assert resp.json() == {"sources": [{"source": "comments", "completed": 2}]}

    def test_sync_unknown_source_404(self, test_settings, monkeypatch):
        import compact_rag.ingestion.sync as sync_module

        async def _raise(*a, **k):
            raise ValueError("Unknown source: 'nope'")

        monkeypatch.setattr(sync_module, "sync_sources", _raise)
        for c in _client(test_settings):
            resp = c.post("/v1/ingestion/sources/sync?source=nope")
            assert resp.status_code == 404
