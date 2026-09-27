"""Tests for dynamic API data sources (fetch + normalize)."""

from __future__ import annotations

from typing import Any

import pytest

from compact_rag.common.exceptions import IngestionError
from compact_rag.ingestion.sources import (
    SourceDefinition,
    fetch_source,
    get_path,
    record_content_hash,
    record_to_source_record,
    render_title,
)


def _definition(**overrides: Any):
    base: dict[str, Any] = {
        "name": "comments",
        "collection": "reviews",
        "base_url": "https://api.example.com",
        "path": "/api/comments",
        "auth_token": "tok123",
    }
    base.update(overrides)
    return SourceDefinition(**base)


def _envelope(items, current, last, code=0):
    return {
        "data": items,
        "code": code,
        "message": "SUCCESS",
        "paginator": {
            "last": last,
            "current": current,
            "numItemsPerPage": 100,
            "first": 1,
            "pageCount": last,
            "totalCount": len(items),
        },
    }


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    """Fake httpx.AsyncClient serving queued pages, recording calls."""

    pages: list = []
    calls: list = []
    failures_before_success: int = 0

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, params: dict | None = None, headers: dict | None = None):
        type(self).calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        if type(self).failures_before_success > 0:
            type(self).failures_before_success -= 1
            raise ConnectionError("boom")
        page = (params or {}).get("page", 1)
        return _FakeResp(type(self).pages[page - 1])


@pytest.fixture
def fake_http(monkeypatch):
    import httpx

    _FakeClient.pages = []
    _FakeClient.calls = []
    _FakeClient.failures_before_success = 0
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    return _FakeClient


class TestFetchSource:
    async def test_paginates_two_pages_with_auth(self, fake_http):
        fake_http.pages = [
            _envelope([{"id": 1, "content": "a"}], 1, 2),
            _envelope([{"id": 2, "content": "b"}], 2, 2),
        ]
        records = await fetch_source(_definition())
        assert [r.external_id for r in records] == ["1", "2"]
        assert len(fake_http.calls) == 2
        assert fake_http.calls[0]["headers"]["x-auth-token"] == "tok123"
        assert fake_http.calls[0]["params"]["page"] == 1
        assert fake_http.calls[0]["params"]["numItemsPerPage"] == 100
        assert fake_http.calls[1]["params"]["page"] == 2

    async def test_custom_auth_header(self, fake_http):
        fake_http.pages = [_envelope([], 1, 1)]
        await fetch_source(_definition(auth_header="Authorization", auth_token="Bearer x"))
        assert fake_http.calls[0]["headers"]["Authorization"] == "Bearer x"

    async def test_error_code_raises(self, fake_http):
        fake_http.pages = [_envelope([], 1, 1, code=5)]
        with pytest.raises(IngestionError, match="code=5"):
            await fetch_source(_definition())

    async def test_transient_failure_retried(self, fake_http):
        fake_http.pages = [_envelope([{"id": 1, "content": "a"}], 1, 1)]
        fake_http.failures_before_success = 2
        records = await fetch_source(_definition())
        assert len(records) == 1
        assert len(fake_http.calls) == 3

    async def test_persistent_failure_raises(self, fake_http):
        fake_http.pages = [_envelope([], 1, 1)]
        fake_http.failures_before_success = 99
        with pytest.raises(IngestionError, match="3 attempts"):
            await fetch_source(_definition())

    async def test_since_filters_old_records(self, fake_http):
        fake_http.pages = [_envelope([
            {"id": 1, "content": "old", "updated_at": "2026-01-01T00:00:00"},
            {"id": 2, "content": "new", "updated_at": "2026-09-27T00:00:00"},
        ], 1, 1)]
        records = await fetch_source(_definition(), since="2026-06-01T00:00:00")
        assert [r.external_id for r in records] == ["2"]

    async def test_missing_items_path_raises(self, fake_http):
        fake_http.pages = [{"code": 0, "paginator": {"current": 1, "last": 1}}]
        with pytest.raises(IngestionError, match="items_path"):
            await fetch_source(_definition())


class TestNormalize:
    def test_record_mapping(self):
        definition = _definition(title_template="评论 #{id} by {author}",
                                 body_fields=["content", "author"])
        record = record_to_source_record("comments", definition, {
            "id": 7, "content": "很好", "author": "amy",
        })
        assert record.external_id == "7"
        assert record.title == "评论 #7 by amy"
        assert "很好" in record.content
        assert record.metadata["source"] == "comments"
        assert record.metadata["external_id"] == "7"

    def test_missing_id_field(self):
        with pytest.raises(ValueError, match="id_field"):
            record_to_source_record("comments", _definition(), {"content": "x"})

    def test_title_template_missing_field(self):
        with pytest.raises(ValueError, match="title_template"):
            render_title("hi {nope}", {"id": 1})

    def test_hash_stable_and_content_sensitive(self):
        definition = _definition()
        r1 = record_to_source_record("c", definition, {"id": 1, "content": "a"})
        r2 = record_to_source_record("c", definition, {"id": 1, "content": "a"})
        r3 = record_to_source_record("c", definition, {"id": 1, "content": "b"})
        assert record_content_hash("c", r1) == record_content_hash("c", r2)
        assert record_content_hash("c", r1) != record_content_hash("c", r3)

    def test_get_path(self):
        assert get_path({"a": {"b": 1}}, "a.b") == 1
        with pytest.raises(KeyError):
            get_path({"a": {}}, "a.b.c")


class TestDefinitionValidation:
    def test_bad_name_rejected(self):
        with pytest.raises(Exception, match="name"):
            SourceDefinition(name="Bad Name!", base_url="https://x")

    def test_defaults_match_expected_envelope(self):
        definition = SourceDefinition(name="reports", base_url="https://x")
        assert definition.items_path == "data"
        assert definition.code_path == "code"
        assert definition.success_code == 0
        assert definition.current_page_field == "current"
        assert definition.last_page_field == "last"
        assert definition.auth_header == "x-auth-token"
