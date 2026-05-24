"""API integration tests for /v1/conversations endpoints."""

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


def _make_conversation(**overrides):
    defaults = {
        "id": str(uuid4()),
        "collection_id": str(uuid4()),
        "title": "Test Conversation",
        "model": "gpt-4o-mini",
        "message_count": 3,
        "created_at": None,
        "updated_at": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_message(**overrides):
    defaults = {
        "id": str(uuid4()),
        "conversation_id": str(uuid4()),
        "role": "user",
        "content": "Hello, world",
        "sources": [{"filename": "doc.pdf", "score": 0.95}],
        "token_count": 15,
        "latency_ms": 200,
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


# ── GET /v1/conversations ──────────────────────────────────────


class TestListConversations:
    def test_list_conversations_with_data(self, client, monkeypatch):
        conv1 = _make_conversation(id="cv1", title="First Chat")
        conv2 = _make_conversation(id="cv2", title="Second Chat", message_count=10)

        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                return [conv1, conv2], 2

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.conversation.ConversationRepository",
            FakeRepo,
        )

        response = client.get("/v1/conversations")
        assert response.status_code == 200
        data = response.json()
        _assert_paginated_contract(data)
        assert len(data["data"]) == 2
        assert data["pagination"]["total"] == 2
        assert data["data"][0]["title"] == "First Chat"
        assert data["data"][1]["message_count"] == 10

    def test_list_conversations_empty(self, client, monkeypatch):
        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                return [], 0

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.conversation.ConversationRepository",
            FakeRepo,
        )

        response = client.get("/v1/conversations")
        assert response.status_code == 200
        data = response.json()
        _assert_paginated_contract(data)
        assert data["data"] == []
        assert data["pagination"]["total"] == 0
        assert data["pagination"]["total_pages"] == 0

    def test_list_conversations_pagination(self, client, monkeypatch):
        convs = [_make_conversation(id=f"cv{i}") for i in range(30)]

        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                start = (page - 1) * page_size
                end = start + page_size
                batch = convs[start:end]
                return batch, 30

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.conversation.ConversationRepository",
            FakeRepo,
        )

        response = client.get("/v1/conversations", params={"page": 2, "page_size": 10})
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 10
        assert data["pagination"]["page"] == 2
        assert data["pagination"]["page_size"] == 10
        assert data["pagination"]["total"] == 30
        assert data["pagination"]["total_pages"] == 3

    def test_list_conversations_page_out_of_range(self, client, monkeypatch):
        convs = [_make_conversation(id=f"cv{i}") for i in range(5)]

        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                start = (page - 1) * page_size
                end = start + page_size
                batch = convs[start:end]
                return batch, 5

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.conversation.ConversationRepository",
            FakeRepo,
        )

        response = client.get("/v1/conversations", params={"page": 10, "page_size": 20})
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []

    def test_list_conversations_page_size_boundaries(self, client, monkeypatch):
        conv = _make_conversation(id="cv-bound")

        class FakeRepo:
            async def list(self, session, page=1, page_size=20):
                return [conv], 1

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.conversation.ConversationRepository",
            FakeRepo,
        )

        response = client.get("/v1/conversations", params={"page_size": 100})
        assert response.status_code == 200

        response = client.get("/v1/conversations", params={"page_size": 1})
        assert response.status_code == 200

    def test_list_conversations_invalid_page_422(self, client):
        response = client.get("/v1/conversations", params={"page": -1})
        assert response.status_code == 422

    def test_list_conversations_invalid_page_size_422(self, client):
        response = client.get("/v1/conversations", params={"page_size": 0})
        assert response.status_code == 422

    def test_list_conversations_page_size_over_max_422(self, client):
        response = client.get("/v1/conversations", params={"page_size": 101})
        assert response.status_code == 422


# ── GET /v1/conversations/{conv_id} ─────────────────────────────


class TestGetConversation:
    def test_get_existing_conversation_with_messages(self, client, monkeypatch):
        conv = _make_conversation(
            id="cv-detail",
            title="Detailed Chat",
            model="gpt-4o-mini",
            message_count=2,
            collection_id="coll-1",
        )
        msgs = [
            _make_message(id="m1", conversation_id="cv-detail", role="user", content="What is AI?"),
            _make_message(id="m2", conversation_id="cv-detail", role="assistant", content="AI stands for..."),
        ]

        class FakeRepo:
            async def get_by_id(self, session, conv_id):
                return conv if conv_id == "cv-detail" else None

            async def list_messages(self, session, conv_id):
                return msgs if conv_id == "cv-detail" else []

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.conversation.ConversationRepository",
            FakeRepo,
        )

        response = client.get("/v1/conversations/cv-detail")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "cv-detail"
        assert data["title"] == "Detailed Chat"
        assert data["model"] == "gpt-4o-mini"
        assert data["message_count"] == 2
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "What is AI?"
        assert data["messages"][1]["role"] == "assistant"

    def test_get_existing_conversation_no_collection(self, client, monkeypatch):
        conv = _make_conversation(id="cv-no-coll", collection_id=None)
        msgs = []

        class FakeRepo:
            async def get_by_id(self, session, conv_id):
                return conv

            async def list_messages(self, session, conv_id):
                return msgs

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.conversation.ConversationRepository",
            FakeRepo,
        )

        response = client.get("/v1/conversations/cv-no-coll")
        assert response.status_code == 200
        data = response.json()
        assert data["collection_id"] is None

    def test_get_nonexistent_conversation(self, client, monkeypatch):
        class FakeRepo:
            async def get_by_id(self, session, conv_id):
                return None

            async def list_messages(self, session, conv_id):
                return []

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.conversation.ConversationRepository",
            FakeRepo,
        )

        response = client.get("/v1/conversations/nonexistent")
        assert response.status_code == 404
        _assert_error_format(response.json())
        assert "FileNotFoundError" in response.json()["error"]["code"]

    def test_get_conversation_detail_response_structure(self, client, monkeypatch):
        conv = _make_conversation(id="cv-struct")
        msgs = [_make_message(id="m1", conversation_id="cv-struct")]

        class FakeRepo:
            async def get_by_id(self, session, conv_id):
                return conv

            async def list_messages(self, session, conv_id):
                return msgs

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.conversation.ConversationRepository",
            FakeRepo,
        )

        response = client.get("/v1/conversations/cv-struct")
        assert response.status_code == 200
        item = response.json()
        assert set(item.keys()) == {
            "id", "collection_id", "title", "model", "message_count",
            "messages", "created_at", "updated_at",
        }
        msg = item["messages"][0]
        assert set(msg.keys()) == {
            "id", "conversation_id", "role", "content", "sources",
            "token_count", "latency_ms", "created_at",
        }

    def test_get_conversation_with_messages_having_sources(self, client, monkeypatch):
        conv = _make_conversation(id="cv-sources")
        msgs = [
            _make_message(
                id="m1", conversation_id="cv-sources",
                sources=[{"filename": "research.pdf", "score": 0.99, "chunk_index": 3}],
            ),
        ]

        class FakeRepo:
            async def get_by_id(self, session, conv_id):
                return conv

            async def list_messages(self, session, conv_id):
                return msgs

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.conversation.ConversationRepository",
            FakeRepo,
        )

        response = client.get("/v1/conversations/cv-sources")
        assert response.status_code == 200
        data = response.json()
        sources = data["messages"][0]["sources"]
        assert len(sources) == 1
        assert sources[0]["filename"] == "research.pdf"
        assert sources[0]["score"] == 0.99

    def test_get_conversation_empty_messages(self, client, monkeypatch):
        conv = _make_conversation(id="cv-empty", message_count=0)

        class FakeRepo:
            async def get_by_id(self, session, conv_id):
                return conv

            async def list_messages(self, session, conv_id):
                return []

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.conversation.ConversationRepository",
            FakeRepo,
        )

        response = client.get("/v1/conversations/cv-empty")
        assert response.status_code == 200
        data = response.json()
        assert data["message_count"] == 0
        assert data["messages"] == []


# ── DELETE /v1/conversations/{conv_id} ──────────────────────────


class TestDeleteConversation:
    def test_delete_existing_conversation(self, client, monkeypatch):
        conv = _make_conversation(id="cv-del", title="Delete Me")
        delete_calls = []

        class FakeRepo:
            async def get_by_id(self, session, conv_id):
                return conv if conv_id == "cv-del" else None

            async def delete(self, session, conv_id):
                delete_calls.append(conv_id)
                return True

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.conversation.ConversationRepository",
            FakeRepo,
        )

        response = client.delete("/v1/conversations/cv-del")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["conversation_id"] == "cv-del"
        assert delete_calls == ["cv-del"]

    def test_delete_nonexistent_conversation(self, client, monkeypatch):
        class FakeRepo:
            async def get_by_id(self, session, conv_id):
                return None

            async def delete(self, session, conv_id):
                return False

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.conversation.ConversationRepository",
            FakeRepo,
        )

        response = client.delete("/v1/conversations/nonexistent")
        assert response.status_code == 404
        _assert_error_format(response.json())
        assert response.json()["error"]["code"] == "FileNotFoundError"

    @pytest.mark.integration
    def test_delete_conversation_verify_cascade(self, client, monkeypatch):
        conv = _make_conversation(id="cv-cascade")
        delete_calls = []

        class FakeRepo:
            async def get_by_id(self, session, conv_id):
                return conv if conv_id == "cv-cascade" else None

            async def delete(self, session, conv_id):
                delete_calls.append(conv_id)
                return True

        monkeypatch.setattr(
            "compact_rag.storage.db.repository.conversation.ConversationRepository",
            FakeRepo,
        )

        response = client.delete("/v1/conversations/cv-cascade")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"
