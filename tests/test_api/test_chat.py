from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from compact_rag.api.deps import _cached_settings, get_rag_pipeline
from compact_rag.api.router import create_app
from compact_rag.storage.schema import RAGCitation, RAGResponse


class MockPipeline:
    def __init__(self):
        self.query = AsyncMock(return_value=RAGResponse(
            id="rag-test-001",
            answer="人工智能是计算机科学的一个重要分支。",
            citations=[
                RAGCitation(
                    doc_id="doc-1",
                    chunk_index=0,
                    page_number=1,
                    filename="ai_intro.pdf",
                    score=0.95,
                    content_snippet="人工智能是计算机科学...",
                )
            ],
            token_usage={"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
            retrieval_latency_ms=12.5,
            generation_latency_ms=200.0,
        ))

    async def query_stream(self, **kwargs):
        for chunk in ["人工", "智能", "是", "重要分支"]:
            yield chunk


class LegacyMockPipeline:
    async def query(self, question, conversation_history=None, collection="default", top_k=10, stream=False):
        return RAGResponse(
            id="rag-legacy-001",
            answer="legacy response",
            citations=[],
            token_usage={"total_tokens": 1},
            retrieval_latency_ms=1,
            generation_latency_ms=1,
        )

    async def query_stream(self, question, conversation_history=None, collection="default", top_k=10):
        yield "legacy"


@pytest.fixture
def client(test_settings):
    _cached_settings.cache_clear()

    mock_pipeline = MockPipeline()

    app = create_app(settings=test_settings)
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_pipeline

    with TestClient(app) as c:
        yield c


class TestChatCompletions:
    def test_valid_chat_request(self, client):
        request_body = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "什么是人工智能？"}
            ],
            "collection": "default",
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=request_body)

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "gpt-4o-mini"
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["finish_reason"] == "stop"
        msg = choice["message"]
        assert msg["role"] == "assistant"
        assert "人工智能" in msg["content"]
        assert len(msg["citations"]) == 1
        assert msg["citations"][0]["filename"] == "ai_intro.pdf"
        assert "usage" in data
        assert data["usage"]["total_tokens"] == 80

    def test_stream_chat_request_uses_query_stream(self, client):
        request_body = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "什么是人工智能？"}],
            "collection": "default",
            "stream": True,
        }

        response = client.post("/v1/chat/completions", json=request_body)

        assert response.status_code == 200
        body = response.text
        assert "text/event-stream" in response.headers.get("content-type", "")
        assert '"role": "assistant"' in body
        assert '"delta": {"content":' in body
        assert "\\u4eba\\u5de5" in body
        assert "\\u667a\\u80fd" in body
        assert "data: [DONE]" in body

    def test_empty_messages_returns_422(self, client):
        request_body = {
            "model": "gpt-4o-mini",
            "messages": [],
            "collection": "default",
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=request_body)
        assert response.status_code == 422


def test_stream_falls_back_for_legacy_pipeline_signature(test_settings):
    _cached_settings.cache_clear()
    app = create_app(settings=test_settings)
    app.dependency_overrides[get_rag_pipeline] = lambda: LegacyMockPipeline()

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "collection": "default",
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert "legacy" in response.text
