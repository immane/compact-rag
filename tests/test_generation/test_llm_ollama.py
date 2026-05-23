from __future__ import annotations

import builtins

import pytest

from compact_rag.generation.llm import OllamaClient


@pytest.fixture
def block_ollama_import(monkeypatch):
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "ollama":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ollama_chat_fallback_without_sdk(monkeypatch, block_ollama_import):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "deepseek-r1:7b",
                "message": {"content": "fallback works"},
                "prompt_eval_count": 3,
                "eval_count": 5,
                "done_reason": "stop",
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.post_calls = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            self.post_calls.append((url, json))
            assert url == "http://127.0.0.1:11434/api/chat"
            assert json["model"] == "deepseek-r1:7b"
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    client = OllamaClient(model="deepseek-r1:7b", host="http://127.0.0.1:11434", timeout=30)
    response = await client.chat(messages=[{"role": "user", "content": "hi"}])

    assert response.content == "fallback works"
    assert response.model == "deepseek-r1:7b"
    assert response.token_usage["prompt_tokens"] == 3
    assert response.token_usage["completion_tokens"] == 5
    assert response.token_usage["total_tokens"] == 8


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ollama_stream_fallback_without_sdk(monkeypatch, block_ollama_import):
    class FakeStreamResponse:
        def raise_for_status(self):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            yield '{"message": {"content": "hello"}}'
            yield '{"message": {"content": " world"}}'
            yield "{invalid-json"
            yield '{"done": true}'

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.stream_calls = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, json):
            self.stream_calls.append((method, url, json))
            assert method == "POST"
            assert url == "http://127.0.0.1:11434/api/chat"
            assert json["stream"] is True
            return FakeStreamResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    client = OllamaClient(model="deepseek-r1:7b", host="http://127.0.0.1:11434", timeout=30)

    chunks = []
    async for chunk in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert chunks == ["hello", " world"]
