"""Extra LLM client tests: Ollama stream 5xx fallback, timeout, JSONDecodeError;
Anthropic/OpenAI tools=None in stream; OpenAIClient chat with tools=None."""

from __future__ import annotations

import asyncio
import builtins
import sys
from types import ModuleType
from unittest.mock import AsyncMock

import pytest

from compact_rag.common.exceptions import LLMTimeoutError
from compact_rag.generation.llm import (
    AnthropicClient,
    OllamaClient,
    OpenAIClient,
)


# ── OllamaClient Stream: SDK 5xx → HTTP fallback ────────────────


class TestOllamaStreamSDK5xxFallback:
    _ollama_fake_module = None

    @classmethod
    def setup_class(cls):
        cls._ollama_fake_module = ModuleType("ollama")
        sys.modules["ollama"] = cls._ollama_fake_module

    @classmethod
    def teardown_class(cls):
        sys.modules.pop("ollama", None)

    @pytest.mark.asyncio
    async def test_sdk_stream_5xx_falls_back_to_http_stream(self, mocker):
        """When SDK stream raises a 5xx error, it falls back to HTTP stream."""
        mock_cls = mocker.MagicMock()
        setattr(self._ollama_fake_module, "AsyncClient", mock_cls)
        mock_sdk = mock_cls.return_value

        class Sdk500Error(Exception):
            status_code = 500

        mock_sdk.chat = AsyncMock(side_effect=Sdk500Error("status code 500"))

        class FakeStreamResp:
            status_code = 200

            def raise_for_status(self):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_lines(self):
                yield '{"message": {}}'
                yield '{"message": {"content": "http"}}'
                yield '{"message": {"content": " stream"}}'
                yield ""

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, json):
                return FakeStreamResp()

        mocker.patch("httpx.AsyncClient", new=FakeAsyncClient)

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        chunks = []
        async for chunk in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert chunks == ["http", " stream"]

    @pytest.mark.asyncio
    async def test_sdk_stream_5xx_status_code_in_text_falls_back_to_http(self, mocker):
        """When SDK stream raises error with `status code 503` in text, falls back to HTTP."""
        mock_cls = mocker.MagicMock()
        setattr(self._ollama_fake_module, "AsyncClient", mock_cls)
        mock_sdk = mock_cls.return_value

        class SdkTextError(Exception):
            status_code = None

        err = SdkTextError("something failed with status code 503")
        mock_sdk.chat = AsyncMock(side_effect=err)

        class FakeStreamResp:
            status_code = 200

            def raise_for_status(self):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_lines(self):
                yield '{"message": {"content": "fallback text"}}'
                yield ""

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, json):
                return FakeStreamResp()

        mocker.patch("httpx.AsyncClient", new=FakeAsyncClient)

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        chunks = []
        async for chunk in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert chunks == ["fallback text"]


# ── OllamaClient HTTP Stream: non-200 and JSONDecodeError ────────


class TestOllamaStreamHTTPerrors:
    @pytest.mark.asyncio
    async def test_http_stream_non_200_response(self, monkeypatch, mocker):
        """HTTP stream with non-200 response raises HTTPStatusError (direct HTTP path)."""
        import httpx as real_httpx

        class FakeStreamResp:
            status_code = 400

            def raise_for_status(self):
                err_response = mocker.MagicMock()
                err_response.status_code = 400
                err_response.text = '{"error": "bad request"}'
                raise real_httpx.HTTPStatusError(
                    "400 Bad Request",
                    request=mocker.MagicMock(),
                    response=err_response,
                )

            async def aread(self):
                return b'{"error": "bad request"}'

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, json):
                return FakeStreamResp()

        # Block SDK import so code goes directly to HTTP streaming
        original_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "ollama" or name.startswith("ollama."):
                raise ImportError("blocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        with pytest.raises(real_httpx.HTTPStatusError):
            async for _ in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
                pass

    @pytest.mark.asyncio
    async def test_http_stream_json_decode_error_skipped(self, monkeypatch):
        """HTTP stream lines with JSONDecodeError are skipped (continue)."""

        original_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "ollama" or name.startswith("ollama."):
                raise ImportError("blocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        class FakeStreamResp:
            status_code = 200

            def raise_for_status(self):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_lines(self):
                yield "not valid json at all"
                yield '{"message": {"content": "valid"}}'
                yield "also not {valid json"
                yield '{"message": {"content": " after"}}'
                yield ""
                yield "garbage"

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, json):
                return FakeStreamResp()

        monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        chunks = []
        async for chunk in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert chunks == ["valid", " after"]


# ── OllamaClient Stream Timeout ──────────────────────────────────


class TestOllamaStreamTimeout:
    @pytest.mark.asyncio
    async def test_stream_timeout_raises_llm_timeout_error(self, monkeypatch):
        """asyncio.TimeoutError during stream raises LLMTimeoutError."""
        original_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "ollama" or name.startswith("ollama."):
                raise ImportError("blocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        class SlowStreamResp:
            status_code = 200

            def raise_for_status(self):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def aiter_lines(self):
                # This must be a synchronous function returning an async iterator
                async def _gen():
                    raise asyncio.TimeoutError("timeout during stream")
                    yield  # never reached

                return _gen()

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, json):
                return SlowStreamResp()

        monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        with pytest.raises(LLMTimeoutError, match="Ollama stream timed out"):
            async for _ in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
                pass


# ── AnthropicClient: tools=None in stream ────────────────────────


class TestAnthropicStreamToolsNone:
    @pytest.fixture
    def anthropic_mock_setup(self, mocker):
        fake = ModuleType("anthropic")
        sys.modules["anthropic"] = fake
        mock_cls = mocker.MagicMock(name="AsyncAnthropic")
        setattr(fake, "AsyncAnthropic", mock_cls)
        yield mock_cls.return_value
        sys.modules.pop("anthropic", None)

    @pytest.mark.asyncio
    async def test_chat_stream_tools_none(self, anthropic_mock_setup, mocker):
        """When tools=None, the 'tools' key should NOT be in kwargs."""
        mock_instance = anthropic_mock_setup

        event = mocker.MagicMock()
        event.type = "content_block_delta"
        event.delta.type = "text_delta"
        event.delta.text = "ok"

        async def _event_iter():
            yield event

        stream_ctx = mocker.MagicMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=stream_ctx)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)
        stream_ctx.__aiter__ = lambda self: _event_iter()

        mock_stream = mocker.MagicMock(return_value=stream_ctx)
        mock_instance.messages.stream = mock_stream

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        async for _ in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
            pass

        call_kwargs = mock_stream.call_args[1]
        assert "tools" not in call_kwargs


# ── OpenAIClient: tools=None in stream and chat ──────────────────


class TestOpenAIClientToolsNone:
    @pytest.fixture
    def mock_openai(self, mocker):
        return mocker.patch("openai.AsyncOpenAI")

    @pytest.mark.asyncio
    async def test_chat_stream_tools_none(self, mock_openai, mocker):
        """When tools=None in stream, 'tools' should NOT be in create kwargs."""
        chunk = mocker.MagicMock()
        chunk.choices = [mocker.MagicMock()]
        chunk.choices[0].delta = mocker.MagicMock()
        chunk.choices[0].delta.content = "ok"

        async def _aiter(_self=None):
            yield chunk

        stream = mocker.MagicMock()
        stream.__aiter__ = _aiter

        mock_instance = mock_openai.return_value
        mock_create = AsyncMock(return_value=stream)
        mock_instance.chat.completions.create = mock_create

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        async for _ in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
            pass

        call_kwargs = mock_create.call_args[1]
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_chat_tools_none(self, mock_openai, mocker):
        """When tools=None, 'tools' should NOT be in create kwargs."""
        choice = mocker.MagicMock()
        choice.message.content = "content"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"

        resp = mocker.MagicMock()
        resp.choices = [choice]
        resp.model = "gpt-4o-mini"
        resp.usage = mocker.MagicMock()
        resp.usage.prompt_tokens = 5
        resp.usage.completion_tokens = 3
        resp.usage.total_tokens = 8

        mock_instance = mock_openai.return_value
        mock_create = AsyncMock(return_value=resp)
        mock_instance.chat.completions.create = mock_create

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        await client.chat(messages=[{"role": "user", "content": "hi"}], tools=None)

        call_kwargs = mock_create.call_args[1]
        assert "tools" not in call_kwargs
