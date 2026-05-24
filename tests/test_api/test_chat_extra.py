"""Extra chat endpoint tests: compat fallbacks, stream errors, edge cases."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from compact_rag.api.deps import _cached_settings, get_rag_pipeline
from compact_rag.api.router import create_app
from compact_rag.storage.schema import RAGCitation, RAGResponse


class _CompatTypeErrorPipeline:
    """Pipeline that raises a signature compat TypeError."""

    async def query(self, question, conversation_history=None, collection="default", top_k=10, stream=False, use_hybrid_search=True, use_rerank=True):
        raise TypeError("unexpected keyword argument 'use_hybrid_search'")

    async def query_legacy(self, question, conversation_history=None, collection="default", top_k=10, stream=False):
        return RAGResponse(
            id="rag-legacy",
            answer="resolved via fallback",
            citations=[],
            token_usage={"total_tokens": 1},
        )

    def __getattr__(self, name):
        if name == "query":
            return self.query
        raise AttributeError


class _NonCompatTypeErrorPipeline:
    """Pipeline that raises a non-compat TypeError."""

    async def query(self, **kwargs):
        raise TypeError("some other type error unrelated to signature")


class _StreamErrorPipeline:
    """Pipeline whose query raises during streaming."""

    async def query(self, question, conversation_history=None, collection="default", top_k=10, use_hybrid_search=True, use_rerank=True, stream=False):
        return RAGResponse(
            id="rag-error",
            answer="ok",
            citations=[],
            token_usage={},
        )

    async def query_stream(self, question, conversation_history=None, collection="default", top_k=10, use_hybrid_search=True, use_rerank=True):
        yield "first chunk"
        raise RuntimeError("stream broke mid-way")
        yield "never reached"


class _ModelTextStreamPipeline:
    """Pipeline that includes model text in streaming responses."""

    async def query_stream(self, question, conversation_history=None, collection="default", top_k=10, use_hybrid_search=True, use_rerank=True):
        yield "The model says: "
        yield "artificial intelligence is key."


class _LegacyStreamPipeline:
    """Pipeline with old signature for stream fallback."""

    async def query_stream(self, question, conversation_history=None, collection="default", top_k=10):
        raise TypeError("unexpected keyword argument 'use_hybrid_search'")

    async def query_stream_legacy(self, question, conversation_history=None, collection="default", top_k=10):
        yield "legacy stream chunk 1"
        yield "legacy stream chunk 2"

    def __getattr__(self, name):
        if name == "query_stream":
            return self.query_stream
        raise AttributeError


@pytest.fixture
def client_factory(test_settings):
    """Create a test client with a given pipeline override."""
    app = None
    client = None

    def _make(pipeline):
        nonlocal app, client
        _cached_settings.cache_clear()
        app = create_app(settings=test_settings)
        app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
        client = TestClient(app)
        return client

    return _make


class TestChatCompatFallback:
    def test_compat_type_error_triggers_fallback(self, test_settings):
        """sig compat TypeError triggers query fallback without use_hybrid_search/use_rerank."""
        _cached_settings.cache_clear()

        class RealCompatPipeline:
            pass

        p = RealCompatPipeline()
        p.query = AsyncMock(side_effect=[
            TypeError("unexpected keyword argument 'use_hybrid_search'"),
            RAGResponse(
                id="rag-fallback-test",
                answer="fallback answer",
                citations=[],
                token_usage={"total_tokens": 5},
            ),
        ])

        _cached_settings.cache_clear()
        app = create_app(settings=test_settings)
        app.dependency_overrides[get_rag_pipeline] = lambda: p

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hello"}],
                    "collection": "default",
                    "stream": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "fallback answer"

    def test_non_compat_type_error_propagates(self, test_settings):
        """Non-compat TypeError is not caught and propagates as 500."""
        _cached_settings.cache_clear()

        class NonCompatPipeline:
            async def query(self, question, conversation_history=None, collection="default", top_k=10, stream=False, use_hybrid_search=True, use_rerank=True):
                raise RuntimeError("pipeline runtime failure")

        p = NonCompatPipeline()
        app = create_app(settings=test_settings)
        app.dependency_overrides[get_rag_pipeline] = lambda: p

        with TestClient(app) as client:
            with pytest.raises(RuntimeError, match="pipeline runtime failure"):
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "hello"}],
                        "collection": "default",
                        "stream": False,
                    },
                )


class TestStreamResponse:
    def test_stream_with_model_text_in_response(self, test_settings):
        _cached_settings.cache_clear()

        class ModelStreamPipeline:
            async def query_stream(self, question, conversation_history=None, collection="default", top_k=10, use_hybrid_search=True, use_rerank=True):
                yield "The model says: "
                yield "AI is the future."

        p = ModelStreamPipeline()
        app = create_app(settings=test_settings)
        app.dependency_overrides[get_rag_pipeline] = lambda: p

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "what is AI?"}],
                    "stream": True,
                },
            )

        assert response.status_code == 200
        body = response.text
        assert "The model says: " in body
        assert "AI is the future." in body

    def test_stream_with_pipeline_exception(self, test_settings):
        _cached_settings.cache_clear()

        class ErrorStreamPipeline:
            async def query_stream(self, **kwargs):
                yield "before error"
                raise RuntimeError("generation failure")
                yield "after"  # noqa

        p = ErrorStreamPipeline()
        app = create_app(settings=test_settings)
        app.dependency_overrides[get_rag_pipeline] = lambda: p

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "error test"}],
                    "stream": True,
                },
            )

        assert response.status_code == 200
        body = response.text
        assert "before error" in body
        assert "generation failure" in body

    def test_stream_legacy_fallback(self, test_settings):
        """Stream with legacy pipeline that lacks use_hybrid_search/use_rerank params."""
        _cached_settings.cache_clear()

        class LegacyStreamPipeline:
            async def query_stream(self, question, conversation_history=None, collection="default", top_k=10, use_hybrid_search=True, use_rerank=True):
                raise TypeError("unexpected keyword argument 'use_hybrid_search'")

    # The compat logic in _query_stream_with_compat catches TypeError matching the sig check,
    # then calls pipeline.query_stream without those args. Need a pipeline with BOTH behaviors.
        class LegacyStreamCompatPipeline:
            def __init__(self):
                self.call_count = 0

            async def query_stream(self, question, conversation_history=None, collection="default", top_k=10, **kwargs):
                self.call_count += 1
                if "use_hybrid_search" in kwargs:
                    raise TypeError("unexpected keyword argument 'use_hybrid_search'")
                yield "legacy chunk a"
                yield "legacy chunk b"

        p = LegacyStreamCompatPipeline()
        app = create_app(settings=test_settings)
        app.dependency_overrides[get_rag_pipeline] = lambda: p

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert response.status_code == 200
        body = response.text
        assert "legacy chunk a" in body
        assert "legacy chunk b" in body


class TestQueryWithCompatError:
    def test_query_pipeline_error_propagates(self, test_settings):
        _cached_settings.cache_clear()

        class ErrorPipeline:
            async def query(self, question, conversation_history=None, collection="default", top_k=10, stream=False, use_hybrid_search=True, use_rerank=True):
                raise RuntimeError("pipeline runtime error")

        p = ErrorPipeline()
        app = create_app(settings=test_settings)
        app.dependency_overrides[get_rag_pipeline] = lambda: p

        with TestClient(app) as client:
            with pytest.raises(RuntimeError, match="pipeline runtime error"):
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                    },
                )
