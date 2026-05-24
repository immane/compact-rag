"""Tests for API dependency injection functions (deps.py)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.api.deps import (
    _cached_settings,
    get_db_session,
    get_embedding_service,
    get_hybrid_retriever,
    get_llm_client,
    get_prompt_manager,
    get_rag_pipeline,
    get_settings,
    get_storage_backend,
    get_vector_store,
)
from compact_rag.embedding.service import EmbeddingService
from compact_rag.generation.llm import OpenAIClient
from compact_rag.generation.prompt import PromptManager
from compact_rag.rag.pipeline import RAGPipeline
from compact_rag.retrieval.retriever import HybridRetriever
from compact_rag.storage.file_storage import LocalFileBackend
from compact_rag.storage.vector_store import VectorStore


@pytest.fixture(autouse=True)
def clear_settings_cache():
    _cached_settings.cache_clear()
    yield
    _cached_settings.cache_clear()


class TestGetSettingsDependency:
    def test_returns_settings(self):
        _cached_settings.cache_clear()
        settings = get_settings()
        assert settings.log_level == "INFO"  # real settings


class TestGetDbSession:
    @pytest.mark.asyncio
    async def test_yields_session(self, test_settings):
        _cached_settings.cache_clear()
        _cached_settings.cache_clear()

        with patch("compact_rag.api.deps.get_settings", return_value=test_settings):
            async for session in get_db_session():
                assert isinstance(session, AsyncSession)
                break


class TestGetLlmClient:
    def test_creates_llm_client(self, test_settings, monkeypatch):
        _cached_settings.cache_clear()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("compact_rag.api.deps.get_settings", return_value=test_settings):
            client = get_llm_client()
            assert isinstance(client, OpenAIClient)
            assert client._model == "gpt-4o-mini"


class TestGetEmbeddingService:
    def test_returns_embedding_service(self, test_settings):
        _cached_settings.cache_clear()

        with patch("compact_rag.api.deps.get_settings", return_value=test_settings):
            svc = get_embedding_service()
            assert isinstance(svc, EmbeddingService)

    def test_singleton_behavior(self, test_settings):
        _cached_settings.cache_clear()
        _cached_settings.cache_clear()

        with patch("compact_rag.api.deps.get_settings", return_value=test_settings):
            svc1 = get_embedding_service()
            svc2 = get_embedding_service()
            assert svc1 is svc2


class TestGetVectorStore:
    def test_returns_vector_store(self, test_settings):
        _cached_settings.cache_clear()

        with patch("compact_rag.api.deps.get_settings", return_value=test_settings):
            vs = get_vector_store()
            assert isinstance(vs, VectorStore)


class TestGetStorageBackend:
    def test_returns_local_backend(self, test_settings):
        _cached_settings.cache_clear()

        with patch("compact_rag.api.deps.get_settings", return_value=test_settings):
            backend = get_storage_backend()
            assert isinstance(backend, LocalFileBackend)


class TestGetPromptManager:
    def test_returns_prompt_manager(self):
        mgr = get_prompt_manager()
        assert isinstance(mgr, PromptManager)


class TestGetHybridRetriever:
    def test_returns_hybrid_retriever(self, test_settings):
        _cached_settings.cache_clear()

        with patch("compact_rag.api.deps.get_settings", return_value=test_settings):
            retriever = get_hybrid_retriever()
            assert isinstance(retriever, HybridRetriever)

    def test_cached_singleton(self, test_settings):
        _cached_settings.cache_clear()

        with patch("compact_rag.api.deps.get_settings", return_value=test_settings):
            r1 = get_hybrid_retriever()
            r2 = get_hybrid_retriever()
            assert r1 is r2


class TestGetRagPipeline:
    def test_wires_all_dependencies(self, test_settings, monkeypatch):
        _cached_settings.cache_clear()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("compact_rag.api.deps.get_settings", return_value=test_settings):
            pipeline = get_rag_pipeline()
            assert isinstance(pipeline, RAGPipeline)
            assert pipeline.retriever is not None
            assert pipeline.llm_client is not None
            assert pipeline.prompt_manager is not None
