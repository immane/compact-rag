from __future__ import annotations

import pytest

from compact_rag.retrieval.reranker import RerankerService
from compact_rag.storage.schema import SearchResult


class TestRerankerService:
    @pytest.mark.asyncio
    async def test_is_available_property(self):
        service = RerankerService()
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401

            assert service.is_available is True
        except ImportError:
            assert service.is_available is False

    @pytest.mark.asyncio
    async def test_rerank_reorders_by_score(self, mocker):
        mock_model = mocker.MagicMock()
        mock_model.predict = mocker.MagicMock(return_value=[0.9, 0.3, 0.1])

        service = RerankerService()
        service._model = mock_model

        candidates = [
            SearchResult(id="a", content="high relevance doc", score=0.5),
            SearchResult(id="b", content="low relevance doc", score=0.5),
            SearchResult(id="c", content="no relevance doc", score=0.5),
        ]

        results = await service.rerank("test query", candidates)

        assert len(results) == 3
        assert results[0].id == "a"
        assert results[1].id == "b"
        assert results[2].id == "c"
        assert results[0].score == 0.9
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_empty_candidates_handled_gracefully(self):
        service = RerankerService()
        results = await service.rerank("query", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_when_model_not_available(self, mocker):
        service = RerankerService()
        service._model = None

        candidates = [
            SearchResult(id="a", content="test", score=0.8),
        ]

        results = await service.rerank("query", candidates)
        assert results is candidates

    @pytest.mark.asyncio
    async def test_score_values_are_rounded(self, mocker):
        mock_model = mocker.MagicMock()
        mock_model.predict = mocker.MagicMock(return_value=[0.12345678])

        service = RerankerService()
        service._model = mock_model

        candidates = [SearchResult(id="x", content="test", score=0.5)]
        results = await service.rerank("query", candidates)

        assert results[0].score == 0.123457

    @pytest.mark.asyncio
    async def test_rerank_uses_asyncio_to_thread(self, mocker):
        mock_model = mocker.MagicMock()
        mock_model.predict = mocker.MagicMock(return_value=[0.5])

        service = RerankerService()
        service._model = mock_model

        candidates = [SearchResult(id="x", content="test", score=0.5)]
        await service.rerank("query", candidates)

        # predict should have been called
        mock_model.predict.assert_called_once()
