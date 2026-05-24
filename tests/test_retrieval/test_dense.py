from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from compact_rag.retrieval.dense import DenseRetriever
from compact_rag.storage.schema import SearchResult


class TestDenseRetriever:
    @pytest.mark.asyncio
    async def test_search_delegates_to_vector_store(self, mocker):
        mock_vs = mocker.MagicMock()
        mock_vs.search = AsyncMock(
            return_value=[
                SearchResult(id="id1", content="content one", score=0.9),
                SearchResult(id="id2", content="content two", score=0.5),
            ]
        )

        retriever = DenseRetriever(mock_vs)
        results = await retriever.search("test query", top_k=5)

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].id == "id1"
        mock_vs.search.assert_called_once_with("test query", top_k=5, where=None)

    @pytest.mark.asyncio
    async def test_search_results_are_search_result_objects(self, mocker):
        mock_vs = mocker.MagicMock()
        mock_vs.search = AsyncMock(
            return_value=[
                SearchResult(id="abc", content="sample content", score=0.8),
            ]
        )

        retriever = DenseRetriever(mock_vs)
        results = await retriever.search("query")

        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].id == "abc"
        assert results[0].content == "sample content"
        assert isinstance(results[0].score, float)

    @pytest.mark.asyncio
    async def test_top_k_passed_through(self, mocker):
        mock_vs = mocker.MagicMock()
        mock_vs.search = AsyncMock(return_value=[])

        retriever = DenseRetriever(mock_vs)
        await retriever.search("query", top_k=3)

        mock_vs.search.assert_called_once_with("query", top_k=3, where=None)

    @pytest.mark.asyncio
    async def test_empty_query_passed_through(self, mocker):
        mock_vs = mocker.MagicMock()
        mock_vs.search = AsyncMock(return_value=[])

        retriever = DenseRetriever(mock_vs)
        await retriever.search("")

        mock_vs.search.assert_called_once_with("", top_k=100, where=None)

    @pytest.mark.asyncio
    async def test_collection_filter_is_passed(self, mocker):
        mock_vs = mocker.MagicMock()
        mock_vs.search = AsyncMock(return_value=[])

        retriever = DenseRetriever(mock_vs)
        await retriever.search("query", collection="my_collection")

        mock_vs.search.assert_called_once_with(
            "query", top_k=100, where={"collection_name": "my_collection"}
        )
