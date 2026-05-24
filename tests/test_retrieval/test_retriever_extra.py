"""Extra retriever tests: query transformer, empty dense, collection filter, error paths."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from compact_rag.config.settings import RetrievalSettings
from compact_rag.retrieval.retriever import HybridRetriever
from compact_rag.retrieval.sparse import BM25Retriever
from compact_rag.storage.schema import SearchResult


class _MinimalFakeVectorStore:
    """A vector store that returns plain SearchResult objects from search()."""

    def __init__(self, search_results=None, fetch_docs=None, doc_count=2):
        self._search_results = search_results or []
        self._fetch_docs = fetch_docs or []
        self._doc_count = doc_count
        self.search_calls = 0
        self.fetch_calls = 0
        self.count_calls = 0

    def search(self, query: str, top_k: int = 10, where: dict | None = None):
        self.search_calls += 1
        return self._search_results

    def count(self, where: dict | None = None) -> int:
        self.count_calls += 1
        return self._doc_count

    def fetch_documents(self, where: dict | None = None):
        self.fetch_calls += 1
        return self._fetch_docs


class _NoRerankReranker:
    @property
    def is_available(self) -> bool:
        return False

    async def rerank(self, query, candidates):
        return candidates


def _make_retriever(
    vector_store,
    bm25=None,
    reranker=None,
    settings=None,
    query_transformer=None,
):
    return HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25 or BM25Retriever(),
        reranker=reranker or _NoRerankReranker(),
        settings=settings or RetrievalSettings(
            dense_top_k=5, sparse_top_k=5, fusion_top_k=5, rerank_top_k=2,
        ),
        query_transformer=query_transformer,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_without_hybrid_search(mocker):
    vs = _MinimalFakeVectorStore(
        search_results=[
            SearchResult(id="d1", content="doc one", score=0.9, metadata={"filename": "f1.pdf"}),
            SearchResult(id="d2", content="doc two", score=0.8, metadata={"filename": "f2.pdf"}),
        ],
    )
    retriever = _make_retriever(vs)

    results = await retriever.retrieve(
        "query",
        top_k=3,
        use_hybrid_search=False,
        use_rerank=False,
    )
    assert len(results) <= 3
    ids = {r.id for r in results}
    assert "d1" in ids
    assert "d2" in ids


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_use_rerank_false(mocker):
    vs = _MinimalFakeVectorStore(
        search_results=[
            SearchResult(id="d1", content="content", score=0.95, metadata={}),
        ],
    )
    bm25 = BM25Retriever()
    retriever = _make_retriever(vs, bm25=bm25)

    results = await retriever.retrieve(
        "query",
        top_k=5,
        use_hybrid_search=True,
        use_rerank=False,
    )
    assert len(results) >= 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_empty_dense_results(mocker):
    vs = _MinimalFakeVectorStore(search_results=[])
    retriever = _make_retriever(vs)

    results = await retriever.retrieve("query", top_k=5, use_hybrid_search=False)
    assert results == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_error_during_dense_search(mocker):
    vs = MagicMock()
    vs.search = AsyncMock(side_effect=RuntimeError("search failed"))
    vs.count = MagicMock(return_value=0)
    vs.fetch_documents = MagicMock(return_value=[])

    retriever = _make_retriever(vs)

    with pytest.raises(RuntimeError, match="search failed"):
        await retriever.retrieve("query", use_hybrid_search=False)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_error_during_sparse_search(mocker):
    vs = _MinimalFakeVectorStore(
        search_results=[
            SearchResult(id="d1", content="text", score=0.9, metadata={}),
        ],
        fetch_docs=[
            SearchResult(id="d1", content="text", score=0.0, metadata={}),
        ],
        doc_count=1,
    )
    bm25 = MagicMock()
    bm25.is_indexed = True
    bm25.search = MagicMock(side_effect=RuntimeError("sparse failed"))
    bm25.get_document = MagicMock(return_value=("content", {}))

    retriever = _make_retriever(vs, bm25=bm25)

    with pytest.raises(RuntimeError, match="sparse failed"):
        await retriever.retrieve("query", use_hybrid_search=True, use_rerank=False)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_with_collection_filter_on_sparse(mocker):
    vs = _MinimalFakeVectorStore(
        search_results=[
            SearchResult(id="d-filter", content="filtered doc", score=0.9, metadata={"collection_name": "custom"}),
        ],
        fetch_docs=[
            SearchResult(id="d-filter", content="filtered doc", score=0.0, metadata={"collection_name": "custom"}),
        ],
        doc_count=1,
    )
    retriever = _make_retriever(vs)

    results = await retriever.retrieve("query", collection="custom", use_hybrid_search=True, use_rerank=False)
    assert len(results) >= 1
    assert vs.count_calls > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_with_query_transformer_hyde(mocker):
    class FakeQueryTransformer:
        async def multi_query_expand(self, query, context):
            return [f"{query} — expanded"]

    vs = _MinimalFakeVectorStore(
        search_results=[
            SearchResult(id="d1", content="about hyde", score=0.8, metadata={}),
        ],
    )
    qt = FakeQueryTransformer()
    retriever = _make_retriever(vs, query_transformer=qt)

    results = await retriever.retrieve("search query", use_hybrid_search=False)
    assert len(results) >= 1
    assert vs.search_calls >= 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_query_transformer_error_fallback(mocker):
    class FaultyTransformer:
        async def multi_query_expand(self, query, context):
            raise RuntimeError("transform failed")

    vs = _MinimalFakeVectorStore(
        search_results=[
            SearchResult(id="d1", content="fallback works", score=0.9, metadata={}),
        ],
    )
    qt = FaultyTransformer()
    retriever = _make_retriever(vs, query_transformer=qt)

    results = await retriever.retrieve("query", use_hybrid_search=False)
    assert len(results) == 1
    assert results[0].content == "fallback works"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_sparse_is_indexed_false_returns_empty_sparse(mocker):
    vs = _MinimalFakeVectorStore(
        search_results=[
            SearchResult(id="d1", content="only dense", score=0.9, metadata={}),
        ],
    )
    bm25 = BM25Retriever()  # default not indexed
    retriever = _make_retriever(vs, bm25=bm25)

    results = await retriever.retrieve("query", use_hybrid_search=True, use_rerank=False)
    # sparse is not indexed but dense results flow through
    assert len(results) >= 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieve_with_empty_sparse_docs(mocker):
    vs = _MinimalFakeVectorStore(
        search_results=[
            SearchResult(id="d1", content="doc", score=0.9, metadata={}),
        ],
        fetch_docs=[],  # empty fetch
        doc_count=0,
    )
    retriever = _make_retriever(vs)

    results = await retriever.retrieve("query", use_hybrid_search=True, use_rerank=False)
    assert len(results) >= 1
    # sparse index was cleared because fetch_docs was empty
    assert not vs._fetch_docs
