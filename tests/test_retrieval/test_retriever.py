from __future__ import annotations

import pytest

from compact_rag.config.settings import RetrievalSettings
from compact_rag.retrieval.retriever import HybridRetriever
from compact_rag.retrieval.sparse import BM25Retriever
from compact_rag.storage.schema import SearchResult


class _FakeVectorStore:
    def __init__(self):
        self.search_calls = 0

    def search(self, query: str, top_k: int = 10, where: dict | None = None):
        self.search_calls += 1
        return [
            SearchResult(id="dense-1", content="python dense document", score=0.9, metadata={"filename": "dense.md"}),
            SearchResult(id="dense-2", content="other dense document", score=0.8, metadata={"filename": "dense2.md"}),
        ]

    def count(self, where: dict | None = None) -> int:
        return 2

    def fetch_documents(self, where: dict | None = None):
        return [
            SearchResult(
                id="dense-1",
                content="python dense document",
                score=0.0,
                metadata={"filename": "dense.md"},
            ),
            SearchResult(
                id="sparse-1",
                content="python sparse only document",
                score=0.0,
                metadata={"filename": "sparse.md"},
            ),
        ]


class _FakeReranker:
    def __init__(self):
        self.last_candidates_len = 0

    @property
    def is_available(self) -> bool:
        return True

    async def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        self.last_candidates_len = len(candidates)
        return candidates


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hybrid_flag_can_disable_sparse_branch():
    vector_store = _FakeVectorStore()
    bm25 = BM25Retriever()
    reranker = _FakeReranker()
    settings = RetrievalSettings(
        dense_top_k=5,
        sparse_top_k=5,
        fusion_top_k=5,
        rerank_top_k=2,
    )

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
        reranker=reranker,
        settings=settings,
        query_transformer=None,
    )

    results = await retriever.retrieve(
        "python",
        top_k=5,
        collection="default",
        use_hybrid_search=False,
    )

    ids = {r.id for r in results}
    assert "dense-1" in ids
    assert "sparse-1" not in ids


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sparse_results_have_content_after_index_build():
    vector_store = _FakeVectorStore()
    bm25 = BM25Retriever()
    reranker = _FakeReranker()
    settings = RetrievalSettings(
        dense_top_k=5,
        sparse_top_k=5,
        fusion_top_k=5,
        rerank_top_k=2,
    )

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
        reranker=reranker,
        settings=settings,
        query_transformer=None,
    )

    results = await retriever.retrieve(
        "python",
        top_k=5,
        collection="default",
        use_hybrid_search=True,
    )

    assert results
    assert all(r.content for r in results)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_top_k_is_applied():
    vector_store = _FakeVectorStore()
    bm25 = BM25Retriever()
    reranker = _FakeReranker()
    settings = RetrievalSettings(
        dense_top_k=5,
        sparse_top_k=5,
        fusion_top_k=5,
        rerank_top_k=1,
    )

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
        reranker=reranker,
        settings=settings,
        query_transformer=None,
    )

    await retriever.retrieve(
        "python",
        top_k=5,
        collection="default",
        use_hybrid_search=False,
        use_rerank=True,
    )

    assert reranker.last_candidates_len == 1
