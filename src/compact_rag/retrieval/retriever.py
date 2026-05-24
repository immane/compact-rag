from __future__ import annotations

from collections import defaultdict

from compact_rag.common.logger import get_logger
from compact_rag.config.settings import RetrievalSettings
from compact_rag.retrieval.dense import DenseRetriever
from compact_rag.retrieval.fusion import rrf_fusion, rsf_fusion
from compact_rag.retrieval.query_transformer import QueryTransformer
from compact_rag.retrieval.reranker import RerankerService
from compact_rag.retrieval.sparse import BM25Retriever
from compact_rag.storage.schema import SearchResult
from compact_rag.storage.vector_store import VectorStore

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        reranker: RerankerService,
        settings: RetrievalSettings,
        query_transformer: QueryTransformer | None = None,
    ) -> None:
        self._dense = DenseRetriever(vector_store)
        self._vector_store = vector_store
        self._sparse = bm25_retriever
        self._reranker = reranker
        self._settings = settings
        self._query_transformer = query_transformer
        self._sparse_index_signature: tuple[str | None, int] | None = None

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        collection: str | None = None,
        use_hybrid_search: bool = True,
        use_rerank: bool = True,
    ) -> list[SearchResult]:
        expanded_queries = await self._expand_queries(query)
        dense_results = await self._retrieve_dense(expanded_queries, collection)

        if use_hybrid_search:
            await self._ensure_sparse_index(collection)
            sparse_results = self._retrieve_sparse(expanded_queries, dense_results)
        else:
            sparse_results = []

        if not use_hybrid_search:
            fused = dense_results[: self._settings.fusion_top_k]
        elif self._settings.fusion_method == "rsf":
            fused = rsf_fusion(
                dense_results,
                sparse_results,
                alpha=self._settings.fusion_alpha,
                top_k=self._settings.fusion_top_k,
            )
        else:
            fused = rrf_fusion(
                dense_results,
                sparse_results,
                top_k=self._settings.fusion_top_k,
            )

        if use_rerank and self._reranker.is_available and fused:
            rerank_top_k = max(1, self._settings.rerank_top_k)
            head_size = min(rerank_top_k, len(fused))
            reranked_head = await self._reranker.rerank(query, fused[:head_size])
            fused = reranked_head + fused[head_size:]

        return fused[:top_k]

    async def _expand_queries(self, query: str) -> list[str]:
        if not self._query_transformer:
            return [query]

        try:
            expanded = await self._query_transformer.multi_query_expand(query, None)
        except Exception as e:
            logger.warning("Query expansion failed", error=str(e))
            return [query]

        unique: list[str] = []
        seen = set()
        for q in [query, *expanded]:
            qq = (q or "").strip()
            if qq and qq not in seen:
                unique.append(qq)
                seen.add(qq)
        return unique or [query]

    async def _retrieve_dense(
        self,
        queries: list[str],
        collection: str | None,
    ) -> list[SearchResult]:
        merged: dict[str, SearchResult] = {}

        for q in queries:
            results = await self._dense.search(
                q,
                top_k=self._settings.dense_top_k,
                collection=collection,
            )
            for result in results:
                existing = merged.get(result.id)
                if existing is None or result.score > existing.score:
                    merged[result.id] = result

        return sorted(merged.values(), key=lambda x: x.score, reverse=True)

    async def _ensure_sparse_index(self, collection: str | None) -> None:
        where = {"collection_name": collection} if collection else None
        doc_count = self._vector_store.count(where=where)
        signature = (collection, doc_count)

        if signature == self._sparse_index_signature and self._sparse.is_indexed:
            return

        if doc_count <= 0:
            self._sparse.clear()
            self._sparse_index_signature = signature
            return

        docs = self._vector_store.fetch_documents(where=where)
        if not docs:
            self._sparse.clear()
            self._sparse_index_signature = signature
            return

        self._sparse.rebuild_index(
            [d.content for d in docs],
            [d.id for d in docs],
            [d.metadata for d in docs],
        )
        self._sparse_index_signature = signature

    def _retrieve_sparse(
        self,
        queries: list[str],
        dense_results: list[SearchResult],
    ) -> list[SearchResult]:
        if not self._sparse.is_indexed:
            return []

        dense_lookup = {r.id: r for r in dense_results}
        sparse_scores: dict[str, float] = defaultdict(float)

        for q in queries:
            for doc_id, score in self._sparse.search(
                q,
                top_k=self._settings.sparse_top_k,
            ):
                if score > sparse_scores[doc_id]:
                    sparse_scores[doc_id] = score

        ranked = sorted(
            sparse_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        results: list[SearchResult] = []
        for doc_id, score in ranked:
            lookup = self._sparse.get_document(doc_id)
            if lookup is not None:
                content, metadata = lookup
            else:
                dense_item = dense_lookup.get(doc_id)
                content = dense_item.content if dense_item else ""
                metadata = dense_item.metadata if dense_item else {}

            results.append(
                SearchResult(
                    id=doc_id,
                    content=content,
                    score=score,
                    metadata=metadata,
                )
            )
        return results
