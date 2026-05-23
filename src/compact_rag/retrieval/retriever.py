from __future__ import annotations

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
        self._sparse = bm25_retriever
        self._reranker = reranker
        self._settings = settings
        self._query_transformer = query_transformer

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        collection: str | None = None,
    ) -> list[SearchResult]:
        dense_results = await self._dense.search(
            query, top_k=self._settings.dense_top_k, collection=collection
        )

        if self._sparse.is_indexed:
            sparse_raw = self._sparse.search(query, top_k=self._settings.sparse_top_k)
            sparse_results = [
                SearchResult(
                    id=doc_id, content="", score=score, metadata={}
                )
                for doc_id, score in sparse_raw
            ]
        else:
            sparse_results = []

        if self._settings.fusion_method == "rsf":
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

        if self._reranker.is_available and fused:
            fused = await self._reranker.rerank(query, fused)

        return fused[:top_k]
