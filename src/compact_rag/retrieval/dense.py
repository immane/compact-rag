from __future__ import annotations

import inspect

from compact_rag.common.logger import get_logger
from compact_rag.storage.schema import SearchResult
from compact_rag.storage.vector_store import VectorStore

logger = get_logger(__name__)


class DenseRetriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store

    async def search(
        self,
        query: str,
        top_k: int = 100,
        collection: str | None = None,
    ) -> list[SearchResult]:
        where = {"collection_name": collection} if collection else None
        results = self._vector_store.search(query, top_k=top_k, where=where)
        if inspect.isawaitable(results):
            return await results
        return results
