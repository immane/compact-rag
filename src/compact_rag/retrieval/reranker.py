from __future__ import annotations

import asyncio

from compact_rag.common.logger import get_logger
from compact_rag.storage.schema import SearchResult

logger = get_logger(__name__)


class RerankerService:
    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        self._model_name = model_name
        self._model = None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
            logger.info(f"CrossEncoder loaded: {model_name}")
        except Exception as e:
            logger.warning(f"CrossEncoder not available: {e}")
            self._model = None

    @property
    def is_available(self) -> bool:
        return self._model is not None

    async def rerank(
        self, query: str, candidates: list[SearchResult]
    ) -> list[SearchResult]:
        if not self.is_available or not candidates:
            return candidates

        pairs = [(query, c.content) for c in candidates]
        scores = await asyncio.to_thread(self._model.predict, pairs)

        scored = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )
        reranked = []
        for result, score in scored:
            reranked.append(
                SearchResult(
                    id=result.id,
                    content=result.content,
                    score=round(float(score), 6),
                    metadata=result.metadata,
                )
            )
        return reranked
