from __future__ import annotations

from compact_rag.storage.schema import SearchResult


def rrf_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    k: int = 60,
    top_k: int = 50,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    docs: dict[str, SearchResult] = {}

    for rank, result in enumerate(dense_results, start=1):
        scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (k + rank)
        docs[result.id] = result

    for rank, result in enumerate(sparse_results, start=1):
        scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (k + rank)
        if result.id not in docs:
            docs[result.id] = result

    sorted_ids = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
    fused = [
        SearchResult(
            id=doc_id,
            content=docs[doc_id].content,
            score=round(scores[doc_id], 6),
            metadata=docs[doc_id].metadata,
        )
        for doc_id in sorted_ids
    ]
    return fused


def rsf_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    alpha: float = 0.5,
    top_k: int = 50,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    docs: dict[str, SearchResult] = {}

    dense_max = dense_results[0].score if dense_results else 0.0
    if dense_max > 0:
        for result in dense_results:
            norm_score = result.score / dense_max
            scores[result.id] = alpha * norm_score
            docs[result.id] = result

    sparse_max = sparse_results[0].score if sparse_results else 0.0
    if sparse_max > 0:
        for result in sparse_results:
            norm_score = result.score / sparse_max
            scores[result.id] = scores.get(result.id, 0.0) + (1 - alpha) * norm_score
            if result.id not in docs:
                docs[result.id] = result

    sorted_ids = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
    fused = [
        SearchResult(
            id=doc_id,
            content=docs[doc_id].content,
            score=round(scores[doc_id], 6),
            metadata=docs[doc_id].metadata,
        )
        for doc_id in sorted_ids
    ]
    return fused
