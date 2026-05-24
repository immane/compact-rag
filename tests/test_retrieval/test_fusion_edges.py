from __future__ import annotations

import pytest

from compact_rag.retrieval.fusion import rrf_fusion, rsf_fusion
from compact_rag.storage.schema import SearchResult


def _make_result(id_: str, score: float) -> SearchResult:
    return SearchResult(id=id_, content=f"content_{id_}", score=score)


class TestRRFFusionEdges:
    def test_rrf_with_k_zero(self):
        dense = [_make_result("a", 0.9), _make_result("b", 0.5)]
        sparse = [_make_result("b", 0.7), _make_result("c", 0.3)]
        result = rrf_fusion(dense, sparse, k=0, top_k=3)
        assert len(result) > 0
        assert all(isinstance(r, SearchResult) for r in result)

    def test_rrf_with_k_one(self):
        dense = [_make_result("a", 0.9), _make_result("b", 0.5)]
        sparse = [_make_result("b", 0.7), _make_result("c", 0.3)]
        result = rrf_fusion(dense, sparse, k=1, top_k=3)
        assert len(result) > 0
        assert all(isinstance(r, SearchResult) for r in result)

    def test_rrf_with_top_k_larger_than_available(self):
        dense = [_make_result("a", 0.9)]
        sparse = [_make_result("b", 0.7)]
        result = rrf_fusion(dense, sparse, k=60, top_k=100)
        assert len(result) == 2

    def test_rrf_with_single_result_in_each_list(self):
        dense = [_make_result("a", 0.9)]
        sparse = [_make_result("b", 0.7)]
        result = rrf_fusion(dense, sparse, k=60, top_k=10)
        assert len(result) == 2

    def test_rrf_with_all_results_same_score(self):
        dense = [
            _make_result("a", 0.5),
            _make_result("b", 0.5),
        ]
        sparse = [
            _make_result("c", 0.5),
            _make_result("d", 0.5),
        ]
        result = rrf_fusion(dense, sparse, k=60, top_k=10)
        assert len(result) == 4

    def test_rrf_with_duplicate_ids_in_one_list(self):
        dense = [
            _make_result("a", 0.9),
            _make_result("a", 0.5),
            _make_result("b", 0.3),
        ]
        sparse = [_make_result("b", 0.7)]
        result = rrf_fusion(dense, sparse, k=60, top_k=10)
        result_ids = [r.id for r in result]
        assert result_ids.count("a") == 1

    def test_rrf_all_empty_returns_empty(self):
        result = rrf_fusion([], [], k=60, top_k=10)
        assert result == []


class TestRSFFusionEdges:
    def test_rsf_with_alpha_zero(self):
        dense = [_make_result("dense_only", 1.0)]
        sparse = [_make_result("sparse_only", 1.0)]
        result = rsf_fusion(dense, sparse, alpha=0.0, top_k=2)
        assert len(result) == 2
        assert any(r.id == "sparse_only" for r in result)

    def test_rsf_with_alpha_one(self):
        dense = [_make_result("dense_only", 1.0)]
        sparse = [_make_result("sparse_only", 1.0)]
        result = rsf_fusion(dense, sparse, alpha=1.0, top_k=2)
        assert len(result) == 2
        assert any(r.id == "dense_only" for r in result)

    def test_rsf_with_alpha_half(self):
        dense = [
            _make_result("a", 1.0),
            _make_result("b", 0.5),
        ]
        sparse = [
            _make_result("a", 0.8),
            _make_result("c", 0.4),
        ]
        result = rsf_fusion(dense, sparse, alpha=0.5, top_k=3)
        assert len(result) == 3

    def test_rsf_with_alpha_negative(self):
        dense = [_make_result("a", 1.0)]
        sparse = [_make_result("b", 1.0)]
        result = rsf_fusion(dense, sparse, alpha=-0.5, top_k=2)
        assert len(result) > 0
        assert all(isinstance(r, SearchResult) for r in result)

    def test_rsf_with_alpha_above_one(self):
        dense = [_make_result("a", 1.0)]
        sparse = [_make_result("b", 1.0)]
        result = rsf_fusion(dense, sparse, alpha=1.5, top_k=2)
        assert len(result) > 0
        assert all(isinstance(r, SearchResult) for r in result)

    def test_rsf_with_single_result_in_each_list(self):
        dense = [_make_result("a", 0.9)]
        sparse = [_make_result("b", 0.7)]
        result = rsf_fusion(dense, sparse, alpha=0.5, top_k=10)
        assert len(result) == 2

    def test_rsf_with_all_results_same_score(self):
        dense = [
            _make_result("a", 0.5),
            _make_result("b", 0.5),
        ]
        sparse = [
            _make_result("a", 0.5),
            _make_result("c", 0.5),
        ]
        result = rsf_fusion(dense, sparse, alpha=0.5, top_k=10)
        assert len(result) == 3

    def test_rsf_with_duplicate_ids_in_one_list(self):
        dense = [
            _make_result("a", 0.9),
            _make_result("a", 0.7),
            _make_result("b", 0.3),
        ]
        sparse = [_make_result("b", 0.7)]
        result = rsf_fusion(dense, sparse, alpha=0.5, top_k=10)
        result_ids = [r.id for r in result]
        assert result_ids.count("a") == 1

    def test_rsf_empty_inputs_returns_empty(self):
        result = rsf_fusion([], [], alpha=0.5, top_k=10)
        assert result == []

    def test_rsf_dense_empty_sparse_has_results(self):
        sparse = [_make_result("a", 0.9), _make_result("b", 0.5)]
        result = rsf_fusion([], sparse, alpha=0.5, top_k=10)
        assert len(result) == 2
        assert result[0].id == "a"

    def test_rsf_sparse_empty_dense_has_results(self):
        dense = [_make_result("a", 0.9), _make_result("b", 0.5)]
        result = rsf_fusion(dense, [], alpha=0.5, top_k=10)
        assert len(result) == 2
        assert result[0].id == "a"

    def test_rsf_with_top_k_larger_than_available(self):
        dense = [_make_result("a", 0.9)]
        sparse = [_make_result("b", 0.7)]
        result = rsf_fusion(dense, sparse, alpha=0.5, top_k=100)
        assert len(result) == 2
