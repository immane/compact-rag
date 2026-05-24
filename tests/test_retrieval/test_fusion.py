from __future__ import annotations


from compact_rag.retrieval.fusion import rrf_fusion, rsf_fusion
from compact_rag.storage.schema import SearchResult


def _make_result(id_: str, score: float) -> SearchResult:
    return SearchResult(id=id_, content=f"content_{id_}", score=score)


class TestRRFFusion:
    def test_basic(self):
        dense = [
            _make_result("a", 0.9),
            _make_result("b", 0.8),
            _make_result("c", 0.3),
        ]
        sparse = [
            _make_result("b", 0.7),
            _make_result("d", 0.6),
            _make_result("a", 0.2),
        ]

        result = rrf_fusion(dense, sparse, k=60, top_k=3)

        assert len(result) == 3
        assert result[0].id in ("a", "b")
        assert all(isinstance(r.score, float) for r in result)
        assert all(r.content for r in result)

    def test_ranking_prefers_docs_in_both_lists(self):
        dense = [_make_result("shared", 0.9), _make_result("only_dense", 0.8)]
        sparse = [_make_result("shared", 0.7), _make_result("only_sparse", 0.6)]

        result = rrf_fusion(dense, sparse, k=60, top_k=3)

        ids_in_order = [r.id for r in result]
        assert ids_in_order[0] == "shared"

    def test_empty_dense(self):
        sparse = [_make_result("a", 0.9), _make_result("b", 0.5)]
        result = rrf_fusion([], sparse, k=60, top_k=2)
        assert len(result) == 2
        assert result[0].id == "a"

    def test_empty_sparse(self):
        dense = [_make_result("x", 0.9)]
        result = rrf_fusion(dense, [], k=60, top_k=2)
        assert len(result) == 1
        assert result[0].id == "x"

    def test_both_empty(self):
        result = rrf_fusion([], [], k=60, top_k=5)
        assert result == []


class TestRSFFusion:
    def test_basic_with_alpha(self):
        dense = [
            _make_result("a", 1.0),
            _make_result("b", 0.5),
        ]
        sparse = [
            _make_result("a", 0.8),
            _make_result("c", 0.4),
        ]

        result = rsf_fusion(dense, sparse, alpha=0.6, top_k=3)

        assert len(result) == 3
        ids = [r.id for r in result]
        assert "a" in ids

    def test_alpha_one_prefers_dense(self):
        dense = [_make_result("dense_only", 1.0), _make_result("shared", 0.5)]
        sparse = [_make_result("sparse_only", 1.0), _make_result("shared", 0.8)]

        result = rsf_fusion(dense, sparse, alpha=1.0, top_k=3)

        ids = [r.id for r in result]
        assert ids[0] == "dense_only"

    def test_alpha_zero_prefers_sparse(self):
        dense = [_make_result("dense_only", 1.0), _make_result("shared", 0.5)]
        sparse = [_make_result("sparse_only", 1.0), _make_result("shared", 0.8)]

        result = rsf_fusion(dense, sparse, alpha=0.0, top_k=3)

        ids = [r.id for r in result]
        assert ids[0] == "sparse_only"

    def test_empty_inputs(self):
        result = rsf_fusion([], [], top_k=5)
        assert result == []
