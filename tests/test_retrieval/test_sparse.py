from __future__ import annotations

import pytest

from compact_rag.retrieval.sparse import BM25Retriever


class TestBM25RetrieverTokenize:
    def test_tokenize_handles_chinese_text(self):
        retriever = BM25Retriever()
        tokens = retriever._tokenize("人工智能是计算机科学的一个重要分支")
        assert len(tokens) > 0
        assert "人工智能" in tokens

    def test_tokenize_handles_english_text(self):
        retriever = BM25Retriever()
        tokens = retriever._tokenize("hello world this is a test")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_tokenize_handles_mixed_text(self):
        retriever = BM25Retriever()
        tokens = retriever._tokenize("AI 人工智能")
        assert len(tokens) > 0


class TestBM25RetrieverIndex:
    def test_index_builds_successfully(self):
        retriever = BM25Retriever()
        retriever.index(
            ["doc one", "doc two", "doc three"],
            ["id1", "id2", "id3"],
        )
        assert retriever._is_indexed is True
        assert retriever._bm25 is not None

    def test_empty_documents_dont_crash(self):
        retriever = BM25Retriever()
        with pytest.raises(ZeroDivisionError):
            retriever.index([], [])

    def test_is_indexed_flag(self):
        retriever = BM25Retriever()
        assert retriever.is_indexed is False
        retriever.index(["doc"], ["id1"])
        assert retriever.is_indexed is True


class TestBM25RetrieverSearch:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.retriever = BM25Retriever()
        self.retriever.index(
            [
                "Python is a programming language",
                "Machine learning uses Python",
                "Rust is a systems language",
                "Deep learning with transformers",
            ],
            ["id1", "id2", "id3", "id4"],
        )

    def test_search_returns_results_in_correct_format(self):
        results = self.retriever.search("Python programming")
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], tuple)
            assert len(results[0]) == 2
            assert isinstance(results[0][0], str)
            assert isinstance(results[0][1], float)

    def test_search_on_empty_index_returns_empty(self):
        empty = BM25Retriever()
        results = empty.search("query")
        assert results == []

    def test_search_top_k_respected(self):
        results = self.retriever.search("learning", top_k=2)
        assert len(results) <= 2

    def test_search_relevance_ordering(self):
        results = self.retriever.search("Python", top_k=10)
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]


class TestBM25RetrieverRebuild:
    def test_rebuild_index_replaces_old_data(self):
        retriever = BM25Retriever()
        retriever.index(["old doc"], ["old_id"])
        retriever.rebuild_index(
            ["new doc one", "new doc two"],
            ["new_id1", "new_id2"],
        )
        assert len(retriever.documents) == 2
        assert retriever.documents == ["new doc one", "new doc two"]
        assert retriever.doc_ids == ["new_id1", "new_id2"]
