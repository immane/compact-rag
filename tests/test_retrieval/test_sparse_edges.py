from __future__ import annotations

import pytest

from compact_rag.retrieval.sparse import BM25Retriever


@pytest.fixture
def indexed_retriever():
    retriever = BM25Retriever()
    retriever.index(
        [
            "Python is a programming language",
            "Machine learning uses Python for data analysis",
            "Rust is a systems programming language",
            "Deep learning with transformers and attention",
        ],
        ["id1", "id2", "id3", "id4"],
    )
    return retriever


class TestSearchEdgeCases:
    def test_search_with_empty_string_query(self, indexed_retriever):
        results = indexed_retriever.search("")
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_search_with_only_stopwords(self, indexed_retriever):
        results = indexed_retriever.search("the a an is of to")
        assert isinstance(results, list)

    def test_search_with_very_long_query(self):
        retriever = BM25Retriever()
        retriever.index(
            ["short doc one", "another short document"],
            ["id1", "id2"],
        )
        long_query = "query " * 300
        results = retriever.search(long_query)
        assert isinstance(results, list)

    def test_search_with_very_long_query_chinese(self):
        retriever = BM25Retriever()
        retriever.index(
            ["人工智能是计算机科学的重要分支", "机器学习使用数据进行训练"],
            ["id1", "id2"],
        )
        long_query = "人工智能" * 200
        results = retriever.search(long_query)
        assert isinstance(results, list)


class TestTokenizeEdgeCases:
    def test_tokenize_with_special_characters(self):
        retriever = BM25Retriever()
        tokens = retriever._tokenize("hello @world #python $test!")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_tokenize_with_numbers(self):
        retriever = BM25Retriever()
        tokens = retriever._tokenize("Python 3.11 released in 2024")
        assert isinstance(tokens, list)
        assert "Python" in tokens

    def test_tokenize_with_mixed_scripts(self):
        retriever = BM25Retriever()
        tokens = retriever._tokenize("AI人工智能Python学习Rust系统")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_tokenize_with_empty_string(self):
        retriever = BM25Retriever()
        tokens = retriever._tokenize("")
        assert tokens == []

    def test_tokenize_with_pure_cjk_chars_triggers_jieba(self):
        retriever = BM25Retriever()
        tokens = retriever._tokenize("   \u4e2d\u6587   ")
        assert isinstance(tokens, list)

    def test_tokenize_with_unicode_symbols(self):
        retriever = BM25Retriever()
        tokens = retriever._tokenize("score: ★★★☆☆  rating 4/5")
        assert isinstance(tokens, list)


class TestGetDocumentEdgeCases:
    def test_get_document_with_non_existent_id(self, indexed_retriever):
        result = indexed_retriever.get_document("non_existent_id")
        assert result is None

    def test_get_document_with_empty_string_id(self, indexed_retriever):
        result = indexed_retriever.get_document("")
        assert result is None

    def test_get_document_after_clear(self, indexed_retriever):
        indexed_retriever.clear()
        result = indexed_retriever.get_document("id1")
        assert result is None


class TestIndexEdgeCases:
    def test_index_with_none_metadata(self):
        retriever = BM25Retriever()
        retriever.index(
            ["doc one", "doc two"],
            ["id1", "id2"],
            metadatas=None,
        )
        assert retriever.is_indexed
        assert retriever.metadatas == [{}, {}]

    def test_index_with_partial_metadata(self):
        retriever = BM25Retriever()
        retriever.index(
            ["doc one", "doc two"],
            ["id1", "id2"],
            metadatas=[{"key": "val"}, {}],
        )
        assert retriever.is_indexed
        assert len(retriever.metadatas) == 2

    def test_index_with_mismatched_lengths_raises_value_error(self):
        retriever = BM25Retriever()
        with pytest.raises(ValueError, match="same length"):
            retriever.index(["doc one"], ["id1", "id2"])

    def test_index_with_mismatched_metadata_length_raises_value_error(self):
        retriever = BM25Retriever()
        with pytest.raises(ValueError, match="metadatas must match"):
            retriever.index(
                ["doc one", "doc two"],
                ["id1", "id2"],
                metadatas=[{"key": "val"}],
            )

    def test_rebuild_with_empty_documents_after_having_data(self):
        retriever = BM25Retriever()
        retriever.index(["doc one", "doc two"], ["id1", "id2"])
        assert retriever.is_indexed
        assert len(retriever.documents) == 2

        retriever.rebuild_index([], [])
        assert not retriever.is_indexed
        assert retriever.documents == []
        assert retriever._bm25 is None


class TestClear:
    def test_clear_resets_all_state(self, indexed_retriever):
        assert indexed_retriever.is_indexed
        assert indexed_retriever._bm25 is not None
        assert len(indexed_retriever.documents) > 0

        indexed_retriever.clear()

        assert not indexed_retriever.is_indexed
        assert indexed_retriever._bm25 is None
        assert indexed_retriever.documents == []
        assert indexed_retriever.doc_ids == []
        assert indexed_retriever.metadatas == []
        assert indexed_retriever._doc_lookup == {}

    def test_clear_is_idempotent(self, indexed_retriever):
        indexed_retriever.clear()
        indexed_retriever.clear()
        assert not indexed_retriever.is_indexed

    def test_search_after_clear_returns_empty(self, indexed_retriever):
        indexed_retriever.clear()
        results = indexed_retriever.search("Python")
        assert results == []


class TestIsIndexed:
    def test_is_indexed_false_before_index(self):
        retriever = BM25Retriever()
        assert not retriever.is_indexed

    def test_is_indexed_true_after_index(self):
        retriever = BM25Retriever()
        retriever.index(["doc"], ["id1"])
        assert retriever.is_indexed

    def test_is_indexed_false_after_clear(self, indexed_retriever):
        indexed_retriever.clear()
        assert not indexed_retriever.is_indexed
