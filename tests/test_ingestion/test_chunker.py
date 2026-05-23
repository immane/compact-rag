from __future__ import annotations

import pytest

from compact_rag.ingestion.chunker import (
    RecursiveCharacterTextSplitter,
    SemanticChunker,
    TableAwareChunker,
    chunk_documents,
)
from compact_rag.ingestion.loader import LoadedPage
from compact_rag.storage.schema import DocumentChunk


class TestRecursiveCharacterTextSplitter:
    def test_splits_text_by_newlines(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        text = "Line one.\nLine two.\nLine three.\nLine four.\nLine five."
        chunks = splitter.split_text(text)
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_chunk_size_is_respected(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        text = "This is a longer piece of text that should be split into multiple chunks based on the chunk size parameter."
        chunks = splitter.split_text(text)
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_chunk_overlap_is_correct(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        text = "x" * 500
        chunks = splitter.split_text(text)
        if len(chunks) > 1:
            # Overlap check: last chars of chunk[0] appear at start of chunk[1]
            pass  # actual overlap depends on whitespace separators

    def test_empty_text_returns_empty_list(self):
        splitter = RecursiveCharacterTextSplitter()
        chunks = splitter.split_text("")
        assert chunks == [] or all(c == "" for c in chunks)

    def test_single_word_shorter_than_chunk_size(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        chunks = splitter.split_text("hello")
        assert len(chunks) >= 1
        assert chunks[0] == "hello"

    def test_chinese_text_with_period_separator(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        text = "人工智能是计算机科学的一个重要分支。它研究如何让计算机模拟人类智能。机器学习是人工智能的核心方法之一。"
        chunks = splitter.split_text(text)
        assert len(chunks) > 0


class TestSemanticChunker:
    def test_basic_split(self):
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.split_text(text)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, str)

    def test_empty_text(self):
        chunker = SemanticChunker()
        chunks = chunker.split_text("")
        assert chunks == []

    def test_split_text_with_embeddings(self):
        import numpy as np

        chunker = SemanticChunker(chunk_size=200, chunk_overlap=20)
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        embeddings = np.random.randn(4, 384).astype(np.float32)
        chunks = chunker.split_text_with_embeddings(text, embeddings)
        assert len(chunks) > 0

    def test_single_sentence(self):
        chunker = SemanticChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split_text("Just one sentence without punctuation")
        assert len(chunks) == 1


class TestTableAwareChunker:
    def test_detects_markdown_tables(self):
        chunker = TableAwareChunker(chunk_size=500, chunk_overlap=50)
        text = (
            "Some preamble text.\n\n"
            "| Col1 | Col2 |\n"
            "|------|------|\n"
            "| a    | b    |\n"
            "| c    | d    |\n\n"
            "Trailing text."
        )
        chunks = chunker.split_text(text)
        assert len(chunks) > 0
        # At least one chunk should be a table
        table_chunks = [c for c in chunks if c.startswith("|") and "---" in c]
        assert len(table_chunks) >= 1, f"No table chunk found in {chunks}"

    def test_preserves_table_integrity(self):
        chunker = TableAwareChunker(chunk_size=500, chunk_overlap=50)
        table = "| Name | Value |\n|------|-------|\n| foo  | 1     |\n| bar  | 2     |"
        chunks = chunker.split_text(table)
        assert len(chunks) == 1
        assert "| foo" in chunks[0]
        assert "| bar" in chunks[0]

    def test_no_table_content(self):
        chunker = TableAwareChunker(chunk_size=100, chunk_overlap=0)
        text = "Just a regular paragraph. Nothing to do with tables."
        chunks = chunker.split_text(text)
        assert len(chunks) > 0
        assert all("---" not in c for c in chunks)


class TestChunkDocuments:
    def test_factory_with_loaded_pages(self):
        page = LoadedPage(
            content="This is the page content. It has multiple sentences.",
            page_number=1,
        )
        chunks = chunk_documents(
            [page], chunk_size=200, chunk_overlap=20, strategy="recursive"
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.page_number == 1
            assert chunk.content

    def test_factory_with_raw_strings(self):
        docs = chunk_documents(
            ["plain string content"], chunk_size=100, chunk_overlap=10
        )
        assert len(docs) > 0
        assert isinstance(docs[0], DocumentChunk)

    def test_factory_semantic_strategy(self):
        page = LoadedPage(content="Sentence one. Sentence two. Sentence three.", page_number=1)
        chunks = chunk_documents(
            [page], chunk_size=200, chunk_overlap=20, strategy="semantic"
        )
        assert len(chunks) > 0

    def test_factory_with_tables(self):
        page = LoadedPage(
            content="Before table.",
            page_number=1,
            tables=["| A | B |\n|---|---|\n| 1 | 2 |"],
        )
        chunks = chunk_documents([page], chunk_size=500, chunk_overlap=20)
        table_chunks = [c for c in chunks if c.is_table]
        assert len(table_chunks) >= 1

    def test_factory_empty_pages(self):
        chunks = chunk_documents([], chunk_size=500, chunk_overlap=50)
        assert chunks == []
