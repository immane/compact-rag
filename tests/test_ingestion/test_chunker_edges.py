from __future__ import annotations

import numpy as np

from compact_rag.ingestion.chunker import (
    RecursiveCharacterTextSplitter,
    SemanticChunker,
    TableAwareChunker,
    chunk_documents,
)
from compact_rag.ingestion.loader import LoadedPage
from compact_rag.storage.schema import DocumentChunk


# ── RecursiveCharacterTextSplitter edge cases ───────────────────


class TestRecursiveCharacterTextSplitterEdges:
    def test_chunk_size_zero_graceful(self):
        """chunk_size=0 should still work (max of 1 used in step)."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=0, chunk_overlap=0)
        chunks = splitter.split_text("abc")
        assert len(chunks) > 0

    def test_chunk_overlap_equals_chunk_size(self):
        """overlap >= chunk_size; step calc = max(chunk_size - overlap, 1) → 1 char steps."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=10)
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = splitter.split_text(text)
        for c in chunks:
            assert len(c) <= 10

    def test_chunk_overlap_exceeds_chunk_size(self):
        """overlap > chunk_size; step calc = 1."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=5, chunk_overlap=100)
        text = "abcdefghij"
        chunks = splitter.split_text(text)
        for c in chunks:
            assert len(c) <= 5

    def test_all_whitespace(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        chunks = splitter.split_text("   \t\n   \n\n  ")
        # Should not crash
        assert isinstance(chunks, list)

    def test_very_long_single_line(self):
        """A single line longer than chunk_size is split by character."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        text = "x" * 500
        chunks = splitter.split_text(text)
        for c in chunks:
            assert len(c) <= 50
        # Should produce roughly 10 chunks
        assert len(chunks) >= 8

    def test_custom_separators(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=30, chunk_overlap=5, separators=[";", ",", " "]
        )
        text = "aaa;bbb,ccc ddd;eee,fff"
        chunks = splitter.split_text(text)
        assert len(chunks) > 0

    def test_newline_as_only_separator(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=30, chunk_overlap=5, separators=["\n"]
        )
        text = "line1\nline2\nline3\nline4\nline5"
        chunks = splitter.split_text(text)
        assert len(chunks) > 0

    def test_no_matching_separator_in_text(self):
        """When none of the preferred separators exist, falls back to empty string split."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=30, chunk_overlap=5, separators=["zzz", "yyy", ""]
        )
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = splitter.split_text(text)
        for c in chunks:
            assert len(c) <= 30


# ── SemanticChunker edge cases ──────────────────────────────────


class TestSemanticChunkerEdges:
    def test_all_same_embeddings_no_breakpoint(self):
        """When all sentences have identical embeddings, chunker still works."""
        chunker = SemanticChunker(chunk_size=200, chunk_overlap=20)
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        embeddings = np.ones((4, 384), dtype=np.float32)
        chunks = chunker.split_text_with_embeddings(text, embeddings)
        assert len(chunks) > 0

    def test_all_different_embeddings(self):
        """Widely varying embeddings may produce many breakpoints."""
        chunker = SemanticChunker(chunk_size=200, chunk_overlap=20)
        text = "A. B. C. D. E. F. G."
        embeddings = np.eye(7, 384, dtype=np.float32)
        chunks = chunker.split_text_with_embeddings(text, embeddings)
        assert len(chunks) > 0

    def test_single_sentence_with_embeddings(self):
        chunker = SemanticChunker(chunk_size=500, chunk_overlap=50)
        text = "Only one sentence here without a period"
        embeddings = np.random.randn(1, 384).astype(np.float32)
        chunks = chunker.split_text_with_embeddings(text, embeddings)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_verify_ignores_embeddings_stub(self):
        """SemanticChunker currently ignores cosine similarity (stub)."""
        chunker = SemanticChunker(chunk_size=500, chunk_overlap=50)
        text = "S1. S2. S3."
        # Same text, different embeddings — should produce same output
        emb1 = np.random.randn(3, 384).astype(np.float32)
        emb2 = np.random.randn(3, 384).astype(np.float32)
        chunks1 = chunker.split_text_with_embeddings(text, emb1)
        chunks2 = chunker.split_text_with_embeddings(text, emb2)
        assert len(chunks1) == len(chunks2)
        # Content should be the same since embeddings are ignored
        assert chunks1 == chunks2

    def test_empty_text_with_embeddings(self):
        chunker = SemanticChunker()
        text = ""
        embeddings = np.array([]).reshape(0, 384)
        chunks = chunker.split_text_with_embeddings(text, embeddings)
        assert chunks == []

    def test_single_character(self):
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split_text("x")
        assert len(chunks) == 1

    def test_only_punctuation(self):
        chunker = SemanticChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split_text(". . . . .")
        assert isinstance(chunks, list)


# ── TableAwareChunker edge cases ────────────────────────────────


class TestTableAwareChunkerEdges:
    def test_nested_tables(self):
        """Content with table-like structure inside another table context."""
        chunker = TableAwareChunker(chunk_size=500, chunk_overlap=50)
        text = (
            "| OuterA | OuterB |\n"
            "|--------|--------|\n"
            "| inner pipe | table |\n\n"
            "| Another | Table |\n"
            "|---------|-------|\n"
            "| x | y |\n"
        )
        chunks = chunker.split_text(text)
        table_chunks = [c for c in chunks if c.startswith("|") and "---" in c]
        assert len(table_chunks) >= 1

    def test_malformed_markdown_tables(self):
        """Missing pipes or mismatched columns."""
        chunker = TableAwareChunker(chunk_size=500, chunk_overlap=50)
        text = (
            "| A | B\n"
            "| --- | --- |\n"
            "| 1 | 2 |\n"
        )
        chunks = chunker.split_text(text)
        assert isinstance(chunks, list)

    def test_different_alignment_syntax(self):
        chunker = TableAwareChunker(chunk_size=500, chunk_overlap=50)
        text = (
            "Some intro.\n\n"
            "| Left | Center | Right |\n"
            "| :--- | :---: | ---: |\n"
            "| a    | b      | c     |\n\n"
            "After table text."
        )
        chunks = chunker.split_text(text)
        table_chunks = [c for c in chunks if c.startswith("|") and "---" in c]
        assert len(table_chunks) >= 1
        assert ":---" in table_chunks[0]

    def test_single_column_table(self):
        chunker = TableAwareChunker(chunk_size=500, chunk_overlap=50)
        text = (
            "| Value |\n"
            "|-------|\n"
            "| 10    |\n"
            "| 20    |\n"
        )
        chunks = chunker.split_text(text)
        assert len(chunks) == 1
        assert "| 10" in chunks[0]

    def test_only_table_no_surrounding_text(self):
        chunker = TableAwareChunker(chunk_size=500, chunk_overlap=50)
        text = (
            "| X | Y |\n"
            "| --- | --- |\n"
            "| 1 | 2 |\n"
        )
        chunks = chunker.split_text(text)
        assert len(chunks) == 1
        assert chunks[0].startswith("|")

    def test_table_with_empty_cells(self):
        chunker = TableAwareChunker(chunk_size=500, chunk_overlap=50)
        text = (
            "| A | B | C |\n"
            "|---|---|---|\n"
            "| 1 |   | 3 |\n"
            "|   | 2 |   |\n"
        )
        chunks = chunker.split_text(text)
        assert len(chunks) == 1

    def test_text_with_pipe_but_not_table(self):
        """Lines starting with | but not ending with | are NOT detected as table rows."""
        chunker = TableAwareChunker(chunk_size=500, chunk_overlap=50)
        text = (
            "Regular paragraph.\n\n"
            "| this is not\n"
            "| a table\n"
            "| really\n\n"
            "More text."
        )
        chunks = chunker.split_text(text)
        # These lines don't end with | so they're treated as regular text
        assert any("Regular paragraph" in c for c in chunks)
        assert any("More text" in c for c in chunks)

    def test_empty_input(self):
        chunker = TableAwareChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split_text("")
        assert chunks == []

    def test_table_immediately_at_start(self):
        chunker = TableAwareChunker(chunk_size=500, chunk_overlap=50)
        text = (
            "| Name | Score |\n"
            "|------|-------|\n"
            "| A    | 100   |\n"
        )
        chunks = chunker.split_text(text)
        assert len(chunks) == 1
        assert "| Name" in chunks[0]

    def test_table_surrounded_by_context_lines(self):
        """Context line immediately before/after table is included in the table chunk."""
        chunker = TableAwareChunker(chunk_size=500, chunk_overlap=50)
        text = (
            "Here is a table:\n"
            "| Item | Qty |\n"
            "|------|-----|\n"
            "| Pen  | 3   |\n"
            "And that was the table."
        )
        chunks = chunker.split_text(text)
        table_chunks = [c for c in chunks if "---" in c]
        assert len(table_chunks) >= 1


# ── chunk_documents edge cases ──────────────────────────────────


class TestChunkDocumentsEdges:
    def test_strategy_recursive_uses_table_aware_chunker(self):
        """'recursive' strategy actually uses TableAwareChunker internally."""
        page = LoadedPage(
            content="| A | B |\n| --- | --- |\n| 1 | 2 |",
            page_number=1,
        )
        chunks = chunk_documents(
            [page], chunk_size=500, chunk_overlap=50, strategy="recursive"
        )
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, DocumentChunk)

    def test_semantic_strategy(self):
        page = LoadedPage(
            content="First sentence. Second sentence. Third sentence.",
            page_number=1,
        )
        chunks = chunk_documents(
            [page], chunk_size=200, chunk_overlap=20, strategy="semantic"
        )
        assert len(chunks) > 0

    def test_unknown_strategy_defaults_to_table_aware(self):
        """Unknown strategy falls back to TableAwareChunker (else branch)."""
        page = LoadedPage(content="Some text", page_number=1)
        chunks = chunk_documents(
            [page], chunk_size=500, chunk_overlap=50, strategy="unknown_strategy"
        )
        assert len(chunks) >= 1

    def test_mixed_pages_with_and_without_tables(self):
        pages = [
            LoadedPage(
                content="| A | B |\n| --- | --- |\n| 1 | 2 |",
                page_number=1,
                tables=[],
            ),
            LoadedPage(
                content="Plain text paragraph.",
                page_number=2,
                tables=[],
            ),
        ]
        chunks = chunk_documents(pages, chunk_size=500, chunk_overlap=50)
        table_chunks = [c for c in chunks if c.is_table]
        assert len(table_chunks) >= 1
        assert len(chunks) >= 2

    def test_all_table_pages(self):
        pages = [
            LoadedPage(
                content="| A | B |\n| --- | --- |\n| 1 | 2 |",
                page_number=1,
                tables=["| X | Y |\n| --- | --- |\n| a | b |"],
            ),
        ]
        chunks = chunk_documents(pages, chunk_size=500, chunk_overlap=50)
        assert all(c.is_table for c in chunks)

    def test_raw_string_input(self):
        """chunk_documents accepts raw strings."""
        chunks = chunk_documents(["raw text"], chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 0
        assert isinstance(chunks[0], DocumentChunk)

    def test_empty_chunks_skipped(self):
        """Whitespace-only chunks from splitting should be skipped."""
        page = LoadedPage(content="\n\n   \n\n", page_number=1)
        chunks = chunk_documents([page], chunk_size=500, chunk_overlap=50)
        # All whitespace chunks stripped out
        for c in chunks:
            assert c.content.strip() != ""

    def test_custom_chunk_size_and_overlap(self):
        page = LoadedPage(content="x" * 2000, page_number=1)
        chunks = chunk_documents([page], chunk_size=100, chunk_overlap=20)
        for c in chunks:
            assert len(c.content) <= 100

    def test_page_metadata_passed_through(self):
        page = LoadedPage(
            content="Hello world.",
            page_number=3,
            metadata={"source": "test", "author": "nobody"},
        )
        chunks = chunk_documents([page], chunk_size=500, chunk_overlap=50)
        for c in chunks:
            assert c.metadata.get("source") == "test"
            assert c.metadata.get("author") == "nobody"

    def test_page_tables_appended_as_chunks(self):
        page = LoadedPage(
            content="Text before table.",
            page_number=1,
            tables=[
                "| Col1 | Col2 |\n| --- | --- |\n| a | b |",
                "| X | Y |\n| --- | --- |\n| 1 | 2 |",
            ],
        )
        chunks = chunk_documents([page], chunk_size=500, chunk_overlap=50)
        table_chunks = [c for c in chunks if c.is_table]
        assert len(table_chunks) == 2
        for tc in table_chunks:
            assert tc.content.startswith("|")
            assert "---" in tc.content

    def test_empty_tables_skipped(self):
        page = LoadedPage(
            content="Text.",
            page_number=1,
            tables=["   ", ""],
        )
        chunks = chunk_documents([page], chunk_size=500, chunk_overlap=50)
        table_chunks = [c for c in chunks if c.is_table]
        assert len(table_chunks) == 0

    def test_zero_chunk_overlap(self):
        page = LoadedPage(content="x" * 1000, page_number=1)
        chunks = chunk_documents([page], chunk_size=100, chunk_overlap=0)
        assert len(chunks) >= 1
