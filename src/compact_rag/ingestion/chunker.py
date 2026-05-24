from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from compact_rag.common.logger import get_logger
from compact_rag.storage.schema import DocumentChunk

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)

_TABLE_SEPARATOR_PATTERN = re.compile(r"^\|[-| :]+\|$")


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", "，", ",", " ", ""]

    def split_text(self, text: str) -> list[str]:
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        chunks: list[str] = []
        separator = separators[-1]
        new_separators: list[str] = []

        for i, s in enumerate(separators):
            _sep = s if s else " "
            if _sep == "":
                separator = s
                break
            if _sep in text:
                separator = s
                new_separators = separators[i + 1 :]
                break

        _sep = separator if separator else " "
        splits = _split_on_separator(text, _sep)
        good_splits: list[str] = []

        for s in splits:
            if _len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = _merge_splits(good_splits, _sep, self.chunk_size, self.chunk_overlap)
                    chunks.extend(merged_text)
                    good_splits = []
                if new_separators:
                    other_chunks = self._split_text(s, new_separators)
                    chunks.extend(other_chunks)
                else:
                    for i in range(0, len(s), max(self.chunk_size - self.chunk_overlap, 1)):
                        chunks.append(s[i : i + self.chunk_size])

        if good_splits:
            merged_text = _merge_splits(good_splits, _sep, self.chunk_size, self.chunk_overlap)
            chunks.extend(merged_text)

        return chunks


def _split_on_separator(text: str, separator: str) -> list[str]:
    if separator:
        return text.split(separator)
    return list(text)


def _merge_splits(
    splits: list[str], separator: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    docs: list[str] = []
    current: list[str] = []
    current_len = 0

    for s in splits:
        s_len = _len(s)
        if current_len + s_len + (len(current) * _len(separator)) <= chunk_size:
            current.append(s)
            current_len += s_len
        else:
            if current:
                docs.append(separator.join(current))
                overlap_splits = _calculate_overlap(current, chunk_overlap, separator)
                current = overlap_splits if overlap_splits else []
                current_len = sum(_len(x) for x in current)
            current.append(s)
            current_len += s_len

    if current:
        docs.append(separator.join(current))

    return docs


def _calculate_overlap(
    current: list[str], chunk_overlap: int, separator: str
) -> list[str]:
    if not current or chunk_overlap <= 0:
        return []
    overlap_text = separator.join(current)
    if _len(overlap_text) <= chunk_overlap:
        return []
    remain = overlap_text[-(min(chunk_overlap, _len(overlap_text) * 2 // 3)) :]
    return _split_on_separator(remain, separator) if separator else list(remain)


def _len(text: str) -> int:
    return len(text)


class SemanticChunker:
    """Chunker based on embedding similarity (stub implementation).

    Uses sentence boundary detection and cosine similarity to find
    semantic breakpoints. Full semantic chunking requires embedding service.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.7,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold

    def split_text(self, text: str) -> list[str]:
        sentences = _split_sentences(text)
        if not sentences:
            return []
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            s_len = len(sentence)
            if current_len + s_len > self.chunk_size and current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            current.append(sentence)
            current_len += s_len

        if current:
            chunks.append(" ".join(current))

        return chunks

    def split_text_with_embeddings(
        self, text: str, embeddings: np.ndarray
    ) -> list[str]:
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return [text] if text else []

        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            s_len = len(sentence)

            if current_len + s_len > self.chunk_size:
                chunks.append(" ".join(current))
                current = []
                current_len = 0

            current.append(sentence)
            current_len += s_len

        if current:
            chunks.append(" ".join(current))

        return chunks


def _split_sentences(text: str) -> list[str]:
    sentence_endings = re.compile(r"(?<=[。！？.!?])\s+")
    parts = sentence_endings.split(text)
    result: list[str] = []
    for part in parts:
        part = part.strip()
        if part:
            result.append(part)
    return result or [text.strip()] if text.strip() else []


class TableAwareChunker:
    """Chunker that detects markdown tables and keeps them intact."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        segments = self._extract_table_segments(text)
        chunks: list[str] = []

        for segment in segments:
            if segment["is_table"]:
                chunks.append(segment["content"])
            else:
                if len(segment["content"]) <= self.chunk_size:
                    chunks.append(segment["content"])
                else:
                    sub_chunks = splitter.split_text(segment["content"])
                    chunks.extend(sub_chunks)

        return chunks

    def _extract_table_segments(self, text: str) -> list[dict]:
        lines = text.split("\n")
        segments: list[dict] = []
        buffer: list[str] = []
        in_table = False
        table_lines: list[str] = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("|") and "---" in stripped.replace(" ", ""):
                if not in_table and buffer:
                    buffer.pop()
                    in_table = True
                    table_lines = [buffer.pop()] if buffer else []
                    table_lines.append(stripped)
                elif in_table:
                    table_lines.append(stripped)
                continue

            is_table_row = stripped.startswith("|") and stripped.endswith("|")
            if in_table:
                if is_table_row:
                    table_lines.append(stripped)
                else:
                    prev_line = lines[i - 1].strip() if i > 0 else ""
                    context_before = lines[i - 2].strip() if i > 1 and not lines[i - 2].strip().startswith("|") else prev_line

                    if buffer:
                        non_table_before = "\n".join(buffer)
                        if non_table_before.strip():
                            segments.append({"content": non_table_before, "is_table": False})
                        buffer = []

                    context_parts: list[str] = []
                    if context_before and not context_before.startswith("|") and not _TABLE_SEPARATOR_PATTERN.match(context_before):
                        context_parts.append(context_before)
                    context_parts.extend(table_lines)
                    if stripped and not stripped.startswith("|"):
                        context_parts.append(stripped)

                    segments.append({"content": "\n".join(context_parts), "is_table": True})
                    table_lines = []
                    in_table = False
                    if stripped:
                        buffer = []
            else:
                if is_table_row:
                    if buffer:
                        if buffer[-1].strip():
                            context_before = buffer.pop()
                        else:
                            context_before = ""
                        non_table = "\n".join(buffer)
                        if non_table.strip():
                            segments.append({"content": non_table, "is_table": False})
                        buffer = [context_before] if context_before else []
                    table_lines = [stripped]
                    in_table = True
                else:
                    buffer.append(line)

        if table_lines:
            if buffer:
                non_table_before = "\n".join(buffer)
                if non_table_before.strip():
                    segments.append({"content": non_table_before, "is_table": False})
            segments.append({"content": "\n".join(table_lines), "is_table": True})
        elif buffer:
            remaining = "\n".join(buffer)
            if remaining.strip():
                segments.append({"content": remaining, "is_table": False})

        return segments

    def _is_table_line(self, line: str) -> bool:
        stripped = line.strip()
        if _TABLE_SEPARATOR_PATTERN.match(stripped):
            return True
        return stripped.startswith("|") and stripped.endswith("|")


def chunk_documents(
    pages: list,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = "recursive",
) -> list[DocumentChunk]:
    from compact_rag.ingestion.loader import LoadedPage

    if strategy == "semantic":
        chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        chunker = TableAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page in pages:
        content = page.content if isinstance(page, LoadedPage) else str(page)
        page_number = page.page_number if isinstance(page, LoadedPage) else None
        page_tables = page.tables if isinstance(page, LoadedPage) else []
        metadata = page.metadata if isinstance(page, LoadedPage) else {}

        text_chunks = chunker.split_text(content)

        for tc in text_chunks:
            tc = tc.strip()
            if not tc:
                continue
            is_table = tc.startswith("|") and "---" in tc and tc.endswith("|")

            all_chunks.append(
                DocumentChunk(
                    content=tc,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    is_table=is_table,
                    token_count=len(tc),
                    content_hash=hashlib.sha256(tc.encode()).hexdigest(),
                    metadata=metadata,
                )
            )
            chunk_index += 1

        for table_md in page_tables:
            if table_md.strip():
                all_chunks.append(
                    DocumentChunk(
                        content=table_md.strip(),
                        page_number=page_number,
                        chunk_index=chunk_index,
                        is_table=True,
                        token_count=len(table_md),
                        content_hash=hashlib.sha256(table_md.encode()).hexdigest(),
                        metadata=metadata,
                    )
                )
                chunk_index += 1

    return all_chunks
