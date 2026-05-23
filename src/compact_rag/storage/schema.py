"""Shared Pydantic data models for storage layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A document chunk — bridge between vector store and SQL."""

    content: str
    page_number: int | None = None
    chunk_index: int = 0
    is_table: bool = False
    token_count: int = 0
    content_hash: str = ""
    metadata: dict = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A single search result from vector or hybrid retrieval."""

    id: str
    content: str
    score: float
    metadata: dict = Field(default_factory=dict)


class IngestionResult(BaseModel):
    """Result of a single file ingestion operation."""

    doc_id: str
    filename: str
    status: str  # completed | skipped | failed
    chunk_count: int = 0
    table_count: int = 0
    error_message: str | None = None
    duration_ms: float = 0.0


class RAGCitation(BaseModel):
    """Citation reference for a RAG response."""

    doc_id: str
    chunk_index: int
    page_number: int | None = None
    filename: str = ""
    score: float = 0.0
    content_snippet: str = ""


class RAGResponse(BaseModel):
    """Complete RAG query response."""

    id: str
    answer: str
    citations: list[RAGCitation] = Field(default_factory=list)
    token_usage: dict = Field(default_factory=dict)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0


class ChatResponse(BaseModel):
    """LLM chat response."""

    content: str
    tool_calls: list[dict] | None = None
    token_usage: dict = Field(default_factory=dict)
    model: str = ""
    finish_reason: str = "stop"
