"""Request/Response Pydantic models for the API layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Pagination ─────────────────────────────────────────────────


class PaginationMeta(BaseModel):
    page: int = 1
    page_size: int = 20
    total: int = 0
    total_pages: int = 0


class PaginatedResponse(BaseModel):
    data: list[Any] = Field(default_factory=list)
    pagination: PaginationMeta = Field(default_factory=PaginationMeta)


# ── Chat ───────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str  # system, user, assistant, tool
    content: str
    name: str | None = None


class RetrievalOptions(BaseModel):
    top_k: int = Field(default=10, ge=1, le=100)
    rerank: bool = True
    hybrid_search: bool = True


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: list[ChatMessage] = Field(min_length=1)
    collection: str = "default"
    retrieval: RetrievalOptions = Field(default_factory=RetrievalOptions)
    tools: list[dict] | None = None
    stream: bool = False
    conversation_id: str | None = None
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class ChatCitation(BaseModel):
    doc_id: str
    filename: str
    page_number: int | None = None
    chunk_index: int = 0
    score: float = 0.0
    content_snippet: str = ""


class ChatMessageResponse(BaseModel):
    role: str = "assistant"
    content: str = ""
    citations: list[ChatCitation] = Field(default_factory=list)


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessageResponse = Field(default_factory=ChatMessageResponse)
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatChoice] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ── Collections ────────────────────────────────────────────────


class CollectionCreateRequest(BaseModel):
    name: str
    description: str = ""
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    chunk_size: int = 500
    chunk_overlap: int = 50


class CollectionResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    embedding_model: str = ""
    chunk_size: int = 500
    chunk_overlap: int = 50
    document_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


# ── Documents ──────────────────────────────────────────────────


class DocumentResponse(BaseModel):
    id: str
    collection_id: str = ""
    filename: str = ""
    file_type: str = ""
    file_size: int = 0
    file_hash: str = ""
    page_count: int | None = None
    chunk_count: int = 0
    table_count: int = 0
    status: str = "pending"
    error_message: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class DocumentIngestResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: int = 0
    table_count: int = 0
    error_message: str | None = None
    duration_ms: float = 0.0


class IngestUrlRequest(BaseModel):
    url: str
    collection: str = "default"


# ── Conversations ──────────────────────────────────────────────


class ConversationResponse(BaseModel):
    id: str
    collection_id: str | None = None
    title: str = "New Conversation"
    model: str = ""
    message_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


class MessageResponse(BaseModel):
    id: str
    conversation_id: str = ""
    role: str = ""
    content: str = ""
    sources: list[dict] | None = None
    token_count: int | None = None
    latency_ms: int | None = None
    created_at: str | None = None


class ConversationDetailResponse(BaseModel):
    id: str
    collection_id: str | None = None
    title: str = ""
    model: str = ""
    message_count: int = 0
    messages: list[MessageResponse] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None


# ── Ingestion Jobs ─────────────────────────────────────────────


class IngestionJobResponse(BaseModel):
    id: str
    collection_id: str = ""
    status: str = "pending"
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    errors: list[dict] | None = None
    started_at: str | None = None
    completed_at: str | None = None
    created_at: str | None = None


# ── API Keys ───────────────────────────────────────────────────


class ApiKeyCreateRequest(BaseModel):
    name: str
    permissions: list[str] = Field(default_factory=lambda: ["read"])


class ApiKeyResponse(BaseModel):
    id: str
    name: str = ""
    key_prefix: str = ""
    permissions: list[str] = Field(default_factory=list)
    is_active: bool = True
    expires_at: str | None = None
    created_at: str | None = None


class ApiKeyCreateResponse(BaseModel):
    id: str
    name: str = ""
    key: str = ""
    permissions: list[str] = Field(default_factory=list)
    created_at: str | None = None


class ApiKeyUpdateRequest(BaseModel):
    is_active: bool | None = None
    name: str | None = None
    permissions: list[str] | None = None


# ── Health / System ────────────────────────────────────────────


class HealthResponse(BaseModel):
    api: str = "ok"  # ok | degraded | error | disabled
    database: str = "degraded"
    chromadb: str = "degraded"
    storage: str = "degraded"


class InfoResponse(BaseModel):
    version: str
    database_url: str
    embedding_model: str
    embedding_dimension: int
    llm_provider: str
    llm_model: str
    storage_backend: str
    log_level: str


# ── Error ──────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)
    request_id: str = ""


class ErrorResponse(BaseModel):
    error: ErrorDetail
