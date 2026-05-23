"""Unified exception hierarchy for compact-rag.

All custom exceptions inherit from CompactRAGException, which automatically
generates a request_id (UUID4) for distributed tracing.
"""

from __future__ import annotations

from uuid import uuid4


class CompactRAGException(Exception):
    """Base exception for all compact-rag errors.

    Attributes:
        message: Human-readable error description.
        details: Optional dict with additional error context.
        cause: Optional original exception that triggered this error.
        request_id: Auto-generated UUID4 for request tracing.
    """

    def __init__(
        self,
        message: str,
        details: dict | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        self.cause = cause
        self.request_id = str(uuid4())
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


# ── Configuration ──────────────────────────────────────────────


class ConfigurationError(CompactRAGException):
    """Configuration error (missing or invalid config values)."""


# ── Document Loading ───────────────────────────────────────────


class DocumentLoadError(CompactRAGException):
    """Document loading or parsing failure."""


class UnsupportedFormatError(DocumentLoadError):
    """Unsupported file format."""


class CorruptedFileError(DocumentLoadError):
    """File is corrupted or unreadable."""


# ── Ingestion ──────────────────────────────────────────────────


class IngestionError(CompactRAGException):
    """Ingestion pipeline error."""


class ChunkingError(IngestionError):
    """Chunking operation failure."""


class EmbeddingError(IngestionError):
    """Embedding generation failure."""


# ── Storage ────────────────────────────────────────────────────


class StorageError(CompactRAGException):
    """Base storage layer error."""


class VectorStoreError(StorageError):
    """ChromaDB operation error."""


class DatabaseError(StorageError):
    """Relational database operation error."""


class FileStorageError(StorageError):
    """File storage operation error."""


class StorageBackendError(FileStorageError):
    """Storage backend connection or authentication failure."""


class FileNotFoundError(FileStorageError):
    """File not found in storage backend."""


# ── Retrieval ──────────────────────────────────────────────────


class RetrievalError(CompactRAGException):
    """Retrieval operation error."""


class EmptyResultError(RetrievalError):
    """Empty retrieval result (not a real error, used for graceful degradation)."""


# ── Generation ─────────────────────────────────────────────────


class GenerationError(CompactRAGException):
    """LLM generation error."""


class LLMTimeoutError(GenerationError):
    """LLM API timeout."""


class LLMAuthError(GenerationError):
    """LLM API authentication failure."""


class LLMRateLimitError(GenerationError):
    """LLM API rate limit exceeded."""


class LLMServiceError(GenerationError):
    """LLM provider service error (e.g., remote model runtime returned 5xx)."""


# ── Tool Execution ─────────────────────────────────────────────


class ToolExecutionError(CompactRAGException):
    """Tool execution failure."""


# ── HTTP Status Mapping ────────────────────────────────────────

# Order matters: children MUST come before parents because
# get_http_status() matches via isinstance() in insertion order.
_EXCEPTION_HTTP_STATUS: dict[type[CompactRAGException], int] = {
    # Configuration
    ConfigurationError: 500,
    # Document — children before parent
    UnsupportedFormatError: 400,
    CorruptedFileError: 400,
    DocumentLoadError: 400,
    # Ingestion — children before parent
    ChunkingError: 500,
    EmbeddingError: 500,
    IngestionError: 500,
    # Storage — children before parent (deepest first)
    StorageBackendError: 502,
    FileNotFoundError: 404,
    VectorStoreError: 500,
    DatabaseError: 500,
    FileStorageError: 500,
    StorageError: 500,
    # Retrieval
    EmptyResultError: 200,
    RetrievalError: 500,
    # Generation — children before parent
    LLMTimeoutError: 504,
    LLMAuthError: 401,
    LLMRateLimitError: 429,
    LLMServiceError: 502,
    GenerationError: 500,
    # Tool
    ToolExecutionError: 500,
}


def get_http_status(exc: CompactRAGException) -> int:
    """Get the HTTP status code for a given exception type.

    Args:
        exc: The exception instance.

    Returns:
        HTTP status code integer.
    """
    for exc_type, status in _EXCEPTION_HTTP_STATUS.items():
        if isinstance(exc, exc_type):
            return status
    return 500
