from __future__ import annotations

import uuid

import pytest

from compact_rag.common.exceptions import (
    ChunkingError,
    CompactRAGException,
    ConfigurationError,
    CorruptedFileError,
    DatabaseError,
    DocumentLoadError,
    EmbeddingError,
    EmptyResultError,
    FileNotFoundError,
    FileStorageError,
    GenerationError,
    IngestionError,
    LLMAuthError,
    LLMRateLimitError,
    LLMTimeoutError,
    RetrievalError,
    StorageBackendError,
    StorageError,
    ToolExecutionError,
    UnsupportedFormatError,
    VectorStoreError,
    get_http_status,
)


class TestExceptions:
    def test_base_exception_init(self):
        exc = CompactRAGException("test message")
        assert exc.message == "test message"
        assert exc.details == {}
        assert exc.cause is None
        assert isinstance(exc.request_id, str)

    def test_exception_with_details_and_cause(self):
        cause = ValueError("root")
        exc = ConfigurationError(
            "bad config", details={"key": "database.url"}, cause=cause
        )
        assert exc.message == "bad config"
        assert exc.details == {"key": "database.url"}
        assert exc.cause is cause

    def test_request_id_is_valid_uuid4(self):
        exc = CompactRAGException("test")
        uid = uuid.UUID(exc.request_id, version=4)
        assert str(uid) == exc.request_id

    def test_str_returns_message(self):
        exc = CompactRAGException("something went wrong")
        assert str(exc) == "something went wrong"

    def test_exception_hierarchy(self):
        assert issubclass(ConfigurationError, CompactRAGException)
        assert issubclass(DocumentLoadError, CompactRAGException)
        assert issubclass(UnsupportedFormatError, DocumentLoadError)
        assert issubclass(CorruptedFileError, DocumentLoadError)
        assert issubclass(IngestionError, CompactRAGException)
        assert issubclass(ChunkingError, IngestionError)
        assert issubclass(EmbeddingError, IngestionError)
        assert issubclass(StorageError, CompactRAGException)
        assert issubclass(VectorStoreError, StorageError)
        assert issubclass(DatabaseError, StorageError)
        assert issubclass(FileStorageError, StorageError)
        assert issubclass(StorageBackendError, FileStorageError)
        assert issubclass(FileNotFoundError, FileStorageError)
        assert issubclass(RetrievalError, CompactRAGException)
        assert issubclass(EmptyResultError, RetrievalError)
        assert issubclass(GenerationError, CompactRAGException)
        assert issubclass(LLMTimeoutError, GenerationError)
        assert issubclass(LLMAuthError, GenerationError)
        assert issubclass(LLMRateLimitError, GenerationError)
        assert issubclass(ToolExecutionError, CompactRAGException)

    def test_instance_hierarchy(self):
        exc = UnsupportedFormatError("bad format")
        assert isinstance(exc, UnsupportedFormatError)
        assert isinstance(exc, DocumentLoadError)
        assert isinstance(exc, CompactRAGException)

    @pytest.mark.parametrize(
        "exc_class,expected_status",
        [
            (ConfigurationError, 500),
            (DocumentLoadError, 400),
            (IngestionError, 500),
            (StorageError, 500),
            (FileStorageError, 500),
            (RetrievalError, 500),
            (GenerationError, 500),
            (ToolExecutionError, 500),
        ],
    )
    def test_get_http_status_base_classes(self, exc_class, expected_status):
        exc = exc_class("test")
        assert get_http_status(exc) == expected_status

    def test_get_http_status_uses_parent_status(self):
        assert get_http_status(UnsupportedFormatError("test")) == 400
        assert get_http_status(ChunkingError("test")) == 500
        assert get_http_status(VectorStoreError("test")) == 500
        assert get_http_status(FileStorageError("test")) == 500
        assert get_http_status(StorageBackendError("test")) == 502
        assert get_http_status(FileNotFoundError("test")) == 404
        assert get_http_status(EmptyResultError("test")) == 200
        assert get_http_status(LLMTimeoutError("test")) == 504
        assert get_http_status(LLMAuthError("test")) == 401
        assert get_http_status(LLMRateLimitError("test")) == 429
