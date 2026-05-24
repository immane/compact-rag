from __future__ import annotations

import pytest
from pydantic import ValidationError

from compact_rag.api.schemas import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyResponse,
    ApiKeyUpdateRequest,
    ChatChoice,
    ChatCitation,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatMessageResponse,
    CollectionCreateRequest,
    CollectionResponse,
    ConversationDetailResponse,
    ConversationResponse,
    DocumentIngestResponse,
    DocumentResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    InfoResponse,
    IngestUrlRequest,
    IngestionJobResponse,
    MessageResponse,
    PaginatedResponse,
    PaginationMeta,
    RetrievalOptions,
    UsageInfo,
)


# ── ChatMessage ─────────────────────────────────────────────────

class TestChatMessage:
    def test_valid_message_all_roles(self):
        for role in ("system", "user", "assistant", "tool"):
            msg = ChatMessage(role=role, content="hello")
            assert msg.role == role
            assert msg.content == "hello"
            assert msg.name is None

    def test_optional_name_field(self):
        msg = ChatMessage(role="user", content="hi", name="Alice")
        assert msg.name == "Alice"

    def test_default_name_is_none(self):
        msg = ChatMessage(role="user", content="hi")
        assert msg.name is None

    def test_missing_content_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user")

    def test_missing_role_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(content="hi")


# ── ChatCompletionRequest ───────────────────────────────────────

class TestChatCompletionRequest:
    def test_valid_request(self):
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="hello")],
        )
        assert req.model == "gpt-4"
        assert len(req.messages) == 1
        assert req.collection == "default"
        assert req.stream is False
        assert req.temperature == 0.1

    def test_empty_messages_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model="gpt-4", messages=[])

    def test_missing_model_field_defaults(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
        )
        assert req.model == "gpt-4o-mini"

    def test_stream_true(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        assert req.stream is True

    def test_optional_tools_field(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )
        assert len(req.tools) == 1

    def test_conversation_id_set(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            conversation_id="conv-123",
        )
        assert req.conversation_id == "conv-123"

    def test_temperature_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")],
                temperature=3.0,
            )


# ── RetrievalOptions ────────────────────────────────────────────

class TestRetrievalOptions:
    def test_valid_defaults(self):
        opts = RetrievalOptions()
        assert opts.top_k == 10
        assert opts.rerank is True
        assert opts.hybrid_search is True

    def test_top_k_one(self):
        opts = RetrievalOptions(top_k=1)
        assert opts.top_k == 1

    def test_top_k_hundred(self):
        opts = RetrievalOptions(top_k=100)
        assert opts.top_k == 100

    def test_top_k_zero_raises(self):
        with pytest.raises(ValidationError):
            RetrievalOptions(top_k=0)

    def test_top_k_negative_raises(self):
        with pytest.raises(ValidationError):
            RetrievalOptions(top_k=-5)

    def test_top_k_above_hundred_raises(self):
        with pytest.raises(ValidationError):
            RetrievalOptions(top_k=101)


# ── PaginationMeta ──────────────────────────────────────────────

class TestPaginationMeta:
    def test_defaults(self):
        p = PaginationMeta()
        assert p.page == 1
        assert p.page_size == 20
        assert p.total == 0
        assert p.total_pages == 0

    def test_custom_values(self):
        p = PaginationMeta(page=2, page_size=10, total=45, total_pages=5)
        assert p.page == 2
        assert p.page_size == 10
        assert p.total == 45
        assert p.total_pages == 5

    def test_total_pages_calculated_from_total_and_page_size(self):
        p = PaginationMeta(page=1, page_size=10, total=95, total_pages=10)
        assert p.total_pages == 10

    def test_page_size_zero(self):
        p = PaginationMeta(page=1, page_size=0, total=10, total_pages=0)
        assert p.page_size == 0


# ── PaginatedResponse ───────────────────────────────────────────

class TestPaginatedResponse:
    def test_defaults(self):
        pr = PaginatedResponse()
        assert pr.data == []
        assert isinstance(pr.pagination, PaginationMeta)

    def test_with_data(self):
        pr = PaginatedResponse(
            data=[{"id": 1}, {"id": 2}],
            pagination=PaginationMeta(total=2, total_pages=1),
        )
        assert len(pr.data) == 2
        assert pr.pagination.total == 2


# ── ChatCitation ────────────────────────────────────────────────

class TestChatCitation:
    def test_required_fields_only(self):
        c = ChatCitation(doc_id="d1", filename="f.pdf")
        assert c.doc_id == "d1"
        assert c.filename == "f.pdf"
        assert c.page_number is None
        assert c.chunk_index == 0
        assert c.score == 0.0
        assert c.content_snippet == ""

    def test_all_fields_set(self):
        c = ChatCitation(
            doc_id="d2",
            filename="g.pdf",
            page_number=3,
            chunk_index=5,
            score=0.88,
            content_snippet="some text",
        )
        assert c.doc_id == "d2"
        assert c.filename == "g.pdf"
        assert c.page_number == 3
        assert c.chunk_index == 5
        assert c.score == 0.88
        assert c.content_snippet == "some text"

    def test_missing_doc_id_raises(self):
        with pytest.raises(ValidationError):
            ChatCitation(filename="f.pdf")

    def test_missing_filename_raises(self):
        with pytest.raises(ValidationError):
            ChatCitation(doc_id="d1")


# ── ChatCompletionResponse ──────────────────────────────────────

class TestChatCompletionResponse:
    def test_minimal_construction(self):
        resp = ChatCompletionResponse(id="chat-1")
        assert resp.id == "chat-1"
        assert resp.object == "chat.completion"
        assert resp.created == 0
        assert resp.model == ""
        assert resp.choices == []
        assert isinstance(resp.usage, UsageInfo)

    def test_full_response(self):
        resp = ChatCompletionResponse(
            id="chat-xyz",
            object="chat.completion",
            created=1710000000,
            model="gpt-4",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessageResponse(
                        role="assistant",
                        content="answer",
                        citations=[ChatCitation(doc_id="d1", filename="a.pdf")],
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )
        assert resp.id == "chat-xyz"
        assert resp.model == "gpt-4"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.content == "answer"
        assert resp.choices[0].message.citations[0].doc_id == "d1"
        assert resp.usage.total_tokens == 150

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionResponse()


# ── HealthResponse ──────────────────────────────────────────────

class TestHealthResponse:
    def test_defaults(self):
        h = HealthResponse()
        assert h.api == "ok"
        assert h.database == "ok"
        assert h.chromadb == "ok"
        assert h.storage == "ok"

    def test_custom_statuses(self):
        h = HealthResponse(api="ok", database="ok", chromadb="ok", storage="ok")
        assert h.api == "ok"
        assert h.database == "ok"
        assert h.chromadb == "ok"
        assert h.storage == "ok"

    def test_error_statuses(self):
        h = HealthResponse(api="error", database="error", chromadb="error", storage="error")
        assert h.api == "error"
        assert h.database == "error"
        assert h.chromadb == "error"
        assert h.storage == "error"

    def test_missing_fields_use_defaults(self):
        h = HealthResponse(api="ok")
        assert h.database == "ok"
        assert h.chromadb == "ok"
        assert h.storage == "ok"


# ── InfoResponse ────────────────────────────────────────────────

class TestInfoResponse:
    def test_required_fields(self):
        info = InfoResponse(
            version="1.0.0",
            database_url="sqlite:///db",
            embedding_model="model-x",
            embedding_dimension=768,
            llm_provider="openai",
            llm_model="gpt-4",
            storage_backend="local",
            log_level="INFO",
        )
        assert info.version == "1.0.0"
        assert info.database_url == "sqlite:///db"
        assert info.embedding_model == "model-x"
        assert info.embedding_dimension == 768
        assert info.llm_provider == "openai"
        assert info.llm_model == "gpt-4"
        assert info.storage_backend == "local"
        assert info.log_level == "INFO"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            InfoResponse(version="1.0.0")


# ── ErrorResponse ───────────────────────────────────────────────

class TestErrorResponse:
    def test_minimal_construction(self):
        err = ErrorResponse(
            error=ErrorDetail(code="E001", message="Something went wrong")
        )
        assert err.error.code == "E001"
        assert err.error.message == "Something went wrong"
        assert err.error.details == {}
        assert err.error.request_id == ""

    def test_full_error(self):
        err = ErrorResponse(
            error=ErrorDetail(
                code="E_VALIDATION",
                message="Invalid input",
                details={"field": "email", "reason": "format"},
                request_id="req-abc-123",
            )
        )
        assert err.error.code == "E_VALIDATION"
        assert err.error.message == "Invalid input"
        assert err.error.details == {"field": "email", "reason": "format"}
        assert err.error.request_id == "req-abc-123"

    def test_missing_error_raises(self):
        with pytest.raises(ValidationError):
            ErrorResponse()


# ── CollectionCreateRequest ─────────────────────────────────────

class TestCollectionCreateRequest:
    def test_valid_creation(self):
        req = CollectionCreateRequest(
            name="my-collection",
            description="A test collection",
        )
        assert req.name == "my-collection"
        assert req.description == "A test collection"
        assert req.embedding_model == "BAAI/bge-small-zh-v1.5"
        assert req.chunk_size == 500
        assert req.chunk_overlap == 50

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            CollectionCreateRequest()

    def test_whitespace_only_name(self):
        req = CollectionCreateRequest(name="   ")
        assert req.name == "   "


# ── CollectionResponse ──────────────────────────────────────────

class TestCollectionResponse:
    def test_valid_response(self):
        resp = CollectionResponse(
            id="col-1",
            name="test",
            description="desc",
            embedding_model="model-x",
            document_count=5,
            created_at="2024-01-01T00:00:00Z",
        )
        assert resp.id == "col-1"
        assert resp.name == "test"
        assert resp.document_count == 5

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            CollectionResponse(name="test")


# ── IngestUrlRequest ────────────────────────────────────────────

class TestIngestUrlRequest:
    def test_valid_url(self):
        req = IngestUrlRequest(url="https://example.com/doc.pdf")
        assert req.url == "https://example.com/doc.pdf"
        assert req.collection == "default"

    def test_missing_url_raises(self):
        with pytest.raises(ValidationError):
            IngestUrlRequest()

    def test_invalid_url_format(self):
        req = IngestUrlRequest(url="not-a-valid-url-!!")
        assert req.url == "not-a-valid-url-!!"

    def test_custom_collection(self):
        req = IngestUrlRequest(url="https://example.com/doc.pdf", collection="custom")
        assert req.collection == "custom"


# ── DocumentResponse ────────────────────────────────────────────

class TestDocumentResponse:
    def test_valid_response(self):
        doc = DocumentResponse(
            id="doc-1",
            collection_id="col-1",
            filename="test.pdf",
            file_type="pdf",
            file_size=1024,
            file_hash="abc123",
            status="completed",
        )
        assert doc.id == "doc-1"
        assert doc.status == "completed"
        assert doc.filename == "test.pdf"
        assert doc.file_type == "pdf"

    def test_status_default_is_pending(self):
        doc = DocumentResponse(
            id="doc-1",
            filename="f.pdf",
            file_type="pdf",
            file_size=0,
            file_hash="x",
        )
        assert doc.status == "pending"

    def test_error_status_with_message(self):
        doc = DocumentResponse(
            id="doc-err",
            filename="bad.pdf",
            file_type="pdf",
            file_size=0,
            file_hash="x",
            status="failed",
            error_message="corrupt file",
        )
        assert doc.status == "failed"
        assert doc.error_message == "corrupt file"

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            DocumentResponse()


# ── DocumentIngestResponse ──────────────────────────────────────

class TestDocumentIngestResponse:
    def test_valid_response(self):
        resp = DocumentIngestResponse(
            doc_id="doc-1",
            filename="f.pdf",
            status="completed",
            chunk_count=10,
            table_count=2,
            duration_ms=1500.0,
        )
        assert resp.doc_id == "doc-1"
        assert resp.status == "completed"
        assert resp.chunk_count == 10
        assert resp.table_count == 2
        assert resp.duration_ms == 1500.0

    def test_defaults(self):
        resp = DocumentIngestResponse(doc_id="d1", filename="f.pdf", status="pending")
        assert resp.chunk_count == 0
        assert resp.table_count == 0
        assert resp.error_message is None
        assert resp.duration_ms == 0.0

    def test_error_status(self):
        resp = DocumentIngestResponse(
            doc_id="d1",
            filename="f.pdf",
            status="failed",
            error_message="parsing error",
        )
        assert resp.status == "failed"
        assert resp.error_message == "parsing error"

    def test_missing_doc_id_raises(self):
        with pytest.raises(ValidationError):
            DocumentIngestResponse(filename="f.pdf", status="pending")

    def test_missing_status_raises(self):
        with pytest.raises(ValidationError):
            DocumentIngestResponse(doc_id="d1", filename="f.pdf")


# ── ConversationDetailResponse ──────────────────────────────────

class TestConversationDetailResponse:
    def test_contains_messages_array(self):
        resp = ConversationDetailResponse(
            id="conv-1",
            title="Test",
            messages=[
                MessageResponse(
                    id="msg-1",
                    conversation_id="conv-1",
                    role="user",
                    content="hello",
                )
            ],
            message_count=1,
        )
        assert resp.id == "conv-1"
        assert len(resp.messages) == 1
        assert resp.messages[0].role == "user"
        assert resp.messages[0].content == "hello"

    def test_empty_messages_default(self):
        resp = ConversationDetailResponse(id="conv-2")
        assert resp.messages == []

    def test_message_count(self):
        resp = ConversationDetailResponse(
            id="conv-3",
            messages=[
                MessageResponse(id="m1", role="user", content="a"),
                MessageResponse(id="m2", role="assistant", content="b"),
            ],
            message_count=2,
        )
        assert resp.message_count == 2

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            ConversationDetailResponse()


# ── MessageResponse ─────────────────────────────────────────────

class TestMessageResponse:
    def test_valid_message(self):
        msg = MessageResponse(
            id="msg-1",
            role="user",
            content="hello world",
        )
        assert msg.id == "msg-1"
        assert msg.role == "user"
        assert msg.content == "hello world"

    def test_message_with_sources(self):
        msg = MessageResponse(
            id="msg-1",
            role="assistant",
            content="answer",
            sources=[{"doc_id": "d1", "filename": "a.pdf"}],
            token_count=50,
            latency_ms=200,
        )
        assert len(msg.sources) == 1
        assert msg.token_count == 50
        assert msg.latency_ms == 200

    def test_nullable_fields_default_none(self):
        msg = MessageResponse(id="m1")
        assert msg.sources is None
        assert msg.token_count is None
        assert msg.latency_ms is None
        assert msg.created_at is None


# ── ConversationResponse ────────────────────────────────────────

class TestConversationResponse:
    def test_valid(self):
        resp = ConversationResponse(id="c1", title="Chat", message_count=3)
        assert resp.id == "c1"
        assert resp.title == "Chat"
        assert resp.message_count == 3
        assert resp.created_at is None

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            ConversationResponse()


# ── IngestionJobResponse ────────────────────────────────────────

class TestIngestionJobResponse:
    def test_valid_response(self):
        job = IngestionJobResponse(
            id="job-1",
            status="completed",
            total_files=5,
            processed_files=5,
            total_chunks=100,
        )
        assert job.id == "job-1"
        assert job.status == "completed"
        assert job.total_files == 5
        assert job.total_chunks == 100

    def test_defaults(self):
        job = IngestionJobResponse(id="j1")
        assert job.collection_id == ""
        assert job.status == "pending"
        assert job.total_files == 0
        assert job.processed_files == 0
        assert job.total_chunks == 0
        assert job.errors is None

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            IngestionJobResponse()


# ── ApiKeyCreateRequest ─────────────────────────────────────────

class TestApiKeyCreateRequest:
    def test_valid(self):
        req = ApiKeyCreateRequest(name="my-key")
        assert req.name == "my-key"
        assert req.permissions == ["read"]

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            ApiKeyCreateRequest()

    def test_custom_permissions(self):
        req = ApiKeyCreateRequest(name="admin-key", permissions=["read", "write", "admin"])
        assert req.permissions == ["read", "write", "admin"]


# ── ApiKeyUpdateRequest ─────────────────────────────────────────

class TestApiKeyUpdateRequest:
    def test_no_fields_set(self):
        req = ApiKeyUpdateRequest()
        assert req.is_active is None
        assert req.name is None
        assert req.permissions is None

    def test_set_active_true(self):
        req = ApiKeyUpdateRequest(is_active=True)
        assert req.is_active is True

    def test_set_active_false(self):
        req = ApiKeyUpdateRequest(is_active=False)
        assert req.is_active is False

    def test_set_name(self):
        req = ApiKeyUpdateRequest(name="renamed")
        assert req.name == "renamed"

    def test_set_permissions(self):
        req = ApiKeyUpdateRequest(permissions=["read", "write"])
        assert req.permissions == ["read", "write"]


# ── ApiKeyResponse ──────────────────────────────────────────────

class TestApiKeyResponse:
    def test_valid(self):
        resp = ApiKeyResponse(id="key-1", name="dev-key", key_prefix="ck_abc")
        assert resp.id == "key-1"
        assert resp.is_active is True
        assert resp.expires_at is None

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            ApiKeyResponse()


# ── ApiKeyCreateResponse ───────────────────────────────────────

class TestApiKeyCreateResponse:
    def test_valid(self):
        resp = ApiKeyCreateResponse(
            id="key-1",
            name="new-key",
            key="ck_secret_raw_key",
        )
        assert resp.id == "key-1"
        assert resp.key == "ck_secret_raw_key"

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            ApiKeyCreateResponse()


# ── ChatChoice ──────────────────────────────────────────────────

class TestChatChoice:
    def test_defaults(self):
        choice = ChatChoice()
        assert choice.index == 0
        assert choice.message.role == "assistant"
        assert choice.message.content == ""
        assert choice.finish_reason == "stop"

    def test_custom(self):
        choice = ChatChoice(
            index=0,
            message=ChatMessageResponse(
                role="assistant",
                content="the answer",
                citations=[ChatCitation(doc_id="d1", filename="a.pdf")],
            ),
            finish_reason="length",
        )
        assert choice.finish_reason == "length"
        assert choice.message.content == "the answer"


# ── ChatMessageResponse ─────────────────────────────────────────

class TestChatMessageResponse:
    def test_defaults(self):
        msg = ChatMessageResponse()
        assert msg.role == "assistant"
        assert msg.content == ""
        assert msg.citations == []

    def test_with_citations(self):
        msg = ChatMessageResponse(
            content="answer",
            citations=[ChatCitation(doc_id="d1", filename="f.pdf")],
        )
        assert len(msg.citations) == 1


# ── UsageInfo ───────────────────────────────────────────────────

class TestUsageInfo:
    def test_defaults(self):
        u = UsageInfo()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_custom(self):
        u = UsageInfo(prompt_tokens=50, completion_tokens=30, total_tokens=80)
        assert u.prompt_tokens == 50
        assert u.completion_tokens == 30
        assert u.total_tokens == 80
