"""Comprehensive data model tests covering all 8 tables, JSON fields, cascades,
constraints, default values, and timestamp auto-generation.

Extends tests/test_storage/test_db_models.py with exhaustive model coverage.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError

from compact_rag.storage.db.models import (
    ApiKey,
    Collection,
    Conversation,
    Document,
    DocumentChunk,
    IngestionJob,
    Message,
    StorageFile,
)


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────


async def _create_collection(session, name="model_test_col"):
    col = Collection(name=name)
    session.add(col)
    await session.commit()
    return col


async def _create_document(session, collection_id, filename="test.pdf", **kwargs):
    defaults = {
        "collection_id": collection_id,
        "filename": filename,
        "file_type": "pdf",
        "file_size": 100,
        "file_hash": f"hash_{filename}",
    }
    defaults.update(kwargs)
    doc = Document(**defaults)
    session.add(doc)
    await session.commit()
    return doc


# ────────────────────────────────────────────────────────────────
# Document model
# ────────────────────────────────────────────────────────────────


class TestDocumentModel:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_all_status_values(self, test_session):
        col = await _create_collection(test_session)

        for status in ["pending", "processing", "completed", "failed"]:
            doc = Document(
                collection_id=col.id,
                filename=f"status_{status}.pdf",
                file_type="pdf",
                file_size=50,
                file_hash=f"status_{status}_hash",
                status=status,
            )
            test_session.add(doc)
        await test_session.commit()

        result = await test_session.execute(select(Document.status).distinct())
        statuses = set(result.scalars().all())
        assert statuses == {"pending", "processing", "completed", "failed"}

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_metadata_json_complex_nested(self, test_session):
        col = await _create_collection(test_session)
        doc = Document(
            collection_id=col.id,
            filename="meta_complex.pdf",
            file_type="pdf",
            file_size=100,
            file_hash="complex_meta_hash",
            metadata_={
                "author": {"name": "Jane Doe", "org": "ACME Corp"},
                "tags": ["rag", "ai", "nlp"],
                "pages": [
                    {"num": 1, "type": "text"},
                    {"num": 2, "type": "table", "table_name": "Revenue Q4"},
                ],
                "custom_fields": {"department": "Engineering", "priority": 3},
            },
        )
        test_session.add(doc)
        await test_session.commit()

        result = await test_session.execute(
            select(Document).where(Document.id == doc.id)
        )
        fetched = result.scalar_one()
        meta = fetched.metadata_
        assert meta["author"]["name"] == "Jane Doe"
        assert meta["pages"][1]["table_name"] == "Revenue Q4"
        assert meta["custom_fields"]["priority"] == 3

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_nullable_fields(self, test_session):
        col = await _create_collection(test_session)
        doc = Document(
            collection_id=col.id,
            filename="nullable_test.pdf",
            file_type="pdf",
            file_size=200,
            file_hash="nullable_hash",
        )
        # These fields should default to None/nulls
        assert doc.page_count is None
        assert doc.error_message is None
        assert doc.metadata_ is None

        test_session.add(doc)
        await test_session.commit()

        result = await test_session.execute(
            select(Document).where(Document.id == doc.id)
        )
        fetched = result.scalar_one()
        assert fetched.page_count is None
        assert fetched.error_message is None
        assert fetched.metadata_ is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_default_values(self, test_session):
        col = await _create_collection(test_session)
        doc = Document(
            collection_id=col.id,
            filename="defaults.pdf",
            file_type="pdf",
            file_size=0,
            file_hash="defaults_hash",
        )
        test_session.add(doc)
        await test_session.commit()

        fetched = await test_session.get(Document, doc.id)
        assert fetched.status == "pending"
        assert fetched.chunk_count == 0
        assert fetched.table_count == 0
        assert fetched.file_size == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_file_type_values(self, test_session):
        """Various file_type strings are accepted."""
        col = await _create_collection(test_session)
        for ft in ["pdf", "docx", "txt", "md", "html", "csv", "json"]:
            doc = Document(
                collection_id=col.id,
                filename=f"type_{ft}.{ft}",
                file_type=ft,
                file_size=10,
                file_hash=f"ft_hash_{ft}",
            )
            test_session.add(doc)
        await test_session.commit()

        result = await test_session.execute(select(func.count(Document.id)))
        assert result.scalar() == 7


# ────────────────────────────────────────────────────────────────
# Collection model
# ────────────────────────────────────────────────────────────────


class TestCollectionModel:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_unique_name_constraint(self, test_session):
        col1 = Collection(name="unique_test")
        test_session.add(col1)
        await test_session.commit()

        col2 = Collection(name="unique_test")
        test_session.add(col2)
        with pytest.raises(IntegrityError):
            await test_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_default_values(self, test_session):
        col = Collection(name="defaults_check")
        test_session.add(col)
        await test_session.commit()

        fetched = await test_session.get(Collection, col.id)
        assert fetched.embedding_model == "BAAI/bge-small-zh-v1.5"
        assert fetched.chunk_size == 500
        assert fetched.chunk_overlap == 50
        assert fetched.document_count == 0
        assert fetched.description is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_timestamps_on_create(self, test_session):
        col = Collection(name="timestamps_col")
        test_session.add(col)
        await test_session.commit()

        fetched = await test_session.get(Collection, col.id)
        assert fetched.created_at is not None
        assert fetched.updated_at is not None
        # On create, created_at and updated_at should be the same (within reason)
        assert abs((fetched.updated_at - fetched.created_at).total_seconds()) < 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_update_changes_updated_at(self, test_session):
        col = Collection(name="update_ts_col")
        test_session.add(col)
        await test_session.commit()

        original_updated = col.updated_at

        # Wait a tiny bit so timestamp changes
        import asyncio
        await asyncio.sleep(0.01)

        col.description = "Updated description"
        await test_session.commit()
        await test_session.refresh(col)

        # onupdate should have updated the timestamp.
        # SQLite returns naive datetimes; normalize both for comparison.
        current = col.updated_at.replace(tzinfo=None) if col.updated_at.tzinfo else col.updated_at
        original = original_updated.replace(tzinfo=None) if original_updated.tzinfo else original_updated
        assert current > original

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_name_index_lookup(self, test_session):
        """name column is indexed; verify fast lookup works."""
        col = Collection(name="indexed_name_xyz")
        test_session.add(col)
        await test_session.commit()

        result = await test_session.execute(
            select(Collection).where(Collection.name == "indexed_name_xyz")
        )
        fetched = result.scalar_one()
        assert fetched is not None
        assert fetched.name == "indexed_name_xyz"


# ────────────────────────────────────────────────────────────────
# Conversation model
# ────────────────────────────────────────────────────────────────


class TestConversationModel:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cascade_delete_messages(self, test_session):
        """Deleting a conversation cascades to delete all its messages."""
        conv = Conversation(title="Cascade Conv", model="gpt-4o-mini")
        test_session.add(conv)
        await test_session.commit()

        msg1 = Message(conversation_id=conv.id, role="user", content="Hello")
        msg2 = Message(conversation_id=conv.id, role="assistant", content="Hi!")
        test_session.add_all([msg1, msg2])
        await test_session.commit()

        # Verify messages exist
        count_before = await test_session.scalar(
            select(func.count()).select_from(Message).where(Message.conversation_id == conv.id)
        )
        assert count_before == 2

        await test_session.delete(conv)
        await test_session.commit()

        # Messages should be cascade-deleted
        count_after = await test_session.scalar(
            select(func.count()).select_from(Message).where(Message.conversation_id == conv.id)
        )
        assert count_after == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_model_field_values(self, test_session):
        """Various model identifiers are accepted."""
        for model_name in ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet", "qwen2.5:7b"]:
            conv = Conversation(title=f"Model: {model_name}", model=model_name)
            test_session.add(conv)
        await test_session.commit()

        result = await test_session.execute(select(Conversation.model).distinct())
        models = set(result.scalars().all())
        assert models == {"gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet", "qwen2.5:7b"}

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_default_values(self, test_session):
        conv = Conversation(title="Default Conv")
        test_session.add(conv)
        await test_session.commit()

        fetched = await test_session.get(Conversation, conv.id)
        assert fetched.model == "gpt-4o-mini"
        assert fetched.message_count == 0
        assert fetched.collection_id is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_collection_id_nullable(self, test_session):
        """Conversation can exist without a collection (collection_id=None)."""
        conv = Conversation(title="No Collection", collection_id=None)
        test_session.add(conv)
        await test_session.commit()

        fetched = await test_session.get(Conversation, conv.id)
        assert fetched.collection_id is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_title_default_when_empty(self, test_session):
        """Test that title defaults to 'New Conversation' when not provided."""
        conv = Conversation()
        test_session.add(conv)
        await test_session.commit()

        fetched = await test_session.get(Conversation, conv.id)
        assert fetched.title == "New Conversation"


# ────────────────────────────────────────────────────────────────
# Message model
# ────────────────────────────────────────────────────────────────


class TestMessageModel:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_all_role_values(self, test_session):
        conv = Conversation(title="Role Test")
        test_session.add(conv)
        await test_session.commit()

        for role in ["system", "user", "assistant", "tool"]:
            msg = Message(conversation_id=conv.id, role=role, content=f"Role: {role}")
            test_session.add(msg)
        await test_session.commit()

        result = await test_session.execute(
            select(Message.role).where(Message.conversation_id == conv.id)
        )
        roles = set(result.scalars().all())
        assert roles == {"system", "user", "assistant", "tool"}

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tool_calls_json_field(self, test_session):
        conv = Conversation(title="Tool Call Test")
        test_session.add(conv)
        await test_session.commit()

        tool_calls_data = {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "arguments": '{"query": "What is RAG?"}',
            },
        }

        msg = Message(
            conversation_id=conv.id,
            role="assistant",
            content="",
            tool_calls=tool_calls_data,
        )
        test_session.add(msg)
        await test_session.commit()

        fetched = await test_session.get(Message, msg.id)
        assert fetched.tool_calls == tool_calls_data
        assert fetched.tool_calls["function"]["name"] == "search_knowledge_base"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sources_json_field(self, test_session):
        conv = Conversation(title="Sources Test")
        test_session.add(conv)
        await test_session.commit()

        sources_data = {
            "documents": [
                {"id": "doc1", "filename": "report.pdf", "score": 0.95},
                {"id": "doc2", "filename": "guide.pdf", "score": 0.82},
            ],
            "chunks": [
                {"id": "chunk1", "content": "...", "score": 0.95},
            ],
        }

        msg = Message(
            conversation_id=conv.id,
            role="assistant",
            content="Based on the documents...",
            sources=sources_data,
        )
        test_session.add(msg)
        await test_session.commit()

        fetched = await test_session.get(Message, msg.id)
        assert fetched.sources == sources_data
        assert len(fetched.sources["documents"]) == 2

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_nullable_fields(self, test_session):
        conv = Conversation(title="Nullable Msg")
        test_session.add(conv)
        await test_session.commit()

        msg = Message(
            conversation_id=conv.id,
            role="user",
            content="Simple message",
        )
        test_session.add(msg)
        await test_session.commit()

        fetched = await test_session.get(Message, msg.id)
        assert fetched.tool_calls is None
        assert fetched.sources is None
        assert fetched.token_count is None
        assert fetched.latency_ms is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_numeric_fields(self, test_session):
        conv = Conversation(title="Numeric Msg")
        test_session.add(conv)
        await test_session.commit()

        msg = Message(
            conversation_id=conv.id,
            role="assistant",
            content="Measured response",
            token_count=250,
            latency_ms=1500,
        )
        test_session.add(msg)
        await test_session.commit()

        fetched = await test_session.get(Message, msg.id)
        assert fetched.token_count == 250
        assert fetched.latency_ms == 1500


# ────────────────────────────────────────────────────────────────
# ApiKey model
# ────────────────────────────────────────────────────────────────


class TestApiKeyModel:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_key_hash_unique_constraint(self, test_session):
        key1 = ApiKey(name="key1", key_hash="same_hash_collision", permissions=["read"])
        test_session.add(key1)
        await test_session.commit()

        key2 = ApiKey(name="key2", key_hash="same_hash_collision", permissions=["read"])
        test_session.add(key2)
        with pytest.raises(IntegrityError):
            await test_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_permissions_json(self, test_session):
        key = ApiKey(
            name="admin-key",
            key_hash="perm_hash_001",
            permissions=["read", "write", "admin", "ingestion"],
        )
        test_session.add(key)
        await test_session.commit()

        fetched = await test_session.get(ApiKey, key.id)
        assert fetched.permissions == ["read", "write", "admin", "ingestion"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_permissions_empty_list(self, test_session):
        key = ApiKey(
            name="no-perms-key",
            key_hash="empty_perm_hash",
            permissions=[],
        )
        test_session.add(key)
        await test_session.commit()

        fetched = await test_session.get(ApiKey, key.id)
        assert fetched.permissions == []

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_is_active_default_true(self, test_session):
        key = ApiKey(name="active-default", key_hash="active_hash")
        test_session.add(key)
        await test_session.commit()

        fetched = await test_session.get(ApiKey, key.id)
        assert fetched.is_active is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_expires_at_nullable(self, test_session):
        """expires_at is nullable and defaults to None."""
        key = ApiKey(name="no-expiry", key_hash="noexp_hash")
        test_session.add(key)
        await test_session.commit()

        fetched = await test_session.get(ApiKey, key.id)
        assert fetched.expires_at is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_expires_at_with_value(self, test_session):
        expiry = datetime.now(timezone.utc) + timedelta(days=90)
        key = ApiKey(
            name="expiring-key",
            key_hash="exp_hash",
            expires_at=expiry,
        )
        test_session.add(key)
        await test_session.commit()

        fetched = await test_session.get(ApiKey, key.id)
        assert fetched.expires_at is not None
        # Allow small delta
        assert abs((fetched.expires_at - expiry).total_seconds()) < 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_deactivated_key(self, test_session):
        key = ApiKey(name="deactivated", key_hash="deact_hash", is_active=False)
        test_session.add(key)
        await test_session.commit()

        fetched = await test_session.get(ApiKey, key.id)
        assert fetched.is_active is False


# ────────────────────────────────────────────────────────────────
# IngestionJob model
# ────────────────────────────────────────────────────────────────


class TestIngestionJobModel:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_status_transitions(self, test_session):
        col = await _create_collection(test_session)
        job = IngestionJob(collection_id=col.id, status="pending")
        test_session.add(job)
        await test_session.commit()

        # pending → running
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        await test_session.commit()
        assert job.status == "running"

        # running → completed
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        await test_session.commit()
        assert job.status == "completed"
        assert job.completed_at is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_errors_json_with_list_of_dicts(self, test_session):
        col = await _create_collection(test_session)
        errors_data = {
            "files": [
                {"filename": "bad.pdf", "error": "Corrupted header", "line": 42},
                {"filename": "broken.docx", "error": "Unsupported encryption"},
            ],
            "summary": {
                "total_errors": 2,
                "retryable": True,
            },
        }
        job = IngestionJob(
            collection_id=col.id,
            status="failed",
            total_files=5,
            processed_files=3,
            errors=errors_data,
        )
        test_session.add(job)
        await test_session.commit()

        fetched = await test_session.get(IngestionJob, job.id)
        assert fetched.errors == errors_data
        assert fetched.errors["files"][0]["filename"] == "bad.pdf"
        assert fetched.errors["summary"]["retryable"] is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_errors_is_nullable(self, test_session):
        col = await _create_collection(test_session)
        job = IngestionJob(collection_id=col.id, status="pending", total_files=1)
        test_session.add(job)
        await test_session.commit()

        fetched = await test_session.get(IngestionJob, job.id)
        assert fetched.errors is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_timestamps(self, test_session):
        col = await _create_collection(test_session)
        job = IngestionJob(collection_id=col.id)
        test_session.add(job)
        await test_session.commit()

        fetched = await test_session.get(IngestionJob, job.id)
        assert fetched.created_at is not None
        assert fetched.started_at is None
        assert fetched.completed_at is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_default_values(self, test_session):
        col = await _create_collection(test_session)
        job = IngestionJob(collection_id=col.id)
        test_session.add(job)
        await test_session.commit()

        fetched = await test_session.get(IngestionJob, job.id)
        assert fetched.status == "pending"
        assert fetched.total_files == 0
        assert fetched.processed_files == 0
        assert fetched.total_chunks == 0


# ────────────────────────────────────────────────────────────────
# StorageFile model
# ────────────────────────────────────────────────────────────────


class TestStorageFileModel:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_storage_type_values(self, test_session):
        """All three storage_type values: temp, persistent, archive."""
        for stype in ["temp", "persistent", "archive"]:
            sfile = StorageFile(
                storage_backend="local",
                storage_key=f"{stype}/key/data.bin",
                filename=f"{stype}_file.bin",
                file_size=100,
                storage_type=stype,
            )
            test_session.add(sfile)
        await test_session.commit()

        result = await test_session.execute(
            select(StorageFile.storage_type).distinct()
        )
        types = set(result.scalars().all())
        assert types == {"temp", "persistent", "archive"}

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_document_id_nullable_fk(self, test_session):
        """StorageFile can exist without being linked to a document."""
        sfile = StorageFile(
            storage_backend="local",
            storage_key="orphan/file.pdf",
            filename="orphan.pdf",
            file_size=500,
        )
        test_session.add(sfile)
        await test_session.commit()

        fetched = await test_session.get(StorageFile, sfile.id)
        assert fetched.document_id is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_with_document_relationship(self, test_session):
        col = await _create_collection(test_session)
        doc = await _create_document(test_session, col.id)

        sfile = StorageFile(
            document_id=doc.id,
            storage_backend="minio",
            storage_key="buckets/docs/report.pdf",
            filename="report.pdf",
            file_size=2048,
            content_type="application/pdf",
        )
        test_session.add(sfile)
        await test_session.commit()

        fetched = await test_session.get(StorageFile, sfile.id)
        assert fetched.document_id == doc.id
        assert fetched.storage_backend == "minio"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_content_type_nullable(self, test_session):
        sfile = StorageFile(
            storage_backend="local",
            storage_key="no-type/file.bin",
            filename="file.bin",
            file_size=256,
        )
        test_session.add(sfile)
        await test_session.commit()

        fetched = await test_session.get(StorageFile, sfile.id)
        assert fetched.content_type is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_expires_at_nullable(self, test_session):
        sfile = StorageFile(
            storage_backend="local",
            storage_key="no-expiry/file.pdf",
            filename="file.pdf",
            file_size=100,
        )
        test_session.add(sfile)
        await test_session.commit()

        fetched = await test_session.get(StorageFile, sfile.id)
        assert fetched.expires_at is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_default_storage_type(self, test_session):
        sfile = StorageFile(
            storage_backend="local",
            storage_key="default-type/file.pdf",
            filename="file.pdf",
            file_size=100,
        )
        test_session.add(sfile)
        await test_session.commit()

        fetched = await test_session.get(StorageFile, sfile.id)
        assert fetched.storage_type == "persistent"


# ────────────────────────────────────────────────────────────────
# Timestamp auto-generation on all models
# ────────────────────────────────────────────────────────────────


class TestTimestampAutoGeneration:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_collection_created_at_not_null(self, test_session):
        col = Collection(name="ts_collection")
        test_session.add(col)
        await test_session.commit()
        fetched = await test_session.get(Collection, col.id)
        assert fetched.created_at is not None
        assert isinstance(fetched.created_at, datetime)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_document_created_at_not_null(self, test_session):
        col = await _create_collection(test_session)
        doc = await _create_document(test_session, col.id)
        fetched = await test_session.get(Document, doc.id)
        assert fetched.created_at is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_chunk_created_at_not_null(self, test_session):
        col = await _create_collection(test_session)
        doc = await _create_document(test_session, col.id)
        chunk = DocumentChunk(
            document_id=doc.id, chroma_id="chr-ts", chunk_index=0, page_number=1,
        )
        test_session.add(chunk)
        await test_session.commit()
        fetched = await test_session.get(DocumentChunk, chunk.id)
        assert fetched.created_at is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_conversation_created_at_not_null(self, test_session):
        conv = Conversation(title="TS Conv")
        test_session.add(conv)
        await test_session.commit()
        fetched = await test_session.get(Conversation, conv.id)
        assert fetched.created_at is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_message_created_at_not_null(self, test_session):
        conv = Conversation(title="TS Msg")
        test_session.add(conv)
        await test_session.commit()
        msg = Message(conversation_id=conv.id, role="user", content="TS test")
        test_session.add(msg)
        await test_session.commit()
        fetched = await test_session.get(Message, msg.id)
        assert fetched.created_at is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_ingestion_job_created_at_not_null(self, test_session):
        col = await _create_collection(test_session)
        job = IngestionJob(collection_id=col.id)
        test_session.add(job)
        await test_session.commit()
        fetched = await test_session.get(IngestionJob, job.id)
        assert fetched.created_at is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_api_key_created_at_not_null(self, test_session):
        key = ApiKey(name="TS Key", key_hash="ts_hash")
        test_session.add(key)
        await test_session.commit()
        fetched = await test_session.get(ApiKey, key.id)
        assert fetched.created_at is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_storage_file_created_at_not_null(self, test_session):
        sfile = StorageFile(
            storage_backend="local", storage_key="ts/file.pdf",
            filename="file.pdf", file_size=100,
        )
        test_session.add(sfile)
        await test_session.commit()
        fetched = await test_session.get(StorageFile, sfile.id)
        assert fetched.created_at is not None


# ────────────────────────────────────────────────────────────────
# Cascade delete tests
# ────────────────────────────────────────────────────────────────


class TestCascades:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cascade_delete_collection_to_documents(self, test_session):
        """Deleting a collection cascades to delete its documents."""
        col = await _create_collection(test_session, name="cascade_col_docs")
        doc1 = await _create_document(test_session, col.id, filename="c1.pdf", file_hash="c1_hash")
        doc2 = await _create_document(test_session, col.id, filename="c2.pdf", file_hash="c2_hash")

        await test_session.delete(col)
        await test_session.commit()

        # Documents should be gone
        d1 = await test_session.get(Document, doc1.id)
        d2 = await test_session.get(Document, doc2.id)
        assert d1 is None
        assert d2 is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cascade_delete_collection_to_document_to_chunks(self, test_session):
        """Delete collection → documents cascade → chunks cascade."""
        col = await _create_collection(test_session, name="cascade_full")
        doc = await _create_document(test_session, col.id, filename="deep.pdf", file_hash="deep_hash")

        for i in range(3):
            chunk = DocumentChunk(
                document_id=doc.id, chroma_id=f"cascade_chr_{i}",
                chunk_index=i, page_number=1,
            )
            test_session.add(chunk)
        await test_session.commit()

        # Count before
        chunk_count = await test_session.scalar(
            select(func.count()).select_from(DocumentChunk).where(
                DocumentChunk.document_id == doc.id
            )
        )
        assert chunk_count == 3

        await test_session.delete(col)
        await test_session.commit()

        # Document gone
        assert await test_session.get(Document, doc.id) is None
        # Chunks gone
        chunk_count_after = await test_session.scalar(
            select(func.count()).select_from(DocumentChunk).where(
                DocumentChunk.document_id == doc.id
            )
        )
        assert chunk_count_after == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cascade_delete_conversation_to_messages(self, test_session):
        """Deleting a conversation cascades to delete all messages."""
        conv = Conversation(title="Cascade Conv Msg")
        test_session.add(conv)
        await test_session.commit()

        msg_ids = []
        for i in range(5):
            msg = Message(conversation_id=conv.id, role="user", content=f"Msg {i}")
            test_session.add(msg)
            await test_session.commit()
            msg_ids.append(msg.id)

        await test_session.delete(conv)
        await test_session.commit()

        # All messages should be gone
        for mid in msg_ids:
            assert await test_session.get(Message, mid) is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cascade_delete_document_to_chunks(self, test_session):
        """Direct document delete cascades to chunks."""
        col = await _create_collection(test_session)
        doc = await _create_document(test_session, col.id, filename="direct_cascade.pdf", file_hash="dc_hash")

        chunk_ids = []
        for i in range(3):
            chunk = DocumentChunk(
                document_id=doc.id, chroma_id=f"dc_chr_{i}",
                chunk_index=i, page_number=1,
            )
            test_session.add(chunk)
            await test_session.commit()
            chunk_ids.append(chunk.id)

        # Delete just the document
        await test_session.delete(doc)
        await test_session.commit()

        # Chunks should be cascade-deleted
        for cid in chunk_ids:
            assert await test_session.get(DocumentChunk, cid) is None
