"""Comprehensive repository-layer tests covering all 7 repositories and base CRUD.

Extends tests/test_storage/test_repositories.py with exhaustive coverage of
generic BaseRepository methods and all specialized repository methods.
"""

from __future__ import annotations

import pytest

from compact_rag.storage.db.models import (
    Collection,
    Document,
    DocumentChunk,
    Message,
)
from compact_rag.storage.db.repository.api_key import ApiKeyRepository
from compact_rag.storage.db.repository.chunk import ChunkRepository
from compact_rag.storage.db.repository.collection import CollectionRepository
from compact_rag.storage.db.repository.conversation import (
    ConversationRepository,
    MessageRepository,
)
from compact_rag.storage.db.repository.document import DocumentRepository
from compact_rag.storage.db.repository.ingestion import IngestionJobRepository
from compact_rag.storage.db.repository.storage_file import StorageFileRepository


# ────────────────────────────────────────────────────────────────
# 1. BaseRepository generic methods (tested via DocumentRepository)
# ────────────────────────────────────────────────────────────────


class TestBaseRepository:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_update_nonexistent_returns_none(self, test_session):
        repo = DocumentRepository()
        result = await repo.update(test_session, "nonexistent-id-12345", status="completed")
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_delete_nonexistent_returns_false(self, test_session):
        repo = DocumentRepository()
        result = await repo.delete(test_session, "nonexistent-id-99999")
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_empty_table(self, test_session):
        """list() on a table with no rows returns empty list and zero total."""
        repo = DocumentRepository()
        items, total = await repo.list(test_session, page=1, page_size=20)
        assert items == []
        assert total == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_with_filters(self, test_session):
        """list() filters by a single column."""
        col = Collection(name="list_filter_col")
        test_session.add(col)
        await test_session.commit()

        repo = DocumentRepository()
        await repo.create(
            test_session,
            collection_id=col.id,
            filename="match.pdf",
            file_type="pdf",
            file_size=10,
            file_hash="aaaa",
        )
        await repo.create(
            test_session,
            collection_id=col.id,
            filename="nomatch.pdf",
            file_type="pdf",
            file_size=20,
            file_hash="bbbb",
        )

        # Filter by file_type
        items, total = await repo.list(test_session, file_type="pdf", page_size=50)
        assert len(items) == 2
        assert total == 2

        # Filter that matches nothing
        items, total = await repo.list(test_session, file_type="docx")
        assert items == []
        assert total == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_with_pagination(self, test_session):
        """list() paginates correctly with page and page_size."""
        col = Collection(name="paginate_col")
        test_session.add(col)
        await test_session.commit()

        repo = DocumentRepository()
        for i in range(7):
            await repo.create(
                test_session,
                collection_id=col.id,
                filename=f"doc_{i}.pdf",
                file_type="pdf",
                file_size=100 + i,
                file_hash=f"hash_{i}",
            )

        # Page 1 with 3 items
        page1, total = await repo.list(test_session, page=1, page_size=3)
        assert len(page1) == 3
        assert total == 7

        # Page 2 with 3 items
        page2, total = await repo.list(test_session, page=2, page_size=3)
        assert len(page2) == 3
        assert total == 7

        # Page 3 with 3 items (only 1 left)
        page3, total = await repo.list(test_session, page=3, page_size=3)
        assert len(page3) == 1
        assert total == 7

        # Page beyond range
        page4, total = await repo.list(test_session, page=10, page_size=3)
        assert len(page4) == 0
        assert total == 7

        # Ensure pages are disjoint (no overlap in IDs)
        all_ids = set()
        for item in page1 + page2 + page3:
            assert item.id not in all_ids
            all_ids.add(item.id)
        assert len(all_ids) == 7

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_returns_instance_with_auto_fields(self, test_session):
        """create() returns a model instance with auto-generated id and created_at."""
        col = Collection(name="auto_field_col")
        test_session.add(col)
        await test_session.commit()

        repo = DocumentRepository()
        doc = await repo.create(
            test_session,
            collection_id=col.id,
            filename="autofields.pdf",
            file_type="pdf",
            file_size=42,
            file_hash="autohash42",
        )

        assert doc.id is not None
        assert len(doc.id) == 36  # UUID4
        assert doc.created_at is not None
        assert doc.status == "pending"  # default
        assert doc.chunk_count == 0
        assert doc.table_count == 0
        assert doc.filename == "autofields.pdf"


# ────────────────────────────────────────────────────────────────
# 2. DocumentRepository
# ────────────────────────────────────────────────────────────────


class TestDocumentRepositoryExtended:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_by_collection_valid(self, test_session):
        col = Collection(name="listcol_val")
        test_session.add(col)
        await test_session.commit()

        repo = DocumentRepository()
        await repo.create(
            test_session,
            collection_id=col.id,
            filename="a.pdf",
            file_type="pdf",
            file_size=10,
            file_hash="hash_a",
        )
        await repo.create(
            test_session,
            collection_id=col.id,
            filename="b.pdf",
            file_type="pdf",
            file_size=20,
            file_hash="hash_b",
        )

        items, total = await repo.list_by_collection(test_session, col.id, page=1, page_size=50)
        assert len(items) == 2
        assert total == 2

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_by_collection_empty(self, test_session):
        col = Collection(name="listcol_empty")
        test_session.add(col)
        await test_session.commit()

        repo = DocumentRepository()
        items, total = await repo.list_by_collection(test_session, col.id)
        assert items == []
        assert total == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_update_status_existing(self, test_session):
        col = Collection(name="updstatus_col")
        test_session.add(col)
        await test_session.commit()

        repo = DocumentRepository()
        doc = await repo.create(
            test_session,
            collection_id=col.id,
            filename="status.pdf",
            file_type="pdf",
            file_size=100,
            file_hash="statushash",
        )
        assert doc.status == "pending"

        updated = await repo.update_status(test_session, doc.id, "completed")
        assert updated is not None
        assert updated.status == "completed"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_update_status_with_error_message(self, test_session):
        col = Collection(name="errmsg_col")
        test_session.add(col)
        await test_session.commit()

        repo = DocumentRepository()
        doc = await repo.create(
            test_session,
            collection_id=col.id,
            filename="err.pdf",
            file_type="pdf",
            file_size=100,
            file_hash="errhash",
        )

        updated = await repo.update_status(test_session, doc.id, "failed", error_message="Bad file")
        assert updated is not None
        assert updated.status == "failed"
        assert updated.error_message == "Bad file"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_update_status_nonexistent(self, test_session):
        repo = DocumentRepository()
        result = await repo.update_status(test_session, "no-such-doc", "completed")
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_with_filters_by_collection(self, test_session):
        col1 = Collection(name="filt_col1")
        col2 = Collection(name="filt_col2")
        test_session.add_all([col1, col2])
        await test_session.commit()

        repo = DocumentRepository()
        await repo.create(
            test_session,
            collection_id=col1.id,
            filename="col1.pdf",
            file_type="pdf",
            file_size=10,
            file_hash="chash1",
        )
        await repo.create(
            test_session,
            collection_id=col2.id,
            filename="col2.pdf",
            file_type="pdf",
            file_size=20,
            file_hash="chash2",
        )

        items1, total1 = await repo.list_with_filters(test_session, collection_id=col1.id)
        assert len(items1) == 1
        assert total1 == 1
        assert items1[0].filename == "col1.pdf"

        items2, total2 = await repo.list_with_filters(test_session, collection_id=col2.id)
        assert len(items2) == 1
        assert total2 == 1
        assert items2[0].filename == "col2.pdf"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_with_filters_by_status(self, test_session):
        col = Collection(name="filt_status")
        test_session.add(col)
        await test_session.commit()

        repo = DocumentRepository()
        await repo.create(
            test_session, collection_id=col.id, filename="s1.pdf",
            file_type="pdf", file_size=10, file_hash="sh1",
        )
        # Default status is "pending", so both start as pending
        doc2 = await repo.create(
            test_session, collection_id=col.id, filename="s2.pdf",
            file_type="pdf", file_size=10, file_hash="sh2",
        )
        await repo.update_status(test_session, doc2.id, "completed")

        # Filter pending
        pending, pt = await repo.list_with_filters(test_session, status="pending", page_size=50)
        assert len(pending) == 1
        assert pending[0].status == "pending"

        # Filter completed
        completed, ct = await repo.list_with_filters(test_session, status="completed", page_size=50)
        assert len(completed) == 1
        assert completed[0].status == "completed"

        # Filter unknown status
        items, tt = await repo.list_with_filters(test_session, status="nonexistent")
        assert items == []
        assert tt == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_with_filters_combined(self, test_session):
        """Combined collection + status filters."""
        col1 = Collection(name="comb_col1")
        col2 = Collection(name="comb_col2")
        test_session.add_all([col1, col2])
        await test_session.commit()

        repo = DocumentRepository()
        doc_a = await repo.create(
            test_session, collection_id=col1.id, filename="ca.pdf",
            file_type="pdf", file_size=10, file_hash="cha1",
        )
        await repo.create(
            test_session, collection_id=col2.id, filename="cb.pdf",
            file_type="pdf", file_size=10, file_hash="chb1",
        )
        await repo.update_status(test_session, doc_a.id, "completed")
        # doc_b stays pending

        # Both filters match
        items, total = await repo.list_with_filters(
            test_session, collection_id=col1.id, status="completed", page_size=50,
        )
        assert len(items) == 1
        assert total == 1
        assert items[0].collection_id == col1.id
        assert items[0].status == "completed"

        # Filters that conflict produce empty result
        items, total = await repo.list_with_filters(
            test_session, collection_id=col2.id, status="completed", page_size=50,
        )
        assert items == []
        assert total == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_with_filters_empty_results(self, test_session):
        col = Collection(name="nofilt_col")
        test_session.add(col)
        await test_session.commit()

        repo = DocumentRepository()
        items, total = await repo.list_with_filters(
            test_session, collection_id=col.id, page_size=50,
        )
        assert items == []
        assert total == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_by_hash_not_found(self, test_session):
        """Already tested in test_repositories.py; verify consistency."""
        repo = DocumentRepository()
        doc = await repo.get_by_hash(test_session, "nonexistent_hash_xyz")
        assert doc is None


# ────────────────────────────────────────────────────────────────
# 3. CollectionRepository
# ────────────────────────────────────────────────────────────────


class TestCollectionRepositoryExtended:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_increment_document_count_negative_delta(self, test_session):
        repo = CollectionRepository()
        col = await repo.create(test_session, name="neg_delta_col")
        assert col.document_count == 0

        await repo.increment_document_count(test_session, col.id, delta=10)
        await test_session.refresh(col)
        assert col.document_count == 10

        await repo.increment_document_count(test_session, col.id, delta=-3)
        await test_session.refresh(col)
        assert col.document_count == 7

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_increment_document_count_large_delta(self, test_session):
        repo = CollectionRepository()
        col = await repo.create(test_session, name="large_delta_col")
        assert col.document_count == 0

        await repo.increment_document_count(test_session, col.id, delta=9999)
        await test_session.refresh(col)
        assert col.document_count == 9999

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_increment_document_count_nonexistent_collection(self, test_session):
        """Increment on non-existent collection is a no-op (no error)."""
        repo = CollectionRepository()
        # Should not raise
        await repo.increment_document_count(test_session, "fake-collection-id", delta=5)


# ────────────────────────────────────────────────────────────────
# 4. ChunkRepository
# ────────────────────────────────────────────────────────────────


class TestChunkRepositoryExtended:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_by_chroma_id_found(self, test_session):
        col = Collection(name="chroma_col")
        test_session.add(col)
        await test_session.commit()

        doc = Document(
            collection_id=col.id, filename="chr.pdf",
            file_type="pdf", file_size=10, file_hash="chrhash",
        )
        test_session.add(doc)
        await test_session.commit()

        chunk = DocumentChunk(
            document_id=doc.id, chroma_id="chroma-unique-id-001",
            chunk_index=0, page_number=1,
        )
        test_session.add(chunk)
        await test_session.commit()

        repo = ChunkRepository()
        found = await repo.get_by_chroma_id(test_session, "chroma-unique-id-001")
        assert found is not None
        assert found.chroma_id == "chroma-unique-id-001"
        assert found.document_id == doc.id

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_by_chroma_id_not_found(self, test_session):
        repo = ChunkRepository()
        result = await repo.get_by_chroma_id(test_session, "no-such-chroma-id")
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_delete_by_document_deletes_all_chunks(self, test_session):
        col = Collection(name="delchunks_col")
        test_session.add(col)
        await test_session.commit()

        doc = Document(
            collection_id=col.id, filename="deldoc.pdf",
            file_type="pdf", file_size=10, file_hash="delhash",
        )
        test_session.add(doc)
        await test_session.commit()

        for i in range(5):
            chunk = DocumentChunk(
                document_id=doc.id, chroma_id=f"chr-del-{i}",
                chunk_index=i, page_number=1,
            )
            test_session.add(chunk)
        await test_session.commit()

        repo = ChunkRepository()
        deleted_count = await repo.delete_by_document(test_session, doc.id)
        assert deleted_count == 5

        # Verify chunks are gone
        remaining = await repo.list_by_document(test_session, doc.id)
        assert remaining == []

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_delete_by_document_empty_document(self, test_session):
        """delete_by_document on a document with no chunks returns 0."""
        col = Collection(name="nochunks_col")
        test_session.add(col)
        await test_session.commit()

        doc = Document(
            collection_id=col.id, filename="nochunks.pdf",
            file_type="pdf", file_size=10, file_hash="nochunkhash",
        )
        test_session.add(doc)
        await test_session.commit()

        repo = ChunkRepository()
        deleted_count = await repo.delete_by_document(test_session, doc.id)
        assert deleted_count == 0


# ────────────────────────────────────────────────────────────────
# 5. ConversationRepository
# ────────────────────────────────────────────────────────────────


class TestConversationRepositoryExtended:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_messages_returns_ordered(self, test_session):
        repo = ConversationRepository()
        conv = await repo.create(test_session, title="Msg Order Test", model="gpt-4o-mini")

        msg1 = Message(conversation_id=conv.id, role="system", content="You are helpful.")
        msg2 = Message(conversation_id=conv.id, role="user", content="Hello")
        msg3 = Message(conversation_id=conv.id, role="assistant", content="Hi there!")
        test_session.add_all([msg1, msg2, msg3])
        await test_session.commit()

        messages = await repo.list_messages(test_session, conv.id)
        assert len(messages) == 3
        # Should be ordered by created_at ascending
        timestamps = [m.created_at for m in messages]
        assert timestamps == sorted(timestamps)
        assert [m.role for m in messages] == ["system", "user", "assistant"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_messages_empty_conversation(self, test_session):
        repo = ConversationRepository()
        conv = await repo.create(test_session, title="Empty Conv")
        messages = await repo.list_messages(test_session, conv.id)
        assert messages == []

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_increment_message_count(self, test_session):
        repo = ConversationRepository()
        conv = await repo.create(test_session, title="Incr Msg")
        assert conv.message_count == 0

        await repo.increment_message_count(test_session, conv.id, delta=1)
        await test_session.refresh(conv)
        assert conv.message_count == 1

        await repo.increment_message_count(test_session, conv.id, delta=4)
        await test_session.refresh(conv)
        assert conv.message_count == 5

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_increment_message_count_nonexistent(self, test_session):
        """Increment on non-existent conversation is a no-op."""
        repo = ConversationRepository()
        await repo.increment_message_count(test_session, "fake-conv-id", delta=10)
        # No error expected


# ────────────────────────────────────────────────────────────────
# 6. MessageRepository
# ────────────────────────────────────────────────────────────────


class TestMessageRepositoryExtended:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_by_conversation_ordered_by_created_at(self, test_session):
        conv_repo = ConversationRepository()
        conv = await conv_repo.create(test_session, title="Msg List Conv")

        msg_repo = MessageRepository()
        msg1 = await msg_repo.create(
            test_session, conversation_id=conv.id, role="user", content="First message",
        )
        msg2 = await msg_repo.create(
            test_session, conversation_id=conv.id, role="assistant", content="Second message",
        )
        msg3 = await msg_repo.create(
            test_session, conversation_id=conv.id, role="user", content="Third message",
        )

        messages = await msg_repo.list_by_conversation(test_session, conv.id)
        assert len(messages) == 3
        assert messages[0].id == msg1.id
        assert messages[1].id == msg2.id
        assert messages[2].id == msg3.id
        # Verify chronological order
        for i in range(len(messages) - 1):
            assert messages[i].created_at <= messages[i + 1].created_at

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_by_conversation_empty(self, test_session):
        msg_repo = MessageRepository()
        messages = await msg_repo.list_by_conversation(test_session, "no-conversation")
        assert messages == []

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_with_all_fields(self, test_session):
        conv_repo = ConversationRepository()
        conv = await conv_repo.create(test_session, title="Full Msg")

        msg_repo = MessageRepository()
        msg = await msg_repo.create(
            test_session,
            conversation_id=conv.id,
            role="assistant",
            content="Full content",
            tool_calls={"name": "search", "args": {"query": "test"}},
            sources={"documents": ["doc1", "doc2"]},
            token_count=150,
            latency_ms=320,
        )
        assert msg.role == "assistant"
        assert msg.content == "Full content"
        assert msg.tool_calls == {"name": "search", "args": {"query": "test"}}
        assert msg.sources == {"documents": ["doc1", "doc2"]}
        assert msg.token_count == 150
        assert msg.latency_ms == 320


# ────────────────────────────────────────────────────────────────
# 7. ApiKeyRepository (NEW)
# ────────────────────────────────────────────────────────────────


class TestApiKeyRepository:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_key_with_all_fields(self, test_session):
        repo = ApiKeyRepository()
        key = await repo.create(
            test_session,
            name="service-account-1",
            key_hash="sha256_hash_abc123",
            permissions=["read", "write", "ingestion"],
        )
        assert key.id is not None
        assert key.name == "service-account-1"
        assert key.key_hash == "sha256_hash_abc123"
        assert key.permissions == ["read", "write", "ingestion"]
        assert key.is_active is True
        assert key.created_at is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_by_hash_found(self, test_session):
        repo = ApiKeyRepository()
        await repo.create(
            test_session,
            name="my-api-key",
            key_hash="hash_found_001",
            permissions=["read"],
        )

        found = await repo.get_by_hash(test_session, "hash_found_001")
        assert found is not None
        assert found.name == "my-api-key"
        assert found.permissions == ["read"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_by_hash_not_found(self, test_session):
        repo = ApiKeyRepository()
        result = await repo.get_by_hash(test_session, "hash_not_in_db")
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_by_hash_excludes_inactive(self, test_session):
        """get_by_hash() ONLY returns active keys (is_active=True filter)."""
        repo = ApiKeyRepository()
        key = await repo.create(
            test_session,
            name="inactive-key",
            key_hash="hash_inactive_001",
            permissions=["read"],
        )

        # Found when active
        found = await repo.get_by_hash(test_session, "hash_inactive_001")
        assert found is not None

        # Deactivate
        await repo.set_active(test_session, key.id, False)

        # Should NOT be found anymore (get_by_hash filters is_active=True)
        not_found = await repo.get_by_hash(test_session, "hash_inactive_001")
        assert not_found is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_set_active_toggles(self, test_session):
        repo = ApiKeyRepository()
        key = await repo.create(
            test_session,
            name="toggle-key",
            key_hash="hash_toggle_001",
            permissions=["read"],
        )
        assert key.is_active is True

        # Deactivate
        updated = await repo.set_active(test_session, key.id, False)
        assert updated is not None
        assert updated.is_active is False

        # Reactivate
        updated = await repo.set_active(test_session, key.id, True)
        assert updated is not None
        assert updated.is_active is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_set_active_nonexistent(self, test_session):
        repo = ApiKeyRepository()
        result = await repo.set_active(test_session, "fake-key-id", False)
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_key_with_minimal_fields(self, test_session):
        """Default permissions are ['read']."""
        repo = ApiKeyRepository()
        key = await repo.create(
            test_session,
            name="minimal-key",
            key_hash="hash_minimal_001",
        )
        assert key.permissions == ["read"]
        assert key.is_active is True


# ────────────────────────────────────────────────────────────────
# 8. IngestionJobRepository (NEW)
# ────────────────────────────────────────────────────────────────


class TestIngestionJobRepository:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_job_creates_with_running_status(self, test_session):
        col = Collection(name="ingest_job_col")
        test_session.add(col)
        await test_session.commit()

        repo = IngestionJobRepository()
        job = await repo.create_job(test_session, collection_id=col.id, total_files=3)

        assert job.id is not None
        assert job.collection_id == col.id
        assert job.status == "running"
        assert job.total_files == 3
        assert job.processed_files == 0
        assert job.total_chunks == 0
        assert job.started_at is not None
        assert job.completed_at is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_update_progress(self, test_session):
        col = Collection(name="prog_job_col")
        test_session.add(col)
        await test_session.commit()

        repo = IngestionJobRepository()
        job = await repo.create_job(test_session, collection_id=col.id, total_files=10)

        # Initial progress
        assert job.processed_files == 0
        assert job.total_chunks == 0

        updated = await repo.update_progress(test_session, job.id, processed=5, chunks=42)
        assert updated is not None
        assert updated.processed_files == 5
        assert updated.total_chunks == 42

        # Second update
        updated = await repo.update_progress(test_session, job.id, processed=10, chunks=100)
        assert updated is not None
        assert updated.processed_files == 10
        assert updated.total_chunks == 100

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_update_progress_nonexistent_job(self, test_session):
        repo = IngestionJobRepository()
        result = await repo.update_progress(test_session, "fake-job", processed=1, chunks=5)
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_job(self, test_session):
        col = Collection(name="complete_job_col")
        test_session.add(col)
        await test_session.commit()

        repo = IngestionJobRepository()
        job = await repo.create_job(test_session, collection_id=col.id, total_files=5)
        assert job.completed_at is None
        assert job.status == "running"

        completed = await repo.complete_job(test_session, job.id)
        assert completed is not None
        assert completed.status == "completed"
        assert completed.completed_at is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_job_with_errors(self, test_session):
        col = Collection(name="complete_err_col")
        test_session.add(col)
        await test_session.commit()

        repo = IngestionJobRepository()
        job = await repo.create_job(test_session, collection_id=col.id, total_files=5)

        error_data = {
            "file1.pdf": "Corrupted header",
            "file2.docx": "Unsupported encryption",
        }
        completed = await repo.complete_job(
            test_session, job.id, status="failed", errors=error_data,
        )
        assert completed is not None
        assert completed.status == "failed"
        assert completed.errors == error_data
        assert completed.completed_at is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_job_nonexistent(self, test_session):
        repo = IngestionJobRepository()
        result = await repo.complete_job(test_session, "no-job-id")
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_by_collection(self, test_session):
        col1 = Collection(name="listjobs_col1")
        col2 = Collection(name="listjobs_col2")
        test_session.add_all([col1, col2])
        await test_session.commit()

        repo = IngestionJobRepository()
        await repo.create_job(test_session, collection_id=col1.id, total_files=2)
        await repo.create_job(test_session, collection_id=col1.id, total_files=3)
        await repo.create_job(test_session, collection_id=col2.id, total_files=1)

        # col1 should have 2 jobs
        jobs1, total1 = await repo.list_by_collection(test_session, col1.id, page_size=50)
        assert len(jobs1) == 2
        assert total1 == 2
        assert all(j.collection_id == col1.id for j in jobs1)

        # col2 should have 1 job
        jobs2, total2 = await repo.list_by_collection(test_session, col2.id, page_size=50)
        assert len(jobs2) == 1
        assert total2 == 1
        assert jobs2[0].collection_id == col2.id

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_by_collection_empty(self, test_session):
        """List by a collection that has no jobs."""
        col = Collection(name="nojobs_col")
        test_session.add(col)
        await test_session.commit()

        repo = IngestionJobRepository()
        items, total = await repo.list_by_collection(test_session, col.id)
        assert items == []
        assert total == 0


# ────────────────────────────────────────────────────────────────
# 9. StorageFileRepository (NEW)
# ────────────────────────────────────────────────────────────────


class TestStorageFileRepository:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_file_record(self, test_session):
        col = Collection(name="sfile_col")
        test_session.add(col)
        await test_session.commit()

        doc = Document(
            collection_id=col.id, filename="sfile_doc.pdf",
            file_type="pdf", file_size=100, file_hash="sfilehash",
        )
        test_session.add(doc)
        await test_session.commit()

        repo = StorageFileRepository()
        sfile = await repo.create(
            test_session,
            document_id=doc.id,
            storage_backend="local",
            storage_key="docs/col-id/2024/01/15/hash123.pdf",
            filename="uploaded_file.pdf",
            file_size=1024,
            content_type="application/pdf",
            storage_type="persistent",
        )
        assert sfile.id is not None
        assert sfile.storage_backend == "local"
        assert sfile.storage_key == "docs/col-id/2024/01/15/hash123.pdf"
        assert sfile.filename == "uploaded_file.pdf"
        assert sfile.file_size == 1024
        assert sfile.content_type == "application/pdf"
        assert sfile.storage_type == "persistent"
        assert sfile.created_at is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_by_document_found(self, test_session):
        col = Collection(name="sf_list_doc")
        test_session.add(col)
        await test_session.commit()

        doc = Document(
            collection_id=col.id, filename="mydoc.pdf",
            file_type="pdf", file_size=50, file_hash="sfhash1",
        )
        test_session.add(doc)
        await test_session.commit()

        repo = StorageFileRepository()
        await repo.create(
            test_session, document_id=doc.id, storage_backend="local",
            storage_key="key-1", filename="f1.pdf", file_size=100,
        )
        await repo.create(
            test_session, document_id=doc.id, storage_backend="local",
            storage_key="key-2", filename="f2.pdf", file_size=200,
        )

        files = await repo.list_by_document(test_session, doc.id)
        assert len(files) == 2
        assert files[0].document_id == doc.id
        assert files[1].document_id == doc.id

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_by_document_not_found(self, test_session):
        repo = StorageFileRepository()
        files = await repo.list_by_document(test_session, "no-such-doc-id")
        assert files == []

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_by_type(self, test_session):
        repo = StorageFileRepository()

        await repo.create(
            test_session, storage_backend="local", storage_key="tk1",
            filename="temp1.bin", file_size=10, storage_type="temp",
        )
        await repo.create(
            test_session, storage_backend="local", storage_key="pk1",
            filename="persist1.bin", file_size=20, storage_type="persistent",
        )
        await repo.create(
            test_session, storage_backend="local", storage_key="ak1",
            filename="archive1.bin", file_size=30, storage_type="archive",
        )
        await repo.create(
            test_session, storage_backend="local", storage_key="tk2",
            filename="temp2.bin", file_size=15, storage_type="temp",
        )

        temps, ct = await repo.list_by_type(test_session, "temp", page_size=50)
        assert len(temps) == 2
        assert ct == 2
        assert all(f.storage_type == "temp" for f in temps)

        persists, cp = await repo.list_by_type(test_session, "persistent", page_size=50)
        assert len(persists) == 1
        assert cp == 1
        assert persists[0].storage_type == "persistent"

        archives, ca = await repo.list_by_type(test_session, "archive", page_size=50)
        assert len(archives) == 1
        assert ca == 1
        assert archives[0].storage_type == "archive"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_by_key_found(self, test_session):
        repo = StorageFileRepository()
        await repo.create(
            test_session, storage_backend="minio", storage_key="unique/storage/key.json",
            filename="data.json", file_size=512, content_type="application/json",
        )

        found = await repo.get_by_key(test_session, "unique/storage/key.json")
        assert found is not None
        assert found.storage_backend == "minio"
        assert found.filename == "data.json"
        assert found.content_type == "application/json"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_by_key_not_found(self, test_session):
        repo = StorageFileRepository()
        result = await repo.get_by_key(test_session, "nonexistent/key/path.pdf")
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_storage_file_without_document(self, test_session):
        """StorageFile with document_id=None (orphan file, e.g. temp upload)."""
        repo = StorageFileRepository()
        sfile = await repo.create(
            test_session,
            storage_backend="local",
            storage_key="temp/upload/2024/file.pdf",
            filename="orphan.pdf",
            file_size=2048,
            storage_type="temp",
        )
        assert sfile.document_id is None
        assert sfile.storage_type == "temp"
