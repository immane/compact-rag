from __future__ import annotations

import pytest
from sqlalchemy import select

from compact_rag.storage.db.models import (
    Collection,
    Conversation,
    Document,
    DocumentChunk,
    Message,
)
from compact_rag.storage.db.repository.collection import CollectionRepository
from compact_rag.storage.db.repository.chunk import ChunkRepository
from compact_rag.storage.db.repository.conversation import (
    ConversationRepository,
    MessageRepository,
)
from compact_rag.storage.db.repository.document import DocumentRepository


class TestDocumentRepository:
    @pytest.mark.asyncio
    async def test_create_and_get_by_id(self, test_session):
        repo = DocumentRepository()
        col = Collection(name="docs_repo_col")
        test_session.add(col)
        await test_session.commit()

        doc = await repo.create(
            test_session,
            collection_id=col.id,
            filename="test.pdf",
            file_type="pdf",
            file_size=100,
            file_hash="abcdef123456",
        )
        assert doc.id is not None
        assert doc.filename == "test.pdf"

        fetched = await repo.get_by_id(test_session, doc.id)
        assert fetched is not None
        assert fetched.id == doc.id

    @pytest.mark.asyncio
    async def test_get_by_hash_found(self, test_session):
        repo = DocumentRepository()
        col = Collection(name="hash_col")
        test_session.add(col)
        await test_session.commit()

        await repo.create(
            test_session,
            collection_id=col.id,
            filename="hashdoc.pdf",
            file_type="pdf",
            file_size=50,
            file_hash="unique_hash_123",
        )

        doc = await repo.get_by_hash(test_session, "unique_hash_123")
        assert doc is not None
        assert doc.filename == "hashdoc.pdf"

    @pytest.mark.asyncio
    async def test_get_by_hash_not_found(self, test_session):
        repo = DocumentRepository()
        doc = await repo.get_by_hash(test_session, "nonexistent")
        assert doc is None

    @pytest.mark.asyncio
    async def test_get_by_hash_with_collection_filter(self, test_session):
        repo = DocumentRepository()
        col1 = Collection(name="col_hash_1")
        col2 = Collection(name="col_hash_2")
        test_session.add_all([col1, col2])
        await test_session.commit()

        await repo.create(
            test_session,
            collection_id=col1.id,
            filename="a.pdf",
            file_type="pdf",
            file_size=10,
            file_hash="samehash",
        )
        await repo.create(
            test_session,
            collection_id=col2.id,
            filename="b.pdf",
            file_type="pdf",
            file_size=10,
            file_hash="samehash",
        )

        doc = await repo.get_by_hash(test_session, "samehash", collection_id=col1.id)
        assert doc is not None
        assert doc.collection_id == col1.id

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, test_session):
        repo = DocumentRepository()
        doc = await repo.get_by_id(test_session, "nonexistent-id")
        assert doc is None


class TestCollectionRepository:
    @pytest.mark.asyncio
    async def test_create_and_get_by_name(self, test_session):
        repo = CollectionRepository()
        col = await repo.create(
            test_session, name="my_collection", description="A test collection"
        )
        assert col.name == "my_collection"

        fetched = await repo.get_by_name(test_session, "my_collection")
        assert fetched is not None
        assert fetched.id == col.id
        assert fetched.description == "A test collection"

    @pytest.mark.asyncio
    async def test_get_by_name_not_found(self, test_session):
        repo = CollectionRepository()
        col = await repo.get_by_name(test_session, "no_such_collection")
        assert col is None

    @pytest.mark.asyncio
    async def test_increment_document_count(self, test_session):
        repo = CollectionRepository()
        col = await repo.create(test_session, name="incr_col")
        assert col.document_count == 0

        await repo.increment_document_count(test_session, col.id, delta=5)
        await test_session.refresh(col)
        assert col.document_count == 5

    @pytest.mark.asyncio
    async def test_increment_document_count_negative_delta(self, test_session):
        repo = CollectionRepository()
        col = await repo.create(test_session, name="decr_col")
        assert col.document_count == 0

        await repo.increment_document_count(test_session, col.id, delta=3)
        await test_session.refresh(col)
        assert col.document_count == 3

        await repo.increment_document_count(test_session, col.id, delta=-2)
        await test_session.refresh(col)
        assert col.document_count == 1


class TestChunkRepository:
    @pytest.mark.asyncio
    async def test_list_by_document(self, test_session):
        col = Collection(name="chunk_repo_col")
        test_session.add(col)
        await test_session.commit()

        doc = Document(
            collection_id=col.id,
            filename="chunked.pdf",
            file_type="pdf",
            file_size=100,
            file_hash="chunkhash",
        )
        test_session.add(doc)
        await test_session.commit()

        chunk1 = DocumentChunk(
            document_id=doc.id,
            chroma_id="chroma-1",
            chunk_index=0,
            page_number=1,
        )
        chunk2 = DocumentChunk(
            document_id=doc.id,
            chroma_id="chroma-2",
            chunk_index=1,
            page_number=1,
        )
        test_session.add_all([chunk1, chunk2])
        await test_session.commit()

        repo = ChunkRepository()
        chunks = await repo.list_by_document(test_session, doc.id)
        assert len(chunks) == 2
        assert chunks[0].chunk_index < chunks[1].chunk_index

    @pytest.mark.asyncio
    async def test_list_by_document_empty(self, test_session):
        repo = ChunkRepository()
        chunks = await repo.list_by_document(test_session, "no-doc-id")
        assert chunks == []


class TestConversationRepository:
    @pytest.mark.asyncio
    async def test_create(self, test_session):
        repo = ConversationRepository()
        conv = await repo.create(
            test_session, title="Test Conversation", model="gpt-4o-mini"
        )
        assert conv.id is not None
        assert conv.title == "Test Conversation"
        assert conv.model == "gpt-4o-mini"

        fetched = await repo.get_by_id(test_session, conv.id)
        assert fetched is not None
        assert fetched.id == conv.id


class TestMessageRepository:
    @pytest.mark.asyncio
    async def test_create(self, test_session):
        conv_repo = ConversationRepository()
        conv = await conv_repo.create(test_session, title="Msg Test")

        msg_repo = MessageRepository()
        msg = await msg_repo.create(
            test_session,
            conversation_id=conv.id,
            role="user",
            content="Hello, this is a test message.",
        )
        assert msg.id is not None
        assert msg.role == "user"
        assert msg.content == "Hello, this is a test message."
        assert msg.conversation_id == conv.id

        fetched = await msg_repo.get_by_id(test_session, msg.id)
        assert fetched is not None
