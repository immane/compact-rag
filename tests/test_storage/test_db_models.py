from __future__ import annotations

import pytest
from sqlalchemy import select
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


class TestDbModels:
    @pytest.mark.asyncio
    async def test_create_collection(self, test_session):
        col = Collection(name="test_collection", description="A test")
        test_session.add(col)
        await test_session.commit()

        result = await test_session.execute(
            select(Collection).where(Collection.name == "test_collection")
        )
        fetched = result.scalar_one()
        assert fetched.name == "test_collection"
        assert fetched.description == "A test"
        assert fetched.embedding_model == "BAAI/bge-small-zh-v1.5"
        assert fetched.document_count == 0

    @pytest.mark.asyncio
    async def test_collection_unique_name_constraint(self, test_session):
        col1 = Collection(name="unique_name")
        col2 = Collection(name="unique_name")
        test_session.add(col1)
        await test_session.commit()

        test_session.add(col2)
        with pytest.raises(IntegrityError):
            await test_session.commit()

    @pytest.mark.asyncio
    async def test_create_document_with_relationship(self, test_session):
        col = Collection(name="docs_collection")
        test_session.add(col)
        await test_session.commit()

        doc = Document(
            collection_id=col.id,
            filename="test.pdf",
            file_type="pdf",
            file_size=1024,
            file_hash="abc123",
        )
        test_session.add(doc)
        await test_session.commit()

        result = await test_session.execute(
            select(Document).where(Document.id == doc.id)
        )
        fetched = result.scalar_one()
        assert fetched.collection_id == col.id
        assert fetched.filename == "test.pdf"
        assert fetched.status == "pending"

    @pytest.mark.asyncio
    async def test_cascade_delete_document_chunks(self, test_session):
        col = Collection(name="cascade_col")
        test_session.add(col)
        await test_session.commit()

        doc = Document(
            collection_id=col.id,
            filename="cascade.pdf",
            file_type="pdf",
            file_size=512,
            file_hash="hash123",
        )
        test_session.add(doc)
        await test_session.commit()

        chunk = DocumentChunk(
            document_id=doc.id,
            chroma_id="chroma-123",
            chunk_index=0,
            page_number=1,
            content_hash="chash",
        )
        test_session.add(chunk)
        await test_session.commit()

        await test_session.delete(doc)
        await test_session.commit()

        chunk_result = await test_session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        )
        assert chunk_result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_json_fields(self, test_session):
        col = Collection(name="json_col")
        test_session.add(col)
        await test_session.commit()

        doc = Document(
            collection_id=col.id,
            filename="json_test.pdf",
            file_type="pdf",
            file_size=100,
            file_hash="jsonhash",
            metadata_={"author": "test", "tags": ["ai", "rag"]},
        )
        test_session.add(doc)
        await test_session.commit()

        result = await test_session.execute(
            select(Document).where(Document.id == doc.id)
        )
        fetched = result.scalar_one()
        assert fetched.metadata_ == {"author": "test", "tags": ["ai", "rag"]}

    @pytest.mark.asyncio
    async def test_api_key_permissions_json(self, test_session):
        key = ApiKey(
            name="test-key",
            key_hash="hash12345",
            permissions=["read", "write", "admin"],
        )
        test_session.add(key)
        await test_session.commit()

        result = await test_session.execute(
            select(ApiKey).where(ApiKey.name == "test-key")
        )
        fetched = result.scalar_one()
        assert fetched.permissions == ["read", "write", "admin"]
        assert fetched.is_active is True

    @pytest.mark.asyncio
    async def test_ingestion_job_errors_json(self, test_session):
        col = Collection(name="ingest_col")
        test_session.add(col)
        await test_session.commit()

        job = IngestionJob(
            collection_id=col.id,
            status="completed",
            total_files=3,
            processed_files=2,
            errors={"file3.pdf": "corrupted file"},
        )
        test_session.add(job)
        await test_session.commit()

        result = await test_session.execute(
            select(IngestionJob).where(IngestionJob.collection_id == col.id)
        )
        fetched = result.scalar_one()
        assert fetched.errors == {"file3.pdf": "corrupted file"}

    @pytest.mark.asyncio
    async def test_create_all_eight_models(self, test_session):
        col = Collection(name="all_models_col")
        test_session.add(col)
        await test_session.commit()

        doc = Document(
            collection_id=col.id,
            filename="all.pdf",
            file_type="pdf",
            file_size=200,
            file_hash="allhash",
        )
        test_session.add(doc)
        await test_session.commit()

        chunk = DocumentChunk(
            document_id=doc.id,
            chroma_id="chr-1",
            chunk_index=0,
            page_number=1,
        )
        test_session.add(chunk)

        conv = Conversation(collection_id=col.id, title="Test Conv")
        test_session.add(conv)
        await test_session.commit()

        msg = Message(conversation_id=conv.id, role="user", content="Hello")
        test_session.add(msg)

        job = IngestionJob(collection_id=col.id, status="pending")
        test_session.add(job)

        key = ApiKey(name="api-key", key_hash="hash42", permissions=["read"])
        test_session.add(key)

        sfile = StorageFile(
            document_id=doc.id,
            storage_backend="local",
            storage_key="docs/col1/2024/file.pdf",
            filename="file.pdf",
            file_size=200,
        )
        test_session.add(sfile)

        await test_session.commit()

        collections = (await test_session.execute(select(Collection))).scalars().all()
        documents = (await test_session.execute(select(Document))).scalars().all()
        chunks = (await test_session.execute(select(DocumentChunk))).scalars().all()
        conversations = (await test_session.execute(select(Conversation))).scalars().all()
        messages = (await test_session.execute(select(Message))).scalars().all()
        jobs = (await test_session.execute(select(IngestionJob))).scalars().all()
        keys = (await test_session.execute(select(ApiKey))).scalars().all()
        sfiles = (await test_session.execute(select(StorageFile))).scalars().all()

        assert len(collections) >= 1
        assert len(documents) >= 1
        assert len(chunks) >= 1
        assert len(conversations) >= 1
        assert len(messages) >= 1
        assert len(jobs) >= 1
        assert len(keys) >= 1
        assert len(sfiles) >= 1
