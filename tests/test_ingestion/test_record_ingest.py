"""Tests for IngestionPipeline.ingest_records (API source documents)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from compact_rag.ingestion.pipeline import IngestionPipeline
from compact_rag.ingestion.sources import SourceRecord, record_content_hash


def _record(external_id="1", content="content: hello"):
    return SourceRecord(
        external_id=external_id,
        title=f"#{external_id}",
        content=f"# #{external_id}\n\n{content}",
        metadata={"source": "comments", "external_id": external_id},
    )


def _pipeline(test_settings, session, mocker, monkeypatch, vector_store=None):
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    collection_repo = mocker.Mock()
    collection_repo.get_by_name = mocker.AsyncMock(
        return_value=SimpleNamespace(id="collection-1")
    )
    collection_repo.increment_document_count = mocker.AsyncMock()

    document_repo = mocker.Mock()
    chunk_repo = mocker.Mock()
    job_repo = mocker.Mock()
    job_repo.create_job = mocker.AsyncMock(return_value=SimpleNamespace(id="job-1"))
    job_repo.update_progress = mocker.AsyncMock()
    job_repo.complete_job = mocker.AsyncMock()

    monkeypatch.setattr(
        "compact_rag.storage.db.repository.collection.CollectionRepository",
        lambda *a, **k: collection_repo,
    )
    monkeypatch.setattr(
        "compact_rag.storage.db.repository.document.DocumentRepository",
        lambda *a, **k: document_repo,
    )
    monkeypatch.setattr(
        "compact_rag.storage.db.repository.chunk.ChunkRepository",
        lambda *a, **k: chunk_repo,
    )
    monkeypatch.setattr(
        "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
        lambda *a, **k: job_repo,
    )
    mocker.patch.object(
        pipeline,
        "_get_embedding_service",
        return_value=SimpleNamespace(encode=lambda texts: [[0.1] * 4 for _ in texts]),
    )
    store = vector_store or mocker.Mock()
    if vector_store is None:
        store.add_documents = mocker.Mock(return_value=["chroma-1"])
        store.delete_by_document = mocker.Mock(return_value=0)
    mocker.patch.object(pipeline, "_get_vector_store", return_value=store)
    handles = {
        "collection_repo": collection_repo,
        "document_repo": document_repo,
        "chunk_repo": chunk_repo,
        "job_repo": job_repo,
        "vector_store": store,
    }
    return pipeline, handles


@pytest.mark.asyncio
async def test_ingest_records_skips_unchanged(test_settings, mocker, monkeypatch):
    session = mocker.AsyncMock()
    pipeline, handles = _pipeline(test_settings, session, mocker, monkeypatch)
    record = _record()
    existing = SimpleNamespace(
        id="doc-1", filename="comments#1", chunk_count=2, table_count=0
    )
    handles["document_repo"].get_by_hash = mocker.AsyncMock(return_value=existing)

    results = await pipeline.ingest_records([record], "reviews", "comments")

    assert len(results) == 1
    assert results[0].status == "skipped"
    assert results[0].doc_id == "doc-1"
    handles["document_repo"].create.assert_not_called()
    handles["job_repo"].complete_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_records_creates_new(test_settings, mocker, monkeypatch):
    session = mocker.AsyncMock()
    store = mocker.Mock()
    store.add_documents = mocker.Mock(return_value=["chroma-1"])
    store.delete_by_document = mocker.Mock()
    pipeline, handles = _pipeline(test_settings, session, mocker, monkeypatch, store)
    handles["document_repo"].get_by_hash = mocker.AsyncMock(return_value=None)
    handles["document_repo"].list = mocker.AsyncMock(return_value=([], 0))
    handles["document_repo"].create = mocker.AsyncMock(
        return_value=SimpleNamespace(id="doc-9")
    )
    handles["document_repo"].update = mocker.AsyncMock()
    handles["document_repo"].delete = mocker.AsyncMock()
    handles["chunk_repo"].create = mocker.AsyncMock()

    results = await pipeline.ingest_records([_record()], "reviews", "comments")

    assert results[0].status == "completed"
    assert results[0].doc_id == "doc-9"
    create_kwargs = handles["document_repo"].create.await_args.kwargs
    assert create_kwargs["filename"] == "comments#1"
    assert create_kwargs["file_type"] == "api"
    assert create_kwargs["metadata_"]["external_id"] == "1"
    store.delete_by_document.assert_not_called()
    handles["collection_repo"].increment_document_count.assert_awaited_with(
        session, "collection-1", 1
    )


@pytest.mark.asyncio
async def test_ingest_records_replaces_changed_version(test_settings, mocker, monkeypatch):
    session = mocker.AsyncMock()
    store = mocker.Mock()
    store.add_documents = mocker.Mock(return_value=["chroma-2"])
    store.delete_by_document = mocker.Mock(return_value=1)
    pipeline, handles = _pipeline(test_settings, session, mocker, monkeypatch, store)
    handles["document_repo"].get_by_hash = mocker.AsyncMock(return_value=None)
    slot = SimpleNamespace(id="doc-old", file_hash="oldhash")
    handles["document_repo"].list = mocker.AsyncMock(return_value=([slot], 1))
    handles["document_repo"].create = mocker.AsyncMock(
        return_value=SimpleNamespace(id="doc-new")
    )
    handles["document_repo"].update = mocker.AsyncMock()
    handles["document_repo"].delete = mocker.AsyncMock(return_value=True)
    handles["chunk_repo"].create = mocker.AsyncMock()

    results = await pipeline.ingest_records(
        [_record(content="content: changed")], "reviews", "comments"
    )

    assert results[0].status == "completed"
    assert results[0].doc_id == "doc-new"
    store.delete_by_document.assert_called_once_with("doc-old")
    handles["document_repo"].delete.assert_awaited_once_with(session, "doc-old")


@pytest.mark.asyncio
async def test_ingest_records_failure_isolated(test_settings, mocker, monkeypatch):
    session = mocker.AsyncMock()
    pipeline, handles = _pipeline(test_settings, session, mocker, monkeypatch)
    handles["document_repo"].get_by_hash = mocker.AsyncMock(
        side_effect=[None, RuntimeError("db down")]
    )
    handles["document_repo"].list = mocker.AsyncMock(return_value=([], 0))
    handles["document_repo"].create = mocker.AsyncMock(
        return_value=SimpleNamespace(id="doc-1")
    )
    handles["document_repo"].update = mocker.AsyncMock()
    handles["chunk_repo"].create = mocker.AsyncMock()

    results = await pipeline.ingest_records(
        [_record("1"), _record("2")], "reviews", "comments"
    )

    assert [r.status for r in results] == ["completed", "failed"]
    assert results[1].error_message is not None
    handles["job_repo"].complete_job.assert_awaited_once()


def test_slot_filename_uses_external_id():
    record = _record(external_id="42")
    assert record_content_hash("comments", record) == record_content_hash(
        "comments", _record(external_id="42")
    )
