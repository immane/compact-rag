from __future__ import annotations

from types import SimpleNamespace

import pytest

from compact_rag.ingestion.pipeline import IngestionPipeline


@pytest.mark.asyncio
async def test_ingest_file_updates_job_progress_before_completion(test_settings, tmp_path, monkeypatch, mocker):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4 test")

    session = mocker.AsyncMock()
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    collection_repo = mocker.Mock()
    collection_repo.get_by_name = mocker.AsyncMock(return_value=SimpleNamespace(id="collection-1"))
    collection_repo.increment_document_count = mocker.AsyncMock()

    document_repo = mocker.Mock()
    document_repo.get_by_hash = mocker.AsyncMock(return_value=None)
    document_repo.create = mocker.AsyncMock(return_value=SimpleNamespace(id="doc-1"))
    document_repo.update = mocker.AsyncMock()

    job_repo = mocker.Mock()
    job_repo.create_job = mocker.AsyncMock(return_value=SimpleNamespace(id="job-1"))
    job_repo.update_progress = mocker.AsyncMock()
    job_repo.complete_job = mocker.AsyncMock()

    monkeypatch.setattr(
        "compact_rag.storage.db.repository.collection.CollectionRepository",
        lambda: collection_repo,
    )
    monkeypatch.setattr(
        "compact_rag.storage.db.repository.document.DocumentRepository",
        lambda: document_repo,
    )
    monkeypatch.setattr(
        "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
        lambda: job_repo,
    )

    monkeypatch.setattr(
        "compact_rag.ingestion.pipeline.chunk_documents",
        lambda *args, **kwargs: [
            SimpleNamespace(
                content="chunk text",
                chunk_index=0,
                page_number=1,
                is_table=False,
                token_count=2,
                content_hash="hash-1",
                metadata={},
            )
        ],
    )
    monkeypatch.setattr(
        "compact_rag.ingestion.pipeline.TableExtractor.extract_from_pdf",
        lambda self, path: [],
    )

    mocker.patch.object(
        pipeline,
        "_load_document",
        mocker.AsyncMock(return_value=[SimpleNamespace(page_number=1)]),
    )
    mocker.patch.object(
        pipeline,
        "_get_embedding_service",
        return_value=SimpleNamespace(encode=lambda texts: []),
    )
    mocker.patch.object(
        pipeline,
        "_get_storage_backend",
        return_value=SimpleNamespace(upload_file=mocker.AsyncMock()),
    )

    result = await pipeline.ingest_file(str(file_path), collection_name="default")

    assert result.status == "completed"
    job_repo.update_progress.assert_awaited_once_with(
        session,
        "job-1",
        processed=1,
        chunks=1,
    )
    job_repo.complete_job.assert_awaited_once_with(
        session,
        "job-1",
        status="completed",
    )