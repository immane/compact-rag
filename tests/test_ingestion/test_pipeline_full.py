from __future__ import annotations

from types import SimpleNamespace

import pytest

from compact_rag.common.exceptions import (
    DocumentLoadError,
    IngestionError,
    UnsupportedFormatError,
)
from compact_rag.ingestion.pipeline import IngestionPipeline
from compact_rag.storage.schema import IngestionResult


def _patch_repos(mocker, collection_repo=None, document_repo=None, job_repo=None, chunk_repo=None):
    """Helper to mock all repositories used by the pipeline."""
    if collection_repo:
        mocker.patch(
            "compact_rag.storage.db.repository.collection.CollectionRepository",
            return_value=collection_repo,
        )
    if document_repo:
        mocker.patch(
            "compact_rag.storage.db.repository.document.DocumentRepository",
            return_value=document_repo,
        )
    if job_repo:
        mocker.patch(
            "compact_rag.storage.db.repository.ingestion.IngestionJobRepository",
            return_value=job_repo,
        )
    if chunk_repo:
        mocker.patch(
            "compact_rag.storage.db.repository.chunk.ChunkRepository",
            return_value=chunk_repo,
        )


# ── ingest_file: full flow ──────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_file_full_flow(test_settings, tmp_path, mocker):
    """Full ingest_file flow with mocked dependencies."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("test content", encoding="utf-8")

    session = mocker.AsyncMock()
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    collection_repo = mocker.Mock()
    collection_repo.get_by_name = mocker.AsyncMock(return_value=SimpleNamespace(id="collection-1"))
    collection_repo.increment_document_count = mocker.AsyncMock()

    document_repo = mocker.Mock()
    document_repo.get_by_hash = mocker.AsyncMock(return_value=None)
    document_repo.create = mocker.AsyncMock(return_value=SimpleNamespace(id="doc-123"))
    document_repo.update = mocker.AsyncMock()

    job_repo = mocker.Mock()
    job_repo.create_job = mocker.AsyncMock(return_value=SimpleNamespace(id="job-abc"))
    job_repo.update_progress = mocker.AsyncMock()
    job_repo.complete_job = mocker.AsyncMock()

    chunk_repo = mocker.Mock()
    chunk_repo.create = mocker.AsyncMock()

    _patch_repos(mocker, collection_repo, document_repo, job_repo, chunk_repo)

    mocker.patch(
        "compact_rag.ingestion.pipeline.chunk_documents",
        return_value=[
            SimpleNamespace(
                content="chunk 1", chunk_index=0, page_number=1,
                is_table=False, token_count=2, content_hash="h1", metadata={},
            ),
            SimpleNamespace(
                content="chunk 2", chunk_index=1, page_number=1,
                is_table=False, token_count=2, content_hash="h2", metadata={},
            ),
        ],
    )

    mocker.patch.object(pipeline, "_load_document", mocker.AsyncMock(
        return_value=[SimpleNamespace(page_number=1)]))
    mocker.patch.object(pipeline, "_get_embedding_service",
                        return_value=SimpleNamespace(encode=lambda texts: [[0.1] * 384 for _ in texts]))
    mocker.patch.object(pipeline, "_get_storage_backend",
                        return_value=SimpleNamespace(upload_file=mocker.AsyncMock()))
    mock_vs = mocker.Mock()
    mock_vs.add_documents = mocker.Mock(return_value=["chroma-1", "chroma-2"])
    mocker.patch.object(pipeline, "_get_vector_store", return_value=mock_vs)

    result = await pipeline.ingest_file(str(file_path), collection_name="default")

    assert result.status == "completed"
    assert result.doc_id == "doc-123"
    assert result.chunk_count == 2
    assert result.table_count == 0
    collection_repo.increment_document_count.assert_awaited_once()
    job_repo.complete_job.assert_awaited_once_with(session, "job-abc", status="completed")


# ── ingest_file: force=True re-ingests ──────────────────────────


@pytest.mark.asyncio
async def test_ingest_file_force_reingests(test_settings, tmp_path, mocker):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4 test content")

    session = mocker.AsyncMock()
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    collection_repo = mocker.Mock()
    collection_repo.get_by_name = mocker.AsyncMock(return_value=SimpleNamespace(id="col-1"))
    collection_repo.increment_document_count = mocker.AsyncMock()

    document_repo = mocker.Mock()
    document_repo.get_by_hash = mocker.AsyncMock(return_value=SimpleNamespace(
        id="existing-doc", filename="sample.pdf", chunk_count=5, table_count=1))
    document_repo.create = mocker.AsyncMock(return_value=SimpleNamespace(id="new-doc"))
    document_repo.update = mocker.AsyncMock()

    job_repo = mocker.Mock()
    job_repo.create_job = mocker.AsyncMock(return_value=SimpleNamespace(id="job-f"))
    job_repo.update_progress = mocker.AsyncMock()
    job_repo.complete_job = mocker.AsyncMock()

    _patch_repos(mocker, collection_repo, document_repo, job_repo, mocker.Mock())

    mocker.patch(
        "compact_rag.ingestion.pipeline.chunk_documents",
        return_value=[
            SimpleNamespace(content="re-chunk", chunk_index=0, page_number=1,
                            is_table=False, token_count=2, content_hash="rh", metadata={})
        ],
    )

    mocker.patch.object(pipeline, "_load_document", mocker.AsyncMock(
        return_value=[SimpleNamespace(page_number=1)]))
    mocker.patch.object(pipeline, "_get_embedding_service",
                        return_value=SimpleNamespace(encode=lambda texts: []))
    mocker.patch.object(pipeline, "_get_storage_backend",
                        return_value=SimpleNamespace(upload_file=mocker.AsyncMock()))

    result = await pipeline.ingest_file(str(file_path), collection_name="default", force=True)

    assert result.status == "completed"
    assert result.doc_id == "new-doc"


# ── ingest_file: skipped when already exists ────────────────────


@pytest.mark.asyncio
async def test_ingest_file_skips_when_exists(test_settings, tmp_path, mocker):
    file_path = tmp_path / "existing.txt"
    file_path.write_text("already here", encoding="utf-8")

    session = mocker.AsyncMock()
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    collection_repo = mocker.Mock()
    collection_repo.get_by_name = mocker.AsyncMock(return_value=SimpleNamespace(id="col-1"))

    document_repo = mocker.Mock()
    document_repo.get_by_hash = mocker.AsyncMock(return_value=SimpleNamespace(
        id="skip-doc", filename="existing.txt", chunk_count=3, table_count=0))

    _patch_repos(mocker, collection_repo, document_repo)

    result = await pipeline.ingest_file(str(file_path))

    assert result.status == "skipped"
    assert result.doc_id == "skip-doc"


# ── ingest_file: load failure marked as failed ──────────────────


@pytest.mark.asyncio
async def test_ingest_file_load_failure_marked_failed(test_settings, tmp_path, mocker):
    file_path = tmp_path / "bad.pdf"
    file_path.write_bytes(b"not a real pdf")

    session = mocker.AsyncMock()
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    collection_repo = mocker.Mock()
    collection_repo.get_by_name = mocker.AsyncMock(return_value=SimpleNamespace(id="col-1"))

    document_repo = mocker.Mock()
    document_repo.get_by_hash = mocker.AsyncMock(return_value=None)

    _patch_repos(mocker, collection_repo, document_repo)

    mocker.patch.object(pipeline, "_load_document", side_effect=DocumentLoadError("Cannot load"))

    result = await pipeline.ingest_file(str(file_path))

    assert result.status == "failed"
    assert result.error_message is not None
    assert "DocumentLoadError" in result.error_message


# ── ingest_file: embed failure falls through gracefully ─────────


@pytest.mark.asyncio
async def test_ingest_file_embed_failure_graceful(test_settings, tmp_path, mocker):
    file_path = tmp_path / "test.txt"
    file_path.write_text("content", encoding="utf-8")

    session = mocker.AsyncMock()
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    collection_repo = mocker.Mock()
    collection_repo.get_by_name = mocker.AsyncMock(return_value=SimpleNamespace(id="col-2"))
    collection_repo.increment_document_count = mocker.AsyncMock()

    document_repo = mocker.Mock()
    document_repo.get_by_hash = mocker.AsyncMock(return_value=None)
    document_repo.create = mocker.AsyncMock(return_value=SimpleNamespace(id="doc-embed-fail"))
    document_repo.update = mocker.AsyncMock()

    job_repo = mocker.Mock()
    job_repo.create_job = mocker.AsyncMock(return_value=SimpleNamespace(id="job-embed-fail"))
    job_repo.update_progress = mocker.AsyncMock()
    job_repo.complete_job = mocker.AsyncMock()

    _patch_repos(mocker, collection_repo, document_repo, job_repo, mocker.Mock())

    mocker.patch(
        "compact_rag.ingestion.pipeline.chunk_documents",
        return_value=[
            SimpleNamespace(content="chunk", chunk_index=0, page_number=1,
                            is_table=False, token_count=1, content_hash="xx", metadata={})
        ],
    )

    mocker.patch.object(pipeline, "_load_document", mocker.AsyncMock(
        return_value=[SimpleNamespace(page_number=1)]))
    mocker.patch.object(pipeline, "_get_storage_backend",
                        return_value=SimpleNamespace(upload_file=mocker.AsyncMock()))

    mock_embedding = mocker.Mock()
    mock_embedding.encode.side_effect = RuntimeError("Embedding server down")
    mocker.patch.object(pipeline, "_get_embedding_service", return_value=mock_embedding)

    result = await pipeline.ingest_file(str(file_path))
    assert result.status == "failed"
    assert result.error_message is not None
    session.rollback.assert_awaited_once()


# ── ingest_file: non-existent collection creates one ────────────


@pytest.mark.asyncio
async def test_ingest_file_creates_collection_if_missing(test_settings, tmp_path, mocker):
    file_path = tmp_path / "test.txt"
    file_path.write_text("data", encoding="utf-8")

    session = mocker.AsyncMock()
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    collection_repo = mocker.Mock()
    collection_repo.get_by_name = mocker.AsyncMock(return_value=None)
    collection_repo.create = mocker.AsyncMock(return_value=SimpleNamespace(id="new-col-9"))
    collection_repo.increment_document_count = mocker.AsyncMock()

    document_repo = mocker.Mock()
    document_repo.get_by_hash = mocker.AsyncMock(return_value=None)
    document_repo.create = mocker.AsyncMock(return_value=SimpleNamespace(id="doc-newcol"))
    document_repo.update = mocker.AsyncMock()

    job_repo = mocker.Mock()
    job_repo.create_job = mocker.AsyncMock(return_value=SimpleNamespace(id="job-newcol"))
    job_repo.update_progress = mocker.AsyncMock()
    job_repo.complete_job = mocker.AsyncMock()

    _patch_repos(mocker, collection_repo, document_repo, job_repo, mocker.Mock())

    mocker.patch(
        "compact_rag.ingestion.pipeline.chunk_documents",
        return_value=[
            SimpleNamespace(content="c", chunk_index=0, page_number=1,
                            is_table=False, token_count=1, content_hash="c", metadata={})
        ],
    )

    mocker.patch.object(pipeline, "_load_document", mocker.AsyncMock(
        return_value=[SimpleNamespace(page_number=1)]))
    mocker.patch.object(pipeline, "_get_embedding_service",
                        return_value=SimpleNamespace(encode=lambda texts: []))
    mocker.patch.object(pipeline, "_get_storage_backend",
                        return_value=SimpleNamespace(upload_file=mocker.AsyncMock()))

    result = await pipeline.ingest_file(str(file_path), collection_name="brand-new-collection")

    assert result.status == "completed"
    collection_repo.create.assert_awaited_once()
    # call is collection_repo.create(session, name=..., embedding_model=..., chunk_size=..., chunk_overlap=...)
    assert collection_repo.create.await_args[1]["name"] == "brand-new-collection"


# ── ingest_directory ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_directory_processes_multiple_files(test_settings, tmp_path, mocker):
    dir_path = tmp_path / "docs"
    dir_path.mkdir()
    (dir_path / "a.txt").write_text("content a", encoding="utf-8")
    (dir_path / "b.txt").write_text("content b", encoding="utf-8")
    (dir_path / "c.csv").write_text("unsupported", encoding="utf-8")

    session = mocker.AsyncMock()
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    results = [
        IngestionResult(doc_id="d1", filename="a.txt", status="completed", chunk_count=1),
        IngestionResult(doc_id="d2", filename="b.txt", status="completed", chunk_count=1),
    ]
    mocker.patch.object(pipeline, "ingest_file", mocker.AsyncMock(side_effect=results))

    output = await pipeline.ingest_directory(str(dir_path))
    assert len(output) == 2
    assert all(r.status == "completed" for r in output)


@pytest.mark.asyncio
async def test_ingest_directory_skips_unsupported_extensions(test_settings, tmp_path, mocker):
    dir_path = tmp_path / "mixed"
    dir_path.mkdir()
    (dir_path / "keep.txt").write_text("valid", encoding="utf-8")
    (dir_path / "skip.csv").write_text("nope", encoding="utf-8")
    (dir_path / "skip.jpg").write_bytes(b"\xff\xd8")

    session = mocker.AsyncMock()
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    mocker.patch.object(
        pipeline, "ingest_file",
        mocker.AsyncMock(return_value=IngestionResult(doc_id="d", filename="keep.txt", status="completed")),
    )

    output = await pipeline.ingest_directory(str(dir_path))
    assert len(output) == 1
    assert output[0].filename == "keep.txt"


@pytest.mark.asyncio
async def test_ingest_directory_handles_individual_failures(test_settings, tmp_path, mocker):
    dir_path = tmp_path / "partial"
    dir_path.mkdir()
    (dir_path / "ok.txt").write_text("fine", encoding="utf-8")
    (dir_path / "bad.txt").write_text("also fine", encoding="utf-8")

    session = mocker.AsyncMock()
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    mocker.patch.object(
        pipeline, "ingest_file",
        mocker.AsyncMock(side_effect=[
            IngestionResult(doc_id="d1", filename="ok.txt", status="completed"),
            Exception("Internal failure"),
        ]),
    )

    output = await pipeline.ingest_directory(str(dir_path))
    assert len(output) == 2
    assert output[0].status == "completed"
    assert output[1].status == "failed"
    assert output[1].error_message is not None
    assert "Internal failure" in output[1].error_message


# ── ingest_url ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_url_valid_download(test_settings, mocker):
    session = mocker.AsyncMock()
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/plain"}
    mock_response.content = b"downloaded content"
    mock_response.raise_for_status = mocker.MagicMock()

    mock_client = mocker.AsyncMock()
    mock_client.get = mocker.AsyncMock(return_value=mock_response)

    mocker.patch("httpx.AsyncClient", return_value=mocker.AsyncMock(
        __aenter__=mocker.AsyncMock(return_value=mock_client),
        __aexit__=mocker.AsyncMock(return_value=None),
    ))

    ingest_result = IngestionResult(doc_id="url-doc", filename="download.txt", status="completed", chunk_count=1)
    mocker.patch.object(pipeline, "ingest_file", mocker.AsyncMock(return_value=ingest_result))

    result = await pipeline.ingest_url("https://example.com/doc.txt", collection_name="default")

    assert result.status == "completed"
    assert result.doc_id == "url-doc"


@pytest.mark.asyncio
async def test_ingest_url_invalid_url_raises_error(test_settings, mocker):
    session = mocker.AsyncMock()
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    mock_response = mocker.MagicMock()
    mock_response.raise_for_status.side_effect = Exception("404 Not Found")

    mock_client = mocker.AsyncMock()
    mock_client.get = mocker.AsyncMock(return_value=mock_response)

    mocker.patch("httpx.AsyncClient", return_value=mocker.AsyncMock(
        __aenter__=mocker.AsyncMock(return_value=mock_client),
        __aexit__=mocker.AsyncMock(return_value=None),
    ))

    with pytest.raises(IngestionError, match="Failed to download URL"):
        await pipeline.ingest_url("https://example.com/bad.pdf")


@pytest.mark.asyncio
async def test_ingest_url_unsupported_extension(test_settings, mocker):
    session = mocker.AsyncMock()
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    with pytest.raises(UnsupportedFormatError):
        await pipeline.ingest_url("https://example.com/video.mp4")


@pytest.mark.asyncio
async def test_ingest_url_no_extension_detects_from_content_type(test_settings, mocker):
    session = mocker.AsyncMock()
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/pdf"}
    mock_response.content = b"%PDF-1.4"
    mock_response.raise_for_status = mocker.MagicMock()

    mock_client = mocker.AsyncMock()
    mock_client.get = mocker.AsyncMock(return_value=mock_response)

    mocker.patch("httpx.AsyncClient", return_value=mocker.AsyncMock(
        __aenter__=mocker.AsyncMock(return_value=mock_client),
        __aexit__=mocker.AsyncMock(return_value=None),
    ))

    ingest_result = IngestionResult(doc_id="detected", filename="download.pdf", status="completed")
    mocker.patch.object(pipeline, "ingest_file", mocker.AsyncMock(return_value=ingest_result))

    result = await pipeline.ingest_url("https://example.com/download")
    assert result.status == "completed"


# ── Session management ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_creates_session_when_not_provided(test_settings, tmp_path, mocker):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    pipeline = IngestionPipeline(settings=test_settings, session=None)

    mock_factory = mocker.MagicMock()
    mock_session = mocker.MagicMock()
    mock_factory.return_value = mock_session

    mocker.patch(
        "compact_rag.storage.db.engine.create_engine",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "compact_rag.storage.db.engine.create_session_factory",
        return_value=mock_factory,
    )

    collection_repo = mocker.Mock()
    collection_repo.get_by_name = mocker.AsyncMock(return_value=SimpleNamespace(id="col-1"))

    document_repo = mocker.Mock()
    document_repo.get_by_hash = mocker.AsyncMock(return_value=SimpleNamespace(
        id="existing", filename="sample.txt", chunk_count=1, table_count=0))

    _patch_repos(mocker, collection_repo, document_repo)

    result = await pipeline.ingest_file(str(file_path))
    assert result.status == "skipped"


@pytest.mark.asyncio
async def test_pipeline_uses_provided_session(test_settings, tmp_path, mocker):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    session = mocker.AsyncMock()
    pipeline = IngestionPipeline(settings=test_settings, session=session)

    collection_repo = mocker.Mock()
    collection_repo.get_by_name = mocker.AsyncMock(return_value=SimpleNamespace(id="col-1"))

    document_repo = mocker.Mock()
    document_repo.get_by_hash = mocker.AsyncMock(return_value=SimpleNamespace(
        id="existing", filename="sample.txt", chunk_count=1, table_count=0))

    _patch_repos(mocker, collection_repo, document_repo)

    result = await pipeline.ingest_file(str(file_path))
    assert result.status == "skipped"
