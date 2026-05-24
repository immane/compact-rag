"""Document management endpoints."""

from __future__ import annotations

import os
import tempfile
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.api.deps import (
    get_db_session,
    get_settings,
    get_storage_backend,
    verify_api_key,
)
from compact_rag.api.schemas import (
    DocumentIngestResponse,
    DocumentResponse,
    IngestUrlRequest,
    PaginatedResponse,
    PaginationMeta,
)
from compact_rag.common.exceptions import FileNotFoundError, UnsupportedFormatError
from compact_rag.common.logger import get_logger
from compact_rag.config.settings import Settings
from compact_rag.storage.file_storage import StorageBackend, build_storage_key

logger = get_logger(__name__)
router = APIRouter(tags=["Documents"])


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    collection: str = Form(default="default"),
    force: bool = Form(default=False),
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(get_storage_backend),
    _api_key: str | None = Depends(verify_api_key),
):
    """Upload and ingest a document file."""
    if not file.filename:
        raise UnsupportedFormatError("No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ingestion.supported_extensions:
        raise UnsupportedFormatError(
            f"Unsupported file format: {ext}. "
            f"Supported: {settings.ingestion.supported_extensions}"
        )

    # Save uploaded file to temp location
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)
    content = await file.read()

    with open(tmp_path, "wb") as f:
        f.write(content)

    # Build storage key for temp file
    temp_key = f"temp/{uuid4().hex[:12]}/{file.filename}"

    try:
        await storage.upload_bytes(content, temp_key)

        # Run ingestion pipeline
        from compact_rag.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline(
            settings=settings,
            session=session,
        )
        result = await pipeline.ingest_file(
            file_path=tmp_path,
            collection_name=collection,
            force=force,
        )

        # Persist original file
        persist_key = build_storage_key(
            collection_id=collection,
            filename=file.filename,
            category="docs",
        )
        await storage.upload_bytes(content, persist_key)

        return DocumentIngestResponse(
            doc_id=result.doc_id,
            filename=result.filename,
            status=result.status,
            chunk_count=result.chunk_count,
            table_count=result.table_count,
            error_message=result.error_message,
            duration_ms=result.duration_ms,
        )
    except Exception as e:
        logger.error("Document ingestion failed", filename=file.filename, error=str(e))
        return DocumentIngestResponse(
            doc_id="",
            filename=file.filename or "",
            status="failed",
            error_message=str(e),
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


@router.post("/documents/ingest-url", response_model=DocumentIngestResponse)
async def ingest_document_url(
    request: IngestUrlRequest,
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    _api_key: str | None = Depends(verify_api_key),
):
    """Ingest a document from a URL."""
    import httpx

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(request.url)
        resp.raise_for_status()

    filename = request.url.split("/")[-1] or "document.txt"
    content = resp.content

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=f"_{filename}", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from compact_rag.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline(settings=settings, session=session)
        result = await pipeline.ingest_file(
            file_path=tmp_path,
            collection_name=request.collection,
        )

        return DocumentIngestResponse(
            doc_id=result.doc_id,
            filename=result.filename,
            status=result.status,
            chunk_count=result.chunk_count,
            table_count=result.table_count,
            error_message=result.error_message,
            duration_ms=result.duration_ms,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.get("/documents", response_model=PaginatedResponse)
async def list_documents(
    collection: str | None = Query(None),
    status: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
):
    """List ingested documents with optional filters."""
    from compact_rag.storage.db.repository.document import DocumentRepository

    repo = DocumentRepository()
    results, total = await repo.list_with_filters(
        session,
        collection_id=collection,
        status=status,
        page=page,
        page_size=page_size,
    )

    total_pages = max(1, (total + page_size - 1) // page_size) if total > 0 else 0
    return PaginatedResponse(
        data=[_doc_to_response(d) for d in results],
        pagination=PaginationMeta(
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
        ),
    )


@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Get document details by ID."""
    from compact_rag.storage.db.repository.document import DocumentRepository

    repo = DocumentRepository()
    doc = await repo.get_by_id(session, doc_id)
    if doc is None:
        raise FileNotFoundError(f"Document not found: {doc_id}")
    return _doc_to_response(doc)


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    session: AsyncSession = Depends(get_db_session),
    _api_key: str | None = Depends(verify_api_key),
):
    """Delete a document and all its chunks (ChromaDB + SQL)."""
    from compact_rag.storage.db.repository.document import DocumentRepository

    doc_repo = DocumentRepository()
    # chunk_repo variable removed as it was unused

    doc = await doc_repo.get_by_id(session, doc_id)
    if doc is None:
        raise FileNotFoundError(f"Document not found: {doc_id}")

    # Delete from ChromaDB
    try:
        from compact_rag.api.deps import get_vector_store

        store = get_vector_store()
        await store.delete_by_document(doc_id)
    except Exception as e:
        logger.warning("Failed to delete from ChromaDB", doc_id=doc_id, error=str(e))

    # Delete from SQL (cascades to chunks)
    await doc_repo.delete(session, doc_id)
    await session.commit()

    return {"status": "deleted", "doc_id": doc_id}


def _doc_to_response(doc) -> DocumentResponse:
    return DocumentResponse(
        id=str(doc.id),
        collection_id=str(doc.collection_id or ""),
        filename=doc.filename or "",
        file_type=doc.file_type or "",
        file_size=doc.file_size or 0,
        file_hash=doc.file_hash or "",
        page_count=doc.page_count,
        chunk_count=doc.chunk_count or 0,
        table_count=doc.table_count or 0,
        status=doc.status or "pending",
        error_message=doc.error_message,
        created_at=str(doc.created_at) if doc.created_at else None,
        updated_at=str(doc.updated_at) if doc.updated_at else None,
    )
