"""Ingestion job monitoring endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.api.deps import get_db_session
from compact_rag.api.schemas import (
    IngestionJobResponse,
    PaginatedResponse,
    PaginationMeta,
)
from compact_rag.common.exceptions import FileNotFoundError
from compact_rag.common.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Ingestion"])


@router.get("/ingestion-jobs", response_model=PaginatedResponse)
async def list_ingestion_jobs(
    status: str | None = Query(None),
    collection: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
):
    """List ingestion jobs."""
    from compact_rag.storage.db.repository.ingestion import IngestionJobRepository

    repo = IngestionJobRepository()
    results, total = await repo.list(
        session,
        page=page,
        page_size=page_size,
        status=status,
        collection_id=collection,
    )
    total_pages = max(1, (total + page_size - 1) // page_size) if total > 0 else 0

    return PaginatedResponse(
        data=[
            IngestionJobResponse(
                id=str(j.id),
                collection_id=str(j.collection_id or ""),
                status=j.status or "pending",
                total_files=j.total_files or 0,
                processed_files=j.processed_files or 0,
                total_chunks=j.total_chunks or 0,
                errors=j.errors,
                started_at=str(j.started_at) if j.started_at else None,
                completed_at=str(j.completed_at) if j.completed_at else None,
                created_at=str(j.created_at) if j.created_at else None,
            )
            for j in results
        ],
        pagination=PaginationMeta(
            page=page, page_size=page_size, total=total, total_pages=total_pages
        ),
    )


@router.get("/ingestion-jobs/{job_id}", response_model=IngestionJobResponse)
async def get_ingestion_job(
    job_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Get ingestion job details."""
    from compact_rag.storage.db.repository.ingestion import IngestionJobRepository

    repo = IngestionJobRepository()
    job = await repo.get_by_id(session, job_id)
    if job is None:
        raise FileNotFoundError(f"Ingestion job not found: {job_id}")

    return IngestionJobResponse(
        id=str(job.id),
        collection_id=str(job.collection_id or ""),
        status=job.status or "pending",
        total_files=job.total_files or 0,
        processed_files=job.processed_files or 0,
        total_chunks=job.total_chunks or 0,
        errors=job.errors,
        started_at=str(job.started_at) if job.started_at else None,
        completed_at=str(job.completed_at) if job.completed_at else None,
        created_at=str(job.created_at) if job.created_at else None,
    )
