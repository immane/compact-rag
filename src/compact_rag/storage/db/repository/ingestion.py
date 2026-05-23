"""IngestionJob repository."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.storage.db.models import IngestionJob
from compact_rag.storage.db.repository.base import BaseRepository


class IngestionJobRepository(BaseRepository[IngestionJob]):
    model = IngestionJob

    async def create_job(
        self, session: AsyncSession, collection_id: str, total_files: int = 1
    ) -> IngestionJob:
        """Create a new ingestion job."""
        return await self.create(
            session,
            collection_id=collection_id,
            total_files=total_files,
            status="running",
            started_at=datetime.now(timezone.utc),
        )

    async def update_progress(
        self,
        session: AsyncSession,
        job_id: str,
        processed: int,
        chunks: int,
    ) -> IngestionJob | None:
        """Update job progress."""
        return await self.update(
            session,
            job_id,
            processed_files=processed,
            total_chunks=chunks,
        )

    async def complete_job(
        self,
        session: AsyncSession,
        job_id: str,
        status: str = "completed",
        errors: dict | None = None,
    ) -> IngestionJob | None:
        """Complete a job with final status."""
        kwargs = {
            "status": status,
            "completed_at": datetime.now(timezone.utc),
        }
        if errors:
            kwargs["errors"] = errors
        return await self.update(session, job_id, **kwargs)

    async def list_by_collection(
        self,
        session: AsyncSession,
        collection_id: str,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[IngestionJob], int]:
        """List jobs for a specific collection."""
        return await self.list(
            session, page=page, page_size=page_size, collection_id=collection_id
        )
