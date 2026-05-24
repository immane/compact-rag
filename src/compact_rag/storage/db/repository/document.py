"""Document repository."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.storage.db.models import Document
from compact_rag.storage.db.repository.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    model = Document

    async def get_by_hash(
        self, session: AsyncSession, file_hash: str, collection_id: str | None = None
    ) -> Document | None:
        """Find a document by file hash (dedup check)."""
        stmt = select(Document).where(Document.file_hash == file_hash)
        if collection_id:
            stmt = stmt.where(Document.collection_id == collection_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_collection(
        self,
        session: AsyncSession,
        collection_id: str,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Document], int]:
        """List documents in a collection with pagination."""
        return await self.list(
            session, page=page, page_size=page_size, collection_id=collection_id
        )

    async def update_status(
        self,
        session: AsyncSession,
        doc_id: str,
        status: str,
        error_message: str | None = None,
    ) -> Document | None:
        """Update document processing status."""
        kwargs = {"status": status}
        if error_message is not None:
            kwargs["error_message"] = error_message
        return await self.update(session, doc_id, **kwargs)

    async def list_with_filters(
        self,
        session: AsyncSession,
        collection_id: str | None = None,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Document], int]:
        """List documents with optional filters."""
        stmt = select(Document)

        if collection_id:
            stmt = stmt.where(Document.collection_id == collection_id)
        if status:
            stmt = stmt.where(Document.status == status)

        # Count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await session.execute(count_stmt)
        total = total_result.scalar() or 0

        # Paginate
        offset = (page - 1) * page_size
        stmt = stmt.order_by(Document.created_at.desc()).offset(offset).limit(page_size)

        result = await session.execute(stmt)
        items = list(result.scalars().all())

        return items, total
