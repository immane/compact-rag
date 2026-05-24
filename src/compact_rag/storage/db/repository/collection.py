"""Collection repository."""

from __future__ import annotations

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.storage.db.models import Collection, Document
from compact_rag.storage.db.repository.base import BaseRepository


class CollectionRepository(BaseRepository[Collection]):
    model = Collection

    async def list_with_realtime_document_count(
        self,
        session: AsyncSession,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[tuple[Collection, int]], int]:
        """List collections with realtime document counts via COUNT(*).

        Returns:
            Tuple of ((collection, doc_count) list, total_collection_count).
        """
        total_stmt = select(func.count()).select_from(Collection)
        total_result = await session.execute(total_stmt)
        total = int(total_result.scalar() or 0)

        offset = (page - 1) * page_size
        stmt = (
            select(Collection, func.count(Document.id).label("doc_count"))
            .outerjoin(Document, Document.collection_id == Collection.id)
            .group_by(Collection.id)
            .order_by(Collection.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await session.execute(stmt)
        rows = [(row[0], int(row[1] or 0)) for row in result.all()]
        return rows, total

    async def get_by_name(self, session: AsyncSession, name: str) -> Collection | None:
        """Find a collection by its unique name."""
        stmt = select(Collection).where(Collection.name == name)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def increment_document_count(
        self, session: AsyncSession, collection_id: str, delta: int = 1
    ) -> None:
        """Atomically increment/decrement the document count."""
        stmt = (
            update(Collection)
            .where(Collection.id == collection_id)
            .values(document_count=Collection.document_count + delta)
        )
        await session.execute(stmt)
        await session.flush()
