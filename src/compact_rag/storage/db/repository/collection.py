"""Collection repository."""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.storage.db.models import Collection
from compact_rag.storage.db.repository.base import BaseRepository


class CollectionRepository(BaseRepository[Collection]):
    model = Collection

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
