"""StorageFile repository."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.storage.db.models import StorageFile
from compact_rag.storage.db.repository.base import BaseRepository


class StorageFileRepository(BaseRepository[StorageFile]):
    model = StorageFile

    async def list_by_document(
        self, session: AsyncSession, document_id: str
    ) -> list[StorageFile]:
        """List storage files for a document."""
        stmt = (
            select(StorageFile)
            .where(StorageFile.document_id == document_id)
            .order_by(StorageFile.created_at.desc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_type(
        self,
        session: AsyncSession,
        storage_type: str,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[StorageFile], int]:
        """List files by storage type (temp/persistent/archive)."""
        return await self.list(
            session, page=page, page_size=page_size, storage_type=storage_type
        )

    async def get_by_key(self, session: AsyncSession, storage_key: str) -> StorageFile | None:
        """Find a storage file by its storage key."""
        stmt = select(StorageFile).where(StorageFile.storage_key == storage_key)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
