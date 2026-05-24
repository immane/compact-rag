"""ApiKey repository."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.storage.db.models import ApiKey
from compact_rag.storage.db.repository.base import BaseRepository


class ApiKeyRepository(BaseRepository[ApiKey]):
    model = ApiKey

    async def get_by_hash(self, session: AsyncSession, key_hash: str) -> ApiKey | None:
        """Find an API key by its SHA256 hash."""
        stmt = select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def set_active(
        self, session: AsyncSession, key_id: str, is_active: bool
    ) -> ApiKey | None:
        """Activate or deactivate an API key."""
        return await self.update(session, key_id, is_active=is_active)
