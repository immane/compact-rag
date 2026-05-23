"""DocumentChunk repository."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.storage.db.models import DocumentChunk
from compact_rag.storage.db.repository.base import BaseRepository


class ChunkRepository(BaseRepository[DocumentChunk]):
    model = DocumentChunk

    async def list_by_document(
        self, session: AsyncSession, document_id: str
    ) -> list[DocumentChunk]:
        """List all chunks for a document, ordered by chunk_index."""
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_chroma_id(
        self, session: AsyncSession, chroma_id: str
    ) -> DocumentChunk | None:
        """Find a chunk by its ChromaDB ID."""
        stmt = select(DocumentChunk).where(DocumentChunk.chroma_id == chroma_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def delete_by_document(self, session: AsyncSession, document_id: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        chunks = await self.list_by_document(session, document_id)
        for chunk in chunks:
            await session.delete(chunk)
        return len(chunks)
