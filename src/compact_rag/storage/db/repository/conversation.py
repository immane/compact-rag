"""Conversation and Message repository."""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.storage.db.models import Conversation, Message
from compact_rag.storage.db.repository.base import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):
    model = Conversation

    async def list_messages(
        self, session: AsyncSession, conversation_id: str
    ) -> list[Message]:
        """Get all messages for a conversation, ordered by creation time."""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def increment_message_count(
        self, session: AsyncSession, conversation_id: str, delta: int = 1
    ) -> None:
        """Atomically increment the message count."""
        stmt = (
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(message_count=Conversation.message_count + delta)
        )
        await session.execute(stmt)
        await session.flush()


class MessageRepository(BaseRepository[Message]):
    model = Message

    async def list_by_conversation(
        self, session: AsyncSession, conversation_id: str
    ) -> list[Message]:
        """List all messages for a conversation."""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())
