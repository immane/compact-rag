"""Conversation history endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.api.deps import get_db_session
from compact_rag.api.schemas import (
    ConversationDetailResponse,
    ConversationResponse,
    MessageResponse,
    PaginatedResponse,
    PaginationMeta,
)
from compact_rag.common.exceptions import FileNotFoundError
from compact_rag.common.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Conversations"])


@router.get("/conversations", response_model=PaginatedResponse)
async def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
):
    """List conversation history."""
    from compact_rag.storage.db.repository.conversation import ConversationRepository

    repo = ConversationRepository()
    results, total = await repo.list(session, page=page, page_size=page_size)
    total_pages = max(1, (total + page_size - 1) // page_size) if total > 0 else 0

    return PaginatedResponse(
        data=[
            ConversationResponse(
                id=str(c.id),
                collection_id=str(c.collection_id) if c.collection_id else None,
                title=c.title or "New Conversation",
                model=c.model or "",
                message_count=c.message_count or 0,
                created_at=str(c.created_at) if c.created_at else None,
                updated_at=str(c.updated_at) if c.updated_at else None,
            )
            for c in results
        ],
        pagination=PaginationMeta(
            page=page, page_size=page_size, total=total, total_pages=total_pages
        ),
    )


@router.get("/conversations/{conv_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conv_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Get conversation details with message history."""
    from compact_rag.storage.db.repository.conversation import ConversationRepository

    repo = ConversationRepository()
    conv = await repo.get_by_id(session, conv_id)
    if conv is None:
        raise FileNotFoundError(f"Conversation not found: {conv_id}")

    messages = await repo.list_messages(session, conv_id)

    return ConversationDetailResponse(
        id=str(conv.id),
        collection_id=str(conv.collection_id) if conv.collection_id else None,
        title=conv.title or "New Conversation",
        model=conv.model or "",
        message_count=conv.message_count or 0,
        messages=[
            MessageResponse(
                id=str(m.id),
                conversation_id=str(m.conversation_id),
                role=m.role or "",
                content=m.content or "",
                sources=m.sources,
                token_count=m.token_count,
                latency_ms=m.latency_ms,
                created_at=str(m.created_at) if m.created_at else None,
            )
            for m in messages
        ],
        created_at=str(conv.created_at) if conv.created_at else None,
        updated_at=str(conv.updated_at) if conv.updated_at else None,
    )


@router.delete("/conversations/{conv_id}")
async def delete_conversation(
    conv_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Delete a conversation and its messages."""
    from compact_rag.storage.db.repository.conversation import ConversationRepository

    repo = ConversationRepository()
    conv = await repo.get_by_id(session, conv_id)
    if conv is None:
        raise FileNotFoundError(f"Conversation not found: {conv_id}")

    await repo.delete(session, conv_id)
    await session.commit()
    return {"status": "deleted", "conversation_id": conv_id}
