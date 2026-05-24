"""Collection management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.api.deps import get_db_session, verify_api_key
from compact_rag.api.schemas import (
    CollectionCreateRequest,
    CollectionResponse,
    PaginatedResponse,
    PaginationMeta,
)
from compact_rag.common.exceptions import ConfigurationError
from compact_rag.common.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Collections"])


@router.get("/collections", response_model=PaginatedResponse)
async def list_collections(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
):
    """List all document collections."""
    from compact_rag.storage.db.repository.collection import CollectionRepository

    repo = CollectionRepository()
    if hasattr(repo, "list_with_realtime_document_count"):
        results_with_count, total = await repo.list_with_realtime_document_count(
            session,
            page=page,
            page_size=page_size,
        )
    else:
        # Backward-compatible fallback for tests/mocks that only implement list().
        collections, total = await repo.list(session, page=page, page_size=page_size)
        results_with_count = [(c, c.document_count or 0) for c in collections]
    total_pages = max(1, (total + page_size - 1) // page_size) if total > 0 else 0

    return PaginatedResponse(
        data=[
            CollectionResponse(
                id=str(c.id),
                name=c.name,
                description=c.description or "",
                embedding_model=c.embedding_model or "",
                chunk_size=c.chunk_size or 500,
                chunk_overlap=c.chunk_overlap or 50,
                document_count=doc_count,
                created_at=str(c.created_at) if c.created_at else None,
                updated_at=str(c.updated_at) if c.updated_at else None,
            )
            for c, doc_count in results_with_count
        ],
        pagination=PaginationMeta(
            page=page, page_size=page_size, total=total, total_pages=total_pages
        ),
    )


@router.post("/collections", response_model=CollectionResponse)
async def create_collection(
    request: CollectionCreateRequest,
    session: AsyncSession = Depends(get_db_session),
    _api_key: str | None = Depends(verify_api_key),
):
    """Create a new document collection."""
    from compact_rag.storage.db.repository.collection import CollectionRepository

    repo = CollectionRepository()

    try:
        collection = await repo.create(
            session,
            name=request.name,
            description=request.description,
            embedding_model=request.embedding_model,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        await session.commit()

        return CollectionResponse(
            id=str(collection.id),
            name=collection.name,
            description=collection.description or "",
            embedding_model=collection.embedding_model or "",
            chunk_size=collection.chunk_size or 500,
            chunk_overlap=collection.chunk_overlap or 50,
            document_count=0,
            created_at=str(collection.created_at),
            updated_at=str(collection.updated_at),
        )
    except Exception as e:
        await session.rollback()
        raise ConfigurationError(f"Failed to create collection: {e}") from e


@router.delete("/collections/{name}")
async def delete_collection(
    name: str,
    session: AsyncSession = Depends(get_db_session),
    _api_key: str | None = Depends(verify_api_key),
):
    """Delete a collection by name."""
    from compact_rag.storage.db.repository.collection import CollectionRepository

    repo = CollectionRepository()
    collection = await repo.get_by_name(session, name)
    if collection is None:
        from compact_rag.common.exceptions import FileNotFoundError

        raise FileNotFoundError(f"Collection not found: {name}")

    await repo.delete(session, collection.id)
    await session.commit()
    return {"status": "deleted", "name": name}
