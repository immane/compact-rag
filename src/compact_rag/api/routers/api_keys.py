"""API key management endpoints."""

from __future__ import annotations

import hashlib
import secrets

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.api.deps import get_db_session, verify_api_key
from compact_rag.api.schemas import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyResponse,
    ApiKeyUpdateRequest,
    PaginatedResponse,
    PaginationMeta,
)
from compact_rag.common.exceptions import ConfigurationError, FileNotFoundError
from compact_rag.common.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(
    tags=["API Keys"],
    dependencies=[Depends(verify_api_key)],
)


def _generate_api_key() -> tuple[str, str]:
    """Generate API key and hash. Returns (raw_key, hash)."""
    raw = f"cr-{secrets.token_hex(32)}"
    key_hash = hashlib.sha256(raw.encode()).hexdigest()
    return raw, key_hash


@router.get("/api-keys", response_model=PaginatedResponse)
async def list_api_keys(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
):
    """List all API keys (hash masked)."""
    from compact_rag.storage.db.repository.api_key import ApiKeyRepository

    repo = ApiKeyRepository()
    results, total = await repo.list(session, page=page, page_size=page_size)
    total_pages = max(1, (total + page_size - 1) // page_size) if total > 0 else 0

    return PaginatedResponse(
        data=[
            ApiKeyResponse(
                id=str(k.id),
                name=k.name or "",
                key_prefix=k.key_hash[:8] + "..." if k.key_hash else "",
                permissions=k.permissions or ["read"],
                is_active=k.is_active or False,
                expires_at=str(k.expires_at) if k.expires_at else None,
                created_at=str(k.created_at) if k.created_at else None,
            )
            for k in results
        ],
        pagination=PaginationMeta(
            page=page, page_size=page_size, total=total, total_pages=total_pages
        ),
    )


@router.post("/api-keys", response_model=ApiKeyCreateResponse)
async def create_api_key(
    request: ApiKeyCreateRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Create a new API key. Returns the raw key only once."""
    from compact_rag.storage.db.repository.api_key import ApiKeyRepository

    raw_key, key_hash = _generate_api_key()
    repo = ApiKeyRepository()

    try:
        key = await repo.create(
            session,
            name=request.name,
            key_hash=key_hash,
            permissions=request.permissions,
            is_active=True,
        )
        await session.commit()

        return ApiKeyCreateResponse(
            id=str(key.id),
            name=key.name or "",
            key=raw_key,
            permissions=key.permissions or ["read"],
            created_at=str(key.created_at) if key.created_at else None,
        )
    except Exception as e:
        await session.rollback()
        raise ConfigurationError(f"Failed to create API key: {e}") from e


@router.patch("/api-keys/{key_id}", response_model=ApiKeyResponse)
async def update_api_key(
    key_id: str,
    request: ApiKeyUpdateRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Update API key (activate/deactivate, rename, change permissions)."""
    from compact_rag.storage.db.repository.api_key import ApiKeyRepository

    repo = ApiKeyRepository()
    key = await repo.get_by_id(session, key_id)
    if key is None:
        raise FileNotFoundError(f"API key not found: {key_id}")

    updates = {}
    if request.is_active is not None:
        updates["is_active"] = request.is_active
    if request.name is not None:
        updates["name"] = request.name
    if request.permissions is not None:
        updates["permissions"] = request.permissions

    key = await repo.update(session, key_id, **updates)
    await session.commit()

    return ApiKeyResponse(
        id=str(key.id),
        name=key.name or "",
        key_prefix=key.key_hash[:8] + "..." if key.key_hash else "",
        permissions=key.permissions or ["read"],
        is_active=key.is_active or False,
        expires_at=str(key.expires_at) if key.expires_at else None,
        created_at=str(key.created_at) if key.created_at else None,
    )


@router.delete("/api-keys/{key_id}")
async def delete_api_key(
    key_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Delete an API key."""
    from compact_rag.storage.db.repository.api_key import ApiKeyRepository

    repo = ApiKeyRepository()
    key = await repo.get_by_id(session, key_id)
    if key is None:
        raise FileNotFoundError(f"API key not found: {key_id}")

    await repo.delete(session, key_id)
    await session.commit()
    return {"status": "deleted", "key_id": key_id}
