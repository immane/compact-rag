"""Base repository with common CRUD operations."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

TModel = TypeVar("TModel", bound=DeclarativeBase)


class BaseRepository(Generic[TModel]):
    """Generic base repository with common CRUD operations."""

    model: type[TModel]

    async def create(self, session: AsyncSession, **kwargs: Any) -> TModel:
        """Create a new record.

        Args:
            session: Database session.
            **kwargs: Model field values.

        Returns:
            Created model instance.
        """
        instance = self.model(**kwargs)
        session.add(instance)
        await session.flush()
        return instance

    async def get_by_id(self, session: AsyncSession, id_val: str) -> TModel | None:
        """Get a record by primary key ID.

        Args:
            session: Database session.
            id_val: UUID primary key value.

        Returns:
            Model instance or None.
        """
        stmt = select(self.model).where(self.model.id == id_val)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def list(
        self,
        session: AsyncSession,
        page: int = 1,
        page_size: int = 20,
        **filters: Any,
    ) -> tuple[list[TModel], int]:
        """List records with pagination.

        Args:
            session: Database session.
            page: Page number (1-indexed).
            page_size: Items per page.
            **filters: Column=value filters.

        Returns:
            Tuple of (items, total_count).
        """
        stmt = select(self.model)

        for key, value in filters.items():
            if value is not None and hasattr(self.model, key):
                stmt = stmt.where(getattr(self.model, key) == value)

        # Count total
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await session.execute(count_stmt)
        total = total_result.scalar() or 0

        # Paginate
        offset = (page - 1) * page_size
        stmt = stmt.order_by(self.model.created_at.desc()).offset(offset).limit(page_size)

        result = await session.execute(stmt)
        items = list(result.scalars().all())

        return items, total

    async def update(
        self, session: AsyncSession, id_val: str, **kwargs: Any
    ) -> TModel | None:
        """Update a record by ID.

        Args:
            session: Database session.
            id_val: UUID primary key.
            **kwargs: Fields to update.

        Returns:
            Updated model instance or None.
        """
        instance = await self.get_by_id(session, id_val)
        if instance is None:
            return None

        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        await session.flush()
        return instance

    async def delete(self, session: AsyncSession, id_val: str) -> bool:
        """Delete a record by ID.

        Args:
            session: Database session.
            id_val: UUID primary key.

        Returns:
            True if deleted, False if not found.
        """
        instance = await self.get_by_id(session, id_val)
        if instance is None:
            return False

        await session.delete(instance)
        await session.flush()
        return True
