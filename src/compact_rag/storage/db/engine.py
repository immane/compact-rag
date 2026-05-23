"""SQLAlchemy async engine factory and session management."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from compact_rag.config.settings import DatabaseSettings


def create_engine(settings: DatabaseSettings) -> AsyncEngine:
    """Create an async SQLAlchemy engine from settings.

    Supports:
    - sqlite+aiosqlite:/// (dev, zero-config)
    - mysql+asyncmy:// (production)

    Args:
        settings: Database configuration.

    Returns:
        AsyncEngine instance.
    """
    connect_args: dict = {}

    if settings.url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_async_engine(
        settings.url,
        echo=settings.echo,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        connect_args=connect_args,
        # MySQL-specific pool_pre_ping to handle stale connections
        pool_pre_ping=True,
    )


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory.

    Args:
        engine: AsyncEngine instance.

    Returns:
        async_sessionmaker configured with expire_on_commit=False.
    """
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def get_session(engine: AsyncEngine | None = None):
    """FastAPI dependency: yield an async session and close it after use.

    Usage:
        @app.get("/")
        async def route(session: AsyncSession = Depends(get_session)):
            ...
    """
    if engine is None:
        from compact_rag.config.settings import get_settings

        settings = get_settings()
        engine = create_engine(settings.database)

    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
