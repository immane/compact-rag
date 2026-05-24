"""SQLAlchemy async engine factory and session management."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from pathlib import Path
import importlib
import sys

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
        # Ensure parent directory exists for SQLite paths so tests/CI
        # can create the DB file. URL format: sqlite+aiosqlite:///absolute/path
        try:
            path = settings.url.split(":::", 1)[1]
        except Exception:
            path = None
        if path:
            parent = Path(path).parent
            if not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)

    # Handle MySQL driver fallbacks: prefer aiomysql if asyncmy not installed
    if "mysql+asyncmy" in settings.url:
        try:
            importlib.import_module("asyncmy")
        except Exception:
            # asyncmy not available; try aiomysql instead
            try:
                importlib.import_module("aiomysql")
                settings = DatabaseSettings(**{**settings.model_dump(), "url": settings.url.replace("mysql+asyncmy", "mysql+aiomysql")})
            except Exception:
                # Leave URL as-is; create_async_engine will raise if driver missing
                pass

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
