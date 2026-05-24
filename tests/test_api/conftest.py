"""Shared fixtures for test_api — ensures every test gets a fresh DB with tables."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.fixture(autouse=True)
def _init_test_db_tables(test_settings):
    """Create all ORM tables in the per-test temp SQLite DB before each test."""
    from compact_rag.storage.db.models import Base

    async def _create() -> None:
        engine = create_async_engine(test_settings.database.url, echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(_create())
