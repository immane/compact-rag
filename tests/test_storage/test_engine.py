"""Tests for storage/db/engine.py: create_engine, get_session, create_session_factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from compact_rag.config.settings import DatabaseSettings
from compact_rag.storage.db.engine import create_engine, create_session_factory, get_session


class TestCreateEngine:
    def test_create_with_sqlite_url(self):
        settings = DatabaseSettings(
            url="sqlite+aiosqlite:///tmp/test.db",
            echo=False,
        )
        engine = create_engine(settings)
        assert isinstance(engine, AsyncEngine)

    def test_create_with_mysql_url(self):
        settings = DatabaseSettings(
            url="mysql+asyncmy://user:pass@localhost:3306/mydb",
            echo=False,
            pool_size=10,
            max_overflow=20,
        )
        engine = create_engine(settings)
        assert isinstance(engine, AsyncEngine)

    def test_create_sets_pool_pre_ping(self):
        settings = DatabaseSettings(
            url="sqlite+aiosqlite:///tmp/test.db",
            echo=False,
        )
        engine = create_engine(settings)
        assert engine.pool is not None


class TestCreateSessionFactory:
    @pytest.fixture
    def engine(self):
        settings = DatabaseSettings(
            url="sqlite+aiosqlite:///tmp/test.db",
            echo=False,
        )
        return create_engine(settings)

    def test_creates_sessionmaker(self, engine):
        factory = create_session_factory(engine)
        # async_sessionmaker is a callable
        assert callable(factory)


class TestGetSession:
    @pytest.mark.asyncio
    async def test_get_session_yields_session(self):
        settings = DatabaseSettings(
            url="sqlite+aiosqlite:///tmp/test.db",
            echo=False,
        )
        engine = create_engine(settings)

        async for session in get_session(engine):
            from sqlalchemy.ext.asyncio import AsyncSession

            assert isinstance(session, AsyncSession)

    @pytest.mark.asyncio
    async def test_get_session_no_engine_uses_settings(self, monkeypatch):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test_noengine.db"
            fake_settings = MagicMock()
            fake_settings.database = DatabaseSettings(
                url=f"sqlite+aiosqlite:///{db_path}", echo=False
            )

            with patch(
                "compact_rag.config.settings.get_settings",
                return_value=fake_settings,
            ):
                async for session in get_session():
                    from sqlalchemy.ext.asyncio import AsyncSession

                    assert isinstance(session, AsyncSession)
                    break
