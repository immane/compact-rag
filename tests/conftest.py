"""Global pytest fixtures for compact-rag tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(scope="session")
def test_dir():
    """Create a temporary test directory."""
    with tempfile.TemporaryDirectory(prefix="compact_rag_test_") as tmp:
        yield tmp


@pytest.fixture
def test_settings():
    """Create test Settings with in-memory SQLite and temp directories."""
    from compact_rag.config.settings import (
        ChromaDBSettings,
        DatabaseSettings,
        EmbeddingSettings,
        IngestionSettings,
        LLMSettings,
        RetrievalSettings,
        Settings,
        StorageSettings,
        AdminSettings,
        LocalStorageSettings,
    )

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        chroma_path = os.path.join(tmp, "chromadb")
        storage_path = os.path.join(tmp, "storage")

        yield Settings(
            database=DatabaseSettings(
                url=f"sqlite+aiosqlite:///{db_path}",
                echo=False,
            ),
            embedding=EmbeddingSettings(
                model_name="all-MiniLM-L6-v2",
                device="cpu",
            ),
            chromadb=ChromaDBSettings(
                persist_directory=chroma_path,
                collection_name="test_default",
            ),
            retrieval=RetrievalSettings(
                dense_top_k=10,
                sparse_top_k=10,
                fusion_top_k=5,
                rerank_top_k=3,
            ),
            llm=LLMSettings(
                provider="openai",
                model="gpt-4o-mini",
            ),
            ingestion=IngestionSettings(
                chunk_size=200,
                chunk_overlap=20,
            ),
            storage=StorageSettings(
                backend="local",
                local=LocalStorageSettings(
                    root_dir=storage_path,
                    base_url="http://localhost:8000/files",
                ),
            ),
            admin=AdminSettings(host="127.0.0.1", port=8501),
            log_level="DEBUG",
        )


@pytest.fixture
async def test_db_engine(test_settings):
    """Create a test async SQLAlchemy engine."""
    engine = create_async_engine(
        test_settings.database.url,
        echo=False,
    )
    from compact_rag.storage.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def test_session(test_db_engine):
    """Provide an async SQLAlchemy session."""
    async_session = sessionmaker(
        test_db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session


@pytest.fixture
def mock_embedding_service(mocker):
    """Mock EmbeddingService that returns fixed-dimension random vectors."""
    import numpy as np

    mock = mocker.MagicMock()
    mock.encode = mocker.AsyncMock(
        return_value=np.random.randn(3, 384).astype(np.float32)
    )
    mock.encode_query = mocker.AsyncMock(
        return_value=np.random.randn(384).astype(np.float32)
    )
    mock.dimension = 384
    return mock


@pytest.fixture
def mock_llm_client(mocker):
    """Mock LLM client returning fixed ChatResponse."""
    from compact_rag.storage.schema import ChatResponse

    mock = mocker.MagicMock()
    mock.chat = mocker.AsyncMock(
        return_value=ChatResponse(
            content="This is a test response.",
            token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            model="test-model",
        )
    )
    mock.chat_stream = mocker.AsyncMock()
    return mock


@pytest.fixture
def mock_chromadb_client(mocker):
    """Mock ChromaDB PersistentClient."""
    mock_client = mocker.MagicMock()
    mock_collection = mocker.MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.list_collections.return_value = []
    return mock_client


@pytest.fixture
def sample_text():
    """Standard test text for chunking/embedding tests."""
    return (
        "人工智能是计算机科学的一个重要分支。它研究如何让计算机模拟人类智能。\n\n"
        "机器学习是人工智能的核心方法之一。通过数据训练模型，使其具备预测能力。\n\n"
        "深度学习使用多层神经网络来处理复杂的模式识别任务。\n\n"
        "自然语言处理让计算机理解和生成人类语言。\n\n"
        "计算机视觉使机器能够理解和分析图像和视频内容。"
    )


@pytest.fixture
def sample_chunks(sample_text):
    """Standard test chunks."""
    from compact_rag.storage.schema import DocumentChunk

    paragraphs = sample_text.split("\n\n")
    return [
        DocumentChunk(
            content=p.strip(),
            chunk_index=i,
            page_number=1,
            is_table=False,
            token_count=len(p.split()),
            content_hash=f"hash_{i}",
            metadata={},
        )
        for i, p in enumerate(paragraphs)
        if p.strip()
    ]


@pytest.fixture
def test_fixtures_dir():
    """Path to test fixtures directory."""
    path = Path(__file__).parent / "fixtures"
    path.mkdir(exist_ok=True)
    return path


def patch_cached_settings(monkeypatch, test_settings) -> None:
    """Monkeypatch _cached_settings so get_db_session() resolves to test DB.

    Call this in any fixture or test that uses create_app(settings=test_settings)
    without overriding get_db_session.  This prevents the default (production)
    database URL from leaking into tests.
    """
    from compact_rag.api.deps import _cached_settings

    _cached_settings.cache_clear()
    monkeypatch.setattr(
        "compact_rag.api.deps._cached_settings",
        lambda: test_settings,
    )


async def _init_test_db(url: str) -> None:
    """Create all ORM tables in the test database."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from compact_rag.storage.db.models import Base

    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


async def _drop_test_db(url: str) -> None:
    """Drop all ORM tables from the test database."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from compact_rag.storage.db.models import Base

    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
