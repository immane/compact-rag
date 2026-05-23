"""FastAPI dependency injection functions — lazy-load all heavy components."""

from __future__ import annotations

from functools import lru_cache
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.config.settings import Settings, get_settings


@lru_cache()
def _cached_settings() -> Settings:
    return get_settings()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session, auto-closed after request."""
    from compact_rag.storage.db.engine import create_engine, create_session_factory

    settings = _cached_settings()
    engine = create_engine(settings.database)
    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


def get_llm_client():
    """Get configured LLM client (lazy import to avoid loading all providers)."""
    from compact_rag.generation.llm import LLMFactory

    return LLMFactory.create(_cached_settings().llm)


def get_embedding_service():
    """Get embedding service singleton (lazy import)."""
    from compact_rag.embedding.service import EmbeddingService

    return EmbeddingService(_cached_settings().embedding)


def get_vector_store():
    """Get vector store instance (lazy import)."""
    from compact_rag.storage.vector_store import VectorStore

    return VectorStore(
        _cached_settings().chromadb,
        get_embedding_service(),
    )


def get_storage_backend():
    """Get configured storage backend (lazy import)."""
    from compact_rag.storage.file_storage import get_storage_backend as _get

    return _get(_cached_settings().storage)


def get_prompt_manager():
    """Get prompt manager instance."""
    from compact_rag.generation.prompt import PromptManager

    return PromptManager()


def get_hybrid_retriever():
    """Get hybrid retriever instance."""
    from compact_rag.retrieval.retriever import HybridRetriever

    from compact_rag.retrieval.reranker import RerankerService
    from compact_rag.retrieval.sparse import BM25Retriever

    settings = _cached_settings()
    vector_store = get_vector_store()
    bm25 = BM25Retriever()
    reranker = RerankerService()

    return HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
        reranker=reranker,
        settings=settings.retrieval,
    )


def get_rag_pipeline():
    """Get full RAG pipeline instance."""
    from compact_rag.rag.pipeline import RAGPipeline

    return RAGPipeline(
        retriever=get_hybrid_retriever(),
        llm_client=get_llm_client(),
        prompt_manager=get_prompt_manager(),
    )
