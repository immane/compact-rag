"""Extra RAGPipeline tests: collection_name, stream error during retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from compact_rag.generation.prompt import PromptManager
from compact_rag.rag.pipeline import RAGPipeline
from compact_rag.storage.schema import ChatResponse, SearchResult


class _FakeRetriever:
    def __init__(self, results=None, raise_on=None):
        self.results = results or []
        self.raise_on = raise_on
        self.last_calls = []

    async def retrieve(self, query, collection, top_k, use_hybrid_search=True, use_rerank=True):
        self.last_calls.append({
            "query": query,
            "collection": collection,
            "top_k": top_k,
            "use_hybrid_search": use_hybrid_search,
            "use_rerank": use_rerank,
        })
        if self.raise_on == "retrieve":
            raise RuntimeError("retrieval stream error")
        return self.results


class _FakeLLM:
    def __init__(self, response=None, stream_chunks=None):
        self._response = response
        self._stream_chunks = stream_chunks or []

    async def chat(self, messages):
        if self._response:
            return self._response
        return ChatResponse(
            content="default answer",
            token_usage={"total_tokens": 10},
            model="test-model",
        )

    async def chat_stream(self, messages):
        for chunk in self._stream_chunks:
            yield chunk


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_with_collection_name_passed_to_retriever():
    retriever = _FakeRetriever(
        results=[SearchResult(id="d1", content="test", score=0.9, metadata={})],
    )
    llm = _FakeLLM(ChatResponse(content="answer", token_usage={"total_tokens": 5}, model="m"))
    pipeline = RAGPipeline(
        retriever=retriever,
        llm_client=llm,
        prompt_manager=PromptManager(),
    )
    await pipeline.query(question="q", collection="my-collection")
    assert retriever.last_calls[-1]["collection"] == "my-collection"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_stream_with_error_during_retrieval():
    retriever = _FakeRetriever(raise_on="retrieve")
    llm = _FakeLLM(stream_chunks=[])

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_client=llm,
        prompt_manager=PromptManager(),
    )
    with pytest.raises(RuntimeError, match="retrieval stream error"):
        async for _ in pipeline.query_stream(question="fail"):
            pass


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_stream_with_collection_name():
    retriever = _FakeRetriever(results=[])
    llm = _FakeLLM(stream_chunks=["c1", "c2"])

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_client=llm,
        prompt_manager=PromptManager(),
    )
    chunks = []
    async for c in pipeline.query_stream(question="q", collection="special"):
        chunks.append(c)

    assert chunks == ["c1", "c2"]
    assert retriever.last_calls[-1]["collection"] == "special"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_stream_saves_conversation_with_collection_name():
    msg_repo = MagicMock()
    msg_repo.create = AsyncMock(return_value=MagicMock())
    conv_repo = MagicMock()
    conv_repo.increment_message_count = AsyncMock()

    retriever = _FakeRetriever(
        results=[SearchResult(id="x", content="ctx", score=0.9, metadata={})],
    )
    llm = _FakeLLM(stream_chunks=["hello", " world"])

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_client=llm,
        prompt_manager=PromptManager(),
        conversation_repo=conv_repo,
        message_repo=msg_repo,
    )
    collected = []
    async for c in pipeline.query_stream(
        question="save me",
        collection="persist-col",
        conversation_id="conv-99",
    ):
        collected.append(c)

    assert collected == ["hello", " world"]
    assert retriever.last_calls[-1]["collection"] == "persist-col"
