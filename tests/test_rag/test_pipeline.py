from __future__ import annotations

import pytest

from compact_rag.generation.prompt import PromptManager
from compact_rag.rag.pipeline import RAGPipeline
from compact_rag.storage.schema import ChatResponse


class _FakeRetriever:
    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int,
        use_hybrid_search: bool = True,
        use_rerank: bool = True,
    ):
        return []


class _FakeLLM:
    async def chat(self, messages):
        return ChatResponse(
            content="ok from llm",
            token_usage={"total_tokens": 12},
            model="test-model",
        )

    async def chat_stream(self, messages):
        if False:
            yield ""


@pytest.mark.unit
def test_build_messages_uses_prompt_manager_render_api():
    pipeline = RAGPipeline(
        retriever=_FakeRetriever(),
        llm_client=_FakeLLM(),
        prompt_manager=PromptManager(),
    )

    messages = pipeline._build_messages("hello", [{"role": "assistant", "content": "hi"}])

    assert messages[0]["role"] == "system"
    assert "智能知识库助手" in messages[0]["content"]
    assert messages[-1]["content"] == "hello"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_accepts_chatresponse_model_result():
    pipeline = RAGPipeline(
        retriever=_FakeRetriever(),
        llm_client=_FakeLLM(),
        prompt_manager=PromptManager(),
    )

    result = await pipeline.query(question="hello", collection="default", top_k=1)

    assert result.answer == "ok from llm"
    assert result.token_usage["total_tokens"] == 12


@pytest.mark.unit
def test_augment_messages_targets_latest_user_turn():
    pipeline = RAGPipeline(
        retriever=_FakeRetriever(),
        llm_client=_FakeLLM(),
        prompt_manager=PromptManager(),
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
    ]

    augmented = pipeline._augment_messages(messages, "ctx")

    assert "Retrieved Context" not in augmented[1]["content"]
    assert "Retrieved Context" in augmented[-1]["content"]
    assert "User Question: new question" in augmented[-1]["content"]
