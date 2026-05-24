from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from compact_rag.generation.prompt import PromptManager
from compact_rag.rag.pipeline import RAGPipeline
from compact_rag.storage.schema import RAGCitation, RAGResponse, SearchResult, ChatResponse


class _FakeMessageRepo:
    def __init__(self):
        self.created_messages = []
        self.create = AsyncMock(side_effect=self._create)

    async def _create(self, _session, **kwargs):
        self.created_messages.append(kwargs)
        return MagicMock()


class _FakeConversationRepo:
    def __init__(self):
        self.increment_calls = []
        self.increment_message_count = AsyncMock(side_effect=self._increment)

    async def _increment(self, _session, conv_id, count):
        self.increment_calls.append((conv_id, count))


def _fake_search_result(chunk_id, content, score=0.8, **meta):
    return SearchResult(
        id=chunk_id,
        content=content,
        score=score,
        metadata={
            "filename": f"doc_{chunk_id}.pdf",
            "chunk_index": 0,
            "page_number": 1,
            **meta,
        },
    )


class FakeRetriever:
    def __init__(self, results=None):
        self.results = results or []
        self.last_query = None
        self.last_collection = None
        self.last_top_k = None
        self.last_use_hybrid = None
        self.last_use_rerank = None

    async def retrieve(self, query, collection, top_k, use_hybrid_search=True, use_rerank=True):
        self.last_query = query
        self.last_collection = collection
        self.last_top_k = top_k
        self.last_use_hybrid = use_hybrid_search
        self.last_use_rerank = use_rerank
        return self.results


class FakeLLM:
    def __init__(self, response=None):
        self._response = response

    async def chat(self, messages):
        return self._response if self._response else ChatResponse(
            content="default answer",
            token_usage={"total_tokens": 5},
            model="test-model",
        )

    async def chat_stream(self, messages):
        if self._response and isinstance(self._response, list):
            for chunk in self._response:
                yield chunk


@pytest.mark.unit
class TestRAGPipelineInit:
    def test_all_dependencies_wired_correctly(self):
        retriever = FakeRetriever()
        llm = FakeLLM()
        prompt_mgr = PromptManager()
        conv_repo = _FakeConversationRepo()
        msg_repo = _FakeMessageRepo()
        tool_engine = MagicMock()

        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=llm,
            prompt_manager=prompt_mgr,
            conversation_repo=conv_repo,
            message_repo=msg_repo,
            tool_engine=tool_engine,
        )

        assert pipeline.retriever is retriever
        assert pipeline.llm_client is llm
        assert pipeline.prompt_manager is prompt_mgr
        assert pipeline.conversation_repo is conv_repo
        assert pipeline.message_repo is msg_repo
        assert pipeline.tool_engine is tool_engine

    def test_optional_dependencies_default_to_none(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(),
            llm_client=FakeLLM(),
            prompt_manager=PromptManager(),
        )
        assert pipeline.conversation_repo is None
        assert pipeline.message_repo is None
        assert pipeline.tool_engine is None


@pytest.mark.unit
@pytest.mark.asyncio
class TestRAGPipelineQuery:
    async def test_query_returns_ragresponse(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever([_fake_search_result("d1", "some content")]),
            llm_client=FakeLLM(ChatResponse(
                content="answer", token_usage={"total_tokens": 10}, model="m"
            )),
            prompt_manager=PromptManager(),
        )
        result = await pipeline.query(question="q1")
        assert isinstance(result, RAGResponse)
        assert result.answer == "answer"
        assert result.token_usage["total_tokens"] == 10
        assert result.id.startswith("rag-")
        assert result.retrieval_latency_ms >= 0
        assert result.generation_latency_ms >= 0

    async def test_query_stream_true_still_uses_chat(self):
        llm = FakeLLM(ChatResponse(content="stream-mode", token_usage={"total_tokens": 3}, model="x"))
        pipeline = RAGPipeline(
            retriever=FakeRetriever(),
            llm_client=llm,
            prompt_manager=PromptManager(),
        )
        result = await pipeline.query(question="q", stream=True)
        assert result.answer == "stream-mode"

    async def test_query_retrieval_empty_still_returns_answer(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever([]),
            llm_client=FakeLLM(ChatResponse(content="no docs", token_usage={}, model="m")),
            prompt_manager=PromptManager(),
        )
        result = await pipeline.query(question="no data")
        assert result.answer == "no docs"
        assert result.citations == []

    async def test_query_with_tool_engine_calls_execute(self):
        tool_engine = MagicMock()
        tool_engine.execute = AsyncMock(return_value=([], {"tool": "result"}))

        pipeline = RAGPipeline(
            retriever=FakeRetriever(),
            llm_client=FakeLLM(),
            prompt_manager=PromptManager(),
            tool_engine=tool_engine,
        )
        result = await pipeline.query(question="use tool")

        tool_engine.execute.assert_awaited_once()
        assert result.answer == "default answer"

    async def test_query_tool_engine_error_is_handled(self):
        tool_engine = MagicMock()
        tool_engine.execute = AsyncMock(side_effect=RuntimeError("tool boom"))

        pipeline = RAGPipeline(
            retriever=FakeRetriever(),
            llm_client=FakeLLM(),
            prompt_manager=PromptManager(),
            tool_engine=tool_engine,
        )
        result = await pipeline.query(question="use tool")
        assert result.answer == "default answer"

    async def test_query_llm_returns_dict_not_chatresponse(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(),
            llm_client=FakeLLM({"content": "dict answer", "token_usage": {"total_tokens": 7}}),
            prompt_manager=PromptManager(),
        )
        result = await pipeline.query(question="dict test")
        assert result.answer == "dict answer"
        assert result.token_usage["total_tokens"] == 7

    async def test_query_llm_returns_arbitrary_object(self):
        class WeirdObj:
            def __str__(self):
                return "weird string"
        pipeline = RAGPipeline(
            retriever=FakeRetriever(),
            llm_client=FakeLLM(WeirdObj()),
            prompt_manager=PromptManager(),
        )
        result = await pipeline.query(question="weird")
        assert result.answer == "weird string"

    async def test_query_retrieval_error_propagates(self):
        retriever = FakeRetriever()
        retriever.retrieve = AsyncMock(side_effect=RuntimeError("retrieval failed"))

        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=FakeLLM(),
            prompt_manager=PromptManager(),
        )
        with pytest.raises(RuntimeError, match="retrieval failed"):
            await pipeline.query(question="error")

    async def test_query_save_conversation_error_still_returns_response(self):
        msg_repo = _FakeMessageRepo()
        msg_repo.create = AsyncMock(side_effect=RuntimeError("db down"))

        conv_repo = _FakeConversationRepo()
        conv_repo.increment_message_count = AsyncMock()

        pipeline = RAGPipeline(
            retriever=FakeRetriever(),
            llm_client=FakeLLM(),
            prompt_manager=PromptManager(),
            conversation_repo=conv_repo,
            message_repo=msg_repo,
        )
        result = await pipeline.query(question="save", conversation_id="conv-1")
        assert result.answer == "default answer"

    async def test_query_saves_conversation_and_messages(self):
        msg_repo = _FakeMessageRepo()
        conv_repo = _FakeConversationRepo()

        pipeline = RAGPipeline(
            retriever=FakeRetriever([_fake_search_result("d99", "c", score=0.9)]),
            llm_client=FakeLLM(ChatResponse(
                content="saved answer",
                token_usage={"total_tokens": 20},
                model="m",
            )),
            prompt_manager=PromptManager(),
            conversation_repo=conv_repo,
            message_repo=msg_repo,
        )
        await pipeline.query(question="save this", conversation_id="conv-1")

        assert len(msg_repo.created_messages) == 2
        roles = [m["role"] for m in msg_repo.created_messages]
        assert roles == ["user", "assistant"]
        assert msg_repo.created_messages[0]["content"] == "save this"
        assert msg_repo.created_messages[1]["content"] == "saved answer"
        assert len(msg_repo.created_messages[1].get("sources", [])) == 1
        assert conv_repo.increment_calls == [("conv-1", 2)]

    async def test_query_no_conversation_id_skips_save(self):
        msg_repo = _FakeMessageRepo()
        conv_repo = _FakeConversationRepo()

        pipeline = RAGPipeline(
            retriever=FakeRetriever(),
            llm_client=FakeLLM(),
            prompt_manager=PromptManager(),
            conversation_repo=conv_repo,
            message_repo=msg_repo,
        )
        result = await pipeline.query(question="no save", conversation_id=None)
        assert result.answer == "default answer"
        assert len(msg_repo.created_messages) == 0

    async def test_query_passes_collection_name_to_retriever(self):
        retriever = FakeRetriever()
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=FakeLLM(),
            prompt_manager=PromptManager(),
        )
        await pipeline.query(question="col test", collection="custom-col")
        assert retriever.last_collection == "custom-col"

    async def test_query_passes_top_k_params_to_retriever(self):
        retriever = FakeRetriever()
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=FakeLLM(),
            prompt_manager=PromptManager(),
        )
        await pipeline.query(question="tk", top_k=5, use_hybrid_search=False, use_rerank=False)
        assert retriever.last_top_k == 5
        assert retriever.last_use_hybrid is False
        assert retriever.last_use_rerank is False


@pytest.mark.unit
class TestBuildContext:
    def test_empty_results_returns_placeholder(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(), llm_client=FakeLLM(), prompt_manager=PromptManager()
        )
        ctx = pipeline._build_context([])
        assert ctx == "No relevant documents found."

    def test_multiple_documents_formatted_correctly(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(), llm_client=FakeLLM(), prompt_manager=PromptManager()
        )
        results = [
            _fake_search_result("d1", "content one", filename="a.pdf"),
            _fake_search_result("d2", "content two", filename="b.pdf"),
        ]
        ctx = pipeline._build_context(results)
        assert "[Document 1] (Source: a.pdf)" in ctx
        assert "content one" in ctx
        assert "[Document 2] (Source: b.pdf)" in ctx
        assert "content two" in ctx
        assert "---" in ctx

    def test_fallback_content_from_str(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(), llm_client=FakeLLM(), prompt_manager=PromptManager()
        )
        results = [_fake_search_result("d1", "hello")]
        results[0].metadata = {}
        ctx = pipeline._build_context(results)
        assert "[Document 1] (Source: unknown)" in ctx
        assert "hello" in ctx


@pytest.mark.unit
class TestBuildCitations:
    def test_correct_citation_objects(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(), llm_client=FakeLLM(), prompt_manager=PromptManager()
        )
        results = [
            _fake_search_result("id-1", "abcdefghij" * 30, score=0.95,
                                chunk_index=3, page_number=5),
        ]
        citations = pipeline._build_citations(results)
        assert len(citations) == 1
        c = citations[0]
        assert isinstance(c, RAGCitation)
        assert c.doc_id == "id-1"
        assert c.filename == "doc_id-1.pdf"
        assert c.score == 0.95
        assert c.chunk_index == 3
        assert c.page_number == 5
        assert len(c.content_snippet) == 200

    def test_missing_metadata_defaults(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(), llm_client=FakeLLM(), prompt_manager=PromptManager()
        )
        results = [_fake_search_result("x", "short")]
        results[0].metadata = {}
        citations = pipeline._build_citations(results)
        c = citations[0]
        assert c.filename == ""
        assert c.score == 0.8
        assert c.chunk_index == 0


@pytest.mark.unit
class TestBuildMessages:
    def test_uses_prompt_manager_system_prompt(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(), llm_client=FakeLLM(), prompt_manager=PromptManager()
        )
        messages = pipeline._build_messages("hi", [])
        assert messages[0]["role"] == "system"
        assert "智能知识库助手" in messages[0]["content"]
        assert messages[-1] == {"role": "user", "content": "hi"}

    def test_truncates_long_history_to_last_20(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(), llm_client=FakeLLM(), prompt_manager=PromptManager()
        )
        history = [{"role": "user", "content": f"msg-{i}"} for i in range(30)]
        messages = pipeline._build_messages("final", history)

        user_msgs = [m for m in messages if m["role"] == "user"]
        assert user_msgs[0]["content"] == "msg-10"
        assert user_msgs[-1]["content"] == "final"
        assert len(user_msgs) == 21

    def test_passes_conversation_history_to_messages(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(), llm_client=FakeLLM(), prompt_manager=PromptManager()
        )
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        messages = pipeline._build_messages("q2", history)
        assert messages == [
            {"role": "system", "content": messages[0]["content"]},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]

    def test_history_messages_default_role_to_user(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(), llm_client=FakeLLM(), prompt_manager=PromptManager()
        )
        history = [{"content": "bare msg"}]
        messages = pipeline._build_messages("q", history)
        assert messages[1]["role"] == "user"

    def test_history_messages_default_empty_content(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(), llm_client=FakeLLM(), prompt_manager=PromptManager()
        )
        history = [{"role": "assistant"}]
        messages = pipeline._build_messages("q", history)
        assert messages[1]["content"] == ""


@pytest.mark.unit
class TestAugmentMessages:
    def test_targets_latest_user_turn(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(), llm_client=FakeLLM(), prompt_manager=PromptManager()
        )
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer1"},
            {"role": "user", "content": "latest"},
        ]
        augmented = pipeline._augment_messages(messages, "ctx")
        assert "first" in augmented[0]["content"]
        assert "Retrieved Context" not in augmented[0]["content"]
        assert "Retrieved Context" in augmented[2]["content"]
        assert "User Question: latest" in augmented[2]["content"]

    def test_no_user_role_fallback_appends(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(), llm_client=FakeLLM(), prompt_manager=PromptManager()
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "only assistant"},
        ]
        augmented = pipeline._augment_messages(messages, "ctx")
        assert len(augmented) == 3
        assert augmented[-1]["role"] == "user"
        assert "Retrieved Context" in augmented[-1]["content"]

    def test_single_user_message_augments_it(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(), llm_client=FakeLLM(), prompt_manager=PromptManager()
        )
        messages = [{"role": "user", "content": "only"}]
        augmented = pipeline._augment_messages(messages, "ctx")
        assert "Retrieved Context" in augmented[0]["content"]
        assert "User Question: only" in augmented[0]["content"]


@pytest.mark.unit
@pytest.mark.asyncio
class TestQueryStream:
    async def test_yields_string_chunks(self):
        pipeline = RAGPipeline(
            retriever=FakeRetriever(),
            llm_client=FakeLLM(chunks := ["chunk1", "chunk2", "chunk3"]),
            prompt_manager=PromptManager(),
        )

        collected = []
        async for chunk in pipeline.query_stream(question="stream q"):
            collected.append(chunk)

        assert collected == ["chunk1", "chunk2", "chunk3"]

    async def test_stream_error_propagates(self):
        async def broken_stream(messages):
            yield "first"
            raise RuntimeError("stream fail")
            yield "never"

        llm = FakeLLM()
        llm.chat_stream = broken_stream

        pipeline = RAGPipeline(
            retriever=FakeRetriever(),
            llm_client=llm,
            prompt_manager=PromptManager(),
        )
        collected = []
        with pytest.raises(RuntimeError, match="stream fail"):
            async for chunk in pipeline.query_stream(question="fail"):
                collected.append(chunk)
        assert collected == ["first"]

    async def test_stream_saves_conversation_after_yield(self):
        msg_repo = _FakeMessageRepo()
        conv_repo = _FakeConversationRepo()

        pipeline = RAGPipeline(
            retriever=FakeRetriever(),
            llm_client=FakeLLM(["a", "b"]),
            prompt_manager=PromptManager(),
            conversation_repo=conv_repo,
            message_repo=msg_repo,
        )
        chunks = []
        async for chunk in pipeline.query_stream(question="stream save", conversation_id="c1"):
            chunks.append(chunk)

        assert chunks == ["a", "b"]
        assert len(msg_repo.created_messages) == 2
        assert msg_repo.created_messages[1]["content"] == "ab"

    async def test_stream_save_error_is_handled(self):
        msg_repo = _FakeMessageRepo()
        msg_repo.create = AsyncMock(side_effect=RuntimeError("db error"))
        conv_repo = _FakeConversationRepo()

        pipeline = RAGPipeline(
            retriever=FakeRetriever(),
            llm_client=FakeLLM(["x"]),
            prompt_manager=PromptManager(),
            conversation_repo=conv_repo,
            message_repo=msg_repo,
        )
        chunks = []
        async for chunk in pipeline.query_stream(question="err save", conversation_id="c1"):
            chunks.append(chunk)

        assert chunks == ["x"]

    async def test_stream_with_tool_engine(self):
        tool_engine = MagicMock()
        tool_engine.execute = AsyncMock(return_value=([{"role": "tool", "content": "t"}], {}))

        pipeline = RAGPipeline(
            retriever=FakeRetriever(),
            llm_client=FakeLLM(["one"]),
            prompt_manager=PromptManager(),
            tool_engine=tool_engine,
        )
        collected = []
        async for chunk in pipeline.query_stream(question="with tool"):
            collected.append(chunk)

        assert collected == ["one"]
        tool_engine.execute.assert_awaited_once()

    async def test_stream_tool_engine_error_handled(self):
        tool_engine = MagicMock()
        tool_engine.execute = AsyncMock(side_effect=RuntimeError("tool fail"))

        pipeline = RAGPipeline(
            retriever=FakeRetriever(),
            llm_client=FakeLLM(["ok"]),
            prompt_manager=PromptManager(),
            tool_engine=tool_engine,
        )
        collected = []
        async for chunk in pipeline.query_stream(question="tool err"):
            collected.append(chunk)

        assert collected == ["ok"]
