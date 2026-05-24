from __future__ import annotations

import json

import pytest

from compact_rag.common.exceptions import ToolExecutionError
from compact_rag.storage.schema import ChatResponse
from compact_rag.tool.engine import ToolEngine
from compact_rag.tool.schema import Tool


def _greet(name: str) -> str:
    """Greet a person."""
    return f"Hello, {name}!"


def _failing_tool(x: int) -> str:
    """Always fails."""
    raise ValueError("intentional error")


def _weather(city: str) -> str:
    """Get weather."""
    return f"Sunny in {city}"


def _no_required_params(prefix: str = "") -> str:
    """No required params."""
    return f"{prefix}ok"


def _make_flaky(fail_count):
    counter = {"calls": 0}

    def flaky_fn(x: int) -> str:
        counter["calls"] += 1
        if counter["calls"] <= fail_count:
            raise ValueError(f"flaky failure #{counter['calls']}")
        return f"success on attempt {counter['calls']}"

    return flaky_fn, counter


class TestToolEngineFull:
    @pytest.fixture
    def two_tools(self):
        return [Tool(_greet), Tool(_failing_tool), Tool(_no_required_params)]

    @pytest.fixture
    def engine(self, two_tools):
        return ToolEngine(two_tools, max_retries=1)

    def test_get_openai_tools_empty_list(self):
        engine = ToolEngine([], max_retries=0)
        assert engine.get_openai_tools() == []

    def test_get_openai_tools_multiple_formats(self, engine):
        tools = engine.get_openai_tools()
        assert len(tools) == 3
        names = [t["function"]["name"] for t in tools]
        assert "_greet" in names
        assert "_failing_tool" in names
        assert "_no_required_params" in names
        for t in tools:
            assert t["type"] == "function"
            assert "function" in t
            assert "parameters" in t["function"]

    @pytest.mark.asyncio
    async def test_execute_nested_function_name_format(self, engine):
        result = await engine.execute_tool_call({
            "id": "call_1",
            "function": {
                "name": "_greet",
                "arguments": '{"name": "World"}',
            },
        })
        assert result["role"] == "tool"
        assert result["name"] == "_greet"
        assert result["content"] == "Hello, World!"
        assert result["tool_call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_execute_top_level_name_format(self, engine):
        result = await engine.execute_tool_call({
            "name": "_greet",
            "function": {"arguments": '{"name": "TopLevel"}'},
            "id": "top_1",
        })
        assert result["content"] == "Hello, TopLevel!"

    @pytest.mark.asyncio
    async def test_execute_with_dict_arguments(self, engine):
        result = await engine.execute_tool_call({
            "function": {"name": "_greet", "arguments": {"name": "Dict"}},
            "id": "call_dict",
        })
        assert result["content"] == "Hello, Dict!"

    @pytest.mark.asyncio
    async def test_execute_with_json_string_arguments(self, engine):
        result = await engine.execute_tool_call({
            "function": {"name": "_greet", "arguments": '{"name": "JSON"}'},
            "id": "call_json",
        })
        assert result["content"] == "Hello, JSON!"

    @pytest.mark.asyncio
    async def test_execute_invalid_json_arguments_falls_back_to_empty_dict(self, engine):
        result = await engine.execute_tool_call({
            "function": {"name": "_no_required_params", "arguments": "not-json-at-all"},
            "id": "call_bad",
        })
        assert result["content"] == "ok"

    @pytest.mark.asyncio
    async def test_execute_missing_name_unknown_tool(self, engine):
        result = await engine.execute_tool_call({
            "id": "missing",
            "function": {"arguments": "{}"},
        })
        assert "not found" in result["content"]
        assert "unknown" not in result["content"].lower()

    @pytest.mark.asyncio
    async def test_retry_fails_once_then_succeeds(self):
        flaky_fn, counter = _make_flaky(fail_count=1)
        tool = Tool(flaky_fn)
        engine = ToolEngine([tool], max_retries=2)

        result = await engine.execute_tool_call({
            "function": {"name": "flaky_fn", "arguments": '{"x": 1}'},
            "id": "c1",
        })
        assert result["content"] == "success on attempt 2"
        assert counter["calls"] == 2

    @pytest.mark.asyncio
    async def test_retry_fails_all_attempts_raises(self):
        engine = ToolEngine([Tool(_failing_tool)], max_retries=2)
        with pytest.raises(ToolExecutionError) as exc_info:
            await engine.execute_tool_call({
                "function": {"name": "_failing_tool", "arguments": '{"x": 1}'},
                "id": "c2",
            })
        assert "_failing_tool" in str(exc_info.value)
        assert "failed after" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retry_max_retries_zero(self):
        flaky_fn, counter = _make_flaky(fail_count=1)
        tool = Tool(flaky_fn)
        engine = ToolEngine([tool], max_retries=0)
        with pytest.raises(ToolExecutionError):
            await engine.execute_tool_call({
                "function": {"name": "flaky_fn", "arguments": '{"x": 1}'},
                "id": "c_zero",
            })
        assert counter["calls"] == 1

    @pytest.mark.asyncio
    async def test_retry_max_retries_one(self):
        flaky_fn, counter = _make_flaky(fail_count=1)
        tool = Tool(flaky_fn)
        engine = ToolEngine([tool], max_retries=1)
        result = await engine.execute_tool_call({
            "function": {"name": "flaky_fn", "arguments": '{"x": 1}'},
            "id": "c_one",
        })
        assert result["content"] == "success on attempt 2"
        assert counter["calls"] == 2

    @pytest.mark.asyncio
    async def test_retry_max_retries_three(self):
        flaky_fn, counter = _make_flaky(fail_count=3)
        tool = Tool(flaky_fn)
        engine = ToolEngine([tool], max_retries=3)
        result = await engine.execute_tool_call({
            "function": {"name": "flaky_fn", "arguments": '{"x": 1}'},
            "id": "c_three",
        })
        assert result["content"] == "success on attempt 4"
        assert counter["calls"] == 4

    @pytest.mark.asyncio
    async def test_run_loop_single_round_no_tool_calls(self, mocker):
        engine = ToolEngine([Tool(_greet)])
        mock_client = mocker.MagicMock()
        mock_client.chat = mocker.AsyncMock(return_value=ChatResponse(
            content="Final answer.",
            tool_calls=None,
        ))

        result = await engine.run_loop(mock_client, [], [])
        assert result == "Final answer."

    @pytest.mark.asyncio
    async def test_run_loop_multiple_rounds_with_tool_calls(self, mocker):
        engine = ToolEngine([Tool(_weather)])
        tc = {"function": {"name": "_weather", "arguments": '{"city": "Paris"}'}, "id": "tc1"}
        mock_client = mocker.MagicMock()
        mock_client.chat = mocker.AsyncMock(side_effect=[
            ChatResponse(content="", tool_calls=[tc]),
            ChatResponse(content="The weather is Sunny in Paris.", tool_calls=None),
        ])

        result = await engine.run_loop(mock_client, [], [])
        assert result == "The weather is Sunny in Paris."
        assert mock_client.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_run_loop_max_rounds_exhausted(self, mocker):
        engine = ToolEngine([Tool(_weather)])
        tc = {"function": {"name": "_weather", "arguments": '{"city": "Paris"}'}, "id": "tc1"}
        mock_client = mocker.MagicMock()
        responses = [ChatResponse(content="", tool_calls=[tc])] * 2
        responses.append(ChatResponse(content="Out of rounds.", tool_calls=None))
        mock_client.chat = mocker.AsyncMock(side_effect=responses)

        result = await engine.run_loop(mock_client, [], [], max_rounds=2)
        assert result == "Out of rounds."
        assert mock_client.chat.call_count == 3

    @pytest.mark.asyncio
    async def test_run_loop_tool_adds_messages_to_conversation(self, mocker):
        engine = ToolEngine([Tool(_weather)])
        tc = {"function": {"name": "_weather", "arguments": '{"city": "Paris"}'}, "id": "tc1"}
        mock_client = mocker.MagicMock()
        mock_client.chat = mocker.AsyncMock(side_effect=[
            ChatResponse(content="Let me check.", tool_calls=[tc]),
            ChatResponse(content="Done.", tool_calls=None),
        ])

        messages = [{"role": "user", "content": "What is the weather?"}]
        result = await engine.run_loop(mock_client, messages, [])

        assert result == "Done."
        message_roles = [m["role"] for m in messages]
        assert "assistant" in message_roles
        assert "tool" in message_roles

    @pytest.mark.asyncio
    async def test_run_loop_no_content_during_tool_calls(self, mocker):
        engine = ToolEngine([Tool(_weather)])
        tc = {"function": {"name": "_weather", "arguments": '{"city": "Berlin"}'}, "id": "tc2"}
        mock_client = mocker.MagicMock()
        mock_client.chat = mocker.AsyncMock(side_effect=[
            ChatResponse(content="", tool_calls=[tc]),
            ChatResponse(content="It is sunny in Berlin.", tool_calls=None),
        ])

        messages = []
        result = await engine.run_loop(mock_client, messages, [])
        assert result == "It is sunny in Berlin."
