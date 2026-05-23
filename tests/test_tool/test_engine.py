from __future__ import annotations

import pytest

from compact_rag.common.exceptions import ToolExecutionError
from compact_rag.tool.engine import ToolEngine
from compact_rag.tool.schema import Tool


def _greet(name: str) -> str:
    """Greet a person."""
    return f"Hello, {name}!"


def _failing_tool(x: int) -> str:
    """Always fails."""
    raise ValueError("intentional error")


class TestToolEngine:
    @pytest.fixture
    def engine(self):
        tools = [Tool(_greet), Tool(_failing_tool)]
        return ToolEngine(tools, max_retries=1)

    def test_get_openai_tools_format(self, engine):
        tools = engine.get_openai_tools()
        assert len(tools) == 2
        for t in tools:
            assert t["type"] == "function"
            assert "function" in t
            assert "name" in t["function"]
            assert "parameters" in t["function"]

    @pytest.mark.asyncio
    async def test_execute_tool_call_success(self, engine):
        result = await engine.execute_tool_call({
            "function": {"name": "_greet", "arguments": '{"name": "World"}'},
            "id": "call_1",
        })
        assert result["role"] == "tool"
        assert result["name"] == "_greet"
        assert result["content"] == "Hello, World!"
        assert result["tool_call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_execute_tool_call_with_dict_arguments(self, engine):
        result = await engine.execute_tool_call({
            "function": {"name": "_greet", "arguments": {"name": "Dict"}},
            "id": "call_2",
        })
        assert result["content"] == "Hello, Dict!"

    @pytest.mark.asyncio
    async def test_execute_tool_call_unknown_tool(self, engine):
        result = await engine.execute_tool_call({
            "function": {"name": "unknown_tool", "arguments": "{}"},
            "id": "call_3",
        })
        assert "not found" in result["content"]
        assert "_greet" in result["content"]

    @pytest.mark.asyncio
    async def test_execute_tool_call_with_retry(self, engine):
        with pytest.raises(ToolExecutionError) as exc_info:
            await engine.execute_tool_call({
                "function": {"name": "_failing_tool", "arguments": '{"x": 42}'},
                "id": "call_4",
            })
        assert "_failing_tool" in str(exc_info.value)
        assert "failed after" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_tool_with_top_level_name(self, engine):
        result = await engine.execute_tool_call({
            "name": "_greet",
            "function": {"arguments": '{"name": "TopLevel"}'},
            "id": "top_1",
        })
        assert result["content"] == "Hello, TopLevel!"

    @pytest.mark.asyncio
    async def test_execute_with_invalid_json_arguments(self, engine):
        with pytest.raises(ToolExecutionError):
            await engine.execute_tool_call({
                "function": {"name": "_greet", "arguments": "not-json"},
                "id": "call_5",
            })
