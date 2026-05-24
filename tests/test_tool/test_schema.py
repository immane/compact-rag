from __future__ import annotations

import json


from compact_rag.tool.schema import Tool


def _simple_fn(name: str, count: int = 1) -> str:
    """A simple test function."""
    return f"simple: {name}, {count}"


def _no_docstring_fn(x: int) -> int:
    return x * 2


def _multi_type_fn(text: str, amount: float, active: bool, items: int = 10) -> dict:
    """Multi-type function with mixed params."""
    return {"text": text, "amount": amount, "active": active, "items": items}


class TestToolSchema:
    def test_build_schema_generates_correct_json_schema(self):
        tool = Tool(_simple_fn)
        schema = tool.schema
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["count"]["default"] == 1
        assert schema["required"] == ["name"]

    def test_to_openai_tool_format(self):
        tool = Tool(_simple_fn)
        openai_tool = tool.to_openai_tool()
        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "_simple_fn"
        assert openai_tool["function"]["description"] == "A simple test function."
        assert "parameters" in openai_tool["function"]

    def test_description_from_docstring(self):
        tool = Tool(_simple_fn)
        assert tool.description == "A simple test function."

        tool2 = Tool(_no_docstring_fn)
        assert tool2.description == ""

    def test_multi_type_parameters(self):
        tool = Tool(_multi_type_fn)
        schema = tool.schema
        props = schema["properties"]
        assert props["text"]["type"] == "string"
        assert props["amount"]["type"] == "number"
        assert props["active"]["type"] == "boolean"
        assert props["items"]["type"] == "integer"
        assert props["items"]["default"] == 10
        assert schema["required"] == ["text", "amount", "active"]

    def test_execute_returns_json_string(self):
        tool = Tool(_multi_type_fn)
        result = tool.execute(text="hello", amount=3.14, active=True)
        parsed = json.loads(result)
        assert parsed["text"] == "hello"
        assert parsed["amount"] == 3.14
        assert parsed["active"] is True
        assert parsed["items"] == 10

    def test_execute_string_result_returned_directly(self):
        tool = Tool(_simple_fn)
        result = tool.execute(name="test", count=5)
        assert result == "simple: test, 5"

    def test_name_matches_function_name(self):
        tool = Tool(_simple_fn)
        assert tool.name == "_simple_fn"
