from __future__ import annotations

import json

import pytest

from compact_rag.tool.schema import Tool


def _no_params():
    return "done"


def _with_args(*args):
    return str(args)


def _with_kwargs(**kwargs):
    return str(kwargs)


def _with_defaults(
    name: str = "alice",
    count: int = 42,
    ratio: float = 3.14,
    active: bool = True,
):
    return f"{name}-{count}-{ratio}-{active}"


def _no_hints(name):
    return f"hello {name}"


def _complex_hints(items: list, mapping: dict):
    return f"{items}, {mapping}"


def _returns_dict() -> dict:
    return {"key": "value"}


def _returns_list() -> list:
    return [1, 2, 3]


def _returns_int() -> int:
    return 42


def _returns_none() -> None:
    return None


def _raises_exception():
    raise ValueError("intentional failure")


class _NonSerializable:
    def __str__(self):
        return "nonserializable"


def _returns_nonserializable():
    return _NonSerializable()


def _has_docstring():
    """A helpful description."""
    pass


def _no_docstring():
    pass


def _partially_typed(x, y: int):
    """Mixed type hints."""
    return x, y


class TestToolSchemaFull:
    def test_no_params_empty_schema(self):
        tool = Tool(_no_params)
        assert tool.schema["type"] == "object"
        assert tool.schema["properties"] == {}
        assert tool.schema["required"] == []

    def test_args_handled_gracefully(self):
        tool = Tool(_with_args)
        props = tool.schema["properties"]
        assert "args" in props
        assert props["args"]["type"] == "string"
        assert "args" in tool.schema["required"]

    def test_kwargs_handled_gracefully(self):
        tool = Tool(_with_kwargs)
        props = tool.schema["properties"]
        assert "kwargs" in props
        assert props["kwargs"]["type"] == "string"
        assert "kwargs" in tool.schema["required"]

    def test_default_parameter_values(self):
        tool = Tool(_with_defaults)
        props = tool.schema["properties"]
        assert props["name"]["type"] == "string"
        assert props["name"]["default"] == "alice"
        assert props["count"]["type"] == "integer"
        assert props["count"]["default"] == 42
        assert props["ratio"]["type"] == "number"
        assert props["ratio"]["default"] == 3.14
        assert props["active"]["type"] == "boolean"
        assert props["active"]["default"] is True
        assert tool.schema["required"] == []

    def test_no_type_hint_defaults_to_string(self):
        tool = Tool(_no_hints)
        props = tool.schema["properties"]
        assert props["name"]["type"] == "string"

    def test_complex_type_defaults_to_string(self):
        tool = Tool(_complex_hints)
        props = tool.schema["properties"]
        assert props["items"]["type"] == "string"
        assert props["mapping"]["type"] == "string"

    def test_partial_hints_mixed(self):
        tool = Tool(_partially_typed)
        props = tool.schema["properties"]
        assert props["x"]["type"] == "string"
        assert props["y"]["type"] == "integer"

    def test_execute_returns_dict_json(self):
        tool = Tool(_returns_dict)
        result = tool.execute()
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_execute_returns_list_json(self):
        tool = Tool(_returns_list)
        result = tool.execute()
        parsed = json.loads(result)
        assert parsed == [1, 2, 3]

    def test_execute_returns_int_json(self):
        tool = Tool(_returns_int)
        result = tool.execute()
        assert result == "42"

    def test_execute_returns_none_as_null_string(self):
        tool = Tool(_returns_none)
        result = tool.execute()
        assert result == "null"

    def test_execute_raises_exception_propagates(self):
        tool = Tool(_raises_exception)
        with pytest.raises(ValueError, match="intentional failure"):
            tool.execute()

    def test_execute_nonserializable_falls_back_to_str(self):
        tool = Tool(_returns_nonserializable)
        result = tool.execute()
        assert result == "nonserializable"

    def test_to_openai_tool_format(self):
        tool = Tool(_no_params)
        ot = tool.to_openai_tool()
        assert ot["type"] == "function"
        assert "function" in ot
        assert ot["function"]["name"] == "_no_params"
        assert ot["function"]["description"] == ""
        assert "parameters" in ot["function"]

    def test_name_matches_function_name_exactly(self):
        tool = Tool(_no_params)
        assert tool.name == "_no_params"

    def test_description_from_docstring(self):
        tool1 = Tool(_has_docstring)
        assert tool1.description == "A helpful description."

    def test_empty_docstring_returns_empty_string(self):
        tool = Tool(_no_docstring)
        assert tool.description == ""
