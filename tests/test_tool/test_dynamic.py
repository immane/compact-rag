"""Tests for generic runtime-defined tools."""

from __future__ import annotations

import json

import httpx
import pytest

from compact_rag.tool.dynamic import DynamicToolDefinition, build_dynamic_tools
from compact_rag.tool.engine import ToolEngine
from compact_rag.rag.pipeline import extract_tool_links


def _http_definition(**patch):
    value = {
        "name": "fetch_record",
        "description": "Fetch a record by identifier",
        "kind": "http",
        "parameters": {
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
        },
        "method": "GET",
        "url": "https://api.example.test/records/{{record_id}}",
        "headers": {"x-auth-token": "token-1"},
        "query": {"expand": "true"},
    }
    value.update(patch)
    return value


class TestDynamicToolDefinition:
    def test_rejects_bad_name(self):
        with pytest.raises(ValueError):
            DynamicToolDefinition(**_http_definition(name="bad name"))

    def test_rejects_unsupported_method(self):
        with pytest.raises(ValueError):
            DynamicToolDefinition(**_http_definition(method="DELETE"))

    def test_builds_openai_function_schema(self):
        tools = build_dynamic_tools([_http_definition()])
        assert len(tools) == 1
        assert tools[0].to_openai_tool()["function"]["name"] == "fetch_record"
        assert tools[0].schema["required"] == ["record_id"]

    def test_omits_disabled_definitions(self):
        assert build_dynamic_tools([_http_definition(enabled=False)]) == []


class TestDynamicHttpExecution:
    async def test_get_template_auth_and_query(self, monkeypatch):
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"id": "r-17", "name": "report"}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                captured["timeout"] = kwargs["timeout"]

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def request(self, method, url, **kwargs):
                captured.update({"method": method, "url": url, **kwargs})
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
        engine = ToolEngine(build_dynamic_tools([_http_definition()]), max_retries=0)
        result = await engine.execute_tool_call({
            "id": "call-1",
            "function": {
                "name": "fetch_record",
                "arguments": json.dumps({"record_id": "r 17"}),
            },
        })
        assert json.loads(result["content"]) == {"id": "r-17", "name": "report"}
        assert captured["method"] == "GET"
        assert captured["url"] == "https://api.example.test/records/r 17"
        assert captured["headers"]["x-auth-token"] == "token-1"
        assert captured["params"] == {"expand": "true"}

    async def test_post_interpolates_nested_body(self, monkeypatch):
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"url": "https://example.test/action/1"}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def request(self, method, url, **kwargs):
                captured.update({"method": method, "url": url, **kwargs})
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
        definition = _http_definition(
            name="create_action",
            method="POST",
            url="https://api.example.test/actions",
            query={},
            body={"record": {"id": "{{record_id}}"}},
        )
        tool = build_dynamic_tools([definition])[0]
        result = await tool.fn(record_id="r-9")
        assert result["url"].endswith("/1")
        assert captured["json"] == {"record": {"id": "r-9"}}
        assert captured["params"] is None

    async def test_missing_template_argument_is_error(self):
        tool = build_dynamic_tools([_http_definition()])[0]
        with pytest.raises(ValueError, match="missing argument"):
            await tool.fn()


class TestGenericToolLinks:
    def test_extracts_nested_urls_from_any_tool(self):
        links = extract_tool_links([
            {
                "role": "tool",
                "name": "submit_request",
                "content": json.dumps({"data": {"result": {"url": "https://example.test/r/1"}}}),
            }
        ])
        assert len(links) == 1
        assert links[0].url == "https://example.test/r/1"
        assert links[0].source == "submit_request"
