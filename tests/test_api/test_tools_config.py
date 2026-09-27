"""Tests for runtime-defined tools configuration and execution."""

from __future__ import annotations

from fastapi.testclient import TestClient

from compact_rag.api.deps import _cached_settings, get_tool_engine
from compact_rag.api.router import create_app


def _client(test_settings):
    _cached_settings.cache_clear()
    app = create_app(settings=test_settings)
    with TestClient(app) as client:
        yield client


def _http_tool(**overrides):
    definition = {
        "name": "lookup_inventory",
        "description": "Look up inventory by SKU",
        "kind": "http",
        "enabled": True,
        "parameters": {
            "type": "object",
            "properties": {"sku": {"type": "string"}},
            "required": ["sku"],
        },
        "method": "GET",
        "url": "https://inventory.example.test/items/{{sku}}",
        "headers": {"x-auth-token": "secret-token"},
        "query": {},
        "timeout": 5,
    }
    definition.update(overrides)
    return definition


class TestDynamicToolConfig:
    def test_get_empty_and_engine_disabled_when_no_tools(self, test_settings):
        for client in _client(test_settings):
            response = client.get("/v1/config/tools")
            assert response.status_code == 200
            assert response.json() == {"enabled": True, "tools": []}
            assert get_tool_engine(test_settings) is None

    def test_put_and_secret_masking(self, test_settings):
        for client in _client(test_settings):
            response = client.put("/v1/config/tools", json={
                "enabled": True,
                "tools": [_http_tool()],
            })
            assert response.status_code == 200
            tool = response.json()["tools"][0]
            assert tool["name"] == "lookup_inventory"
            assert tool["headers"] == {"x-auth-token": ""}
            assert tool["headers_configured"] == {"x-auth-token": True}
            assert "secret-token" not in response.text
            engine = get_tool_engine(test_settings)
            assert engine is not None
            assert [t["function"]["name"] for t in engine.get_openai_tools()] == [
                "lookup_inventory"
            ]

    def test_invalid_and_duplicate_tools_rejected(self, test_settings):
        for client in _client(test_settings):
            invalid = client.put("/v1/config/tools", json={"tools": [{"name": "bad name"}]})
            assert invalid.status_code == 400
            duplicate = client.put("/v1/config/tools", json={
                "tools": [_http_tool(), _http_tool()],
            })
            assert duplicate.status_code == 400

    def test_empty_header_keeps_existing_secret(self, test_settings):
        for client in _client(test_settings):
            client.put("/v1/config/tools", json={"tools": [_http_tool()]})
            update = _http_tool(headers={"x-auth-token": ""})
            client.put("/v1/config/tools", json={"tools": [update]})
            configured = client.get("/v1/config/tools").json()["tools"][0]
            assert configured["headers_configured"]["x-auth-token"] is True

    def test_switch_off_disables_tool_engine(self, test_settings):
        for client in _client(test_settings):
            client.put("/v1/config/tools", json={"enabled": False, "tools": [_http_tool()]})
            assert get_tool_engine(test_settings) is None

    def test_vector_search_definition(self, test_settings):
        for client in _client(test_settings):
            tool = {
                "name": "search_reports",
                "description": "Search reports",
                "kind": "vector_search",
                "parameters": {
                    "type": "object",
                    "properties": {"term": {"type": "string"}},
                    "required": ["term"],
                },
                "collection": "reports",
                "query_argument": "term",
            }
            response = client.put("/v1/config/tools", json={"tools": [tool]})
            assert response.status_code == 200
            assert get_tool_engine(test_settings) is not None

    def test_test_endpoint_unknown(self, test_settings):
        for client in _client(test_settings):
            assert client.post("/v1/config/tools/nope/test", json={}).status_code == 404
