"""Admin-defined tools backed by vector search or generic HTTP APIs."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

from compact_rag.common.logger import get_logger
from compact_rag.tool.schema import Tool

logger = get_logger(__name__)

_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
_TEMPLATE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class DynamicToolDefinition(BaseModel):
    """Generic tool declaration editable at runtime."""

    name: str
    description: str
    kind: str = "http"  # http | vector_search
    enabled: bool = True
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )

    # HTTP execution configuration
    method: str = "GET"
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    query: dict[str, Any] = Field(default_factory=dict)
    body: Any = None
    timeout: int = Field(default=20, ge=1, le=300)

    # Vector-search execution configuration
    collection: str = "default"
    query_argument: str = "query"
    top_k: int = Field(default=5, ge=1, le=100)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError(
                "name must be 1-64 letters/digits/underscores and start with a letter"
            )
        return value

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        if value not in {"http", "vector_search"}:
            raise ValueError("kind must be 'http' or 'vector_search'")
        return value

    @field_validator("method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        value = value.upper()
        if value not in {"GET", "POST"}:
            raise ValueError("method must be GET or POST")
        return value

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type", "object") != "object":
            raise ValueError("parameters schema must have type 'object'")
        if not isinstance(value.get("properties", {}), dict):
            raise ValueError("parameters.properties must be an object")
        return value


def _render(value: Any, arguments: dict[str, Any]) -> Any:
    """Recursively interpolate ``{{argument}}`` placeholders in config values."""
    if isinstance(value, dict):
        return {k: _render(v, arguments) for k, v in value.items()}
    if isinstance(value, list):
        return [_render(v, arguments) for v in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in arguments:
            raise ValueError(f"Tool template references missing argument: {key}")
        val = arguments[key]
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        return str(val)

    return _TEMPLATE.sub(replace, value)


def _template_headers(
    headers: dict[str, str], arguments: dict[str, Any]
) -> dict[str, str]:
    return {key: _render(value, arguments) for key, value in headers.items()}


async def _execute_http(definition: DynamicToolDefinition, arguments: dict) -> Any:
    if not definition.url:
        return {"error": f"HTTP tool '{definition.name}' has no URL configured."}
    url = _render(definition.url, arguments)
    headers = _template_headers(definition.headers, arguments)
    query = _render(definition.query, arguments)
    body = _render(definition.body, arguments) if definition.body is not None else None
    async with httpx.AsyncClient(timeout=definition.timeout, trust_env=False) as client:
        response = await client.request(
            definition.method,
            url,
            headers=headers,
            params=query if definition.method == "GET" else None,
            json=body if definition.method == "POST" and body is not None else None,
        )
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {"text": response.text}


def build_dynamic_tools(definitions: list[dict], retriever_provider=None) -> list[Tool]:
    """Build runtime Tool objects from validated tool declarations."""
    tools = []
    for raw in definitions:
        definition = DynamicToolDefinition(**raw)
        if not definition.enabled:
            continue

        async def execute(_definition=definition, **arguments):
            if _definition.kind == "http":
                return await _execute_http(_definition, arguments)
            provider = retriever_provider
            if provider is None:
                from compact_rag.api.deps import get_hybrid_retriever

                provider = get_hybrid_retriever
            retriever = provider()
            query = arguments.get(_definition.query_argument)
            if not query:
                return {
                    "error": f"Missing query argument '{_definition.query_argument}'."
                }
            results = await retriever.retrieve(
                query=str(query),
                collection=_definition.collection,
                top_k=_definition.top_k,
                use_hybrid_search=True,
                use_rerank=True,
            )
            return [
                {
                    "id": result.id,
                    "content": result.content,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                for result in results
            ]

        execute.__name__ = definition.name
        execute.__doc__ = definition.description
        tool = Tool(execute)
        tool.name = definition.name
        tool.description = definition.description
        tool.schema = definition.parameters
        tools.append(tool)
    return tools
