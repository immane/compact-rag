from __future__ import annotations

import re
from typing import Callable

from compact_rag.tool.engine import ToolEngine
from compact_rag.tool.schema import Tool

_SELECT_PATTERN = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_FORBIDDEN_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def retrieve_docs(query: str, top_k: int = 3) -> str:
    """从知识库中检索与查询最相关的文档内容。返回相关文档的摘要。"""
    return f"Retrieving documents for: {query} (top_k={top_k})"


def query_database(sql: str) -> str:
    """执行数据库查询。仅允许 SELECT 语句，禁止 INSERT/UPDATE/DELETE/DROP。返回 JSON 格式的查询结果。"""
    if _FORBIDDEN_PATTERN.search(sql):
        return '{"error": "Only SELECT statements are allowed for security reasons."}'
    if not _SELECT_PATTERN.search(sql):
        return '{"error": "Only SELECT statements are permitted."}'
    return f'{{"result": "Query executed: {sql}"}}'


RAG_TOOLS = [Tool(retrieve_docs), Tool(query_database)]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def register_function(self, fn: Callable):
        tool = Tool(fn)
        self._tools[tool.name] = tool
        return fn

    def get_all(self) -> list[Tool]:
        return list(self._tools.values())

    def get_engine(self, max_retries: int = 2) -> ToolEngine:
        return ToolEngine(self.get_all(), max_retries=max_retries)
