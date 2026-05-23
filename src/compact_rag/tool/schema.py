from __future__ import annotations

import inspect
import json
from typing import Callable, get_type_hints

from compact_rag.common.logger import get_logger

logger = get_logger(__name__)

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


class Tool:
    def __init__(self, fn: Callable) -> None:
        self.fn = fn
        self.name = fn.__name__
        self.description = (fn.__doc__ or "").strip()
        self.schema = self._build_schema()

    def _build_schema(self) -> dict:
        sig = inspect.signature(self.fn)
        hints = get_type_hints(self.fn)
        properties: dict = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, str)
            json_type = _TYPE_MAP.get(param_type, "string")

            prop: dict = {"type": json_type}
            if param.default is not inspect.Parameter.empty:
                prop["default"] = param.default
            else:
                required.append(param_name)

            properties[param_name] = prop

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }

    def execute(self, **kwargs) -> str:
        result = self.fn(**kwargs)
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False)
        except TypeError:
            return str(result)
