from __future__ import annotations

import json

from compact_rag.common.exceptions import ToolExecutionError
from compact_rag.common.logger import get_logger
from compact_rag.tool.schema import Tool

logger = get_logger(__name__)


class ToolEngine:
    def __init__(self, tools: list[Tool], max_retries: int = 2) -> None:
        self._tool_map: dict[str, Tool] = {t.name: t for t in tools}
        self.max_retries = max_retries

    def get_openai_tools(self) -> list[dict]:
        return [t.to_openai_tool() for t in self._tool_map.values()]

    async def execute_tool_call(self, tool_call: dict) -> dict:
        name = (
            tool_call.get("function", {})
            .get("name", "")
            or tool_call.get("name", "")
        )
        tool_call_id = (
            tool_call.get("id", "")
            or tool_call.get("function", {}).get("id", name)
        )

        arguments_str = (
            tool_call.get("function", {})
            .get("arguments", "{}")
            or tool_call.get("arguments", "{}")
        )

        if isinstance(arguments_str, dict):
            kwargs = arguments_str
        else:
            try:
                kwargs = json.loads(arguments_str)
            except (json.JSONDecodeError, TypeError):
                kwargs = {}

        tool = self._tool_map.get(name)
        if tool is None:
            return {
                "role": "tool",
                "name": name,
                "content": f"Tool '{name}' not found. Available: {list(self._tool_map.keys())}",
                "tool_call_id": tool_call_id,
            }

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = tool.execute(**kwargs)
                return {
                    "role": "tool",
                    "name": name,
                    "content": result,
                    "tool_call_id": tool_call_id,
                }
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Tool '{name}' failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )

        raise ToolExecutionError(
            f"Tool '{name}' failed after {self.max_retries + 1} attempts",
            details={"tool_name": name, "arguments": kwargs},
            cause=last_error,
        )

    async def run_loop(
        self,
        llm_client,
        messages: list[dict],
        tools: list[dict],
        max_rounds: int = 5,
    ) -> str:
        for _round in range(max_rounds):
            response = await llm_client.chat(messages=messages, tools=tools)

            if not response.tool_calls:
                return response.content

            for tc in response.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [tc],
                    }
                )
                tool_result = await self.execute_tool_call(tc)
                messages.append(tool_result)

            messages.append(
                {
                    "role": "system",
                    "content": "Please continue. If you have all the tool results needed, provide your final answer.",
                }
            )

        final_response = await llm_client.chat(messages=messages)
        return final_response.content
