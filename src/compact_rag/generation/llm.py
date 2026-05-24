from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import AsyncGenerator
import os

from compact_rag.common.exceptions import (
    ConfigurationError,
    GenerationError,
    LLMServiceError,
    LLMAuthError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from compact_rag.common.logger import get_logger
from compact_rag.config.settings import LLMSettings
from compact_rag.storage.schema import ChatResponse

logger = get_logger(__name__)


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class LLMClient(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.1,
    ) -> AsyncGenerator[str, None]:
        ...

    def supports_tool_calling(self) -> bool:
        return False


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: int = 60,
    ) -> None:
        from openai import AsyncOpenAI

        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
        )

    def supports_tool_calling(self) -> bool:
        return True

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        try:
            kwargs = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
            response = await self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]
            return ChatResponse(
                content=choice.message.content or "",
                tool_calls=tool_calls,
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                model=response.model,
                finish_reason=choice.finish_reason or "stop",
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "unauthorized" in error_msg or "invalid_api_key" in error_msg:
                raise LLMAuthError(str(e), cause=e)
            if "429" in error_msg or "rate" in error_msg:
                raise LLMRateLimitError(str(e), cause=e)
            if "timeout" in error_msg:
                raise LLMTimeoutError(str(e), cause=e)
            raise

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.1,
    ) -> AsyncGenerator[str, None]:
        try:
            kwargs = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            if tools:
                kwargs["tools"] = tools
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg:
                raise LLMTimeoutError(str(e), cause=e)
            raise


class AnthropicClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout: int = 60,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ConfigurationError(
                "anthropic package not installed. Install with: pip install anthropic"
            )
        self._model = model
        self._client = AsyncAnthropic(
            api_key=api_key or "dummy",
            timeout=timeout,
        )

    def supports_tool_calling(self) -> bool:
        return True

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        try:
            system_msg = ""
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg += msg["content"] + "\n"
                else:
                    user_messages.append(msg)

            kwargs = {
                "model": self._model,
                "messages": user_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system_msg.strip():
                kwargs["system"] = system_msg.strip()
            if tools:
                kwargs["tools"] = tools

            response = await self._client.messages.create(**kwargs)
            content = ""
            tool_calls = None
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": block.input,
                        },
                    })

            return ChatResponse(
                content=content,
                tool_calls=tool_calls,
                token_usage={
                    "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                    "completion_tokens": response.usage.output_tokens if response.usage else 0,
                    "total_tokens": (
                        response.usage.input_tokens + response.usage.output_tokens
                        if response.usage
                        else 0
                    ),
                },
                model=response.model,
                finish_reason=response.stop_reason or "stop",
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "unauthorized" in error_msg:
                raise LLMAuthError(str(e), cause=e)
            if "429" in error_msg or "rate" in error_msg:
                raise LLMRateLimitError(str(e), cause=e)
            if "timeout" in error_msg:
                raise LLMTimeoutError(str(e), cause=e)
            raise

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.1,
    ) -> AsyncGenerator[str, None]:
        try:
            system_msg = ""
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg += msg["content"] + "\n"
                else:
                    user_messages.append(msg)

            kwargs = {
                "model": self._model,
                "messages": user_messages,
                "max_tokens": 2048,
                "temperature": temperature,
            }
            if system_msg.strip():
                kwargs["system"] = system_msg.strip()
            if tools:
                kwargs["tools"] = tools

            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        yield event.delta.text
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg:
                raise LLMTimeoutError(str(e), cause=e)
            raise


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self._sdk_client = None
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = timeout

        try:
            import ollama

            self._sdk_client = ollama.AsyncClient(host=self._host)
        except ImportError:
            # Fallback to direct HTTP API so local-model usage does not depend on SDK install.
            logger.warning(
                "ollama package not installed, falling back to direct HTTP API",
                host=self._host,
            )

    def supports_tool_calling(self) -> bool:
        return True

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        try:
            kwargs = {
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            if tools:
                kwargs["tools"] = tools

            # Try SDK first, allow a single retry for transient server errors
            sdk_attempts = 2 if self._sdk_client is not None else 0
            sdk_exception = None
            for attempt in range(sdk_attempts):
                try:
                    response = await asyncio.wait_for(
                        self._sdk_client.chat(**kwargs), timeout=self._timeout
                    )
                    sdk_exception = None
                    break
                except Exception as e:
                    sdk_exception = e
                    # Detect server-side status codes
                    status_code = getattr(e, "status_code", None)
                    if status_code is None:
                        try:
                            if hasattr(e, "args") and e.args:
                                text = " ".join(str(a) for a in e.args)
                            else:
                                text = str(e)
                        except Exception:
                            text = str(e)
                        import re

                        m = re.search(r"status\s*code[:=]?\s*([0-9]{3})", text, re.IGNORECASE)
                        if m:
                            try:
                                status_code = int(m.group(1))
                            except Exception:
                                status_code = None

                    # If transient server error, wait and retry once
                    if status_code is not None and 500 <= int(status_code) < 600 and attempt + 1 < sdk_attempts:
                        backoff = 0.5 * (2 ** attempt)
                        logger.warning(
                            "Ollama SDK attempt %s failed with status %s, retrying after %.1fs",
                            attempt + 1,
                            status_code,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    # Otherwise, break to fall back to HTTP if available
                    break

            # If SDK didn't produce a response, try HTTP fallback (if configured)
            if self._sdk_client is None or sdk_exception is not None:
                import httpx

                http_attempts = 3
                last_http_exc = None
                for attempt in range(http_attempts):
                    try:
                        async with httpx.AsyncClient(
                            timeout=self._timeout,
                            trust_env=False,
                        ) as client:
                            resp = await client.post(f"{self._host}/api/chat", json=kwargs)
                            try:
                                resp.raise_for_status()
                            except httpx.HTTPStatusError as he:
                                # Log body for debugging and decide whether to retry
                                body = resp.text
                                status = resp.status_code
                                logger.warning(
                                    "Ollama HTTP fallback returned status %s (attempt %s): %s",
                                    status,
                                    attempt + 1,
                                    body[:400],
                                )
                                last_http_exc = he
                                if 500 <= status < 600 and attempt + 1 < http_attempts:
                                    await asyncio.sleep(0.5 * (2 ** attempt))
                                    continue
                                # Non-retryable HTTP error
                                raise LLMServiceError(
                                    f"Ollama HTTP API error {status}", details={"body": body}, cause=he
                                )
                            # Success
                            response = resp.json()
                            last_http_exc = None
                            break
                    except httpx.RequestError as rexc:
                        last_http_exc = rexc
                        logger.warning(
                            "Ollama HTTP request error (attempt %s): %s",
                            attempt + 1,
                            str(rexc),
                        )
                        if attempt + 1 < http_attempts:
                            await asyncio.sleep(0.5 * (2 ** attempt))
                            continue
                        raise LLMServiceError("Ollama HTTP request failed", cause=rexc)

                # If both SDK and HTTP failed, raise a GenerationError so API maps to appropriate HTTP status
                if (self._sdk_client is not None and sdk_exception is not None) and last_http_exc is not None:
                    raise LLMServiceError(
                        "Ollama SDK and HTTP fallback both failed",
                        details={"sdk_error": str(sdk_exception), "http_error": str(last_http_exc)},
                        cause=sdk_exception,
                    )
            # If SDK succeeded, `response` is already set from the SDK call above.

            tool_calls = None
            if response.get("message", {}).get("tool_calls"):
                tool_calls = response["message"]["tool_calls"]

            return ChatResponse(
                content=response.get("message", {}).get("content", ""),
                tool_calls=tool_calls,
                token_usage={
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
                },
                model=response.get("model", self._model),
                finish_reason=response.get("done_reason", "stop"),
            )
        except asyncio.TimeoutError as e:
            raise LLMTimeoutError(f"Ollama request timed out after {self._timeout}s", cause=e)
        except GenerationError:
            # propagate GenerationError up so API layer can convert to HTTP status
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "refused" in error_msg:
                raise LLMTimeoutError(str(e), cause=e)
            raise

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.1,
    ) -> AsyncGenerator[str, None]:
        try:
            kwargs = {
                "model": self._model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                },
            }
            if tools:
                kwargs["tools"] = tools

            if self._sdk_client is not None:
                try:
                    stream = await self._sdk_client.chat(**kwargs)
                    async for chunk in stream:
                        if chunk.get("message", {}).get("content"):
                            yield chunk["message"]["content"]
                    return
                except Exception as e:
                    # Detect 5xx and fall back to HTTP streaming
                    status_code = getattr(e, "status_code", None)
                    if status_code is None:
                        try:
                            if hasattr(e, "args") and e.args:
                                text = " ".join(str(a) for a in e.args)
                            else:
                                text = str(e)
                        except Exception:
                            text = str(e)
                        import re

                        m = re.search(r"status\s*code[:=]?\s*([0-9]{3})", text, re.IGNORECASE)
                        if m:
                            try:
                                status_code = int(m.group(1))
                            except Exception:
                                status_code = None

                    if status_code is not None and 500 <= int(status_code) < 600:
                        logger.warning(
                            "Ollama SDK stream failed with status %s, falling back to HTTP stream",
                            status_code,
                        )
                        import httpx
                        import httpx

                        try:
                            async with httpx.AsyncClient(
                                timeout=self._timeout,
                                trust_env=False,
                            ) as client:
                                async with client.stream("POST", f"{self._host}/api/chat", json=kwargs) as resp:
                                    try:
                                        resp.raise_for_status()
                                    except httpx.HTTPStatusError as he:
                                        body = await resp.aread()
                                        raise LLMServiceError(
                                            f"Ollama HTTP stream error {resp.status_code}",
                                            details={"body": body.decode("utf-8", errors="replace")},
                                            cause=he,
                                        )
                                    async for line in resp.aiter_lines():
                                        if not line:
                                            continue
                                        try:
                                            chunk = json.loads(line)
                                        except json.JSONDecodeError:
                                            continue
                                        content = chunk.get("message", {}).get("content")
                                        if content:
                                            yield content
                        except httpx.RequestError as rexc:
                            raise LLMServiceError("Ollama HTTP stream request failed", cause=rexc)
                        return
                    raise

            import httpx

            async with httpx.AsyncClient(
                timeout=self._timeout,
                trust_env=False,
            ) as client:
                async with client.stream("POST", f"{self._host}/api/chat", json=kwargs) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content = chunk.get("message", {}).get("content")
                        if content:
                            yield content
        except asyncio.TimeoutError as e:
            raise LLMTimeoutError(f"Ollama stream timed out", cause=e)
        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "refused" in error_msg:
                raise LLMTimeoutError(str(e), cause=e)
            raise


class LLMFactory:
    @staticmethod
    def create(settings: LLMSettings) -> LLMClient:
        provider = settings.provider
        if provider == LLMProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ConfigurationError("OPENAI_API_KEY must be set in the environment")
            return OpenAIClient(
                model=settings.model,
                api_key=api_key,
                api_base=settings.api_base,
                timeout=settings.timeout,
            )
        elif provider == LLMProvider.ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ConfigurationError("ANTHROPIC_API_KEY must be set in the environment")
            return AnthropicClient(
                model=settings.model,
                api_key=api_key,
                timeout=settings.timeout,
            )
        elif provider == LLMProvider.OLLAMA:
            return OllamaClient(
                model=settings.model,
                host=settings.api_base or "http://localhost:11434",
                timeout=settings.timeout,
            )
        raise ConfigurationError(f"Unknown LLM provider: {provider}")
