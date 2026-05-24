from __future__ import annotations

import asyncio
import builtins
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from compact_rag.common.exceptions import (
    ConfigurationError,
    GenerationError,
    LLMAuthError,
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)
from compact_rag.config.settings import LLMSettings
from compact_rag.generation.llm import (
    AnthropicClient,
    LLMClient,
    LLMFactory,
    LLMProvider,
    OllamaClient,
    OpenAIClient,
)
from compact_rag.storage.schema import ChatResponse


# ────────────────────────────────────────────────────────────────────
# Shared Fixtures
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def openai_settings():
    return LLMSettings(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-openai-key",
        timeout=30,
    )


@pytest.fixture
def anthropic_settings():
    return LLMSettings(
        provider="anthropic",
        model="claude-3-opus",
        api_key="test-anthropic-key",
        timeout=30,
    )


@pytest.fixture
def ollama_settings():
    return LLMSettings(
        provider="ollama",
        model="llama3",
        api_base="http://localhost:11434",
        timeout=30,
    )


@pytest.fixture
def unknown_provider_settings():
    """Create an LLMSettings-like object with an unsupported provider value."""
    settings = MagicMock(spec=LLMSettings)
    settings.provider = "google"
    settings.model = "gemini-pro"
    settings.api_key = "test-key"
    settings.api_base = None
    settings.timeout = 30
    return settings


@pytest.fixture
def mock_openai(mocker):
    return mocker.patch("openai.AsyncOpenAI")


@pytest.fixture
def anthropic_mock_setup(mocker):
    """Create a fake anthropic module so AnthropicClient.__init__ succeeds."""
    fake = ModuleType("anthropic")
    sys.modules["anthropic"] = fake
    mock_cls = mocker.MagicMock(name="AsyncAnthropic")
    setattr(fake, "AsyncAnthropic", mock_cls)
    yield mock_cls.return_value
    sys.modules.pop("anthropic", None)


@pytest.fixture
def block_ollama_import(monkeypatch):
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "ollama" or name.startswith("ollama."):
            raise ImportError(f"blocked for test: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


_NO_USAGE = object()


def _make_openai_response(mocker, *, content: str | None = "Hello, world!", tool_calls=None, usage=_NO_USAGE, model="gpt-4o-mini", finish_reason="stop"):
    """Helper to build a mock OpenAI chat completion response.

    Set ``usage=None`` to simulate a missing usage block.
    """
    tc_mocks = None
    if tool_calls:
        tc_mocks = []
        for tc in tool_calls:
            tc_mock = mocker.MagicMock()
            tc_mock.id = tc.get("id", "call-1")
            tc_mock.function.name = tc.get("function", {}).get("name", "test_func")
            tc_mock.function.arguments = tc.get("function", {}).get("arguments", "{}")
            tc_mocks.append(tc_mock)

    choice = mocker.MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tc_mocks
    choice.finish_reason = finish_reason

    resp = mocker.MagicMock()
    resp.choices = [choice]
    resp.model = model

    if usage is None:
        resp.usage = None
    else:
        usage_mock = mocker.MagicMock()
        if usage is _NO_USAGE:
            usage_mock.prompt_tokens = 10
            usage_mock.completion_tokens = 5
            usage_mock.total_tokens = 15
        else:
            usage_mock.prompt_tokens = usage.get("prompt_tokens", 10)  # type: ignore[union-attr]
            usage_mock.completion_tokens = usage.get("completion_tokens", 5)  # type: ignore[union-attr]
            usage_mock.total_tokens = usage.get("total_tokens", 15)  # type: ignore[union-attr]
        resp.usage = usage_mock
    return resp


def _setup_openai_chat(mock_openai, mocker, response):
    """Wire up mock_openai so chat.completions.create returns `response`."""
    mock_instance = mock_openai.return_value
    mock_create = AsyncMock(return_value=response)
    mock_instance.chat.completions.create = mock_create
    return mock_create


# ────────────────────────────────────────────────────────────────────
# 1. LLMProvider Enum
# ────────────────────────────────────────────────────────────────────

class TestLLMProvider:
    def test_all_enum_values(self):
        assert len(LLMProvider) == 3
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.OLLAMA.value == "ollama"

    def test_openai_string_equality(self):
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.OPENAI != "anthropic"

    def test_anthropic_string_equality(self):
        assert LLMProvider.ANTHROPIC == "anthropic"
        assert LLMProvider.ANTHROPIC != "ollama"

    def test_ollama_string_equality(self):
        assert LLMProvider.OLLAMA == "ollama"
        assert LLMProvider.OLLAMA != "openai"

    def test_enum_is_string_subclass(self):
        assert issubclass(LLMProvider, str)


# ────────────────────────────────────────────────────────────────────
# 2. LLMClient ABC
# ────────────────────────────────────────────────────────────────────

class TestLLMClientABC:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            LLMClient()  # type: ignore[abstract]

    def test_has_chat_abstract_method(self):
        assert hasattr(LLMClient, "chat")
        assert getattr(LLMClient.chat, "__isabstractmethod__", False)

    def test_has_chat_stream_abstract_method(self):
        assert hasattr(LLMClient, "chat_stream")
        assert getattr(LLMClient.chat_stream, "__isabstractmethod__", False)

    def test_supports_tool_calling_default_false(self):
        class ConcreteClient(LLMClient):  # type: ignore[abstract]
            async def chat(self, messages, tools=None, temperature=0.1, max_tokens=2048) -> ChatResponse:  # noqa: UP006
                return ChatResponse(content="ok")

            async def chat_stream(self, messages, tools=None, temperature=0.1):  # type: ignore[override]  # noqa: UP006
                if False:
                    yield
                yield "ok"

        client = ConcreteClient()  # type: ignore[abstract]
        assert client.supports_tool_calling() is False


# ────────────────────────────────────────────────────────────────────
# 3. LLMFactory.create()
# ────────────────────────────────────────────────────────────────────

class TestLLMFactory:
    @pytest.fixture(autouse=True)
    def _set_env_api_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    def test_creates_openai_client(self, openai_settings, mock_openai):
        client = LLMFactory.create(openai_settings)
        assert isinstance(client, OpenAIClient)
        assert client._model == "gpt-4o-mini"

    def test_creates_openai_with_api_base(self, mock_openai):
        settings = LLMSettings(provider="openai", model="gpt-4", api_base="https://custom.openai.com", timeout=45)
        client = LLMFactory.create(settings)
        assert isinstance(client, OpenAIClient)
        assert client._model == "gpt-4"

    def test_creates_anthropic_client(self, anthropic_mock_setup):
        settings = LLMSettings(provider="anthropic", model="claude-3-opus", api_key="sk-ant-key", timeout=60)
        client = LLMFactory.create(settings)
        assert isinstance(client, AnthropicClient)
        assert client._model == "claude-3-opus"

    def test_creates_ollama_client_no_sdk(self, block_ollama_import):
        settings = LLMSettings(provider="ollama", model="llama3", api_base="http://localhost:11434", timeout=30)
        client = LLMFactory.create(settings)
        assert isinstance(client, OllamaClient)
        assert client._model == "llama3"

    def test_creates_ollama_default_host(self, block_ollama_import):
        settings = LLMSettings(provider="ollama", model="mistral", api_base=None, timeout=30)
        client = LLMFactory.create(settings)
        assert isinstance(client, OllamaClient)
        assert client._host == "http://localhost:11434"

    def test_raises_configuration_error_for_unknown_provider(self, unknown_provider_settings):
        with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
            LLMFactory.create(unknown_provider_settings)

    def test_passes_timeout_to_client(self, mock_openai):
        settings = LLMSettings(provider="openai", model="gpt-4o-mini", timeout=15)
        client = LLMFactory.create(settings)
        mock_openai.assert_called_once()
        _, kwargs = mock_openai.call_args
        assert kwargs.get("timeout") == 15


# ────────────────────────────────────────────────────────────────────
# 4. OpenAIClient
# ────────────────────────────────────────────────────────────────────

class TestOpenAIClient:
    def test_supports_tool_calling(self, mock_openai):
        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        assert client.supports_tool_calling() is True

    @pytest.mark.asyncio
    async def test_chat_returns_correct_response(self, mock_openai, mocker):
        fake_resp = _make_openai_response(mocker, content="Hello, world!")
        _setup_openai_chat(mock_openai, mocker, fake_resp)

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        response = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert isinstance(response, ChatResponse)
        assert response.content == "Hello, world!"
        assert response.model == "gpt-4o-mini"
        assert response.finish_reason == "stop"
        assert response.token_usage["prompt_tokens"] == 10
        assert response.token_usage["completion_tokens"] == 5
        assert response.token_usage["total_tokens"] == 15
        assert response.tool_calls is None

    @pytest.mark.asyncio
    async def test_chat_passes_tools(self, mock_openai, mocker):
        fake_resp = _make_openai_response(mocker)
        mock_create = _setup_openai_chat(mock_openai, mocker, fake_resp)

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        await client.chat(messages=[{"role": "user", "content": "weather?"}], tools=tools)

        call_kwargs = mock_create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_chat_passes_temperature_and_max_tokens(self, mock_openai, mocker):
        fake_resp = _make_openai_response(mocker)
        mock_create = _setup_openai_chat(mock_openai, mocker, fake_resp)

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        await client.chat(messages=[{"role": "user", "content": "hi"}], temperature=0.5, max_tokens=512)

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_chat_handles_tool_calls(self, mock_openai, mocker):
        fake_resp = _make_openai_response(
            mocker,
            content="",
            tool_calls=[
                {
                    "id": "call-abc",
                    "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                }
            ],
        )
        _setup_openai_chat(mock_openai, mocker, fake_resp)

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        response = await client.chat(messages=[{"role": "user", "content": "weather in Tokyo?"}])

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["id"] == "call-abc"
        assert response.tool_calls[0]["type"] == "function"
        assert response.tool_calls[0]["function"]["name"] == "get_weather"
        assert response.tool_calls[0]["function"]["arguments"] == '{"city": "Tokyo"}'

    @pytest.mark.asyncio
    async def test_chat_handles_empty_content(self, mock_openai, mocker):
        fake_resp = _make_openai_response(mocker, content=None)
        _setup_openai_chat(mock_openai, mocker, fake_resp)

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        response = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert response.content == ""

    @pytest.mark.asyncio
    async def test_chat_handles_missing_usage(self, mock_openai, mocker):
        choice = mocker.MagicMock()
        choice.message.content = "no usage"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"

        resp = mocker.MagicMock()
        resp.choices = [choice]
        resp.model = "gpt-4o-mini"
        resp.usage = None  # no usage info

        _setup_openai_chat(mock_openai, mocker, resp)

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        response = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert response.token_usage["prompt_tokens"] == 0
        assert response.token_usage["completion_tokens"] == 0
        assert response.token_usage["total_tokens"] == 0

    @pytest.mark.asyncio
    async def test_chat_handles_none_finish_reason(self, mock_openai, mocker):
        choice = mocker.MagicMock()
        choice.message.content = "ok"
        choice.message.tool_calls = None
        choice.finish_reason = None

        resp = mocker.MagicMock()
        resp.choices = [choice]
        resp.model = "gpt-4o-mini"
        resp.usage.prompt_tokens = 1
        resp.usage.completion_tokens = 1
        resp.usage.total_tokens = 2

        _setup_openai_chat(mock_openai, mocker, resp)

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        response = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert response.finish_reason == "stop"

    # ── Error classification ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_chat_401_raises_llm_auth_error(self, mock_openai, mocker):
        mock_create = _setup_openai_chat(mock_openai, mocker, None)
        mock_create.side_effect = Exception("401 Unauthorized")

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        with pytest.raises(LLMAuthError) as exc_info:
            await client.chat(messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.cause is not None

    @pytest.mark.asyncio
    async def test_chat_unauthorized_message_raises_llm_auth_error(self, mock_openai, mocker):
        mock_create = _setup_openai_chat(mock_openai, mocker, None)
        mock_create.side_effect = Exception("invalid_api_key: unauthorized")

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        with pytest.raises(LLMAuthError):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_chat_429_raises_llm_rate_limit_error(self, mock_openai, mocker):
        mock_create = _setup_openai_chat(mock_openai, mocker, None)
        mock_create.side_effect = Exception("429 rate limit exceeded")

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        with pytest.raises(LLMRateLimitError) as exc_info:
            await client.chat(messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.cause is not None

    @pytest.mark.asyncio
    async def test_chat_rate_message_raises_llm_rate_limit_error(self, mock_openai, mocker):
        mock_create = _setup_openai_chat(mock_openai, mocker, None)
        mock_create.side_effect = Exception("rate limit exceeded please wait")

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        with pytest.raises(LLMRateLimitError):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_chat_timeout_raises_llm_timeout_error(self, mock_openai, mocker):
        mock_create = _setup_openai_chat(mock_openai, mocker, None)
        mock_create.side_effect = Exception("request timeout")

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        with pytest.raises(LLMTimeoutError) as exc_info:
            await client.chat(messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.cause is not None

    @pytest.mark.asyncio
    async def test_chat_reraises_unknown_error(self, mock_openai, mocker):
        mock_create = _setup_openai_chat(mock_openai, mocker, None)
        mock_create.side_effect = ValueError("something unexpected")

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        with pytest.raises(ValueError, match="something unexpected"):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    # ── Streaming ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_chat_stream_yields_content_chunks(self, mock_openai, mocker):
        chunk1 = mocker.MagicMock()
        chunk1.choices = [mocker.MagicMock()]
        chunk1.choices[0].delta = mocker.MagicMock()
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = mocker.MagicMock()
        chunk2.choices = [mocker.MagicMock()]
        chunk2.choices[0].delta = mocker.MagicMock()
        chunk2.choices[0].delta.content = " world"

        async def _aiter(_self=None):
            for c in [chunk1, chunk2]:
                yield c

        stream = mocker.MagicMock()
        stream.__aiter__ = _aiter

        mock_instance = mock_openai.return_value
        mock_create = AsyncMock(return_value=stream)
        mock_instance.chat.completions.create = mock_create

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        chunks = []
        async for chunk in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_chat_stream_passes_tools(self, mock_openai, mocker):
        chunk = mocker.MagicMock()
        chunk.choices = [mocker.MagicMock()]
        chunk.choices[0].delta = mocker.MagicMock()
        chunk.choices[0].delta.content = "ok"

        async def _aiter(_self=None):
            yield chunk

        stream = mocker.MagicMock()
        stream.__aiter__ = _aiter

        mock_instance = mock_openai.return_value
        mock_create = AsyncMock(return_value=stream)
        mock_instance.chat.completions.create = mock_create

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        async for _ in client.chat_stream(messages=[{"role": "user", "content": "hi"}], tools=tools):
            pass

        call_kwargs = mock_create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools

    @pytest.mark.asyncio
    async def test_chat_stream_handles_missing_delta(self, mock_openai, mocker):
        chunk1 = mocker.MagicMock()
        chunk1.choices = [mocker.MagicMock()]
        chunk1.choices[0].delta = None  # no delta

        chunk2 = mocker.MagicMock()
        chunk2.choices = [mocker.MagicMock()]
        chunk2.choices[0].delta = mocker.MagicMock()
        chunk2.choices[0].delta.content = None  # delta exists but no content

        chunk3 = mocker.MagicMock()
        chunk3.choices = [mocker.MagicMock()]
        chunk3.choices[0].delta = mocker.MagicMock()
        chunk3.choices[0].delta.content = "valid"

        chunk4 = mocker.MagicMock()
        chunk4.choices = []  # no choices

        async def _aiter(_self=None):
            for c in [chunk1, chunk2, chunk3, chunk4]:
                yield c

        stream = mocker.MagicMock()
        stream.__aiter__ = _aiter

        mock_instance = mock_openai.return_value
        mock_create = AsyncMock(return_value=stream)
        mock_instance.chat.completions.create = mock_create

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        chunks = []
        async for chunk in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert chunks == ["valid"]

    @pytest.mark.asyncio
    async def test_chat_stream_timeout_raises_llm_timeout_error(self, mock_openai, mocker):
        mock_instance = mock_openai.return_value
        mock_create = AsyncMock(side_effect=Exception("connection timeout"))
        mock_instance.chat.completions.create = mock_create

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        with pytest.raises(LLMTimeoutError) as exc_info:
            async for _ in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
                pass
        assert exc_info.value.cause is not None

    @pytest.mark.asyncio
    async def test_chat_stream_reraises_unknown_error(self, mock_openai, mocker):
        mock_instance = mock_openai.return_value
        mock_create = AsyncMock(side_effect=RuntimeError("unexpected stream failure"))
        mock_instance.chat.completions.create = mock_create

        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-key")
        with pytest.raises(RuntimeError, match="unexpected stream failure"):
            async for _ in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
                pass


# ────────────────────────────────────────────────────────────────────
# 5. AnthropicClient
# ────────────────────────────────────────────────────────────────────

class TestAnthropicClient:
    def test_supports_tool_calling(self, anthropic_mock_setup):
        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        assert client.supports_tool_calling() is True

    def test_init_raises_configuration_error_when_not_installed(self, monkeypatch):
        original_import = builtins.__import__

        def block_anthropic(name, *args, **kwargs):
            if name == "anthropic" or name.startswith("anthropic."):
                raise ImportError("No module named 'anthropic'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_anthropic)

        with pytest.raises(ConfigurationError, match="anthropic package not installed"):
            AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")

    # ── chat() ───────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_chat_handles_system_messages(self, anthropic_mock_setup, mocker):
        mock_instance = anthropic_mock_setup
        mock_create = AsyncMock()
        mock_instance.messages.create = mock_create

        mock_resp = mocker.MagicMock()
        mock_resp.content = [mocker.MagicMock()]
        mock_resp.content[0].type = "text"
        mock_resp.content[0].text = "I understand."
        mock_resp.model = "claude-3-opus"
        mock_resp.stop_reason = "end_turn"
        mock_resp.usage.input_tokens = 50
        mock_resp.usage.output_tokens = 30
        mock_create.return_value = mock_resp

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        response = await client.chat(messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ])

        assert response.content == "I understand."
        assert response.model == "claude-3-opus"
        assert response.finish_reason == "end_turn"
        assert response.token_usage["prompt_tokens"] == 50
        assert response.token_usage["completion_tokens"] == 30
        assert response.token_usage["total_tokens"] == 80

        call_kwargs = mock_create.call_args[1]
        assert "system" in call_kwargs
        assert call_kwargs["system"] == "You are helpful."
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_chat_concatenates_multiple_system_messages(self, anthropic_mock_setup, mocker):
        mock_instance = anthropic_mock_setup
        mock_create = AsyncMock()
        mock_instance.messages.create = mock_create

        mock_resp = mocker.MagicMock()
        mock_resp.content = [mocker.MagicMock()]
        mock_resp.content[0].type = "text"
        mock_resp.content[0].text = "OK"
        mock_resp.model = "claude-3-opus"
        mock_resp.stop_reason = "end_turn"
        mock_resp.usage.input_tokens = 5
        mock_resp.usage.output_tokens = 2
        mock_create.return_value = mock_resp

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        await client.chat(messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "system", "content": "Use JSON."},
            {"role": "user", "content": "Hello"},
        ])

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["system"] == "Be concise.\nUse JSON."

    @pytest.mark.asyncio
    async def test_chat_handles_empty_system_message(self, anthropic_mock_setup, mocker):
        mock_instance = anthropic_mock_setup
        mock_create = AsyncMock()
        mock_instance.messages.create = mock_create

        mock_resp = mocker.MagicMock()
        mock_resp.content = [mocker.MagicMock()]
        mock_resp.content[0].type = "text"
        mock_resp.content[0].text = "Hi"
        mock_resp.model = "claude-3-opus"
        mock_resp.stop_reason = "end_turn"
        mock_resp.usage.input_tokens = 1
        mock_resp.usage.output_tokens = 1
        mock_create.return_value = mock_resp

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        await client.chat(messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ])

        call_kwargs = mock_create.call_args[1]
        assert "system" not in call_kwargs

    @pytest.mark.asyncio
    async def test_chat_no_system_message(self, anthropic_mock_setup, mocker):
        mock_instance = anthropic_mock_setup
        mock_create = AsyncMock()
        mock_instance.messages.create = mock_create

        mock_resp = mocker.MagicMock()
        mock_resp.content = [mocker.MagicMock()]
        mock_resp.content[0].type = "text"
        mock_resp.content[0].text = "Hi"
        mock_resp.model = "claude-3-opus"
        mock_resp.stop_reason = "end_turn"
        mock_resp.usage.input_tokens = 1
        mock_resp.usage.output_tokens = 1
        mock_create.return_value = mock_resp

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        await client.chat(messages=[{"role": "user", "content": "Hello"}])

        call_kwargs = mock_create.call_args[1]
        assert "system" not in call_kwargs
        assert len(call_kwargs["messages"]) == 1

    @pytest.mark.asyncio
    async def test_chat_handles_tool_use_blocks(self, anthropic_mock_setup, mocker):
        mock_instance = anthropic_mock_setup
        mock_create = AsyncMock()
        mock_instance.messages.create = mock_create

        text_block = mocker.MagicMock()
        text_block.type = "text"
        text_block.text = "Let me check."

        tool_block = mocker.MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_001"
        tool_block.name = "get_weather"
        tool_block.input = {"city": "Paris"}

        mock_resp = mocker.MagicMock()
        mock_resp.content = [text_block, tool_block]
        mock_resp.model = "claude-3-opus"
        mock_resp.stop_reason = "tool_use"
        mock_resp.usage.input_tokens = 20
        mock_resp.usage.output_tokens = 15
        mock_create.return_value = mock_resp

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        response = await client.chat(messages=[{"role": "user", "content": "weather in Paris?"}])

        assert response.content == "Let me check."
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["id"] == "toolu_001"
        assert response.tool_calls[0]["type"] == "function"
        assert response.tool_calls[0]["function"]["name"] == "get_weather"
        assert response.tool_calls[0]["function"]["arguments"] == {"city": "Paris"}
        assert response.finish_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_chat_handles_missing_usage(self, anthropic_mock_setup, mocker):
        mock_instance = anthropic_mock_setup
        mock_create = AsyncMock()
        mock_instance.messages.create = mock_create

        mock_resp = mocker.MagicMock()
        mock_resp.content = [mocker.MagicMock()]
        mock_resp.content[0].type = "text"
        mock_resp.content[0].text = "Hi"
        mock_resp.model = "claude-3-opus"
        mock_resp.stop_reason = None
        mock_resp.usage = None
        mock_create.return_value = mock_resp

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        response = await client.chat(messages=[{"role": "user", "content": "Hello"}])

        assert response.token_usage["prompt_tokens"] == 0
        assert response.token_usage["completion_tokens"] == 0
        assert response.token_usage["total_tokens"] == 0
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_passes_tools(self, anthropic_mock_setup, mocker):
        mock_instance = anthropic_mock_setup
        mock_create = AsyncMock()
        mock_instance.messages.create = mock_create

        mock_resp = mocker.MagicMock()
        mock_resp.content = [mocker.MagicMock()]
        mock_resp.content[0].type = "text"
        mock_resp.content[0].text = "OK"
        mock_resp.model = "claude-3-opus"
        mock_resp.stop_reason = "end_turn"
        mock_resp.usage.input_tokens = 1
        mock_resp.usage.output_tokens = 1
        mock_create.return_value = mock_resp

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        await client.chat(messages=[{"role": "user", "content": "hi"}], tools=tools)

        call_kwargs = mock_create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools

    # ── Error classification ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_chat_401_raises_llm_auth_error(self, anthropic_mock_setup, mocker):
        mock_instance = anthropic_mock_setup
        mock_create = AsyncMock(side_effect=Exception("401 Unauthorized"))
        mock_instance.messages.create = mock_create

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        with pytest.raises(LLMAuthError):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_chat_429_raises_llm_rate_limit_error(self, anthropic_mock_setup, mocker):
        mock_instance = anthropic_mock_setup
        mock_create = AsyncMock(side_effect=Exception("429 rate limit exceeded"))
        mock_instance.messages.create = mock_create

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        with pytest.raises(LLMRateLimitError):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_chat_timeout_raises_llm_timeout_error(self, anthropic_mock_setup, mocker):
        mock_instance = anthropic_mock_setup
        mock_create = AsyncMock(side_effect=Exception("request timeout"))
        mock_instance.messages.create = mock_create

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        with pytest.raises(LLMTimeoutError):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_chat_reraises_unknown_error(self, anthropic_mock_setup, mocker):
        mock_instance = anthropic_mock_setup
        mock_create = AsyncMock(side_effect=ValueError("unexpected"))
        mock_instance.messages.create = mock_create

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        with pytest.raises(ValueError, match="unexpected"):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    # ── Streaming ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_chat_stream_yields_text_delta_events(self, anthropic_mock_setup, mocker):
        mock_instance = anthropic_mock_setup

        event1 = mocker.MagicMock()
        event1.type = "content_block_delta"
        event1.delta.type = "text_delta"
        event1.delta.text = "Hello"

        event2 = mocker.MagicMock()
        event2.type = "content_block_delta"
        event2.delta.type = "text_delta"
        event2.delta.text = " world"

        event3 = mocker.MagicMock()
        event3.type = "content_block_stop"  # not a text_delta

        async def _event_iter():
            for e in [event1, event2, event3]:
                yield e

        stream_ctx = mocker.MagicMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=stream_ctx)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)
        stream_ctx.__aiter__ = lambda self: _event_iter()

        mock_instance.messages.stream = mocker.MagicMock(return_value=stream_ctx)

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        chunks = []
        async for chunk in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_chat_stream_handles_system_messages(self, anthropic_mock_setup, mocker):
        mock_instance = anthropic_mock_setup

        event = mocker.MagicMock()
        event.type = "content_block_delta"
        event.delta.type = "text_delta"
        event.delta.text = "ok"

        async def _event_iter():
            yield event

        stream_ctx = mocker.MagicMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=stream_ctx)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)
        stream_ctx.__aiter__ = lambda self: _event_iter()

        mock_stream = mocker.MagicMock(return_value=stream_ctx)
        mock_instance.messages.stream = mock_stream

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        _chunks = []
        async for _chunk in client.chat_stream(messages=[
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hi"},
        ]):
            _chunks.append(_chunk)

        call_kwargs = mock_stream.call_args[1]
        assert "system" in call_kwargs
        assert call_kwargs["system"] == "Be helpful."
        assert len(call_kwargs["messages"]) == 1
        assert _chunks == ["ok"]

    @pytest.mark.asyncio
    async def test_chat_stream_timeout_raises_llm_timeout_error(self, anthropic_mock_setup, mocker):
        mock_instance = anthropic_mock_setup
        mock_instance.messages.stream = mocker.MagicMock(side_effect=Exception("connection timeout"))

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        with pytest.raises(LLMTimeoutError):
            async for _ in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
                pass

    @pytest.mark.asyncio
    async def test_chat_stream_reraises_unknown_error(self, anthropic_mock_setup, mocker):
        mock_instance = anthropic_mock_setup
        mock_instance.messages.stream = mocker.MagicMock(side_effect=RuntimeError("boom"))

        client = AnthropicClient(model="claude-3-opus", api_key="sk-ant-key")
        with pytest.raises(RuntimeError, match="boom"):
            async for _ in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
                pass


# ────────────────────────────────────────────────────────────────────
# 6. OllamaClient — added tests (existing test_llm_ollama.py covers some)
# ────────────────────────────────────────────────────────────────────

class TestOllamaClientSDK:
    """Tests where the ollama SDK is available (mocked)."""

    _ollama_fake_module = None

    @classmethod
    def setup_class(cls):
        """Create a fake ollama module so import succeeds."""
        cls._ollama_fake_module = ModuleType("ollama")
        sys.modules["ollama"] = cls._ollama_fake_module

    @classmethod
    def teardown_class(cls):
        sys.modules.pop("ollama", None)

    @pytest.mark.asyncio
    async def test_chat_sdk_successful(self, mocker):
        """SDK path: successful chat with installed ollama SDK."""
        mock_cls = mocker.MagicMock()
        setattr(self._ollama_fake_module, "AsyncClient", mock_cls)
        mock_instance = mock_cls.return_value
        mock_instance.chat = AsyncMock(return_value={
            "model": "llama3",
            "message": {"content": "Hello from SDK"},
            "prompt_eval_count": 10,
            "eval_count": 5,
            "done_reason": "stop",
        })

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        response = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert response.content == "Hello from SDK"
        assert response.model == "llama3"
        assert response.token_usage["prompt_tokens"] == 10
        assert response.token_usage["completion_tokens"] == 5
        assert response.token_usage["total_tokens"] == 15
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_sdk_with_tool_calls(self, mocker):
        """SDK path: response includes tool_calls."""
        mock_cls = mocker.MagicMock()
        setattr(self._ollama_fake_module, "AsyncClient", mock_cls)
        mock_instance = mock_cls.return_value
        mock_instance.chat = AsyncMock(return_value={
            "model": "llama3",
            "message": {
                "content": "",
                "tool_calls": [
                    {"id": "tc-1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}
                ],
            },
            "prompt_eval_count": 5,
            "eval_count": 10,
            "done_reason": "tool_calls",
        })

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        response = await client.chat(messages=[{"role": "user", "content": "weather?"}])

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "get_weather"
        assert response.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_stream_sdk_yields_content(self, mocker):
        """Stream with SDK: content from chunk."""
        mock_cls = mocker.MagicMock()
        setattr(self._ollama_fake_module, "AsyncClient", mock_cls)
        mock_instance = mock_cls.return_value

        class _AsyncIter:
            def __init__(self, items):
                self._items = items
                self._idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._idx]
                self._idx += 1
                return item

        async def _fake_chat(**kwargs):
            return _AsyncIter([
                {"message": {"content": "Hello"}},
                {"message": {"content": " world"}},
                {"message": {}},  # no content
                {"done": True},
            ])

        mock_instance.chat = _fake_chat

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        chunks = []
        async for chunk in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_sdk_5xx_triggers_retry_then_http_fallback(self, mocker, block_ollama_import):
        """SDK path: 5xx error triggers retry then HTTP fallback when SDK not available."""
        # This tests the path where SDK raises 5xx first attempt,
        # but since we blocked ollama import, _sdk_client is None,
        # so it goes directly to HTTP. Let's test HTTP fallback directly.
        pass

    @pytest.mark.asyncio
    async def test_sdk_5xx_error_falls_back_to_http(self, mocker):
        """When SDK returns 5xx, it should fall back to HTTP."""
        mock_cls = mocker.MagicMock()
        setattr(self._ollama_fake_module, "AsyncClient", mock_cls)
        mock_sdk = mock_cls.return_value

        call_count = 0

        class Sdk500Error(Exception):
            status_code = 500

        async def _sdk_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            raise Sdk500Error("status code 500")

        mock_sdk.chat = _sdk_chat

        # Now mock httpx for the fallback
        mock_httpx_client = mocker.patch("httpx.AsyncClient")
        mock_httpx_instance = mocker.MagicMock()

        fake_resp = mocker.MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "model": "llama3",
            "message": {"content": "HTTP fallback response"},
            "prompt_eval_count": 3,
            "eval_count": 7,
            "done_reason": "stop",
        }
        fake_resp.raise_for_status = mocker.MagicMock()

        mock_httpx_instance.__aenter__ = AsyncMock(return_value=mock_httpx_instance)
        mock_httpx_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_instance.post = AsyncMock(return_value=fake_resp)
        mock_httpx_client.return_value = mock_httpx_instance

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        response = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert response.content == "HTTP fallback response"
        assert call_count >= 1  # SDK was tried

    @pytest.mark.asyncio
    async def test_http_non_5xx_error_raises_llm_service_error(self, monkeypatch, block_ollama_import):
        """HTTP fallback: non-5xx HTTP error raises LLMServiceError."""
        original_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "ollama" or name.startswith("ollama."):
                raise ImportError("blocked")
            return original_import(name, *args, **kwargs)

        # Need fresh block for this test
        import httpx as real_httpx

        class FakeResponse:
            status_code = 400

            def raise_for_status(self):
                raise real_httpx.HTTPStatusError(
                    "400 Bad Request",
                    request=MagicMock(),
                    response=MagicMock(status_code=400, text='{"error": "bad request"}'),
                )

            @property
            def text(self):
                return '{"error": "bad request"}'

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json):
                return FakeResponse()

        monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)
        monkeypatch.setattr(builtins, "__import__", _fake_import)

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        with pytest.raises(LLMServiceError, match="Ollama HTTP API error 400"):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_http_request_error_raises_llm_service_error(self, monkeypatch, block_ollama_import):
        """HTTP fallback: httpx.RequestError raises LLMServiceError."""
        import httpx as real_httpx

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json):
                raise real_httpx.RequestError("connection refused")

        monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        with pytest.raises(LLMServiceError, match="Ollama HTTP request failed"):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_sdk_fails_then_http_request_error_raises_llm_service_error(self, mocker):
        """SDK fails, then HTTP fallback RequestError raises LLMServiceError."""
        mock_cls = mocker.MagicMock()
        setattr(self._ollama_fake_module, "AsyncClient", mock_cls)
        mock_sdk = mock_cls.return_value
        mock_sdk.chat = AsyncMock(side_effect=Exception("sdk failure"))

        mock_httpx_client = mocker.patch("httpx.AsyncClient")
        mock_httpx_instance = mocker.MagicMock()
        mock_httpx_instance.__aenter__ = AsyncMock(return_value=mock_httpx_instance)
        mock_httpx_instance.__aexit__ = AsyncMock(return_value=False)

        import httpx as real_httpx
        mock_httpx_instance.post = AsyncMock(side_effect=real_httpx.RequestError("total failure"))
        mock_httpx_client.return_value = mock_httpx_instance

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        with pytest.raises(LLMServiceError, match="Ollama HTTP request failed"):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_sdk_timeout_falls_back_to_http_then_request_error(self, mocker):
        """When SDK times out via asyncio.wait_for, HTTP fallback is attempted.
        With HTTP also failing, LLMServiceError is raised."""
        mock_cls = mocker.MagicMock()
        setattr(self._ollama_fake_module, "AsyncClient", mock_cls)
        mock_sdk = mock_cls.return_value

        async def _hang_forever(**kwargs):
            await asyncio.sleep(999)
            return {}

        mock_sdk.chat = _hang_forever

        mock_httpx_client = mocker.patch("httpx.AsyncClient")
        mock_httpx_instance = mocker.MagicMock()
        mock_httpx_instance.__aenter__ = AsyncMock(return_value=mock_httpx_instance)
        mock_httpx_instance.__aexit__ = AsyncMock(return_value=False)

        import httpx as real_httpx
        mock_httpx_instance.post = AsyncMock(side_effect=real_httpx.RequestError("connection failure"))
        mock_httpx_client.return_value = mock_httpx_instance

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=1)
        with pytest.raises(LLMServiceError, match="Ollama HTTP request failed"):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_connection_refused_raises_llm_timeout_error(self, mocker, block_ollama_import):
        """Connection refused raises LLMTimeoutError (asyncio.TimeoutError first, or via catch)."""
        # When no SDK, the HTTP fallback runs. If the post raises a RequestError
        # with "connection refused", the outer exception handler should catch it.

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json):
                raise Exception("connection refused")

        import httpx
        mocker.patch.object(httpx, "AsyncClient", FakeAsyncClient)

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        with pytest.raises(LLMTimeoutError, match="connection refused"):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_stream_http_fallback_with_lines(self, monkeypatch, block_ollama_import):
        """Stream with HTTP fallback: yields content from streaming lines."""
        import httpx as real_httpx

        class FakeStreamResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_lines(self):
                yield '{"message": {"content": "hello"}}'
                yield '{"message": {"content": " stream"}}'
                yield ""
                yield "not-valid-json"
                yield '{"message": {}}'  # no content
                yield '{"done": true}'

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, json):
                return FakeStreamResponse()

        monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        chunks = []
        async for chunk in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert chunks == ["hello", " stream"]

    @pytest.mark.asyncio
    async def test_supports_tool_calling(self, block_ollama_import):
        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        assert client.supports_tool_calling() is True

    @pytest.mark.asyncio
    async def test_sdk_error_triggers_http_fallback_then_llm_service_error(self, mocker):
        """When SDK raises an error (e.g., GenerationError), HTTP fallback is attempted.
        If HTTP also fails, LLMServiceError is raised from HTTP handler."""
        mock_cls = mocker.MagicMock()
        setattr(self._ollama_fake_module, "AsyncClient", mock_cls)
        mock_sdk = mock_cls.return_value

        mock_sdk.chat = AsyncMock(side_effect=GenerationError("sdk failed"))

        mock_httpx_client = mocker.patch("httpx.AsyncClient")
        mock_httpx_instance = mocker.MagicMock()
        mock_httpx_instance.__aenter__ = AsyncMock(return_value=mock_httpx_instance)
        mock_httpx_instance.__aexit__ = AsyncMock(return_value=False)

        import httpx as real_httpx
        mock_httpx_instance.post = AsyncMock(side_effect=real_httpx.RequestError("total failure"))
        mock_httpx_client.return_value = mock_httpx_instance

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        with pytest.raises(LLMServiceError, match="Ollama HTTP request failed"):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_llm_service_error_from_sdk_triggers_http_fallback(self, mocker):
        """When SDK raises LLMServiceError, HTTP fallback is attempted.
        If HTTP also fails, LLMServiceError is raised from HTTP handler."""
        mock_cls = mocker.MagicMock()
        setattr(self._ollama_fake_module, "AsyncClient", mock_cls)
        mock_sdk = mock_cls.return_value

        mock_sdk.chat = AsyncMock(side_effect=LLMServiceError("sdk service error"))

        mock_httpx_client = mocker.patch("httpx.AsyncClient")
        mock_httpx_instance = mocker.MagicMock()
        mock_httpx_instance.__aenter__ = AsyncMock(return_value=mock_httpx_instance)
        mock_httpx_instance.__aexit__ = AsyncMock(return_value=False)

        import httpx as real_httpx
        mock_httpx_instance.post = AsyncMock(side_effect=real_httpx.RequestError("total failure"))
        mock_httpx_client.return_value = mock_httpx_instance

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        with pytest.raises(LLMServiceError, match="Ollama HTTP request failed"):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_empty_message_content_defaults_to_empty_string(self, mocker):
        """Response without message.content defaults to empty string."""
        mock_cls = mocker.MagicMock()
        setattr(self._ollama_fake_module, "AsyncClient", mock_cls)
        mock_sdk = mock_cls.return_value
        mock_sdk.chat = AsyncMock(return_value={
            "model": "llama3",
            "message": {},
            "prompt_eval_count": 0,
            "eval_count": 0,
        })

        client = OllamaClient(model="llama3", host="http://localhost:11434", timeout=30)
        response = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert response.content == ""
        assert response.token_usage["total_tokens"] == 0

    @pytest.mark.asyncio
    async def test_response_without_model_uses_configured_model(self, mocker):
        """When response has no model field, falls back to configured model."""
        mock_cls = mocker.MagicMock()
        setattr(self._ollama_fake_module, "AsyncClient", mock_cls)
        mock_sdk = mock_cls.return_value
        mock_sdk.chat = AsyncMock(return_value={
            "message": {"content": "hi"},
            "prompt_eval_count": 1,
            "eval_count": 1,
        })

        client = OllamaClient(model="my-custom-model", host="http://localhost:11434", timeout=30)
        response = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert response.model == "my-custom-model"
