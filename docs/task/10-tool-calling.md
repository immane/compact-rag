# 任务 09: 生成层 LLM 抽象

> **依赖**: 01-配置管理, 02-公共基础设施 | **优先级**: P0 | **预计工时**: 6h

## 目标

实现 LLM 客户端抽象层，通过接口 + 工厂模式支持 OpenAI / Anthropic / Ollama 三种后端，并提供 Jinja2 提示词模板管理。

## 产出文件

```
src/compact_rag/generation/
├── __init__.py
├── llm.py                 # LLM 客户端抽象层 + 各 Provider 实现
└── prompt.py              # Jinja2 提示词模板管理
```

## 详细需求

### 1. `llm.py` — LLM 客户端抽象

```python
class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

class ChatResponse(BaseModel):
    content: str
    tool_calls: list[dict] | None
    token_usage: dict          # {prompt_tokens, completion_tokens, total_tokens}
    model: str
    finish_reason: str

class LLMClient(ABC):
    """LLM 客户端抽象基类"""

    @abstractmethod
    async def chat(self, messages: list[dict], tools: list[dict] = None,
                   temperature: float = 0.1, max_tokens: int = 2048) -> ChatResponse:
        """发送对话，返回 ChatResponse"""

    @abstractmethod
    async def chat_stream(self, messages: list[dict], tools: list[dict] = None,
                          temperature: float = 0.1) -> AsyncGenerator[str, None]:
        """流式对话，yield token 片段"""

    def supports_tool_calling(self) -> bool:
        """是否支持原生 tool calling"""
```

**三种 Provider 实现**：

| Provider | 客户端 | 构造函数 |
|----------|--------|---------|
| `OpenAIClient` | `openai.AsyncOpenAI` | `model, api_key, api_base` |
| `AnthropicClient` | `anthropic.AsyncAnthropic` | `model, api_key` |
| `OllamaClient` | `ollama.AsyncClient` | `model, host` |

```python
class LLMFactory:
    @staticmethod
    def create(settings: LLMSettings) -> LLMClient:
        """工厂方法，根据 provider 字段返回对应客户端"""
        if settings.provider == "openai":
            return OpenAIClient(settings.model, settings.api_key, settings.api_base)
        elif settings.provider == "anthropic":
            return AnthropicClient(settings.model, settings.api_key)
        elif settings.provider == "ollama":
            return OllamaClient(settings.model, settings.api_base or "http://localhost:11434")
        raise ConfigurationError(f"Unknown LLM provider: {settings.provider}")
```

### 2. `prompt.py` — 提示词模板管理

```python
from jinja2 import Environment, FileSystemLoader

class PromptManager:
    """Jinja2 提示词模板管理器"""

    def __init__(self, template_dir: str = None):
        """
        - 加载内置默认模板
        - 可选：从自定义目录加载覆盖模板
        - 使用 Jinja2 Environment + FileSystemLoader
        """

    def render_system_prompt(self, collections: list[str]) -> str:
        """渲染系统提示词"""

    def render_rag_context(self, documents: list[dict]) -> str:
        """渲染 RAG 上下文（检索到的文档块）"""

    def register_template(self, name: str, template_str: str):
        """注册自定义模板"""

    def render(self, name: str, **kwargs) -> str:
        """渲染指定模板"""
```

**默认 RAG 系统提示词**：

```
你是一个智能知识库助手，基于提供的文档内容回答用户问题。

规则：
1. 仅基于提供的文档内容回答，不编造信息
2. 如果文档中没有相关信息，诚实告知用户
3. 回答要简洁准确，在末尾标注引用的文档来源
4. 当文档中包含表格时，保留 Markdown 表格格式
5. 当用户问及数据时，可调用相关工具获取精确信息

可用集合：{{ collections | join(", ") }}
```

**RAG 上下文模板**：

```
{% for doc in documents %}
---
[来源 {{ loop.index }}] 文件: {{ doc.filename }}
页码: {{ doc.page_number }}

{{ doc.content }}
{% endfor %}
```

## 验收标准

- [ ] OpenAIClient 可调用 OpenAI API 并返回正常响应
- [ ] AnthropicClient 可调用 Anthropic API 并返回正常响应
- [ ] OllamaClient 可调用本地 Ollama 服务
- [ ] `LLMFactory.create()` 根据 provider 配置返回正确客户端
- [ ] `chat_stream` 正确 yield token（SSE 兼容）
- [ ] PromptManager 正确渲染系统提示词和上下文模板
- [ ] Tool calling 模式下工具定义正确传递给 LLM
- [ ] 认证失败时抛出 `LLMAuthError`
- [ ] 超时时抛出 `LLMTimeoutError`
- [ ] 速率限制时抛出 `LLMRateLimitError`
