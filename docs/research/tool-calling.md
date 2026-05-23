# 低门槛 Tool Calling 实现方案研究

> 研究日期：2026-05-23
> 研究目标：为 compact-rag 系统寻找最简单、最实用的 Tool Calling（工具调用/函数调用）实现方案

---

## 目录

1. [Tool Calling 的基本原理](#1-tool-calling-的基本原理)
2. [主流方案对比](#2-主流方案对比)
3. [轻量级实现方案](#3-轻量级实现方案)
4. [完整的 Python 示例代码](#4-完整的-python-示例代码)
5. [错误处理与重试策略](#5-错误处理与重试策略)
6. [在 RAG 系统中集成的建议](#6-在-rag-系统中集成的建议)
7. [推荐方案总结](#7-推荐方案总结)

---

## 1. Tool Calling 的基本原理

### 1.1 什么是 Tool Calling

Tool Calling（工具调用，也称 Function Calling / 函数调用）是指大语言模型（LLM）在生成回复的过程中，输出结构化的函数调用请求，由开发者执行实际的函数逻辑，并将结果返回给模型，最终模型基于结果生成自然语言回复。

### 1.2 核心流程

```
用户提问
    |
    v
[1] LLM 接收消息 + 工具定义（JSON Schema）
    |
    v
[2] LLM 判断是否需要调用工具
    |--- 不需要工具 -> 直接生成文本回复
    |--- 需要工具 -> 生成结构化 tool_calls
    |
    v
[3] 开发者解析 tool_calls，执行对应函数
    |
    v
[4] 将函数执行结果以 tool role 发回给 LLM
    |
    v
[5] LLM 生成最终自然语言回复
    |
    v
返回给用户
```

### 1.3 关键技术点

- **工具定义格式**：使用 JSON Schema 描述函数的名称、参数和返回值
- **模型原生支持**：主流 LLM API（OpenAI、Anthropic、Mistral、Ollama）都已原生支持
- **结构化输出**：模型输出的是结构化 JSON，非自由文本
- **循环执行**：模型可能需要多次调用工具（如先查用户信息、再查订单）
- **并行调用**：部分模型支持一次返回多个 tool_calls（并行执行）

### 1.4 JSON Schema 工具定义示例

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取指定城市的当前天气",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "城市名称，如 北京、上海"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "温度单位"
        }
      },
      "required": ["city"]
    }
  }
}
```

---

## 2. 主流方案对比

### 2.1 方案总览

| 方案 | 类型 | 复杂度 | 模型支持 | 本地部署 | 适用场景 |
|------|------|--------|----------|----------|----------|
| **OpenAI Function Calling** | 原生 API | 低 | GPT-4o, GPT-4, GPT-3.5 | 否 | 云端生产环境 |
| **Anthropic Tool Use** | 原生 API | 低 | Claude 3/4 系列 | 否 | 云端生产环境 |
| **Mistral Function Calling** | 原生 API | 低 | Mistral Large/Small | 否 | 云端生产环境 |
| **Ollama Tool Calling** | 原生 API | 低 | Llama 3.1+, Mistral, Qwen | 是 | 本地开发/隐私敏感 |
| **LiteLLM** | 统一封装层 | 中 | 100+ 模型 | 取决于后端 | 多供应商切换 |
| **LangChain Agent** | 框架层 | 高 | 多种 | 取决于后端 | 复杂 Agent 流程 |

### 2.2 OpenAI Function Calling

**优势**：
- 生态最成熟，文档最完善
- 支持并行函数调用（一次返回多个 tool_calls）
- 支持 `tool_choice` 控制（auto / required / specific tool）
- 支持结构化输出（Structured Outputs）与工具调用结合

**用法**：
```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京天气怎么样？"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }]
)
```

### 2.3 Anthropic Tool Use

**差异点（与 OpenAI 相比）**：
- tools 定义格式略有不同（使用 `input_schema` 而非 `parameters`）
- 使用 `tool_use` content block 而非 `tool_calls` 字段
- 结果通过 `tool_result` content block 返回

```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[{
        "name": "get_weather",
        "description": "获取天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }],
    messages=[{"role": "user", "content": "北京天气怎么样？"}]
)
```

### 2.4 Ollama Tool Calling（本地推荐）

**优势**：
- 完全本地运行，零成本
- 支持多种开源模型（Llama 3.1、Mistral、Qwen 等）
- API 兼容 OpenAI 格式
- Python 客户端简单易用

**支持工具调用的模型**（截至 2025 年）：
- Llama 3.1 / 3.2 / 3.3
- Mistral Nemo / Mistral Small
- Qwen 2.5 / Qwen 3
- Firefunction v2
- Command R+

```python
import ollama

response = ollama.chat(
    model="llama3.1",
    messages=[{"role": "user", "content": "北京天气怎么样？"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }]
)

# 解析工具调用
if response["message"].get("tool_calls"):
    for tc in response["message"]["tool_calls"]:
        print(tc["function"]["name"])
        print(tc["function"]["arguments"])
```

### 2.5 LiteLLM（统一接口层）

**优势**：
- 一套代码兼容所有主流 LLM 供应商
- 自动处理各供应商的格式差异
- 内置函数重试、降级逻辑

```python
from litellm import completion

# 同一套代码，只需改 model 参数
response = completion(
    model="openai/gpt-4o",          # 或 anthropic/claude-sonnet-4
    messages=[{"role": "user", "content": "北京天气怎么样？"}],
    tools=[...]
)
```

### 2.6 各方案复杂度 vs 能力对比

```
能力
  ^
  |           LangChain Agent
  |           /
  |     LiteLLM
  |     /
  |    OpenAI/Anthropic 原生
  |    /
  |   Ollama 原生
  |
  +------------------------>
  低                     高
  复杂度

推荐路径：Ollama/OpenAI 原生 -> LiteLLM -> LangChain（按需升级）
```

---

## 3. 轻量级实现方案

### 3.1 方案选择原则

对于 compact-rag 这样的中小型 RAG 系统，我们的原则是：
1. **能不依赖框架就不依赖** - 减少依赖、提高可维护性
2. **优先选择原生 API** - 直接调用 OpenAI/Ollama 的 API
3. **工具封装最小化** - 只需一个 Tool 类 + 一个执行引擎
4. **渐进式增强** - 从简单开始，按需增加功能

### 3.2 最小工具定义框架（约 80 行代码）

以下是一个不依赖任何外部框架的 Tool Calling 实现，支持：
- 从 Python 函数自动生成 JSON Schema
- 多工具路由和执行
- 错误处理
- 类型提示

```python
"""轻量级 Tool Calling 框架 - 不依赖 LangChain 等重量级库"""
import inspect
import json
from typing import Any, Callable, Dict, List, Optional, get_type_hints


class Tool:
    """工具定义，封装一个 Python 函数及其 JSON Schema"""

    def __init__(self, fn: Callable):
        self.fn = fn
        self.name = fn.__name__
        self.description = fn.__doc__ or ""
        self.schema = self._build_schema()

    def _build_schema(self) -> Dict:
        """从函数签名和类型注解生成 JSON Schema"""
        sig = inspect.signature(self.fn)
        hints = get_type_hints(self.fn)

        properties = {}
        required = []

        for name, param in sig.parameters.items():
            if name == "return":
                continue

            type_map = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
            }

            param_type = hints.get(name, str)
            json_type = type_map.get(param_type, "string")

            prop = {"type": json_type}

            if param.default is not inspect.Parameter.empty:
                pass  # 有默认值则非必需
            else:
                required.append(name)

            properties[name] = prop

        schema = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        return schema

    def to_openai_tool(self) -> Dict:
        """转换为 OpenAI/Ollama 兼容的 tool 定义"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            }
        }

    def execute(self, **kwargs) -> Any:
        """执行工具函数"""
        return self.fn(**kwargs)


class ToolEngine:
    """工具执行引擎：路由工具调用、执行、处理错误"""

    def __init__(self, tools: List[Tool]):
        self._tool_map = {t.name: t for t in tools}

    def get_openai_tools(self) -> List[Dict]:
        """获取所有工具的 OpenAI 格式定义"""
        return [t.to_openai_tool() for t in self._tool_map.values()]

    def execute_tool_call(self, tool_call: Dict) -> Dict:
        """
        执行单个工具调用

        Args:
            tool_call: 包含 name 和 arguments 的字典
                格式: {"function": {"name": "xxx", "arguments": '{"key": "val"}'}}

        Returns:
            {"role": "tool", "name": str, "content": str, "tool_call_id": str}
        """
        try:
            fn_name = tool_call["function"]["name"]
            fn_args = json.loads(tool_call["function"]["arguments"])

            if fn_name not in self._tool_map:
                return {
                    "role": "tool",
                    "name": fn_name,
                    "content": json.dumps({"error": f"未知工具: {fn_name}"}),
                    "tool_call_id": tool_call.get("id", ""),
                }

            tool = self._tool_map[fn_name]
            result = tool.execute(**fn_args)

            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)

            return {
                "role": "tool",
                "name": fn_name,
                "content": result,
                "tool_call_id": tool_call.get("id", ""),
            }

        except json.JSONDecodeError as e:
            return {
                "role": "tool",
                "name": tool_call.get("function", {}).get("name", "unknown"),
                "content": json.dumps({"error": f"参数解析失败: {str(e)}"}),
                "tool_call_id": tool_call.get("id", ""),
            }
        except Exception as e:
            return {
                "role": "tool",
                "name": tool_call.get("function", {}).get("name", "unknown"),
                "content": json.dumps({"error": f"执行失败: {str(e)}"}),
                "tool_call_id": tool_call.get("id", ""),
            }
```

### 3.3 使用示例

```python
# 1. 定义工具函数
def get_weather(city: str, unit: str = "celsius") -> str:
    """获取指定城市的当前天气"""
    weather_data = {
        "北京": {"temp": 25, "condition": "晴"},
        "上海": {"temp": 28, "condition": "多云"},
    }
    info = weather_data.get(city, {"temp": "未知", "condition": "未知"})
    return json.dumps({"city": city, "temperature": info["temp"], "condition": info["condition"]})

def search_database(query: str, limit: int = 5) -> str:
    """搜索数据库中的记录"""
    results = [f"结果{i}: 关于 {query} 的第 {i} 条记录" for i in range(1, limit + 1)]
    return json.dumps({"query": query, "results": results})

# 2. 注册工具
tools = [Tool(get_weather), Tool(search_database)]
engine = ToolEngine(tools)

# 3. 获取工具定义，传给 LLM
tool_defs = engine.get_openai_tools()
```

### 3.4 更完善的方案：Instructor

[Instructor](https://github.com/jxnl/instructor) 是一个轻量级的 Python 库，将 Pydantic 模型与 LLM API 结合。

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())

class Weather(BaseModel):
    city: str
    temperature: int
    unit: str = "celsius"

# 直接获取结构化输出
weather = client.chat.completions.create(
    model="gpt-4o",
    response_model=Weather,
    messages=[{"role": "user", "content": "北京天气怎么样？"}]
)
print(weather.city, weather.temperature)
```

---

## 4. 完整的 Python 示例代码

### 4.1 OpenAI 版本

```python
import json
from openai import OpenAI

# ========== 1. 定义工具函数 ==========

def get_weather(city: str) -> str:
    """获取天气（模拟）"""
    db = {
        "北京": "25°C, 晴",
        "上海": "28°C, 多云",
        "广州": "32°C, 阵雨",
    }
    result = db.get(city, f"{city}: 暂无数据")
    return json.dumps({"city": city, "weather": result})

def search_hotels(city: str, check_in: str = None) -> str:
    """搜索指定城市的酒店"""
    hotels = {
        "北京": ["北京饭店", "王府井希尔顿", "国贸大酒店"],
        "上海": ["外滩华尔道夫", "浦东香格里拉", "静安瑞吉"],
    }
    result = hotels.get(city, [f"{city}: 暂无酒店数据"])
    return json.dumps({"city": city, "hotels": result, "check_in": check_in})

# ========== 2. 构建工具定义 ==========

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如 北京、上海"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "搜索指定城市的酒店信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "check_in": {"type": "string", "description": "入住日期，格式 YYYY-MM-DD"}
                },
                "required": ["city"]
            }
        }
    }
]

# ========== 3. 工具路由表 ==========

available_functions = {
    "get_weather": get_weather,
    "search_hotels": search_hotels,
}

# ========== 4. 主循环 ==========

def run_tool_calling(messages, model="gpt-4o", max_turns=5):
    """执行 Tool Calling 主循环"""
    client = OpenAI()

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # 如果没有工具调用，直接返回
        if not message.tool_calls:
            return message.content

        # 将 assistant 消息加入对话
        messages.append(message)

        # 执行每个工具调用
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            fn_to_call = available_functions[fn_name]

            print(f"  [调用工具] {fn_name}({fn_args})")
            result = fn_to_call(**fn_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "已达到最大调用轮数"

# ========== 5. 使用 ==========

messages = [{"role": "user", "content": "我明天想去北京旅游，帮我查一下北京的天气和酒店"}]
result = run_tool_calling(messages)
print(f"\n最终回复: {result}")
```

### 4.2 Ollama 本地模型版本

```python
import json
import ollama

# ========== 1. 定义工具函数 ==========

def get_weather(city: str) -> str:
    """获取天气（模拟）"""
    db = {
        "北京": "25°C, 晴",
        "上海": "28°C, 多云",
        "广州": "32°C, 阵雨",
    }
    result = db.get(city, f"{city}: 暂无数据")
    return json.dumps({"city": city, "weather": result})

# ========== 2. 工具定义 ==========

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"],
            },
        },
    },
]

# ========== 3. 调用循环 ==========

def chat_with_tools(model: str = "llama3.1"):
    """与本地模型对话，支持工具调用"""
    messages = [{"role": "user", "content": "北京天气怎么样？"}]

    response = ollama.chat(
        model=model,
        messages=messages,
        tools=tools,
    )

    message = response["message"]

    if message.get("tool_calls"):
        messages.append({
            "role": "assistant",
            "content": message["content"],
            "tool_calls": message["tool_calls"],
        })

        for tc in message["tool_calls"]:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            print(f"  [调用工具] {fn_name}({fn_args})")
            result = get_weather(**fn_args)

            messages.append({
                "role": "tool",
                "content": result,
                "tool_name": fn_name,
            })

        final = ollama.chat(model=model, messages=messages)
        print(f"\n最终回复: {final['message']['content']}")
    else:
        print(f"回复: {message['content']}")

if __name__ == "__main__":
    chat_with_tools()
```

### 4.3 LiteLLM 统一接口版本

```python
import json
from litellm import completion

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    }
]

available_functions = {
    "get_weather": lambda city: json.dumps({"city": city, "weather": "25°C, 晴"})
}

def run_with_provider(model: str, provider_label: str):
    print(f"\n=== 使用 {provider_label} ===")
    messages = [{"role": "user", "content": "北京天气怎么样？"}]

    response = completion(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)
        for tc in message.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            result = available_functions[fn_name](**fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        final = completion(model=model, messages=messages)
        print(f"回复: {final.choices[0].message.content}")

# 一行代码切换供应商
run_with_provider("openai/gpt-4o", "OpenAI GPT-4o")
# run_with_provider("anthropic/claude-sonnet-4", "Anthropic Claude")
# run_with_provider("ollama/llama3.1", "Ollama Llama 3.1")
```

---

## 5. 错误处理与重试策略

### 5.1 常见错误类型

| 错误类型 | 发生阶段 | 原因 | 处理方式 |
|----------|----------|------|----------|
| JSON 解析失败 | 解析工具参数 | 模型输出不合法的 JSON | 捕获异常，要求模型重新生成 |
| 参数校验失败 | 执行工具前 | 缺少必需参数/类型错误 | 返回错误信息给模型，让其修正 |
| 工具执行异常 | 执行工具时 | API 调用超时、数据库连不上 | 重试，或返回错误信息 |
| 工具不存在 | 路由阶段 | 模型幻觉，生成了不存在的工具名 | 返回"未知工具"给模型 |
| 超时 | 任意阶段 | 模型响应太慢 | 设置 timeout，超时后重试 |
| 达到最大轮数 | 循环控制 | 模型陷入无限调用循环 | 设置 max_turns 上限 |

### 5.2 健壮的工具执行引擎

```python
import json
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ToolResult:
    success: bool
    content: str
    error_message: Optional[str] = None


class RobustToolEngine:
    """带错误处理和重试的工具执行引擎"""

    def __init__(
        self,
        tools: Dict[str, Callable],
        max_retries: int = 2,
        retry_delay: float = 0.5,
    ):
        self.tools = tools
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def execute(self, tool_call: Dict) -> ToolResult:
        """执行工具调用，带重试逻辑"""
        fn_name = tool_call.get("function", {}).get("name", "")
        fn_args_raw = tool_call.get("function", {}).get("arguments", "{}")

        if fn_name not in self.tools:
            return ToolResult(
                success=False,
                content=json.dumps({"error": f"未知工具 '{fn_name}', 可用工具: {list(self.tools.keys())}"}),
                error_message=f"未知工具: {fn_name}",
            )

        try:
            fn_args = json.loads(fn_args_raw)
        except json.JSONDecodeError as e:
            return ToolResult(
                success=False,
                content=json.dumps({"error": f"参数 JSON 解析失败: {str(e)}"}),
                error_message=f"JSON 解析失败: {str(e)}",
            )

        last_error = None
        for attempt in range(1 + self.max_retries):
            try:
                result = self.tools[fn_name](**fn_args)
                if not isinstance(result, str):
                    result = json.dumps(result, ensure_ascii=False)
                return ToolResult(success=True, content=result)
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                continue

        return ToolResult(
            success=False,
            content=json.dumps({"error": f"执行失败（已重试 {self.max_retries} 次）: {last_error}"}),
            error_message=last_error,
        )
```

### 5.3 完整的重试循环

```python
def tool_calling_loop(
    client,
    model: str,
    messages: List[Dict],
    tools: List[Dict],
    tool_engine: RobustToolEngine,
    max_rounds: int = 5,
) -> str:
    """
    完整的 Tool Calling 循环，包含错误恢复。
    """
    consecutive_failures = 0

    for round_num in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or "（无回复）"

        messages.append(message)

        all_success = True
        for tc in message.tool_calls:
            result = tool_engine.execute(tc)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result.content,
            })
            if not result.success:
                all_success = False
                consecutive_failures += 1
            else:
                consecutive_failures = 0

        if consecutive_failures >= 3:
            return f"工具调用连续失败 {consecutive_failures} 次，已终止。"

    return "已达到最大对话轮数"
```

---

## 6. 在 RAG 系统中集成的建议

### 6.1 典型集成场景

在 compact-rag 这样的 RAG 系统中，Tool Calling 可以用于：

| 场景 | 工具示例 | 说明 |
|------|----------|------|
| **数据库查询** | query_sql(sql) | 自然语言 -> SQL -> 查询数据库 |
| **文档检索** | retrieve_docs(query, top_k) | 从向量数据库检索相关文档 |
| **API 调用** | call_external_api(endpoint, params) | 调用外部服务获取实时数据 |
| **计算/分析** | calculate(formula) | 执行数学计算或数据分析 |
| **知识库搜索** | search_knowledge_base(keyword) | 搜索内部知识库 |

### 6.2 推荐架构

```
用户查询
    |
    v
[路由判断] - 简单分类器或 LLM 本身
    |
    +-- 需要实时数据 --> [Tool Calling]
    |                     +-- 数据库查询工具
    |                     +-- 文档检索工具
    |                     +-- 天气/新闻等外部 API
    |                     +-- 结果返回 LLM
    |
    +-- 纯知识问答 --> [RAG 直接检索]
                          +-- 向量检索
                          +-- 重排序
                          +-- 生成回答
```

### 6.3 RAG + Tool Calling 集成示例

```python
"""
compact-rag 中集成 Tool Calling 的建议实现
功能：
  1. 文档检索（向量搜索）
  2. 数据库查询
  3. 时间查询（外部 API 示例）
"""

import json
from typing import List, Dict


# ========== RAG 工具函数 ==========

def retrieve_docs(query: str, top_k: int = 3) -> str:
    """
    从知识库中检索与查询相关的文档。

    Args:
        query: 用户的查询文本
        top_k: 返回的文档数量，默认 3
    """
    # 实际项目中替换为你的向量检索逻辑
    # results = vector_store.similarity_search(query, k=top_k)
    results = [
        {"title": f"文档 {i+1}", "content": f"关于 '{query}' 的相关内容 {i+1}...", "score": 0.95 - i * 0.1}
        for i in range(top_k)
    ]
    return json.dumps({"query": query, "results": results}, ensure_ascii=False)


def query_database(sql: str) -> str:
    """
    执行 SQL 查询并返回结果。

    Args:
        sql: SQL 查询语句（仅允许 SELECT）
    """
    if not sql.strip().upper().startswith("SELECT"):
        return json.dumps({"error": "只允许 SELECT 查询"})

    # 模拟结果
    return json.dumps({
        "sql": sql,
        "rows": [{"id": 1, "name": "示例数据 A", "value": 100}],
        "row_count": 1,
    })


# ========== 工具注册 ==========

RAG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_docs",
            "description": "从知识库中检索与查询相关的文档，用于回答用户问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用户的查询内容"},
                    "top_k": {"type": "integer", "description": "返回的文档数量，默认 3"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "执行数据库查询，获取结构化数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL 查询语句（仅支持 SELECT）"}
                },
                "required": ["sql"]
            }
        }
    }
]

RAG_TOOL_FUNCTIONS = {
    "retrieve_docs": retrieve_docs,
    "query_database": query_database,
}


# ========== RAG + Tool Calling 集成函数 ==========

def rag_with_tools(
    query: str,
    model: str = "gpt-4o",
    max_turns: int = 3,
) -> str:
    """结合 RAG 和 Tool Calling 的回答函数"""
    from openai import OpenAI
    client = OpenAI()

    system_prompt = """你是一个智能助手，结合了知识库检索和实时数据查询能力。
当用户的问题需要特定知识时，使用 retrieve_docs 检索知识库。
当用户需要查询数据时，使用 query_database。
回答要简洁准确，并引用信息来源。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=RAG_TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return message.content

        messages.append(message)

        for tc in message.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            fn = RAG_TOOL_FUNCTIONS.get(fn_name)
            if fn:
                result = fn(**fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

    return "查询完成（已达最大轮数限制）"
```

### 6.4 架构设计决策

#### 决策 1：使用什么模型？

| 需求 | 推荐模型 | 理由 |
|------|----------|------|
| 云端、高性能 | gpt-4o / claude-sonnet-4 | 最强的工具调用能力 |
| 云端、低成本 | gpt-4o-mini | 性价比高，工具调用能力扎实 |
| 本地、隐私优先 | llama3.1 / qwen2.5 via Ollama | 完全本地，无需联网 |
| 本地、高性能 | llama3.3-70b via Ollama | 本地最强工具调用 |

#### 决策 2：使用什么框架？

| 如果你的系统 | 推荐方案 |
|-------------|----------|
| 只有一个 LLM 供应商 | 直接使用该供应商的原生 SDK |
| 可能切换供应商 | 使用 LiteLLM 作为统一封装 |
| 需要复杂的 Agent 逻辑 | 考虑 LangChain / LangGraph |
| 需要结构化输出 | 用 Pydantic + Instructor |
| 需要最精简的依赖 | 使用本文中的 ~80 行 Tool 框架 |

#### 决策 3：如何与 RAG 结合？

推荐采用**分层策略**：

```
Layer 1: 简单问答（无需外部信息）
    -> 直接 LLM 回答，无需 Tool Calling

Layer 2: 知识库问答（需要检索文档）
    -> LLM + retrieve_docs 工具，最常用场景

Layer 3: 数据查询（需要实时数据）
    -> LLM + query_database / API 工具，用于报表、分析场景

Layer 4: 复杂任务（混合多个工具）
    -> LLM + 多工具编排，需要 max_turns > 1
```

对于 compact-rag，建议从 Layer 1 + Layer 2 开始，逐步增加 Layer 3 和 Layer 4。

---

## 7. 推荐方案总结

### 7.1 最终推荐

基于对 compact-rag 项目需求的分析（中小型 RAG 系统、需要低门槛、可本地部署），推荐如下：

```
主选方案：
  +---------------------------------------+
  |  Ollama（本地开发/测试）              |
  |  + OpenAI API（生产环境）              |
  |  + 轻量级 Tool 框架（自制）            |
  +---------------------------------------+

备选方案：
  +---------------------------------------+
  |  使用 LiteLLM 统一封装                |
  |  方便日后切换供应商                    |
  +---------------------------------------+

不推荐：
  X LangChain Agent -- 过度设计，依赖过重
  X 自建 Prompt 注入方案 -- 不兼容主流 API
```

### 7.2 实施路线图

```
Phase 1（1-2 天）
  +-- 集成 Ollama，用本地模型实现基础 Tool Calling
  +-- 实现 retrieve_docs 工具（连接 RAG 向量检索）
  +-- 验证端到端流程

Phase 2（3-5 天）
  +-- 增加多工具支持（数据库查询、API 调用）
  +-- 实现错误处理和重试机制
  +-- 添加 LiteLLM 支持，便于切换供应商

Phase 3（按需）
  +-- 增加流式输出的 Tool Calling 支持
  +-- 实现并行工具调用
  +-- 添加监控和日志
  +-- 考虑迁移到 LangGraph（如果 Agent 逻辑变得复杂）
```

### 7.3 关键提醒

1. **Token 消耗**：工具定义本身会消耗大量 token。对于复杂工具，考虑压缩描述文本。
2. **安全边界**：如果工具涉及代码执行或数据库写入，必须在沙箱/事务中执行。
3. **模型选择**：开源模型的 Tool Calling 能力仍然弱于 GPT-4/Claude。如果本地模型调用频繁出错，考虑回退到云端模型。
4. **调试**：使用 OpenAI 和 Ollama 的日志模式可以更容易地调试工具调用过程。
5. **用户体验**：工具调用期间给用户适当的"加载中"反馈，避免用户以为系统无响应。

### 7.4 参考资源

- [OpenAI Function Calling 文档](https://platform.openai.com/docs/guides/function-calling)
- [Ollama Tool Calling 博客](https://ollama.com/blog/tool-support)
- [Ollama Tool Calling API 文档](https://github.com/ollama/ollama/blob/main/docs/api.md#tool-calling)
- [LiteLLM Function Calling 文档](https://docs.litellm.ai/docs/completion/function_call)
- [LangChain Agent 快速开始](https://python.langchain.com/docs/agents/quickstart/)
- [Instructor: 结构化输出库](https://github.com/jxnl/instructor)
- [Anthropic Claude Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview)

---

> **文档维护者**: compact-rag 团队
> **更新日期**: 2026-05-23
> **下一阶段**: 根据 Phase 1 实施结果，更新方案细节
