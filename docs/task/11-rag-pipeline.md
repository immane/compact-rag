# 任务 10: Tool Calling 子系统

> **依赖**: 02-公共基础设施 | **优先级**: P1 | **预计工时**: 8h

## 目标

实现轻量级 Tool Calling 框架，约 80 行核心代码，从 Python 函数自动生成 JSON Schema，支持工具执行、重试和多轮循环。

## 产出文件

```
src/compact_rag/tool/
├── __init__.py
├── schema.py              # Tool 定义 + JSON Schema 自动生成
├── engine.py              # 工具执行引擎（含重试 + 循环）
└── builtin.py             # 内置 RAG 工具（retrieve_docs, query_database）
```

## 详细需求

### 1. `schema.py` — Tool 封装

```python
class Tool:
    """
    工具封装：将 Python 函数包装为 LLM 可调用的 Tool

    从函数签名自动生成 OpenAI 兼容的 JSON Schema：
    - 函数名 → tool name
    - docstring → tool description
    - 类型注解 → parameter types
    - 参数默认值 → required 判断
    """

    def __init__(self, fn: Callable):
        self.fn = fn
        self.name = fn.__name__
        self.description = fn.__doc__ or ""
        self.schema = self._build_schema()

    def _build_schema(self) -> dict:
        """
        使用 inspect.signature + typing.get_type_hints
        类型映射: str→"string", int→"integer", float→"number", bool→"boolean"
        无默认值的参数加入 required 列表
        """

    def to_openai_tool(self) -> dict:
        """转换为 OpenAI tool 格式"""

    def execute(self, **kwargs):
        """执行工具函数"""
```

### 2. `engine.py` — 执行引擎

```python
class ToolEngine:
    """
    工具执行引擎
    职责：路由、参数解析、错误处理、重试
    """

    def __init__(self, tools: list[Tool], max_retries: int = 2):
        self._tool_map = {t.name: t for t in tools}
        self.max_retries = max_retries

    def get_openai_tools(self) -> list[dict]:
        """返回所有工具的 OpenAI 格式定义"""

    async def execute_tool_call(self, tool_call: dict) -> dict:
        """
        执行单个 tool_call
        返回: {"role": "tool", "name": str, "content": str, "tool_call_id": str}

        流程:
        1. 解析 tool_call 中的 name 和 arguments
        2. 在 _tool_map 中查找工具
        3. 执行工具函数
        4. 错误处理 + 重试（最多 max_retries 次）
        5. 返回工具响应消息
        """

    async def run_loop(self, llm_client: LLMClient, messages: list[dict],
                       tools: list[dict], max_rounds: int = 5) -> str:
        """
        完整 Tool Calling 循环:
        1. 发送 messages + tools 给 LLM
        2. 若 LLM 返回 tool_calls → 逐个执行 → 追加结果到 messages → 回到 1
        3. 无 tool_calls → 返回 LLM 的文本回复
        4. 超过 max_rounds → 强制 LLM 总结工具结果
        """
```

### 3. `builtin.py` — 内置工具

```python
def retrieve_docs(query: str, top_k: int = 3) -> str:
    """
    从知识库中检索与查询最相关的文档内容。
    返回相关文档的摘要。
    """
    # 通过闭包或全局注入 hybrid_retriever

def query_database(sql: str) -> str:
    """
    执行数据库查询。
    仅允许 SELECT 语句，禁止 INSERT/UPDATE/DELETE/DROP。
    返回 JSON 格式的查询结果。
    """
    # 通过闭包或全局注入 SQLAlchemy session
    # 需做 SQL 安全检查：仅允许 SELECT

RAG_TOOLS = [Tool(retrieve_docs), Tool(query_database)]
```

### 4. Tool Calling 路线图

```
Phase 1: retrieve_docs (连接 RAG 检索)
Phase 2: query_database (报表查询)
Phase 3: 自定义工具注册接口 (ToolRegistry.register 装饰器)
Phase 4: 并行工具调用 (asyncio.gather 执行多个 tool_call)
```

### 5. 工具注入模式

```python
class ToolRegistry:
    """全局工具注册中心"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """注册工具"""
        self._tools[tool.name] = tool

    def register_function(self, fn: Callable):
        """通过装饰器注册"""
        self.register(Tool(fn))
        return fn

    def get_all(self) -> list[Tool]:
        """获取所有已注册工具"""

    def get_engine(self) -> ToolEngine:
        """创建工具引擎"""
```

## 验收标准

- [ ] `Tool._build_schema()` 从函数签名正确生成 JSON Schema
- [ ] `to_openai_tool()` 输出符合 OpenAI tool 格式
- [ ] `ToolEngine.execute_tool_call` 正确解析参数并执行
- [ ] 工具执行异常时自动重试，超过 max_retries 返回错误
- [ ] `run_loop` 完整循环：LLM 调用工具 → 结果回传 → 继续对话
- [ ] `query_database` 拒绝非 SELECT 语句
- [ ] `retrieve_docs` 正确调用外部注入的 hybrid_retriever
- [ ] 找不到工具时返回友好错误而非崩溃
