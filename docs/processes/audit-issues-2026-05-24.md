# Compact-RAG 全代码库契约合规审计 — 问题清单

> **审计日期**: 2026-05-24  
> **审计基准**: `docs/design/CONTRACTS.md` v1.0 + `docs/design/DESIGN.md` v1.2  
> **审计范围**: 全部源代码 (15 模块, 9 子 agent 并行扫描)  
> **发现问题总数**: ~55 项

---

## 🔴 严重问题 (会导致运行时错误) — 5 项

### #1 RAG Pipeline 调用不存在的 `tool_engine.execute()` 方法

- **文件**: `src/compact_rag/rag/pipeline.py:53,133`
- **契约引用**: CONTRACTS 1.6 — `ToolEngine` 方法: `execute_tool_call()`, `run_loop()`, `get_openai_tools()`
- **问题**: `query()` 和 `query_stream()` 中调用 `self.tool_engine.execute(messages, self.llm_client)`，但 `ToolEngine` 根本没有 `execute` 方法。当 `tool_engine` 不为 `None` 时，会直接抛出 `AttributeError`。
- **修复方案**:
  ```python
  # 替换 pipeline.py:53
  tools = self.tool_engine.get_openai_tools()
  result = await self.tool_engine.run_loop(
      llm_client=self.llm_client,
      messages=messages,
      tools=tools,
  )
  if result:
      messages.append({"role": "assistant", "content": result})
  ```

---

### #2 AnthropicClient 将 OpenAI 格式 tools 直接透传给 Anthropic SDK

- **文件**: `src/compact_rag/generation/llm.py:208-209`
- **契约引用**: CONTRACTS 1.5 — `AnthropicClient` 必须支持 tool calling
- **问题**: 代码将 OpenAI 格式的 tools (`{"type":"function","function":{...}}`) 直接传给 `self._client.messages.create(tools=tools)`，未做格式转换。Anthropic 原生格式要求 `{"name":"...","description":"...","input_schema":{...}}`。这意味着使用 Anthropic 时 tool calling 完全不可用。
- **修复方案**:
  ```python
  def _to_anthropic_tools(self, openai_tools: list[dict]) -> list[dict]:
      result = []
      for t in openai_tools:
          f = t.get("function", {})
          result.append({
              "name": f.get("name", ""),
              "description": f.get("description", ""),
              "input_schema": f.get("parameters", {}),
          })
      return result
  ```

---

### #3 无 API Key 认证中间件

- **文件**: `src/compact_rag/api/deps.py`, `src/compact_rag/api/router.py:50-57`
- **契约引用**: CONTRACTS 3.1 — 端点表中的"建议/可选"认证列
- **问题**: API Key 管理端点存在，但从未在路由层被拦截。所有标记为"建议认证"的端点（POST/DELETE 文档、POST/DELETE 集合、DELETE 对话、API Key 管理、摄入任务查询）实际上**无需任何认证即可访问**。仅有 CORS 中间件注册。
- **修复方案**: 在 `deps.py` 中添加 `async def verify_api_key(x_api_key: str = Header(None))` 依赖，并在各受保护的 router 中注入，或注册 FastAPI 中间件检查 `X-API-Key` Header。

---

### #4 SSE 流式最终块缺少 citations

- **文件**: `src/compact_rag/api/routers/chat.py:131`
- **契约引用**: CONTRACTS 3.6 — 最终 SSE delta 块必须包含 `"citations":[...]`
- **问题**: 流式最终 chunk 只发送 `{"delta":{},"finish_reason":"stop"}`，没有 citations。流式客户端（如管理后台 Playground）永远收不到引用来源。
- **修复方案**: 在流式完成后，将 `RAGResponse.citations` 注入最后的 SSE 事件中：
  ```python
  data: {"id":"...","choices":[{"delta":{"citations":[...]},"finish_reason":"stop","index":0}]}
  ```

---

### #5 citations 使用 chroma_id 而非实际文档 ID

- **文件**: `src/compact_rag/rag/pipeline.py:231`
- **契约引用**: CONTRACTS 2.2 + 4.1 — `RAGCitation.doc_id` 应为文档 ID，ChromaDB metadata 中分别保存 `doc_id` 和 `chroma_id`
- **问题**: 代码 `doc_id = getattr(r, "id", "unknown")` 取的是 `SearchResult.id`（即 chroma_id），而非 `metadata["doc_id"]`。下游无法通过引用追溯到实际文档。
- **修复方案**:
  ```python
  doc_id = getattr(r, "metadata", {}).get("doc_id", "unknown")
  ```

---

## 🟠 高优先级问题 (功能缺失/关键偏差) — 12 项

### #6 SemanticChunker 语义分块是空桩

- **文件**: `src/compact_rag/ingestion/chunker.py:167-193`
- **契约引用**: DESIGN 5.4 — "基于 embedding 相似度阈值检测断点"
- **问题**: `split_text_with_embeddings()` 方法接收 `embeddings` 参数但从不使用。`similarity_threshold` 参数被存储但从不读取。实际行为与普通大小分块完全相同。
- **修复方案**: 实现真正的余弦相似度比较——当相邻句子 embedding 余弦相似度低于 `self.similarity_threshold` 时分段。

---

### #7 IngestionPipeline 构造函数不符合 DI 契约

- **文件**: `src/compact_rag/ingestion/pipeline.py:30-40`
- **契约引用**: CONTRACTS 8.2 — 9 参数构造注入
- **问题**: 契约要求 `__init__(self, settings, loader_factory, chunker, embedding_service, vector_store, doc_repo, chunk_repo, ingestion_repo, storage_backend)`。实际代码只接受 `settings` 和 `session`，其余依赖全部通过私有方法惰性解析。导致无法进行依赖 mock、测试困难，且隐式耦合全局 `get_settings()`。
- **修复方案**: 重构为显式构造注入所有依赖，或更新契约记录惰性解析模式。

---

### #8 BaseLoader.load() 返回类型偏差

- **文件**: `src/compact_rag/ingestion/loader.py:36`
- **契约引用**: CONTRACTS 1.1 — `load(file_path: str) → list[DocumentChunk]`
- **问题**: 实际返回 `list[LoadedPage]`，引入了契约和设计中均未记录的中间类型 `LoadedPage`（含 `page_number, content, tables, metadata`）。后续 chunker 需要知道此类型并做转换。
- **修复方案**: 更新契约包含 `LoadedPage` 类型，或直接让 loader 产出 `DocumentChunk`。

---

### #9 HyDE 查询变换是确定性空桩

- **文件**: `src/compact_rag/retrieval/query_transformer.py:27`
- **契约引用**: DESIGN 5.8.5 — "先让 LLM 生成假设答案，再用假设答案去检索"
- **问题**: `hyde_transform()` 完全忽略传入的 `llm_client`，只在查询前面拼接固定字符串 `"与问题相关的关键事实："`。这不是 HyDE，而是简单的规则变换。
- **修复方案**: 实现真实的 LLM 驱动 HyDE（调用 `llm_client.chat` 生成假设答案并返回其 embedding 用于检索），或文档化当前的退化实现。

---

### #10 OSS/Kodo/S3 存储后端为未实现空桩

- **文件**: `src/compact_rag/storage/file_storage.py:293-304`
- **契约引用**: CONTRACTS 1.8 — 五个后端: Local, MinIO, OSS, Kodo, S3
- **问题**: 仅 `LocalFileBackend` 和 `MinIOBackend` 有实际实现。OSS、Kodo、S3 三种后端调用 `get_storage_backend()` 直接抛异常提示安装 SDK，但即使 SDK 已安装也没有对应的类。
- **修复方案**: 实现 `OSSBackend`、`KodoBackend`、`S3Backend` 类（参考 `MinIOBackend` 实现模式）。

---

### #11 query_stream() 构建了 citations 但从未 yield

- **文件**: `src/compact_rag/rag/pipeline.py:153-174`
- **契约引用**: CONTRACTS 3.6 — SSE 中需要 citations
- **问题**: `query_stream()` 在内部构建 citations（line 160），但仅用于 `_save_conversation`，从未 yield 给调用方。流式调用者收不到任何引用数据。
- **修复方案**: 在流结束后 yield 最后一个包含 citations 的结构化 chunk。

---

### #12 Admin 使用 requests 而非 httpx

- **文件**: `src/compact_rag/admin/client.py:8`
- **契约引用**: DESIGN 5.14.3 / 3.1 — `httpx.Client`
- **问题**: 契约指定 `httpx.Client(timeout=30.0)`，实际使用了 `requests.Session()`。`requests` 不在技术栈清单中，且 `httpx` 是核心依赖之一。
- **修复方案**: 将 `requests.Session` 替换为 `httpx.Client(timeout=30.0)`。

---

### #13 LLMFactory 忽略 settings.api_key

- **文件**: `src/compact_rag/generation/llm.py:621-641`
- **契约引用**: CONTRACTS 2.1 — `LLMSettings.api_key: Optional[str]`（可编程设置）
- **问题**: 对于 OpenAI 和 Anthropic 两个 provider，工厂方法直接调用 `os.getenv("OPENAI_API_KEY")` / `os.getenv("ANTHROPIC_API_KEY")`，完全忽略 `settings.api_key`。如果调用方通过 `LLMSettings(api_key="sk-xxx")` 编程式传入密钥，该值会被无声丢弃。
- **修复方案**:
  ```python
  api_key = settings.api_key or os.getenv("OPENAI_API_KEY")
  ```

---

### #14 config/default.yaml 包含非规范默认值

- **文件**: `config/default.yaml:33-36`
- **契约引用**: CONTRACTS 5.5 — `model: "gpt-4o-mini"`, `max_tokens: 2048`
- **问题**: 文件包含 `model: "deepseek-v4-flash"`、`api_base: "https://api.deepseek.com"`、`max_tokens: 20480`、被注释的 ollama 配置。这些是部署自定义，不应出现在规范默认配置中。
- **修复方案**: 恢复规范默认值。DeepSeek 特定配置移至 `config/deepseek.yaml` 或 `.env`。

---

### #15 production.yaml 自动加载逻辑未实现

- **文件**: `src/compact_rag/config/settings.py:157-158`
- **契约引用**: CONTRACTS 5.1 — 优先级链中 `production.yaml` 应在 `.env` 与 `default.yaml` 之间
- **问题**: 代码注释说会检查 `production.yaml` 存在即自动加载，但实际 `load()` 方法中无此逻辑。
- **修复方案**: 添加 auto-load 逻辑或更新注释/契约。

---

### #16 pyproject.toml 缺少/版本不匹配的依赖

- **文件**: `pyproject.toml`
- **契约引用**: DESIGN Appendix A
- **问题**:
  | 依赖 | 契约版本 | 实际 | 状态 |
  |------|---------|------|------|
  | `asyncmy` | >=0.2 | 不存在 | 缺失 |
  | `anthropic` | >=0.25 | 不存在 | 缺失 |
  | `camelot-py[cv]` | >=0.12 | >=0.11 | 版本低 |
  | `pdfplumber` | >=0.11 | >=0.10 | 版本低 |
  | `ollama` | >=0.4 | >=0.1 | 版本低 |
  | `minio/oss2/qiniu/boto3` | core deps | optional deps | 分类错误 |
  | `pandas` (admin) | >=2.0 | 不存在 | 缺失 |
  | `production` group | `["asyncmy>=0.2"]` | 不存在 | 缺失 |
- **修复方案**: 按契约补全并修正所有依赖。

---

### #17 conftest.py 缺失契约要求的 fixtures

- **文件**: `tests/conftest.py`
- **契约引用**: DESIGN 10.3
- **问题**: 契约描述了 5 个 fixtures，现有代码中：
  | 契约 fixture | 实际 | 状态 |
  |-------------|------|------|
  | `test_db` (async) | `test_db_engine` + `test_session` | 命名偏差 |
  | `test_chromadb` (async) | `mock_chromadb_client` (sync) | 命名/类型偏差 |
  | `test_documents` (async) | `sample_text` + `sample_chunks` | 缺失 |
  | `mock_llm_client` | `mock_llm_client` | ✅ |
  | `test_rag_pipeline` (async) | 不存在 | 缺失 |
- **修复方案**: 按 DESIGN 10.3 添加匹配的 fixtures。

---

## 🟡 中优先级问题 (类型安全/设计偏差) — 16 项

### #18 RAG_CONTEXT 模板从未被使用

- **文件**: `src/compact_rag/rag/pipeline.py:186-194` vs `src/compact_rag/generation/prompt.py:19-27`
- **问题**: `prompt.py` 中定义了 `_DEFAULT_RAG_CONTEXT` Jinja2 模板和 `render_rag_context()` 方法，但 `_build_context()` 用手写的字符串拼接代替。模板成为死代码。
- **修复方案**: `_build_context()` 改用 `self.prompt_manager.render_rag_context(documents)`。

---

### #19 render_system_prompt 未传 collections

- **文件**: `src/compact_rag/rag/pipeline.py:177`
- **问题**: 模板包含 `{{ collections | join(", ") }}` 占位符，但调用时未传参，导致输出"可用集合："后为空。
- **修复方案**: 传入 `collections=[collection]`。

---

### #20 IngestionResult.status 使用 str 而非 Literal

- **文件**: `src/compact_rag/storage/schema.py:34`
- **问题**: `status: str  # completed | skipped | failed` 应改为 `status: Literal["completed", "skipped", "failed"]`。
- **修复方案**: 从 `typing` 导入 `Literal`，修改类型注解。

---

### #21 14 个业务模型字段有不必要的默认值

- **文件**: `src/compact_rag/storage/schema.py`
- **问题**: `DocumentChunk.chunk_index=0`, `RAGCitation.filename=""`, `RAGCitation.score=0.0` 等违反契约的必填语义。允许创建不完整的对象。
- **修复方案**: 移除不必要的默认值，强制调用者提供所有必填字段。

---

### #22 Reranker 仅对 head 重排

- **文件**: `src/compact_rag/retrieval/retriever.py:68-72`
- **问题**: 设计文档显示对全部 fused 结果重排后截断 top_k。实际代码只对前 `rerank_top_k` 个结果重排后拼接未重排的尾部。做性能优化，但设计文档不一致。
- **修复方案**: 更新 DESIGN.md 记录此优化行为。

---

### #23 RRF k 参数不可通过 settings 配置

- **文件**: `src/compact_rag/retrieval/retriever.py:62-65`
- **问题**: RRF k 参数依赖 `fusion.py` 中的默认值 `k=60`，未通过 `RetrievalSettings` 暴露为可配置项。
- **修复方案**: 添加 `fusion_k: int = 60` 到 `RetrievalSettings`，并显式传入。

---

### #24 Tool._build_schema 不处理 Optional/Union 类型

- **文件**: `src/compact_rag/tool/schema.py:36-37`
- **问题**: `_TYPE_MAP` 无法处理 `Optional[int]` → `int | None`，会回退为 `"string"`。导致 LLM 收到错误的参数类型。
- **修复方案**: 使用 `typing.get_origin()` 和 `typing.get_args()` 解包 Union/Optional。

---

### #25 query_database 内置工具为未连接数据库的空桩

- **文件**: `src/compact_rag/tool/builtin.py:21-27`
- **问题**: 实际没有数据库连接，只返回 f-string JSON。也未使用 SQLAlchemy 参数化查询。
- **修复方案**: 接入真实的 `AsyncSession`，使用 `text(sql).execution_options(is_select=True)` 参数化执行。

---

### #26 FileNotFoundError 与 Python 内置异常同名

- **文件**: `src/compact_rag/common/exceptions.py:98`
- **问题**: 自定义 `FileNotFoundError` 与 Python 内置 `FileNotFoundError` 同名。虽然通过显式导入能避免问题，但对未来维护者有歧义风险。
- **修复方案**: 重命名为 `StorageFileNotFoundError`。

---

### #27 chunk_size >= chunk_overlap 未强制

- **文件**: `src/compact_rag/config/settings.py:55-61`
- **问题**: `IngestionSettings` 缺少 Pydantic validator 来强制 `chunk_size >= chunk_overlap`。用户可能设置 `chunk_size=100, chunk_overlap=200` 导致异常。
- **修复方案**: 添加 `@model_validator` 验证此约束。

---

### #28 ingest_directory 不支持 force

- **文件**: `src/compact_rag/ingestion/pipeline.py:284`
- **问题**: `ingest_directory()` 调用 `ingest_file()` 时不传 `force`，无法强制重新摄入整个目录。
- **修复方案**: 添加 `force: bool = False` 参数并透传。

---

### #29 _get_session 每次创建新 engine

- **文件**: `src/compact_rag/ingestion/pipeline.py:374-381`
- **问题**: 每次调用 `_get_session()` 都执行 `create_engine()` + `create_session_factory()` + `factory()`，创建新的 engine 实例和连接池，可能导致 MySQL 连接池耗尽。
- **修复方案**: 缓存 engine 和 session_factory 为实例变量。

---

### #30 缺少通用异常处理器

- **文件**: `src/compact_rag/api/router.py:60`
- **问题**: 仅注册了 `CompactRAGException` 的异常处理器。其他异常（`ValueError`, `KeyError`, `HTTPException`, `ValidationError`）将使用 FastAPI 默认格式，不符合契约统一错误格式 `{error:{code,message,details,request_id}}`。
- **修复方案**: 添加 `@app.exception_handler(Exception)` 通用 handler。

---

### #31 /v1/info 硬编码 embedding_dimension

- **文件**: `src/compact_rag/api/routers/system.py:87`
- **问题**: `embedding_dimension=384` 硬编码。如果加载 BGE-base（768 维）等模型，返回值将不正确。
- **修复方案**: 动态读取 `EmbeddingService.dimension` 属性。

---

### #32 引用包含所有检索结果而非仅 LLM 引用的

- **文件**: `src/compact_rag/rag/pipeline.py:228-243`
- **问题**: `_build_citations` 把所有检索结果都转为引文，未按 LLM 回答中的 `[Document N]` 标记过滤，导致假阳性引用。
- **修复方案**: 解析 LLM 回复中的 `[Document N]` 脚注标记，仅返回被引用的文档。

---

### #33 Admin 缺少 get_version() 和侧边栏版本

- **文件**: `src/compact_rag/admin/app.py`
- **问题**: 契约要求侧边栏显示 `f"版本: {get_version()}"`，但该函数和 UI 元素均不存在。
- **修复方案**: 添加 `get_version()` 函数（读 `compact_rag.__version__`），在侧边栏中显示。

---

## 🔵 低优先级问题 — 20+ 项

| # | 文件 | 问题 |
|---|------|------|
| 34 | `storage/vector_store.py:97` | `page_number` 用 `or 0` 替代 `None` |
| 35 | `storage/vector_store.py:74,127,...` | VectorStore 方法是 sync 而非 async（设计"异步优先"） |
| 36 | `storage/db/models.py:80` | `metadata_` 列名与 SQLAlchemy 内部冲突 |
| 37 | `storage/db/models.py:128` | `Conversation.title` 默认值用英文非契约中文 |
| 38 | `storage/db/engine.py:15` | `create_engine()` 接受 `DatabaseSettings` 而非 `Settings` |
| 39 | `generation/prompt.py:35-36` | `PromptManager.__init__` 编译模板后丢弃返回值 |
| 40 | `tool/schema.py:63` | `Tool.execute()` 返回 `-> str` 而非 `-> Any` |
| 41 | `tool/engine.py:53` | 返回 `tool_call_id` 键名而非契约的 `id` |
| 42 | `generation/llm.py:326` | `OllamaClient.chat()` 杂乱叠加 HTTP fallback/retry 逻辑 |
| 43 | `rag/pipeline.py:34` | `query()` 返回类型标注为 `Any` 而非 `RAGResponse` |
| 44 | `rag/pipeline.py:34-44` | `query()` 参数名与契约不完全匹配 |
| 45 | `rag/pipeline.py:74-77` | `stream` 参数在 `query()` 中无实际效果 |
| 46 | `rag/pipeline.py:179` | 历史消息窗口硬编码截断最后 20 条 |
| 47 | `api/routers/conversations.py:55` | 路径参数 `{conv_id}` 而非契约的 `{id}` |
| 48 | `api/routers/ingestion.py:64` | 路径参数 `{job_id}` 而非契约的 `{id}` |
| 49 | `api/routers/api_keys.py:99` | 路径参数 `{key_id}` 而非契约的 `{id}` |
| 50 | `api/schemas.py:232` | `HealthResponse` 默认值 `"degraded"` 而非 `"ok"` |
| 51 | `admin/client.py:24` | 默认 `base_url` 用 `127.0.0.1` 而非 `localhost` |
| 52 | `admin/client.py:122` | `upload_document` 接收 `bytes` 非 `file_path` |
| 53 | `admin/app.py` | 无 `main()` 函数包装 |
| 54 | `embedding/service.py:105/122` | `encode` 与 `encode_query` 空输入处理不一致 |
| 55 | `retrieval/retriever.py:25` | `query_transformer` 参数未在契约中记载 |

---

## 合规确认项 (PASS)

以下模块和功能**完全符合契约**，无需任何修改：

| 类别 | 确认项 |
|------|--------|
| **异常体系** | 完整 21 类异常层级 + `LLMServiceError` 扩展，映射正确的 HTTP 状态码 |
| **日志** | loguru 结构化日志，API key 脱敏 patcher，开发/生产模式切换 |
| **ORM 模型** | 全部 8 张表存在，所有列、约束、外键、CASCADE 规则正确 |
| **Repository 层** | 9 个 Repository（含 4 个额外扩展）全部实现标准 CRUD |
| **VectorStore** | 6 个契约要求的方法全部实现，metadata 格式符合 CONTRACTS 4.1 |
| **StorageBackend ABC** | 8 个抽象方法全部正确定义，`build_storage_key` 路径策略正确 |
| **Pydantic 模型** | 6 个业务模型全部存在，字段匹配（类型注解偏差见 #20） |
| **Settings 配置** | 全部子模型存在，字段和默认值匹配，YAML 加载正确 |
| **文档加载器** | 5 种格式全部支持，LoaderFactory 注册正确 |
| **表格提取** | Camelot + pdfplumber 后备策略已实现，质量评估函数存在 |
| **Chunker** | Recursive 分隔符匹配 DESIGN 5.4，TableAwareChunker 已实现 |
| **BM25 检索** | rank_bm25 正确使用，jieba 中文分词正确 |
| **RRF/RSF 融合** | RRF 公式 `1/(k+rank)` 正确，RSF 加权融合正确 |
| **Reranker** | CrossEncoder 正确集成，`asyncio.to_thread()` 非阻塞 |
| **LLM 抽象** | 3 个客户端实现，工厂方法正确，ChatResponse 字段完整 |
| **Tool Calling** | Tool/ToolEngine 核心逻辑正确，run_loop 实现正确 |
| **Prompt Manager** | SYSTEM_PROMPT 和 RAG_CONTEXT 模板已定义 |
| **API 端点** | 全部 21 个端点已实现（路径参数命名偏差见 #47-49） |
| **API 分页** | `{data, pagination: {page, page_size, total, total_pages}}` 格式正确 |
| **健康检查** | `/v1/health` 实际检查 DB + ChromaDB + Storage |
| **SSE 流式** | `data:` 前缀 + `\n\n` 分隔 + `[DONE]` 终端正确 |
| **Admin 页面** | 8 个页面 + 3 个组件全部存在，render 函数正确定义 |
| **Admin 方法** | 22 个契约方法全部在 `AdminAPIClient` 中存在 |
| **Alembic** | 配置正确，初始迁移包含 8 张表 |
| **Makefile** | 所有必要命令存在 (install, serve, admin, test, lint, migrate) |
| **__version__** | `src/compact_rag/__init__.py` 正确定义 `"0.1.0"` |

---

## 修复优先级建议

| 优先级 | 问题编号 | 预计工时 | 说明 |
|--------|---------|---------|------|
| **P0 (立即)** | #1, #3 | 2-4h | 运行时崩溃 + 安全漏洞 |
| **P0 (立即)** | #2, #5 | 2-3h | 功能不可用 + 数据错误 |
| **P1 (本周)** | #6, #7, #8, #10, #11, #16 | 8-16h | 核心功能缺失 + 契约严重偏差 |
| **P1 (本周)** | #12, #13, #14, #15 | 3-5h | 技术栈/配置规范的偏差 |
| **P2 (下一轮)** | #4, #9, #17-#33 | 12-20h | 类型安全 + 设计偏差 |
| **P3 (后续)** | #34-#55 | 8-12h | 风格/命名/文档一致性 |
