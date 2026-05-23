# compact-rag 设计契约

> **版本**: v1.0 | **日期**: 2026-05-24 | **基于设计文档**: v1.2

---

## 目录

1. [系统接口契约](#1-系统接口契约)
2. [数据模型契约](#2-数据模型契约)
3. [API 契约](#3-api-契约)
4. [存储契约](#4-存储契约)
5. [配置契约](#5-配置契约)
6. [异常契约](#6-异常契约)
7. [性能契约](#7-性能契约)
8. [模块依赖契约](#8-模块依赖契约)
9. [安全契约](#9-安全契约)

---

## 1. 系统接口契约

> 定义各模块的公开接口（ABC / 工厂函数 / 公共类）。实现方必须遵守接口契约。

### 1.1 文档加载 (`ingestion/loader.py`)

```
┌──────────────────────────────────┐
│         BaseLoader (ABC)         │
├──────────────────────────────────┤
│ + load(file_path: str)           │
│   → list[DocumentChunk]          │
└──────────────────────────────────┘
         △               △
         │               │
┌────────┴────┐  ┌───────┴───────┐
│ PDFLoader   │  │ DOCXLoader    │ ...
│ .pdf        │  │ .docx         │
└─────────────┘  └───────────────┘

LoaderFactory.get_loader(file_path: str) → BaseLoader
```

### 1.2 向量化 (`embedding/service.py`)

```
┌──────────────────────────────────┐
│       EmbeddingService           │
├──────────────────────────────────┤
│ + encode(texts: list[str])       │
│   → np.ndarray                   │
│ + encode_query(query: str)       │
│   → np.ndarray                   │
│ + dimension: int (property)      │
└──────────────────────────────────┘
```

### 1.3 向量存储 (`storage/vector_store.py`)

```
┌──────────────────────────────────┐
│          VectorStore             │
├──────────────────────────────────┤
│ + add_documents(chunks, emb)     │
│   → list[str] (chroma_ids)       │
│ + search(query, top_k, where)    │
│   → list[SearchResult]           │
│ + delete_by_document(doc_id)     │
│ + delete_by_ids(chroma_ids)      │
│ + list_collections() → list[str] │
│ + count(where=None) → int        │
└──────────────────────────────────┘
```

### 1.4 混合检索 (`retrieval/retriever.py`)

```
┌──────────────────────────────────┐
│        HybridRetriever           │
├──────────────────────────────────┤
│ + retrieve(query, top_k,         │
│            collection)           │
│   → list[SearchResult]           │
│                                  │
│ 流程: Dense → Sparse → RRF →    │
│       Cross-Encoder → Top-K      │
└──────────────────────────────────┘

融合函数:
  rrf_fusion(dense, sparse, k=60, top_k=50) → list[SearchResult]
  score(d) = Σ 1 / (k + rank_i(d))
```

### 1.5 LLM 客户端 (`generation/llm.py`)

```
┌──────────────────────────────────┐
│         LLMClient (ABC)          │
├──────────────────────────────────┤
│ + chat(messages, tools,          │
│        temperature)              │
│   → ChatResponse                 │
│ + chat_stream(messages, tools,   │
│        temperature)              │
│   → AsyncGenerator[str]          │
└──────────────────────────────────┘
         △           △          △
         │           │          │
┌────────┴───┐ ┌────┴──────┐ ┌─┴──────────┐
│OpenAIClient│ │Anthropic  │ │OllamaClient│
└────────────┘ └───────────┘ └────────────┘

LLMFactory.create(settings: LLMSettings) → LLMClient
```

### 1.6 Tool Calling (`tool/`)

```
┌──────────────────────────────────┐
│             Tool                 │
├──────────────────────────────────┤
│ + name: str                      │
│ + description: str               │
│ + schema: dict (JSON Schema)     │
│ + to_openai_tool() → dict        │
│ + execute(**kwargs) → Any        │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│          ToolEngine              │
├──────────────────────────────────┤
│ + get_openai_tools() → list[dict]│
│ + execute_tool_call(tool_call)   │
│   → dict (role/name/content/id)  │
│ + run_loop(llm, messages,        │
│            tools, max_rounds=5)  │
│   → str                          │
└──────────────────────────────────┘

内置工具:
  retrieve_docs(query: str, top_k: int = 3) → str
  query_database(sql: str) → str
```

### 1.7 RAG 管线 (`rag/pipeline.py`)

```
┌──────────────────────────────────┐
│          RAGPipeline             │
├──────────────────────────────────┤
│ + query(question, conv_id,       │
│         collection, stream,      │
│         retrieval_top_k)         │
│   → RAGResponse                  │
│ + query_stream(...)              │
│   → AsyncGenerator[str]          │
└──────────────────────────────────┘
```

### 1.8 文件存储 (`storage/file_storage.py`)

```
┌──────────────────────────────────┐
│     StorageBackend (ABC)         │
├──────────────────────────────────┤
│ + upload_file(local, remote)→str │
│ + upload_bytes(data, remote)→str │
│ + download_file(remote, local)   │
│   → str                          │
│ + download_bytes(remote)→bytes   │
│ + delete(remote) → bool          │
│ + list(prefix="") → list[str]    │
│ + get_url(remote, expires)→str   │
│ + exists(remote) → bool          │
└──────────────────────────────────┘
     △     △     △     △     △
     │     │     │     │     │
  Local  MinIO OSS  Kodo   S3

@lru_cache()
get_storage_backend(settings) → StorageBackend
```

### 1.9 管理后台客户端 (`admin/client.py`)

```
┌──────────────────────────────────┐
│        AdminAPIClient            │
├──────────────────────────────────┤
│ 系统:  health(), info()          │
│ 集合:  list/create/delete        │
│ 文档:  list/upload/get/delete    │
│ 摄入:  list_jobs/get_job         │
│ 对话:  list/get/delete           │
│ 问答:  chat/chat_stream          │
│ 密钥:  list/create/toggle/delete │
│ 存储:  list/get_url/delete       │
└──────────────────────────────────┘
```

---

## 2. 数据模型契约

### 2.1 Pydantic 配置模型

```python
# 顶层
class Settings(BaseSettings):
    database: DatabaseSettings
    embedding: EmbeddingSettings
    chromadb: ChromaDBSettings
    retrieval: RetrievalSettings
    llm: LLMSettings
    ingestion: IngestionSettings
    storage: StorageSettings
    admin: AdminSettings
    log_level: str = "INFO"

# 子模型 (含所有字段和默认值)
class DatabaseSettings(BaseModel):
    url: str = "sqlite+aiosqlite:///data/compact-rag.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10

class EmbeddingSettings(BaseModel):
    model_name: str = "BAAI/bge-small-zh-v1.5"
    device: str = "cpu"
    normalize: bool = True
    batch_size: int = 64
    use_onnx: bool = False
    max_seq_length: int = 512

class ChromaDBSettings(BaseModel):
    persist_directory: str = "./data/chromadb"
    collection_name: str = "default"

class RetrievalSettings(BaseModel):
    dense_top_k: int = 100
    sparse_top_k: int = 100
    fusion_top_k: int = 50
    rerank_top_k: int = 10
    fusion_method: Literal["rrf", "rsf"] = "rrf"
    fusion_alpha: float = 0.5

class LLMSettings(BaseModel):
    provider: Literal["openai", "anthropic", "ollama"] = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None        # 留空读环境变量
    api_base: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 60

class IngestionSettings(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunking_strategy: Literal["recursive", "semantic"] = "recursive"
    supported_extensions: list[str] = [".pdf",".docx",".txt",".md",".html"]
```

### 2.2 Pydantic 业务模型

```python
class DocumentChunk(BaseModel):
    content: str
    page_number: int | None
    chunk_index: int
    is_table: bool
    token_count: int
    content_hash: str
    metadata: dict

class SearchResult(BaseModel):
    id: str                  # chroma_id
    content: str
    score: float
    metadata: dict

class IngestionResult(BaseModel):
    doc_id: str
    filename: str
    status: Literal["completed", "skipped", "failed"]
    chunk_count: int
    table_count: int
    error_message: str | None
    duration_ms: float

class RAGCitation(BaseModel):
    doc_id: str
    chunk_index: int
    page_number: int | None
    filename: str
    score: float
    content_snippet: str

class RAGResponse(BaseModel):
    id: str
    answer: str
    citations: list[RAGCitation]
    token_usage: dict
    retrieval_latency_ms: float
    generation_latency_ms: float

class ChatResponse(BaseModel):
    content: str
    tool_calls: list[dict] | None
    token_usage: dict
    model: str
    finish_reason: str
```

### 2.3 SQL 表 Schema

```
collections ────< documents ────< document_chunks [CASCADE]
collections ────< conversations [SET NULL] ────< messages [CASCADE]
collections ────< ingestion_jobs
documents ────< storage_files [SET NULL]
api_keys (独立)
```

| 表 | 关键约束 |
|----|---------|
| `collections` | name UNIQUE |
| `documents` | file_hash 索引 (去重), status ∈ {pending, processing, completed, failed} |
| `document_chunks` | chroma_id NOT NULL, document_id CASCADE DELETE |
| `conversations` | collection_id NULLABLE |
| `messages` | role ∈ {system, user, assistant, tool}, conversation_id CASCADE DELETE |
| `ingestion_jobs` | status ∈ {pending, running, completed, failed} |
| `api_keys` | key_hash UNIQUE |
| `storage_files` | storage_type ∈ {temp, persistent, archive} |

---

## 3. API 契约

### 3.1 端点总表 (21 个)

| 组 | 方法 | 路径 | 认证 |
|----|------|------|------|
| 问答 | `POST` | `/v1/chat/completions` | 可选 |
| 文档 | `POST` | `/v1/documents/ingest` | 建议 |
| 文档 | `POST` | `/v1/documents/ingest-url` | 建议 |
| 文档 | `GET` | `/v1/documents` | 否 |
| 文档 | `GET` | `/v1/documents/{doc_id}` | 否 |
| 文档 | `DELETE` | `/v1/documents/{doc_id}` | 建议 |
| 集合 | `GET` | `/v1/collections` | 否 |
| 集合 | `POST` | `/v1/collections` | 建议 |
| 集合 | `DELETE` | `/v1/collections/{name}` | 建议 |
| 对话 | `GET` | `/v1/conversations` | 否 |
| 对话 | `GET` | `/v1/conversations/{id}` | 否 |
| 对话 | `DELETE` | `/v1/conversations/{id}` | 建议 |
| 摄入 | `GET` | `/v1/ingestion-jobs` | 建议 |
| 摄入 | `GET` | `/v1/ingestion-jobs/{id}` | 建议 |
| 密钥 | `GET` | `/v1/api-keys` | 建议 |
| 密钥 | `POST` | `/v1/api-keys` | 建议 |
| 密钥 | `PATCH` | `/v1/api-keys/{id}` | 建议 |
| 密钥 | `DELETE` | `/v1/api-keys/{id}` | 建议 |
| 系统 | `GET` | `/v1/health` | 否 |
| 系统 | `GET` | `/v1/info` | 否 |
| 系统 | `GET` | `/v1/files/{storage_key}` | 否 (预签名) |

### 3.2 核心问答请求契约

```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "user", "content": "公司今年的营收目标是多少？"}
  ],
  "collection": "finance-2024",
  "retrieval": {
    "top_k": 10,
    "rerank": true,
    "hybrid_search": true
  },
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "retrieve_docs",
        "parameters": {"type": "object", "properties": {...}}
      }
    }
  ],
  "stream": false
}
```

### 3.3 核心问答响应契约

```json
{
  "id": "rag-call-xxx",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "gpt-4o",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "根据公司财报，2024年营收目标为50亿元...",
      "citations": [
        {
          "doc_id": "abc123",
          "filename": "2024-fiscal-plan.pdf",
          "page_number": 5,
          "chunk_index": 3,
          "score": 0.92,
          "content_snippet": "..."
        }
      ]
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 1250,
    "completion_tokens": 180,
    "total_tokens": 1430
  }
}
```

### 3.4 分页响应契约

所有 `GET` 列表端点使用统一格式：

```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total": 156,
    "total_pages": 8
  }
}
```

查询参数: `?page=1&page_size=20` (page_size max=100)

### 3.5 错误响应契约

```json
{
  "error": {
    "code": "DOCUMENT_NOT_FOUND",
    "message": "文档 abc-123 不存在",
    "details": {},
    "request_id": "req-xxx"
  }
}
```

### 3.6 SSE 流式契约

```
data: {"id":"rag-xxx","choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n
data: {"id":"rag-xxx","choices":[{"delta":{"content":"根据"},"index":0}]}\n\n
...\n\n
data: {"id":"rag-xxx","choices":[{"delta":{},"finish_reason":"stop","citations":[...]},"index":0]}\n\n
data: [DONE]\n\n
```

### 3.7 健康检查响应契约

```json
{
  "api": "ok",
  "database": "ok",
  "chromadb": "ok",
  "storage": "ok"
}
```

各组件状态: `"ok"` | `"degraded"` | `"error"` | `"disabled"`

---

## 4. 存储契约

### 4.1 ChromaDB Metadata 格式

每个 chunk 存储时必须包含以下 metadata：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `doc_id` | str | 是 | 关联 `documents.id` |
| `chroma_id` | str | 是 | ChromaDB 自动生成 |
| `chunk_index` | int | 是 | 本文档内块序号 |
| `page_number` | int/None | 否 | 所在页码 |
| `filename` | str | 是 | 源文件名 |
| `collection_name` | str | 是 | 所属集合 |
| `is_table` | bool | 否 | 是否为表格块 |
| `token_count` | int | 否 | Token 估算数 |

### 4.2 双库同步契约

```
写入: ChromaDB.add(ids, embeddings, documents, metadatas)
      → 返回 chroma_id → 同步 INSERT document_chunks (chroma_id, doc_id, chunk_index, ...)

删除: collection.delete(ids=[chroma_ids])
      → SQL DELETE document_chunks WHERE chroma_id IN (...)
      → SQL DELETE documents WHERE id = doc_id

关联: document_chunks.chroma_id ↔ ChromaDB id
```

### 4.3 文件路径策略

```
持久化: docs/{collection_id}/{year}/{month}/{day}/{hash16}{ext}
临时:   temp/{session_id}/{timestamp}_{filename}
归档:   archive/{collection_id}/{year}/{month}/{hash16}{ext}
```

```python
def build_storage_key(collection_id: str, filename: str, 
                      category: str = "docs") -> str:
    now = datetime.utcnow()
    date_path = f"{now.year}/{now.month:02d}/{now.day:02d}"
    file_hash = hashlib.sha256(filename.encode()).hexdigest()[:16]
    ext = Path(filename).suffix
    return f"{category}/{collection_id}/{date_path}/{file_hash}{ext}"
```

### 4.4 文件生命周期契约

```
上传 → StorageBackend.upload(temp/{session_id}/{filename})
  ↓ (解析完成)
持久化 → StorageBackend.upload(docs/{collection_id}/{date}/{hash}{ext})
  ↓ (不再活跃)
清理 → TempFileCleaner (TTL 默认 1 小时, 定时 30 分钟执行)
```

---

## 5. 配置契约

### 5.1 配置加载优先级

```
CLI args (最高)
  ↓
环境变量 (COMPACT_RAG_* 前缀)
  ↓
.env 文件
  ↓
config/production.yaml
  ↓
config/default.yaml
  ↓
Pydantic model defaults (最低)
```

### 5.2 必填环境变量 (生产)

| 变量 | 说明 |
|------|------|
| `COMPACT_RAG_CONFIG` | 配置文件路径 |
| `OPENAI_API_KEY` | OpenAI API Key (若使用 OpenAI) |
| `ANTHROPIC_API_KEY` | Anthropic API Key (若使用 Anthropic) |

### 5.3 可选环境变量

| 变量 | 覆盖字段 | 示例 |
|------|---------|------|
| `DATABASE_URL` | database.url | `mysql+asyncmy://user:pass@host:3306/compact_rag` |
| `OLLAMA_HOST` | llm.api_base | `http://localhost:11434` |
| `LOG_LEVEL` | log_level | `DEBUG` |
| `STORAGE_BACKEND` | storage.backend | `minio` |
| `ADMIN_PASSWORD` | admin.password | `xxx` |
| `COMPACT_RAG_LLM__MODEL` | llm.model | `gpt-4o` (嵌套语法) |

### 5.4 约束字段

| 字段 | 约束 |
|------|------|
| `llm.provider` | `"openai"` / `"anthropic"` / `"ollama"` |
| `retrieval.fusion_method` | `"rrf"` / `"rsf"` |
| `ingestion.chunking_strategy` | `"recursive"` / `"semantic"` |
| `storage.backend` | `"local"` / `"minio"` / `"oss"` / `"kodo"` / `"s3"` |
| `llm.temperature` | 0.0 ~ 2.0 |
| `llm.timeout` | > 0 |
| `chunk_size` | >= `chunk_overlap` |

### 5.5 YAML 配置示例

```yaml
# config/default.yaml
database:
  url: "sqlite+aiosqlite:///data/compact-rag.db"
  echo: false
embedding:
  model_name: "BAAI/bge-small-zh-v1.5"
  device: "cpu"
chromadb:
  persist_directory: "./data/chromadb"
  collection_name: "default"
retrieval:
  dense_top_k: 100
  sparse_top_k: 100
  fusion_top_k: 50
  rerank_top_k: 10
  fusion_method: "rrf"
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: 2048
  timeout: 60
ingestion:
  chunk_size: 500
  chunk_overlap: 50
  chunking_strategy: "recursive"
log_level: "INFO"
```

---

## 6. 异常契约

### 6.1 异常层级 (全 15 种)

```
CompactRAGException (基类, 自动生成 request_id: UUID)
├── ConfigurationError                # HTTP 500
├── DocumentLoadError                 # HTTP 400
│   ├── UnsupportedFormatError        # HTTP 400
│   └── CorruptedFileError            # HTTP 400
├── IngestionError                    # HTTP 500
│   ├── ChunkingError                 # HTTP 500
│   └── EmbeddingError                # HTTP 500
├── StorageError                      # HTTP 500
│   ├── VectorStoreError              # HTTP 500
│   ├── DatabaseError                 # HTTP 500
│   └── FileStorageError              # HTTP 500
│       ├── StorageBackendError       # HTTP 502
│       └── FileNotFoundError         # HTTP 404
├── RetrievalError                    # HTTP 500
│   └── EmptyResultError              # HTTP 200 (降级, 非错误)
├── GenerationError                   # HTTP 500
│   ├── LLMTimeoutError               # HTTP 504
│   ├── LLMAuthError                  # HTTP 401
│   └── LLMRateLimitError             # HTTP 429
└── ToolExecutionError                # HTTP 500
```

### 6.2 异常基类契约

```python
class CompactRAGException(Exception):
    def __init__(self, message: str, details: dict = None, cause: Exception = None):
        self.message = message
        self.details = details or {}
        self.cause = cause
        self.request_id = str(uuid4())
```

### 6.3 降级策略契约

| 场景 | 降级行为 | 日志级别 |
|------|---------|---------|
| BM25 索引为空 | 仅用 Dense 检索 | WARNING |
| Embedding 服务不可用 | 仅用 BM25 | ERROR |
| Cross-Encoder 加载失败 | 跳过重排序 | WARNING |
| LLM API 超时 | 重试 2 次 (指数退避)，仍失败则报错 | ERROR |
| LLM 速率限制 | 指数退避重试 3 次，仍失败则报错 | WARNING |
| 表格提取失败 | 保留原始文本，标记为未解析 | WARNING |
| 云存储后端不可用 | 降级到本地文件存储 | ERROR |
| ChromaDB 写入失败 | 回滚 SQL 事务，标记 job 为 failed | ERROR |
| 数据库连接池耗尽 | 返回 503 | CRITICAL |

### 6.4 全局异常处理器

```python
@app.exception_handler(CompactRAGException)
async def handler(request: Request, exc: CompactRAGException):
    return JSONResponse(
        status_code=_http_status(exc),
        content={
            "error": {
                "code": exc.__class__.__name__,
                "message": str(exc),
                "details": exc.details,
                "request_id": exc.request_id,
            }
        }
    )
```

---

## 7. 性能契约

### 7.1 检索延迟基准 (8 万文档)

| 配置 | 延迟 | 内存 | Recall@10 |
|------|------|------|----------|
| BM25 only | ≤ 15ms | ≤ 120MB | ≥ 0.72 |
| Dense only (MiniLM+ONNX) | ≤ 10ms | ≤ 180MB | ≥ 0.81 |
| Hybrid (RRF) | ≤ 25ms | ≤ 220MB | ≥ 0.87 |
| Hybrid + Cross-Encoder | ≤ 50ms | ≤ 320MB | ≥ 0.91 |

### 7.2 Embedding 吞吐基准

| 方案 | 加速比 | 单条延迟 |
|------|--------|---------|
| PyTorch CPU | 1x | ~4ms |
| ONNX Runtime | 2-3x | ~1.8ms |
| OpenVINO | 3x | ~1.3ms |
| batch_size=64 | 2-3x | — |

### 7.3 表格提取基准

| 工具 | 单页耗时 | 内存 | GPU 需求 |
|------|---------|------|---------|
| pdfplumber | 0.5-2s | 低 | 否 |
| Camelot | 1-3s | 中 | 否 |
| PaddleOCR | 5-20s | 高 | 建议 |

### 7.4 API 吞吐目标

| 端点 | 目标 QPS | 说明 |
|------|---------|------|
| `/v1/chat/completions` | ≥ 10 | (含检索) |
| `/v1/documents/ingest` | ≥ 1 (每个文件) | (含 embedding) |
| `/v1/health` | ≥ 1000 | (纯状态检查) |
| 列表型 GET | ≥ 50 | (分页后) |

### 7.5 存储性价比 (月费估算)

| 后端 | 100GB 存储 | 500GB 流量 | 合计/月 |
|------|-----------|-----------|---------|
| 七牛云 Kodo | ¥11.5 | ¥130 | ¥141.5 |
| 阿里云 OSS | ¥12 | ¥250 | ¥262 |
| AWS S3 | $2.3 | $45 | $47.3 |
| MinIO (自建) | ¥0 | ¥0 | ~¥500-1000 (服务器) |

---

## 8. 模块依赖契约

### 8.1 依赖图 (DAG)

```
01-config ──┬── 02-common ──┬── 03-database ──┬── 04-ingestion
            │               │                 ├── 11-pipeline
            │               │                 └── 12-api
            │               ├── 05-table-extraction
            │               ├── 09-llm ──────── 11-pipeline
            │               ├── 10-tool ─────── 11-pipeline
            │               ├── 13-file-storage
            │               └── 16-errors
            │
            ├── 06-embedding ── 07-vector-store ── 08-retrieval ──┬── 11-pipeline
            │                                                     └── 12-api
            ├── 09-llm
            └── 13-file-storage

14-admin ──── requires: 12-api, 13-file-storage
```

### 8.2 构造注入约定

所有模块通过 `__init__` 接收依赖，不通过全局变量查找：

```python
class IngestionPipeline:
    def __init__(self, settings, loader_factory, chunker,
                 embedding_service, vector_store, doc_repo,
                 chunk_repo, ingestion_repo, storage_backend):
        ...

class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm_client: LLMClient,
                 prompt_manager: PromptManager,
                 conversation_repo: ConversationRepository,
                 message_repo: MessageRepository,
                 tool_engine: ToolEngine | None = None):
        ...
```

### 8.3 单例缓存约定

重量级资源使用 `@lru_cache()` 进程级缓存：

```python
@lru_cache()
def get_settings() -> Settings: ...

@lru_cache()
def get_storage_backend(settings) -> StorageBackend: ...

# EmbeddingService 使用 __new__ 单例
class EmbeddingService:
    _instance = None
    def __new__(cls, settings):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### 8.4 延迟加载约定

不使用的模块不在启动时导入：

```python
# main.py
@app.command()
def serve(config: str = None):
    from compact_rag.api.router import app  # 延迟导入
    uvicorn.run(app, ...)
```

---

## 9. 安全契约

### 9.1 API 密钥安全

- 密钥使用 SHA256 哈希存储，明文仅创建时返回一次
- `key_hash` 唯一索引，防重复
- `is_active` 开关支持即时停用
- `expires_at` 支持过期自动失效
- 权限通过 JSON 数组控制: `["read"]` / `["read", "write"]` / `["admin"]`

### 9.2 管理后台安全

| 措施 | 默认行为 |
|------|---------|
| 绑定地址 | `127.0.0.1` (仅本地) |
| 认证密码 | `ADMIN_PASSWORD` 环境变量 (可选) |
| 操作审计 | 所有操作通过 API 调用，日志记录在服务端 |
| 网络隔离 | 生产环境放内网/VPN 后 |

### 9.3 数据安全

| 措施 | 实现 |
|------|------|
| SQL 注入防护 | SQLAlchemy 参数化查询，`query_database` 仅允许 SELECT |
| 文件上传 | 扩展名白名单 (`.pdf`, `.docx`, `.txt`, `.md`, `.html`) |
| 文件大小 | 通过 `python-multipart` 限制 |
| 预签名 URL | 文件下载通过短时效预签名 URL (默认 1h) |
| 无直接存储暴露 | 文件 URL 通过 `/v1/files/{key}` 代理 |
| API Key 脱敏 | 日志中不输出 API Key |

### 9.4 日志安全

- 生产环境日志自动隐藏 API Key (loguru `patcher`)
- 错误响应不包含堆栈信息 (生产环境)
- `request_id` 贯穿全链路用于审计追踪

---

## 附录 A: 设计决策记录速查

| 编号 | 决策 | 理由 |
|------|------|------|
| D-001 | ChromaDB 而非 Qdrant/Weaviate | 更轻量，Python 原生，无独立服务 |
| D-002 | MySQL/SQLite 而非 PostgreSQL | 团队熟悉 |
| D-003 | 不用 LangChain 核心 | 过度封装 |
| D-004 | 自制 Tool Calling (~80行) | 减少依赖 |
| D-005 | RRF 融合 | 无需归一化，鲁棒性最高 |
| D-006 | rank_bm25 | 零依赖，<5万条足够 |
| D-007 | 异步贯穿全栈 | FastAPI + SQLAlchemy async + httpx async |
| D-008 | Recursive 分块默认 | 通用性好，中文优化 |
| D-009 | 文件存储 ABC + 策略模式 | 与 LLM 抽象一致 |
| D-010 | 开发用 MinIO | S3 兼容，零成本 |
| D-011 | 国内用七牛云 Kodo | 外网流量费最低 (0.26 元/GB) |
| D-012 | 管理后台用 Streamlit | Python 原生，复用 REST API |

## 附录 B: 技术栈契约

| 类别 | 技术 | 最低版本 |
|------|------|---------|
| 语言 | Python | ≥ 3.11 |
| Web 框架 | FastAPI | ≥ 0.110 |
| 数据验证 | Pydantic | ≥ 2.0 |
| ORM | SQLAlchemy (async) | ≥ 2.0 |
| 向量库 | ChromaDB | ≥ 0.5 |
| Embedding | sentence-transformers | ≥ 2.7 |
| 文档解析 | pypdf / python-docx | ≥ 4.0 / ≥ 1.1 |
| LLM SDK | openai / anthropic / ollama | ≥ 1.30 / ≥ 0.25 / ≥ 0.4 |
| 日志 | loguru | ≥ 0.7 |
| 测试 | pytest + pytest-asyncio | ≥ 8.0 |
| 管理后台 | Streamlit (可选) | ≥ 1.35 |

---

> **基于**: [DESIGN.md](./DESIGN.md) v1.2 | 研究: [design-contracts-rag.md](../research/design-contracts-rag.md), [contracts-analysis.md](../research/contracts-analysis.md)
