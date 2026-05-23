# compact-rag 模块契约分析

> 从 `docs/design/DESIGN.md` v1.2 提取 | 日期: 2026-05-24

---

## 模块 → 契约映射总表

| 编号 | 模块 | 接口/类 | 数据模型 | 异常类型 | 配置依赖 |
|------|------|---------|---------|---------|---------|
| 5.1 | 配置管理 | `Settings.load()` | `DatabaseSettings`, `EmbeddingSettings`, `ChromaDBSettings`, `RetrievalSettings`, `LLMSettings`, `IngestionSettings`, `StorageSettings`, `AdminSettings` | `ConfigurationError` | YAML + env vars |
| 5.2 | 公共基础设施 | `get_logger()` | — | 15 种异常 (全层级) | `log_level` |
| 5.3 | 关系数据库 | `create_engine()`, `create_session_factory()`, 8 个 Repository | 8 张 SQLAlchemy ORM 表 | `DatabaseError` | `DatabaseSettings` |
| 5.4 | 文档摄入 | `BaseLoader`, `LoaderFactory`, `RecursiveCharacterTextSplitter`, `SemanticChunker`, `TableAwareChunker`, `IngestionPipeline` | `DocumentChunk`, `IngestionResult` | `DocumentLoadError`, `UnsupportedFormatError`, `CorruptedFileError`, `ChunkingError`, `EmbeddingError`, `IngestionError` | `IngestionSettings` |
| 5.5 | 表格提取 | Camelot→pdfplumber→markdownify→Pandoc, `evaluate_table_quality()` | Markdown 表格格式, `ExtractedTable` | 后备不抛异常 | — |
| 5.6 | 向量化服务 | `EmbeddingService` | `np.ndarray` (float32) | `EmbeddingError` | `EmbeddingSettings` |
| 5.7 | 向量存储 | `VectorStore` | `SearchResult`, ChromaDB metadata (8 字段) | `VectorStoreError` | `ChromaDBSettings` |
| 5.8 | 混合检索 | `DenseRetriever`, `BM25Retriever`, `rrf_fusion()`, `RerankerService`, `HybridRetriever` | `SearchResult` | `RetrievalError`, `EmptyResultError` | `RetrievalSettings` |
| 5.9 | LLM 生成 | `LLMClient` (ABC), `OpenAIClient`, `AnthropicClient`, `OllamaClient`, `LLMFactory`, `PromptManager` | `ChatResponse` | `GenerationError`, `LLMTimeoutError`, `LLMAuthError`, `LLMRateLimitError` | `LLMSettings` |
| 5.10 | Tool Calling | `Tool`, `ToolEngine`, `ToolRegistry` | JSON Schema (OpenAI 兼容), `ToolResult` | `ToolExecutionError` | — |
| 5.11 | RAG 管线 | `RAGPipeline` | `RAGResponse`, `RAGCitation` | 上游所有异常 | 全部 Settings |
| 5.12 | API 层 | 20 个 FastAPI 端点 + `get_rag_pipeline()` 等依赖注入 | 20 个请求/响应 Pydantic 模型 | 全局异常处理器 | `Settings` |
| 5.13 | 文件存储 | `StorageBackend`(ABC), `LocalFileBackend`, `MinIOBackend`, `OSSBackend`, `KodoBackend`, `S3Backend`, `TempFileCleaner`, `get_storage_backend()` | 文件路径: `{category}/{collection_id}/{date}/{hash}{ext}` | `FileStorageError`, `StorageBackendError`, `FileNotFoundError` | `storage.yaml` |
| 5.14 | Streamlit 后台 | `AdminAPIClient`, 8 个页面渲染函数, 3 个组件 | — | httpx 异常 | `AdminSettings` |

---

## ABC 抽象接口清单

### 1. `BaseLoader` (ingestion/loader.py)

```python
class BaseLoader(ABC):
    @abstractmethod
    async def load(self, file_path: str) -> list[DocumentChunk]: ...
```

### 2. `LLMClient` (generation/llm.py)

```python
class LLMClient(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], tools: list[dict] = None,
                   temperature: float = 0.1) -> ChatResponse: ...
    @abstractmethod
    async def chat_stream(self, messages: list[dict], tools: list[dict] = None,
                          temperature: float = 0.1) -> AsyncGenerator[str, None]: ...
```

### 3. `StorageBackend` (storage/file_storage.py)

```python
class StorageBackend(ABC):
    @abstractmethod async def upload_file(self, local_path: str, remote_key: str) -> str: ...
    @abstractmethod async def upload_bytes(self, data: bytes, remote_key: str, content_type: str = "") -> str: ...
    @abstractmethod async def download_file(self, remote_key: str, local_path: str) -> str: ...
    @abstractmethod async def download_bytes(self, remote_key: str) -> bytes: ...
    @abstractmethod async def delete(self, remote_key: str) -> bool: ...
    @abstractmethod async def list(self, prefix: str = "") -> list[str]: ...
    @abstractmethod async def get_url(self, remote_key: str, expires: int = 3600) -> str: ...
    @abstractmethod async def exists(self, remote_key: str) -> bool: ...
```

---

## 数据模型清单

### Pydantic 模型

| 模型 | 用途 | 字段数 |
|------|------|--------|
| `DatabaseSettings` | 数据库配置 | 4 |
| `EmbeddingSettings` | 向量化配置 | 6 |
| `ChromaDBSettings` | ChromaDB 配置 | 2 |
| `RetrievalSettings` | 检索配置 | 6 |
| `LLMSettings` | LLM 配置 | 7 |
| `IngestionSettings` | 摄入配置 | 4 |
| `StorageSettings` | 存储配置 | 嵌套子模型 |
| `AdminSettings` | 管理后台配置 | 3 |
| `Settings` | 顶层聚合 | 8 子模型 + log_level |
| `RAGCitation` | 引用标注输出 | 6 |
| `RAGResponse` | RAG 问答响应 | 6 |
| `SearchResult` | 检索结果 | 4 |
| `DocumentChunk` | 文档分块 | 7 |
| `IngestionResult` | 摄入结果 | 6 |
| `ChatResponse` | LLM 响应 | 4 |
| `ExtractedTable` | 表格提取结果 | 6 |

### SQL ORM 表 (8 张)

| 表 | 主键 | 外键 | 索引 |
|----|------|------|------|
| `collections` | id(UUID) | — | name(UNIQUE) |
| `documents` | id(UUID) | collection_id → collections.id | collection_id, file_hash |
| `document_chunks` | id(UUID) | document_id → documents.id(CASCADE) | document_id |
| `conversations` | id(UUID) | collection_id → collections.id(NULLABLE) | — |
| `messages` | id(UUID) | conversation_id → conversations.id(CASCADE) | conversation_id |
| `ingestion_jobs` | id(UUID) | collection_id → collections.id | — |
| `api_keys` | id(UUID) | — | key_hash(UNIQUE) |
| `storage_files` | id(UUID) | document_id → documents.id(NULLABLE) | — |

### ChromaDB Metadata (8 字段)

```json
{
  "doc_id": "uuid",
  "chroma_id": "auto",
  "chunk_index": 0,
  "page_number": 3,
  "filename": "report.pdf",
  "collection_name": "finance-2024",
  "is_table": false,
  "token_count": 245
}
```

---

## API 端点清单 (20 个)

| 组 | 方法 | 路径 |
|----|------|------|
| 问答 | `POST` | `/v1/chat/completions` |
| 文档 | `POST` | `/v1/documents/ingest` |
| 文档 | `POST` | `/v1/documents/ingest-url` |
| 文档 | `GET` | `/v1/documents` |
| 文档 | `GET` | `/v1/documents/{doc_id}` |
| 文档 | `DELETE` | `/v1/documents/{doc_id}` |
| 集合 | `GET` | `/v1/collections` |
| 集合 | `POST` | `/v1/collections` |
| 集合 | `DELETE` | `/v1/collections/{name}` |
| 对话 | `GET` | `/v1/conversations` |
| 对话 | `GET` | `/v1/conversations/{id}` |
| 对话 | `DELETE` | `/v1/conversations/{id}` |
| 摄入 | `GET` | `/v1/ingestion-jobs` |
| 摄入 | `GET` | `/v1/ingestion-jobs/{id}` |
| 密钥 | `GET` | `/v1/api-keys` |
| 密钥 | `POST` | `/v1/api-keys` |
| 密钥 | `PATCH` | `/v1/api-keys/{id}` |
| 密钥 | `DELETE` | `/v1/api-keys/{id}` |
| 系统 | `GET` | `/v1/health` |
| 系统 | `GET` | `/v1/info` |
| 系统 | `GET` | `/v1/files/{storage_key}` |

---

## 异常层级 (15 种)

```
CompactRAGException
├── ConfigurationError
├── DocumentLoadError
│   ├── UnsupportedFormatError
│   └── CorruptedFileError
├── IngestionError
│   ├── ChunkingError
│   └── EmbeddingError
├── StorageError
│   ├── VectorStoreError
│   ├── DatabaseError
│   └── FileStorageError
│       ├── StorageBackendError
│       └── FileNotFoundError
├── RetrievalError
│   └── EmptyResultError
├── GenerationError
│   ├── LLMTimeoutError
│   ├── LLMAuthError
│   └── LLMRateLimitError
└── ToolExecutionError
```

### HTTP 状态码映射

| 异常 | HTTP | 异常 | HTTP |
|------|------|------|------|
| ConfigurationError | 500 | LLMAuthError | 401 |
| DocumentLoadError | 400 | LLMRateLimitError | 429 |
| UnsupportedFormatError | 400 | LLMTimeoutError | 504 |
| CorruptedFileError | 400 | StorageBackendError | 502 |
| FileNotFoundError | 404 | ToolExecutionError | 500 |
| EmptyResultError | 200 | DatabaseError | 500 |

---

## 设计决策记录

| 编号 | 决策 | 理由 |
|------|------|------|
| D-001 | ChromaDB 而非 Qdrant | 更轻量，Python 原生 |
| D-002 | MySQL/SQLite 而非 PostgreSQL | 团队熟悉 |
| D-003 | 不用 LangChain 核心 | 过度封装 |
| D-004 | 自制 Tool Calling (~80行) | 减少依赖 |
| D-005 | RRF 融合 | 无需归一化，鲁棒性高 |
| D-006 | rank_bm25 | 零依赖，<5万条足够 |
| D-007 | 异步全栈 | 性能一致性 |
| D-008 | 默认 Recursive 分块 | 通用性好 |
| D-009 | 文件存储 ABC + 策略模式 | 与 LLM 抽象一致 |
| D-010 | 开发用 MinIO | S3 兼容，零成本 |
| D-011 | 国内用七牛云 Kodo | 流量费最低 |
| D-012 | 管理后台用 Streamlit | Python 原生，复用 API |
