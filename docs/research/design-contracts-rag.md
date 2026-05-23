# RAG 系统设计契约最佳实践

> 研究日期: 2026-05-24 | 来源: FastAPI / Pydantic / ChromaDB 官方文档 + 生产实践

---

## 1. REST API 契约设计

### 1.1 OpenAI 兼容格式

```
POST /v1/chat/completions
Request:  {model, messages, stream, ...扩展字段: collection, retrieval, tools}
Response: {id, object, created, model, choices[{message{role,content,citations}}], usage}
Stream:   data: {json}\n\n ... data: [DONE]
```

### 1.2 统一分页契约

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

### 1.3 统一错误信封

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {},
    "request_id": "req-xxx"
  }
}
```

### 1.4 SSE 流式契约

```
data: {"id":"...","choices":[{"delta":{"content":"text"}}]}
data: {"id":"...","choices":[{"delta":{},"finish_reason":"stop","citations":[...]}]}
data: [DONE]
```

---

## 2. Python ABC 接口契约

### 2.1 抽象基类模式

```python
from abc import ABC, abstractmethod

class StorageBackend(ABC):
    @abstractmethod
    async def upload_file(self, local_path: str, remote_key: str) -> str: ...
    @abstractmethod
    async def download_file(self, remote_key: str, local_path: str) -> str: ...
    @abstractmethod
    async def delete(self, remote_key: str) -> bool: ...
    @abstractmethod
    async def list(self, prefix: str = "") -> list[str]: ...
    @abstractmethod
    async def exists(self, remote_key: str) -> bool: ...
    @abstractmethod
    async def get_url(self, remote_key: str, expires: int = 3600) -> str: ...
```

### 2.2 策略 + 工厂模式

```python
@lru_cache()
def get_storage_backend(settings) -> StorageBackend:
    """配置驱动切换后端，零代码改动"""

class LLMFactory:
    @staticmethod
    def create(settings: LLMSettings) -> LLMClient:
        """根据 provider 字段返回对应客户端"""
```

### 2.3 依赖注入 (FastAPI Depends)

```python
# deps.py
async def get_settings() -> Settings: ...
async def get_db_session() -> AsyncGenerator[AsyncSession, None]: ...
async def get_rag_pipeline(settings=Depends(get_settings)) -> RAGPipeline: ...

# router.py
@router.post("/v1/chat/completions")
async def chat(request: ChatRequest, pipeline=Depends(get_rag_pipeline)): ...
```

---

## 3. Pydantic 数据模型契约

### 3.1 嵌套配置模型

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='COMPACT_RAG_',
        env_nested_delimiter='__',
        cli_parse_args=True
    )
    database: DatabaseSettings = DatabaseSettings()
    llm: LLMSettings = LLMSettings()

class LLMSettings(BaseModel):
    provider: Literal["openai", "anthropic", "ollama"] = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    timeout: int = Field(default=60, gt=0)
```

### 3.2 请求/响应模型

```python
class RAGResponse(BaseModel):
    id: str
    answer: str
    citations: list[RAGCitation]
    token_usage: dict
    retrieval_latency_ms: float
    generation_latency_ms: float
```

### 3.3 配置优先级链

```
CLI args → 环境变量 → .env → production.yaml → default.yaml → Pydantic 默认值
```

---

## 4. ChromaDB 存储契约

### 4.1 写入契约

| 参数 | 类型 | 必填 |
|------|------|------|
| `ids` | `list[str]` | 是 |
| `documents` | `list[str]` | embeddings 或 documents 二选一 |
| `embeddings` | `list[list[float]]` | embeddings 或 documents 二选一 |
| `metadatas` | `list[dict]` | 否 |

### 4.2 Metadata 类型约束

允许类型: `str`, `int`, `float`, `bool`，及其数组。数组中所有元素必须同类型。

### 4.3 双数据库同步契约

```
ChromaDB.add() → 返回 chroma_id → 同步 INSERT document_chunks
删除文档 → ChromaDB.delete(ids=[...]) → SQL DELETE CASCADE
关联键: chroma_id ↔ document_chunks.chroma_id
```

---

## 5. 错误处理契约

### 5.1 异常层级

```
CompactRAGException (基类, 含 request_id)
├── ConfigurationError
├── DocumentLoadError → UnsupportedFormatError, CorruptedFileError
├── IngestionError → ChunkingError, EmbeddingError
├── StorageError → VectorStoreError, DatabaseError, FileStorageError
├── RetrievalError → EmptyResultError
├── GenerationError → LLMTimeoutError, LLMAuthError, LLMRateLimitError
└── ToolExecutionError
```

### 5.2 HTTP 状态码映射

| 异常 | HTTP |
|------|------|
| DocumentLoadError | 400 |
| FileNotFoundError | 404 |
| LLMAuthError | 401 |
| LLMRateLimitError | 429 |
| LLMTimeoutError | 504 |
| StorageBackendError | 502 |
| DatabaseError | 500 |

### 5.3 降级策略

| 故障 | 降级行为 |
|------|---------|
| BM25 索引为空 | 仅用 Dense 检索 |
| Embedding 不可用 | 仅用 BM25 |
| Cross-Encoder 加载失败 | 跳过重排序 |
| LLM 超时 | 重试 2 次后报错 |
| 云存储不可用 | 降级到本地存储 |

---

## 6. 摄入管道契约

### 6.1 Loader 接口

```python
class BaseLoader(ABC):
    @abstractmethod
    async def load(self, file_path: str) -> list[DocumentChunk]: ...

class LoaderFactory:
    @staticmethod
    def get_loader(file_path: str) -> BaseLoader: ...
```

### 6.2 支持格式

| 格式 | 解析器 | 依赖 |
|------|--------|------|
| `.pdf` | pypdf | pypdf |
| `.docx` | python-docx | python-docx |
| `.txt` | 直接读取 | 零依赖 |
| `.md` | 直接读取 | 零依赖 |
| `.html` | BeautifulSoup + markdownify | bs4, markdownify |

### 6.3 分块策略

- **Recursive**: 分隔符 `["\n\n", "\n", "。", ".", "，", ",", " ", ""]`, chunk_size=500, overlap=50
- **Semantic**: Embedding 余弦相似度阈值断点
- **Table-Aware**: 保持表格完整性，>50 行拆分表头+数据组

### 6.4 表格提取后备链

```
Camelot (Lattice) → pdfplumber → markdownify → Pandoc → PaddleOCR
```

### 6.5 增量摄入

- SHA256 哈希去重
- `force=True` 强制重新摄入

---

## 7. 配置契约

### 7.1 字段约束

| 字段 | 类型/范围 |
|------|----------|
| `llm.provider` | `Literal["openai", "anthropic", "ollama"]` |
| `retrieval.fusion_method` | `Literal["rrf", "rsf"]` |
| `ingestion.chunking_strategy` | `Literal["recursive", "semantic"]` |
| `storage.backend` | `Literal["local", "minio", "oss", "kodo", "s3"]` |
| `embedding.device` | `"cpu"` or `"cuda"` |
| `chunk_size` | >= chunk_overlap |
| `temperature` | 0.0 ~ 2.0 |
| `timeout` | > 0 |

### 7.2 环境变量映射

```
COMPACT_RAG_LLM__API_KEY → settings.llm.api_key
COMPACT_RAG_DATABASE__URL → settings.database.url
```

---

## 8. 性能基准契约

### 8.1 检索延迟 (8 万文档)

| 配置 | 延迟 | 内存 | Recall@10 |
|------|------|------|----------|
| BM25 only | 15ms | 120MB | 0.72 |
| Dense only (ONNX) | 8ms | 180MB | 0.81 |
| Hybrid (RRF) | 20ms | 220MB | 0.87 |
| Hybrid + Cross-Encoder | 35ms | 320MB | 0.91 |

### 8.2 Embedding 加速

| 方案 | 加速比 |
|------|--------|
| ONNX Runtime | 2-3x |
| OpenVINO | 3x |
| int8 量化 | 4x 内存节省 |
| Matryoshka 截断 | 2-4x |

### 8.3 存储性价比

| 服务商 | 存储 (元/GB/月) | 外网流量 (元/GB) |
|--------|----------------|-----------------|
| 七牛云 Kodo | 0.115 | 0.26 |
| 阿里云 OSS | 0.12 | 0.50 |
| AWS S3 | $0.023 | $0.09 |
