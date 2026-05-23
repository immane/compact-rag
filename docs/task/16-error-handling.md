# 任务 16: 错误处理与异常体系

> **依赖**: 02-公共基础设施 | **优先级**: P0 | **预计工时**: 4h

## 目标

定义完整的异常层级和降级策略，确保系统在部分组件故障时优雅降级而非崩溃。

## 产出文件

```
src/compact_rag/common/
└── exceptions.py          # 完整异常类实现（已在任务 02 中创建）
```

*本任务与 02-公共基础设施 共享文件，此处为详细实现指南和降级策略文档。*

## 异常层级

```
CompactRAGException (基类, 包含 request_id)
├── ConfigurationError          # 配置错误
├── DocumentLoadError           # 文档加载/解析失败
│   ├── UnsupportedFormatError  # 不支持的格式
│   └── CorruptedFileError      # 文件损坏
├── IngestionError              # 摄入流程错误
│   ├── ChunkingError           # 分块失败
│   └── EmbeddingError          # 向量化失败
├── StorageError                # 存储层错误
│   ├── VectorStoreError        # ChromaDB 错误
│   ├── DatabaseError           # 关系数据库错误
│   └── FileStorageError        # 文件存储错误
│       ├── StorageBackendError # 后端连接/认证失败
│       └── FileNotFoundError   # 文件不存在
├── RetrievalError              # 检索错误
│   └── EmptyResultError        # 空结果（降级处理用）
├── GenerationError             # LLM 生成错误
│   ├── LLMTimeoutError         # 超时
│   ├── LLMAuthError            # 认证失败
│   └── LLMRateLimitError       # 速率限制
└── ToolExecutionError          # 工具执行错误
```

## 降级策略

系统的每个关键路径都需要实现优雅降级：

| 场景 | 降级行为 | 日志级别 |
|------|---------|---------|
| BM25 索引为空 | 仅用 Dense 检索，不报错 | WARNING |
| Embedding 服务不可用 | 仅用 BM25 检索 | ERROR |
| Cross-Encoder 加载失败 | 跳过重排序，直接返回融合结果 | WARNING |
| LLM API 超时 | 重试 2 次，之后返回 `LLMTimeoutError` | ERROR |
| LLM 速率限制 | 指数退避重试，超过 3 次返回 `LLMRateLimitError` | WARNING |
| 表格提取失败 | 保留原始文本，标记为未解析，继续处理 | WARNING |
| 云存储后端不可用 | 降级到本地文件存储 | ERROR |
| ChromaDB 写入失败 | 回滚 SQL 事务，标记 ingestion_job 为 failed | ERROR |
| 数据库连接池耗尽 | 返回 503 Service Unavailable | CRITICAL |

## 异常处理实现要点

### 1. 基类设计

```python
class CompactRAGException(Exception):
    def __init__(self, message: str, details: dict = None,
                 cause: Exception = None):
        self.message = message
        self.details = details or {}
        self.cause = cause
        self.request_id = str(uuid4())
        super().__init__(message)
```

### 2. FastAPI 全局异常处理器

```python
@app.exception_handler(CompactRAGException)
async def compact_rag_handler(request: Request, exc: CompactRAGException):
    status_code = _get_http_status(exc)
    return JSONResponse(
        status_code=status_code,
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

### 3. HTTP 状态码映射

| 异常 | HTTP 状态码 |
|------|------------|
| `ConfigurationError` | 500 |
| `DocumentLoadError` / `UnsupportedFormatError` / `CorruptedFileError` | 400 |
| `IngestionError` / `ChunkingError` / `EmbeddingError` | 500 |
| `FileNotFoundError` | 404 |
| `StorageBackendError` | 502 |
| `DatabaseError` | 500 |
| `VectorStoreError` | 500 |
| `EmptyResultError` | 200 (空列表) |
| `LLMTimeoutError` | 504 |
| `LLMAuthError` | 401 |
| `LLMRateLimitError` | 429 |
| `ToolExecutionError` | 500 |

### 4. 重试策略实现

```python
import asyncio
from functools import wraps

def retry(max_retries: int = 3, base_delay: float = 1.0,
          exceptions: tuple = (GenerationError, StorageBackendError)):
    """异步重试装饰器，支持指数退避"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Retry {attempt+1}/{max_retries} after {delay}s: {e}")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator
```

## 验收标准

- [ ] 所有异常类可正确实例化并携带 request_id
- [ ] FastAPI 全局异常处理器返回标准错误格式
- [ ] HTTP 状态码映射正确
- [ ] 降级策略在各模块中实现（非仅文档）
- [ ] 重试装饰器在 LLM 超时场景下正常工作
- [ ] 异常信息不泄露敏感数据（如 API key）
