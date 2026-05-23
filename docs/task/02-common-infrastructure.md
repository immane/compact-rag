# 任务 02: 公共基础设施

> **依赖**: 01-配置管理 | **优先级**: P0 | **预计工时**: 3h

## 目标

实现项目级别的公共基础设施：结构化日志系统（loguru）+ 统一异常类体系。

## 产出文件

```
src/compact_rag/common/
├── __init__.py
├── logger.py              # loguru 结构化日志配置
└── exceptions.py          # 统一异常类定义
```

## 详细需求

### 1. `logger.py` — 日志系统

- 基于 **loguru** 配置全局 logger
- 读取 `Settings.log_level` 控制日志级别
- 开发环境：彩色控制台输出（`colorize=True`）
- 生产环境：JSON 序列化格式（`serialize=True`），可接入 ELK/Grafana
- 关键路径需要埋点日志（INFO 级别）：
  - 摄入开始/结束（含文件名、文件大小）
  - 检索耗时（含 query、结果数、latency_ms）
  - LLM 调用耗时（含 model、token 用量）
  - API 请求日志（method、path、status_code、duration）
- 提供 `get_logger(name)` 工厂函数，返回带有模块名称的 logger

```python
# 使用示例
from compact_rag.common.logger import get_logger
logger = get_logger(__name__)
logger.info("Document ingestion started", filename="report.pdf", file_size=1024000)
```

### 2. `exceptions.py` — 异常体系

按以下层级定义异常类，所有异常继承自 `CompactRAGException`：

```
CompactRAGException (基类，含 request_id 字段)
├── ConfigurationError          # 配置错误（缺失/无效字段）
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
│   └── EmptyResultError        # 空结果（非错误，降级处理用）
├── GenerationError             # LLM 生成错误
│   ├── LLMTimeoutError         # 超时
│   ├── LLMAuthError            # 认证失败
│   └── LLMRateLimitError       # 速率限制
└── ToolExecutionError          # 工具执行错误
```

每个异常类需：
- 包含 `message` 和可选的 `details` (dict)
- `__str__` 返回人类可读的错误信息
- 基类 `CompactRAGException` 自动生成 `request_id`（uuid4）

## 验收标准

- [ ] `get_logger(__name__)` 在开发环境输出彩色日志，生产环境输出 JSON
- [ ] 日志级别可通过配置动态切换
- [ ] 所有异常类可正确实例化，`str(e)` 返回有意义的信息
- [ ] 异常层级正确，`except CompactRAGException` 能捕获所有自定义异常
- [ ] 异常类包含 `request_id` 用于请求追踪
