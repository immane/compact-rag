# 任务 01: 配置管理

> **依赖**: 无 | **优先级**: P0 | **预计工时**: 4h

## 目标

实现类型安全的配置管理系统，支持 YAML 文件 + 环境变量双层配置，通过 `pydantic-settings` 提供自动校验和默认值。

## 产出文件

```
src/compact_rag/config/
├── __init__.py
└── settings.py              # pydantic-settings 配置模型

config/
├── default.yaml             # 开发环境默认配置
├── production.yaml          # 生产环境覆盖配置
└── storage.yaml             # 文件存储后端配置
```

## 详细需求

### 1. `settings.py` — 配置模型

使用 `pydantic-settings` 的 `BaseSettings`，定义以下配置子模型：

| 模型 | 关键字段 | 默认值 |
|------|---------|--------|
| `DatabaseSettings` | `url`, `echo`, `pool_size`, `max_overflow` | `sqlite+aiosqlite:///data/compact-rag.db` |
| `EmbeddingSettings` | `model_name`, `device`, `normalize`, `batch_size`, `use_onnx`, `max_seq_length` | `BAAI/bge-small-zh-v1.5`, cpu |
| `ChromaDBSettings` | `persist_directory`, `collection_name` | `./data/chromadb`, `default` |
| `RetrievalSettings` | `dense_top_k`, `sparse_top_k`, `fusion_top_k`, `rerank_top_k`, `fusion_method`, `fusion_alpha` | rrf, 0.5 |
| `LLMSettings` | `provider`, `model`, `api_key`, `api_base`, `temperature`, `max_tokens`, `timeout` | openai, gpt-4o-mini |
| `IngestionSettings` | `chunk_size`, `chunk_overlap`, `chunking_strategy`, `supported_extensions` | 500, 50, recursive |
| `StorageSettings` | `backend`, 子配置 (local/minio/oss/kodo/s3) | minio |
| `AdminSettings` | `host`, `port`, `password` | 127.0.0.1, 8501, None |

外层 `Settings` 类聚合以上子模型，并提供 `load(config_path)` 类方法：
- 从 YAML 文件加载
- 环境变量优先级高于 YAML（通过 `COMPACT_RAG_` 前缀转换，如 `COMPACT_RAG_DATABASE__URL`）
- 支持 `COMPACT_RAG_CONFIG` 环境变量指定配置文件路径

### 2. YAML 配置文件

`config/default.yaml`: 开发环境（SQLite、本地 ChromaDB、本地文件存储、debug 日志）

`config/production.yaml`: 覆盖项（MySQL 连接串、生产日志级别、关闭 SQL echo）

`config/storage.yaml`: 各存储后端的连接参数模板

### 3. 环境变量映射

| 环境变量 | 覆盖字段 |
|----------|---------|
| `COMPACT_RAG_CONFIG` | 配置文件路径 |
| `DATABASE_URL` | `database.url` |
| `OPENAI_API_KEY` | `llm.api_key` |
| `ANTHROPIC_API_KEY` | `llm.api_key` |
| `OLLAMA_HOST` | `llm.api_base` |
| `LOG_LEVEL` | `log_level` |
| `STORAGE_BACKEND` | `storage.backend` |
| `MINIO_ENDPOINT` | `storage.minio.endpoint` |
| `ADMIN_PASSWORD` | `admin.password` |

## 验收标准

- [ ] `Settings.load()` 正确加载 YAML 文件
- [ ] 环境变量 `COMPACT_RAG_DATABASE__URL=mysql://...` 覆盖 YAML 中的 database.url
- [ ] 不存在的配置文件抛出 `ConfigurationError`
- [ ] 缺失必填字段时 pydantic 自动报错
- [ ] 所有字段有合理的默认值，新开发者零配置即可启动
