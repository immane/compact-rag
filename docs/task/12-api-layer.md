# 任务 12: API 层

> **依赖**: 11-RAG 管线编排, 03-关系数据库, 13-文件存储子系统 | **优先级**: P0 | **预计工时**: 12h

## 目标

实现 FastAPI REST API 层，包含 19 个端点，覆盖问答、文档管理、集合管理、对话记录、摄入任务、API 密钥、系统接口。

## 产出文件

```
src/compact_rag/api/
├── __init__.py
├── deps.py                # FastAPI 依赖注入
├── router.py              # 路由注册（按模块分子路由）
├── schemas.py             # 请求/响应 Pydantic 模型
└── routers/
    ├── __init__.py
    ├── chat.py            # /v1/chat/completions
    ├── documents.py       # /v1/documents/*
    ├── collections.py     # /v1/collections/*
    ├── conversations.py   # /v1/conversations/*
    ├── ingestion.py       # /v1/ingestion-jobs/*
    ├── api_keys.py        # /v1/api-keys/*
    └── system.py          # /v1/health, /v1/info, /v1/files/*
```

## 详细需求

### 1. `deps.py` — 依赖注入

```python
# 配置单例
@lru_cache()
def get_settings() -> Settings: ...

# 组件单例（通过 get_settings 获取配置）
def get_llm_client() -> LLMClient: ...
def get_embedding_service() -> EmbeddingService: ...
def get_vector_store() -> VectorStore: ...
def get_storage_backend() -> StorageBackend: ...
def get_hybrid_retriever() -> HybridRetriever: ...
def get_rag_pipeline() -> RAGPipeline: ...
def get_prompt_manager() -> PromptManager: ...

# Session 注入
async def get_db_session() -> AsyncGenerator[AsyncSession, None]: ...
```

### 2. 端点清单

#### 问答核心
| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/chat/completions` | 核心问答（兼容 OpenAI API 格式），支持 `stream: true` |

#### 文档管理
| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/documents/ingest` | 上传文件摄入（`multipart/form-data`） |
| `POST` | `/v1/documents/ingest-url` | 从 URL 摄入 |
| `GET` | `/v1/documents` | 文档列表（分页 + 集合/状态过滤） |
| `GET` | `/v1/documents/{doc_id}` | 文档详情（含 chunks 摘要） |
| `DELETE` | `/v1/documents/{doc_id}` | 删除文档（同时删 ChromaDB + SQL） |

#### 集合管理
| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/collections` | 集合列表 |
| `POST` | `/v1/collections` | 创建集合 |
| `DELETE` | `/v1/collections/{name}` | 删除集合（含关联文档） |

#### 对话记录
| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/conversations` | 对话列表（分页） |
| `GET` | `/v1/conversations/{id}` | 对话详情 + 消息历史 |
| `DELETE` | `/v1/conversations/{id}` | 删除对话 |

#### 摄入任务
| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/ingestion-jobs` | 摄入任务列表（状态/集合过滤） |
| `GET` | `/v1/ingestion-jobs/{id}` | 摄入任务详情 + 错误信息 |

#### API 密钥
| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/api-keys` | 密钥列表 |
| `POST` | `/v1/api-keys` | 创建新密钥 |
| `PATCH` | `/v1/api-keys/{id}` | 更新（激活/停用） |
| `DELETE` | `/v1/api-keys/{id}` | 删除密钥 |

#### 系统
| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/health` | 健康检查（DB/ChromaDB/Storage 连通性） |
| `GET` | `/v1/info` | 系统信息（version, model, stats） |
| `GET` | `/v1/files/{storage_key}` | 文件下载/预览 |

### 3. 关键端点实现要点

**`POST /v1/chat/completions`**:
- 请求体兼容 OpenAI Chat Completions 格式
- 新增 `collection` 字段指定检索集合
- 新增 `retrieval` 字段控制检索参数
- 支持 `stream: true` 返回 SSE
- 支持 `tools` 传入可选工具定义

**`POST /v1/documents/ingest`**:
- 接受 `multipart/form-data`（`file` + `collection` 字段）
- 文件先存到临时目录 → 调用 `ingestion_pipeline.ingest_file()`
- 返回 `IngestionResult`

**`GET /v1/health`**:
- 检查 SQLAlchemy 连接
- 检查 ChromaDB 连接
- 检查 StorageBackend 可用性
- 返回各组件状态 `{"api": "ok", "database": "ok", "chromadb": "ok", "storage": "ok"}`

### 4. 分页响应格式

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

### 5. 错误响应格式

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

### 6. 全局异常处理器

在 `router.py` 注册全局异常处理器，将 `CompactRAGException` 及其子类转换为标准错误响应。

```python
@app.exception_handler(CompactRAGException)
async def compact_rag_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": {"code": exc.__class__.__name__, "message": str(exc),
                           "details": exc.details, "request_id": exc.request_id}}
    )
```

## 验收标准

- [ ] 所有 19 个端点可访问并返回正确格式
- [ ] `/v1/chat/completions` 流式和非流式均正常
- [ ] 文件上传摄入端到端工作
- [ ] 分页参数正确过滤和计数
- [ ] 删除文档同步清理 ChromaDB 和 SQL 数据
- [ ] 异常统一格式返回
- [ ] `/v1/health` 正确报告各组件状态
- [ ] OpenAPI 文档 (`/docs`) 自动生成且可用
