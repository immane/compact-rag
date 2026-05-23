# 任务 03: 关系型数据库层

> **依赖**: 01-配置管理, 02-公共基础设施 | **优先级**: P0 | **预计工时**: 12h

## 目标

实现 SQLAlchemy 2.0 async 的关系型数据库层，包括异步引擎管理、8 张表的 ORM 模型定义、Repository 模式封装、以及 Alembic 数据库迁移。

## 产出文件

```
src/compact_rag/storage/db/
├── __init__.py
├── engine.py              # SQLAlchemy async engine + session 工厂
├── models.py              # ORM 模型定义（8 张表）
└── repository/            # Repository 模式
    ├── __init__.py
    ├── base.py            # BaseRepository 基类
    ├── collection.py      # Collection CRUD
    ├── document.py        # Document CRUD
    ├── chunk.py           # DocumentChunk CRUD
    ├── conversation.py    # Conversation + Messages CRUD
    ├── ingestion.py       # IngestionJob CRUD
    ├── api_key.py         # ApiKey CRUD
    └── storage_file.py    # StorageFile CRUD

src/compact_rag/storage/db/migrations/    # Alembic
├── alembic.ini
├── env.py
├── script.py.mako
└── versions/               # 迁移版本文件
```

## 详细需求

### 1. `engine.py` — 引擎工厂

```python
def create_engine(settings: DatabaseSettings) -> AsyncEngine:
    """根据配置创建异步 SQLAlchemy 引擎"""
    # sqlite+aiosqlite:/// → 开发环境
    # mysql+asyncmy:// → 生产环境
    # pool_size, max_overflow 从 settings 读取

def create_session_factory(engine: AsyncEngine) -> async_sessionmaker:
    """创建 async session 工厂，expire_on_commit=False"""

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖注入使用的 session 生成器"""
```

关键点：
- 开发环境 `sqlite+aiosqlite:///` 零配置，生产环境 `mysql+asyncmy://`
- 两者共享同一套 ORM 模型，无需代码修改
- Session 为异步，与 FastAPI async 模型匹配

### 2. `models.py` — ORM 模型（8 张表）

所有模型使用 `UUID` 作为主键，继承自 SQLAlchemy `DeclarativeBase`。

| 表名 | 对应模块 | 关键字段 |
|------|---------|---------|
| `collections` | 集合 | id, name(UNIQUE), description, embedding_model, chunk_size, chunk_overlap, document_count, created_at, updated_at |
| `documents` | 文档 | id, collection_id(FK), filename, file_type, file_size, file_hash(SHA256), page_count, chunk_count, table_count, status, error_message, metadata(JSON), created_at, updated_at |
| `document_chunks` | 分块映射 | id, document_id(FK CASCADE), chroma_id, chunk_index, page_number, is_table, token_count, content_hash, created_at |
| `conversations` | 对话 | id, collection_id(FK NULLABLE), title, model, message_count, created_at, updated_at |
| `messages` | 消息 | id, conversation_id(FK CASCADE), role, content, tool_calls(JSON), sources(JSON), token_count, latency_ms, created_at |
| `ingestion_jobs` | 摄入任务 | id, collection_id(FK), status, total_files, processed_files, total_chunks, errors(JSON), started_at, completed_at, created_at |
| `api_keys` | API 密钥 | id, name, key_hash(UNIQUE), permissions(JSON), is_active, expires_at, created_at |
| `storage_files` | 存储记录 | id, document_id(FK NULLABLE), storage_backend, storage_key, filename, file_size, content_type, storage_type, expires_at, created_at |

**关系定义**：
- `documents.collection_id` → `collections.id`
- `document_chunks.document_id` → `documents.id` (CASCADE delete)
- `messages.conversation_id` → `conversations.id` (CASCADE delete)
- `conversations.collection_id` → `collections.id` (NULLABLE, SET NULL on delete)
- `ingestion_jobs.collection_id` → `collections.id`
- `storage_files.document_id` → `documents.id` (NULLABLE, SET NULL on delete)

### 3. Repository 模式

`BaseRepository` 提供通用方法：
- `create(session, **kwargs)` — 创建记录
- `get_by_id(session, id)` — 按 ID 查询
- `list(session, page, page_size, **filters)` — 分页列表
- `update(session, id, **kwargs)` — 更新记录
- `delete(session, id)` — 删除记录

各子 Repository 扩展特有方法：

**DocumentRepository**:
- `get_by_hash(session, file_hash)` — 去重检测
- `list_by_collection(session, collection_id)` — 按集合过滤
- `update_status(session, doc_id, status)` — 更新处理状态

**CollectionRepository**:
- `get_by_name(session, name)` — 按名称查询
- `increment_document_count(session, collection_id, delta)` — 原子更新文档计数

**ConversationRepository**:
- `list_messages(session, conversation_id)` — 获取对话所有消息
- `increment_message_count(session, conversation_id)`

**IngestionJobRepository**:
- `create_job(session, collection_id, total_files)` — 创建任务
- `update_progress(session, job_id, processed, chunks)` — 更新进度
- `complete_job(session, job_id, status, errors)` — 完成任务

**ApiKeyRepository**:
- `get_by_hash(session, key_hash)` — 验证密钥
- `set_active(session, key_id, is_active)` — 激活/停用

### 4. Alembic 迁移

- 使用异步 Alembic 配置（`env.py` 中配置 async engine）
- 初始化首次迁移（创建全部 8 张表）
- `alembic.ini` 中 `sqlalchemy.url` 指向开发库
- 迁移命令封装为 Makefile target：`make migrate` / `make migrate-create MSG="..."`

## 验收标准

- [ ] SQLite 模式下 `create_engine` + `create_session_factory` 正常工作
- [ ] 所有 8 张表的 ORM 模型可正确创建（`Base.metadata.create_all`）
- [ ] Repository 各方法的 CRUD 操作和分页功能正常
- [ ] `alembic upgrade head` 正确创建所有表
- [ ] `alembic downgrade -1` 正确回滚
- [ ] ForeignKey 约束和外键级联删除行为正确
- [ ] Session 在 FastAPI 依赖注入中正确管理生命周期
