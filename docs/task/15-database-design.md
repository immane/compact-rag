# 任务 15: 数据库设计

> **依赖**: 03-关系型数据库层 | **优先级**: P0 | **预计工时**: — (设计参考)

## 概述

本文档描述 compact-rag 的双数据库设计：关系型数据库 (SQLAlchemy + Alembic) 用于结构化元数据，向量数据库 (ChromaDB) 用于语义检索。

## 关系型数据库

### 表清单（8 张表）

#### 1. `collections` — 文档集合

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | 集合唯一标识 |
| `name` | VARCHAR(255) | UNIQUE, NOT NULL | 集合名称 (如 "finance-2024") |
| `description` | TEXT | NULLABLE | 集合描述 |
| `embedding_model` | VARCHAR(255) | NOT NULL | 使用的 embedding 模型名 |
| `chunk_size` | INTEGER | DEFAULT 500 | 分块大小 |
| `chunk_overlap` | INTEGER | DEFAULT 50 | 分块重叠 |
| `document_count` | INTEGER | DEFAULT 0 | 文档数量（冗余计数） |
| `created_at` | DATETIME | NOT NULL | 创建时间 |
| `updated_at` | DATETIME | NOT NULL | 更新时间 |

#### 2. `documents` — 文档元数据

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | 文档唯一标识 |
| `collection_id` | UUID | FK → collections.id | 所属集合 |
| `filename` | VARCHAR(500) | NOT NULL | 原始文件名 |
| `file_type` | VARCHAR(20) | NOT NULL | pdf/docx/txt/md/html |
| `file_size` | INTEGER | NOT NULL | 文件大小（字节） |
| `file_hash` | VARCHAR(64) | NOT NULL | SHA256（去重） |
| `page_count` | INTEGER | NULLABLE | 页数 |
| `chunk_count` | INTEGER | DEFAULT 0 | 分块数量 |
| `table_count` | INTEGER | DEFAULT 0 | 表格数 |
| `status` | VARCHAR(20) | DEFAULT 'pending' | pending/processing/completed/failed |
| `error_message` | TEXT | NULLABLE | 失败原因 |
| `metadata` | JSON | NULLABLE | 扩展元数据 |
| `created_at` | DATETIME | NOT NULL | |
| `updated_at` | DATETIME | NOT NULL | |

**索引**: `collection_id`, `file_hash` (去重查询)

#### 3. `document_chunks` — Chunk 索引映射

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | |
| `document_id` | UUID | FK → documents.id, CASCADE | |
| `chroma_id` | VARCHAR(255) | NOT NULL | ChromaDB 对应 ID |
| `chunk_index` | INTEGER | NOT NULL | 本文档内的块序号 |
| `page_number` | INTEGER | NULLABLE | 所在页码 |
| `is_table` | BOOLEAN | DEFAULT FALSE | 是否表格块 |
| `token_count` | INTEGER | NULLABLE | Token 估算数 |
| `content_hash` | VARCHAR(64) | NULLABLE | 内容哈希（更新检测） |
| `created_at` | DATETIME | NOT NULL | |

#### 4. `conversations` — 对话会话

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | |
| `collection_id` | UUID | FK → collections.id, NULLABLE | 关联集合 |
| `title` | VARCHAR(500) | DEFAULT '新对话' | 对话标题 |
| `model` | VARCHAR(100) | NOT NULL | LLM 模型 |
| `message_count` | INTEGER | DEFAULT 0 | 消息数 |
| `created_at` | DATETIME | NOT NULL | |
| `updated_at` | DATETIME | NOT NULL | |

#### 5. `messages` — 对话消息

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | |
| `conversation_id` | UUID | FK → conversations.id, CASCADE | |
| `role` | VARCHAR(20) | NOT NULL | system/user/assistant/tool |
| `content` | TEXT | NOT NULL | 消息内容 |
| `tool_calls` | JSON | NULLABLE | 工具调用记录 |
| `sources` | JSON | NULLABLE | 引用来源 |
| `token_count` | INTEGER | NULLABLE | Token 消耗 |
| `latency_ms` | INTEGER | NULLABLE | 响应延迟 |
| `created_at` | DATETIME | NOT NULL | |

**索引**: `conversation_id`

#### 6. `ingestion_jobs` — 摄入任务跟踪

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | |
| `collection_id` | UUID | FK → collections.id | |
| `status` | VARCHAR(20) | DEFAULT 'pending' | pending/running/completed/failed |
| `total_files` | INTEGER | DEFAULT 0 | |
| `processed_files` | INTEGER | DEFAULT 0 | |
| `total_chunks` | INTEGER | DEFAULT 0 | |
| `errors` | JSON | NULLABLE | 错误汇总 |
| `started_at` | DATETIME | NULLABLE | |
| `completed_at` | DATETIME | NULLABLE | |
| `created_at` | DATETIME | NOT NULL | |

#### 7. `api_keys` — API 认证

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | |
| `name` | VARCHAR(255) | NOT NULL | 备注名 |
| `key_hash` | VARCHAR(255) | UNIQUE, NOT NULL | API Key 哈希 |
| `permissions` | JSON | DEFAULT '["read"]' | 权限列表 |
| `is_active` | BOOLEAN | DEFAULT TRUE | |
| `expires_at` | DATETIME | NULLABLE | |
| `created_at` | DATETIME | NOT NULL | |

#### 8. `storage_files` — 文件存储记录

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | |
| `document_id` | UUID | FK → documents.id, NULLABLE | |
| `storage_backend` | VARCHAR(50) | NOT NULL | local/minio/oss/kodo/s3 |
| `storage_key` | VARCHAR(1000) | NOT NULL | 存储键路径 |
| `filename` | VARCHAR(500) | NOT NULL | 原始文件名 |
| `file_size` | INTEGER | NOT NULL | |
| `content_type` | VARCHAR(100) | NULLABLE | MIME 类型 |
| `storage_type` | VARCHAR(20) | DEFAULT 'persistent' | temp/persistent/archive |
| `expires_at` | DATETIME | NULLABLE | |
| `created_at` | DATETIME | NOT NULL | |

### 外键关系图

```
collections (1) ────< (N) documents
documents (1) ────< (N) document_chunks [CASCADE]
collections (1) ────< (N) conversations [SET NULL]
conversations (1) ────< (N) messages [CASCADE]
collections (1) ────< (N) ingestion_jobs
documents (1) ────< (N) storage_files [SET NULL]
api_keys (独立)
```

## 向量数据库 (ChromaDB)

### Collection 设计

每个逻辑集合对应一个 ChromaDB Collection：

```
Collection Name: {collection_name}
  ↓
  Documents: [{id: chroma_id, embedding: [384 floats],
               metadata: {...}, document: chunk_text}]
```

### Chunk Metadata

| 字段 | 类型 | 说明 |
|------|------|------|
| `doc_id` | str | 关联 documents.id |
| `chroma_id` | str | ChromaDB 自动生成 |
| `chunk_index` | int | 块序号 |
| `page_number` | int/None | 页码 |
| `filename` | str | 源文件名 |
| `collection_name` | str | 所属集合 |
| `is_table` | bool | 是否表格块 |
| `token_count` | int | Token 估算数 |

### 双数据库同步规则

- 每次 `ChromaDB.add()` 后，同步写入 `document_chunks` 表（通过 `chroma_id` 关联）
- 删除文档：先删 ChromaDB (`collection.delete(ids=[...])`)，再删 SQL（`document_chunks` + `documents`）
- 通过 `chroma_id` 和 `doc_id` 关联两套存储

## 验收标准

- [ ] Alembic 迁移能正确创建所有 8 张表
- [ ] ForeignKey 约束和 CASCADE 删除行为正确
- [ ] ChromaDB Collection 正确映射到 SQL collections 表
- [ ] 双数据库增删操作保持一致性
