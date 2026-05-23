# compact-rag 多 Agent 实施执行方案

> **版本**: v1.1 | **日期**: 2026-05-24 | **基于**: [设计文档 v1.2](../design/DESIGN.md) + [任务索引](../task/00-README.md) + [设计契约](../design/CONTRACTS.md)

---

## 目录

1. [Agent 矩阵](#1-agent-矩阵)
2. [执行拓扑](#2-执行拓扑)
3. [Phase 执行计划](#3-phase-执行计划)
4. [Agent 间契约](#4-agent-间契约)
5. [共享上下文](#5-共享上下文)
6. [验证门禁](#6-验证门禁)
7. [回滚与修复策略](#7-回滚与修复策略)

---

## 1. Agent 矩阵

> 共 **14 个 Agent**，每个负责一个或一组紧密耦合的任务。Agent 之间通过产出文件 + 验收标准交接。

| Agent ID | 名称 | 负责任务 | 产出文件数 | 预计耗时 |
|----------|------|---------|-----------|---------|
| **A1** | 项目骨架 | [01(部分)](../task/01-config-management.md), 脚手架 | 15+ | 2h |
| **A2** | 配置+公共 | [01](../task/01-config-management.md), [02](../task/02-common-infrastructure.md), [16](../task/16-error-handling.md) | 6 | 5h |
| **A3** | 关系数据库 | [03](../task/03-relational-database.md), [15](../task/15-database-design.md) | 16+ | 12h |
| **A4** | 文件存储 | [13](../task/13-file-storage.md) | 2 | 8h |
| **A5** | 向量化服务 | [06](../task/06-embedding-service.md) | 2 | 4h |
| **A6** | 向量存储 | [07](../task/07-vector-store.md) | 2 | 6h |
| **A7** | 文档摄入 | [04](../task/04-document-ingestion.md), [05](../task/05-table-extraction.md) | 4 | 14h |
| **A8** | 混合检索 | [08](../task/08-hybrid-retrieval.md) | 6 | 10h |
| **A9** | LLM 生成 | [09](../task/09-llm-generation.md) | 2 | 5h |
| **A10** | Tool Calling | [10](../task/10-tool-calling.md) | 3 | 6h |
| **A11** | RAG 管线 | [11](../task/11-rag-pipeline.md) | 1 | 6h |
| **A12** | API 层 | [12](../task/12-api-layer.md) | 10+ | 12h |
| **A13** | Streamlit 后台 | [14](../task/14-streamlit-admin.md) | 15+ | 14h |
| **A14** | 部署+测试 | [17](../task/17-testing-strategy.md), [18](../task/18-deployment.md), [19](../task/19-performance-optimization.md) | 20+ | 持续 |

---

## 2. 执行拓扑

```
Phase 1 (项目骨架)
  ├── A1: 项目骨架 ───────────── 并行
  ├── A2: 配置+公共 ───────────── 并行 (依赖 A1 的目录结构)
  └── A4: 文件存储 ───────────── 并行 (依赖 A2 的配置+异常)

Phase 2 (数据层)
  └── A3: 关系数据库 ─────────── 依赖 A1+A2

Phase 3 (摄入能力)
  ├── A5: 向量化服务 ─────────── 并行 (依赖 A2 配置)
  ├── A6: 向量存储 ───────────── 并行 (依赖 A5)
  └── A7: 文档摄入 ───────────── 并行 (依赖 A3+A5+A6)

Phase 4 (检索+生成)
  ├── A8: 混合检索 ───────────── 并行 (依赖 A6)
  ├── A9: LLM 生成 ───────────── 并行 (依赖 A2 配置+异常)
  └── A10: Tool Calling ──────── 并行 (依赖 A2 异常)

Phase 5 (集成层)
  ├── A11: RAG 管线 ──────────── 依赖 A3+A8+A9+A10
  └── A12: API 层 ────────────── 依赖 A11 + main.py

Phase 6 (管理后台)
  └── A13: Streamlit 后台 ────── 依赖 A12+A4
                                  (前提: API 服务可启动)

Phase 7 (测试+部署)
  └── A14: 部署+测试 ─────────── 依赖 全部
```

**并行度**: Phase 1 最多 3 agent 并行，Phase 4 最多 3 agent 并行。

### 2.1 Agent → 任务映射

> 对照 [任务索引](../task/00-README.md) 的实施分期。

| 本方案 Phase | Agent | 对应任务 (来自 00-README) | 00-README Phase |
|-------------|-------|--------------------------|-----------------|
| Phase 1 | A1, A2, A4 | [01](../task/01-config-management.md), [02](../task/02-common-infrastructure.md), [13](../task/13-file-storage.md), [16](../task/16-error-handling.md) | Phase 1 (部分) |
| Phase 2 | A3 | [03](../task/03-relational-database.md), [15](../task/15-database-design.md) | Phase 1 (部分) |
| Phase 3 | A5, A6, A7 | [06](../task/06-embedding-service.md), [07](../task/07-vector-store.md), [04](../task/04-document-ingestion.md), [05](../task/05-table-extraction.md) | Phase 2 |
| Phase 4 | A8, A9, A10 | [08](../task/08-hybrid-retrieval.md), [09](../task/09-llm-generation.md), [10](../task/10-tool-calling.md) | Phase 3 + Phase 5 |
| Phase 5 | A11, A12 | [11](../task/11-rag-pipeline.md), [12](../task/12-api-layer.md) | Phase 4 |
| Phase 6 | A13 | [14](../task/14-streamlit-admin.md) | Phase 6 |
| Phase 7 | A14 | [17](../task/17-testing-strategy.md), [18](../task/18-deployment.md), [19](../task/19-performance-optimization.md) | Phase 7 |

> **说明**: 本方案将 00-README 的 Phase 1-5 重新编排为 Phase 1-5，以提高 Agent 并行度和模块解耦。总任务覆盖一致。

---

## 3. Phase 执行计划

### Phase 1 — 项目骨架 (目标: `compact-rag serve` 可启动，`/v1/health` 可达)

| 步骤 | Agent | 输入 | 产出 | 验证 |
|------|-------|------|------|------|
| 1.1 | **A1** | 设计文档 §4 目录结构 | `pyproject.toml`, 全部 `__init__.py`, `config/default.yaml`, `config/production.yaml`, `config/storage.yaml`, `.env.example`, `.gitignore`, `alembic.ini`, `Makefile`, `data/` 目录 | `pip install -e ".[dev]"` 成功 |
| 1.2 | **A2** | A1 产出，设计文档 §5.1 §5.2 §9 | `config/settings.py`, `common/__init__.py`, `common/logger.py`, `common/exceptions.py`, `storage/schema.py` | `python -c "from compact_rag.config.settings import Settings; s=Settings(); print(s.database.url)"` 通过 |
| 1.3 | **A4** | A2 产出，设计文档 §5.13 | `storage/__init__.py`, `storage/file_storage.py` (含 LocalFileBackend) | `python -c "from compact_rag.storage.file_storage import LocalFileBackend"` 通过 |
| 1.4 | **A1** | (继续) | `main.py` (CLI 入口 + uvicorn 启动), `api/__init__.py`, `api/deps.py`(基础), `api/router.py`(仅 health/info) | `compact-rag serve` 启动成功，`curl :8000/v1/health` 返回 200 |

**Phase 1 门禁**: `compact-rag serve` 启动 → `GET /v1/health` → `{"api":"ok"}`

---

### Phase 2 — 数据层 (目标: 8 张表可创建，Alembic 迁移可执行)

| 步骤 | Agent | 输入 | 产出 | 验证 |
|------|-------|------|------|------|
| 2.1 | **A3** | A1+A2 产出，设计文档 §5.3 §7 | `storage/db/engine.py`, `storage/db/models.py` (8 张表), `storage/db/repository/*.py` (7 个 repository), `migrations/` (首次迁移) | `alembic upgrade head` → SQLite 中 8 张表存在 |

**Phase 2 门禁**: 
```bash
alembic upgrade head
python -c "
from compact_rag.storage.db.engine import create_engine
from compact_rag.storage.db.models import Base
# 验证所有表存在
"
```

---

### Phase 3 — 摄入能力 (目标: `ingest_file()` 端到端可执行)

| 步骤 | Agent | 输入 | 产出 | 验证 |
|------|-------|------|------|------|
| 3.1 | **A5** | A2 产出，设计文档 §5.6 | `embedding/__init__.py`, `embedding/service.py` | `EmbeddingService.encode(["test"])` 返回 `(1, 384)` |
| 3.2 | **A6** | A5 产出，设计文档 §5.7 | `storage/vector_store.py` (完善) | `VectorStore.search("test")` 可执行 |
| 3.3 | **A7** | A2+A3+A5+A6 产出，设计文档 §5.4 §5.5 | `ingestion/loader.py`, `ingestion/chunker.py`, `ingestion/table_extractor.py`, `ingestion/pipeline.py` | `IngestionPipeline.ingest_file("test.pdf")` 端到端通过 |

**Phase 3 门禁**:
```python
pipeline = IngestionPipeline(...)
result = await pipeline.ingest_file("tests/fixtures/sample.pdf", "default")
assert result.status == "completed"
assert result.chunk_count > 0
# 验证 ChromaDB + SQL 数据一致
```

---

### Phase 4 — 检索+生成 (目标: 独立模块可单测通过)

| 步骤 | Agent | 输入 | 产出 | 验证 |
|------|-------|------|------|------|
| 4.1 | **A8** | A6 产出，设计文档 §5.8 | `retrieval/dense.py`, `retrieval/sparse.py`, `retrieval/fusion.py`, `retrieval/reranker.py`, `retrieval/query_transformer.py`, `retrieval/retriever.py` | `HybridRetriever.retrieve("test query")` 返回结果 |
| 4.2 | **A9** | A2 产出，设计文档 §5.9 | `generation/llm.py`, `generation/prompt.py` | `LLMFactory.create(settings).chat([...])` 正常调用 (mock 测试) |
| 4.3 | **A10** | A2 产出，设计文档 §5.10 | `tool/schema.py`, `tool/engine.py`, `tool/builtin.py` | `ToolEngine.execute_tool_call({...})` 正确路由执行 |

**Phase 4 门禁**: 各模块独立可导入且方法签名正确，mock 测试通过。

---

### Phase 5 — 集成层 (目标: `/v1/chat/completions` 端到端问答可达)

| 步骤 | Agent | 输入 | 产出 | 验证 |
|------|-------|------|------|------|
| 5.1 | **A11** | A3+A8+A9+A10 产出，设计文档 §5.11 | `rag/pipeline.py` | `RAGPipeline.query("测试问题")` 返回 `RAGResponse` |
| 5.2 | **A12** | A11+A1 产出，设计文档 §5.12 §8 | `api/router.py`(全端点), `api/schemas.py`, `api/deps.py`(完善), `api/routers/*.py` (7 个子路由) | 20 个端点均可访问 |

**Phase 5 门禁**:
```bash
# 启动服务
compact-rag serve &

# 文档摄入
curl -X POST :8000/v1/documents/ingest \
  -F "file=@tests/fixtures/sample.pdf" \
  -F "collection=default"

# 问答
curl -X POST :8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"总结"}],"collection":"default"}'

# 验证响应含 answer + citations
```

---

### Phase 6 — 管理后台 (目标: `streamlit run` 可启动，8 页可导航)

| 步骤 | Agent | 输入 | 产出 | 验证 |
|------|-------|------|------|------|
| 6.1 | **A13** | A12+A4 产出，设计文档 §5.14 | `admin/client.py`, `admin/app.py`, `admin/pages/*.py` (8 页), `admin/components/*.py` (3 组件) | `streamlit run admin/app.py` → 浏览器 8 页可导航 |

**Phase 6 门禁**: Streamlit 仪表盘正确显示统计数据，RAG 问答台可完成检索→回答流程。

---

### Phase 7 — 测试+部署

| 步骤 | Agent | 输入 | 产出 | 验证 |
|------|-------|------|------|------|
| 7.1 | **A14** | 全部产出，任务 17 | `tests/conftest.py`, 全部测试文件 | `pytest` 全部通过，覆盖率 > 85% |
| 7.2 | **A14** | 全部产出，任务 18 | `Dockerfile`, `Makefile` 完善 | `docker build` 成功 |
| 7.3 | **A14** | 全部产出，任务 19 | 性能优化代码 (lazy load, batch, etc.) | 检索延迟符合基准 |

---

## 4. Agent 间契约

> 每个 Agent 在启动前读取指定的参考文件，产出后由下一级 Agent 消费。

### 4.1 Agent 输入契约

| Agent | 必须读取的参考文件 | 消费的前置产出 |
|-------|------------------|---------------|
| **A1** | 设计文档 §1 §2 §3 §4, 任务 01 18 | — |
| **A2** | 设计文档 §5.1 §5.2 §9, 任务 01 02 16, CONTRACTS §2 §5 §6 | A1 的目录结构 |
| **A3** | 设计文档 §5.3 §7, 任务 03 15, CONTRACTS §2 §4 | A1+A2 的 `Settings`, `exceptions` |
| **A4** | 设计文档 §5.13, 任务 13, CONTRACTS §1 §4 | A2 的 `Settings`, `exceptions` |
| **A5** | 设计文档 §5.6, 任务 06, CONTRACTS §7 | A2 的 `EmbeddingSettings` |
| **A6** | 设计文档 §5.7, 任务 07, CONTRACTS §4 | A5 的 `EmbeddingService` |
| **A7** | 设计文档 §5.4 §5.5 §6, 任务 04 05, CONTRACTS §1 §4 §6 | A3 的 repo, A5 的 service, A6 的 store |
| **A8** | 设计文档 §5.8, 任务 08, CONTRACTS §1 §7 | A6 的 `VectorStore` |
| **A9** | 设计文档 §5.9, 任务 09, CONTRACTS §1 | A2 的 `LLMSettings`, `exceptions` |
| **A10** | 设计文档 §5.10, 任务 10, CONTRACTS §1 | A2 的 `exceptions` |
| **A11** | 设计文档 §5.11 §6, 任务 11, CONTRACTS §1 §2 | A3+A8+A9+A10 全部 |
| **A12** | 设计文档 §5.12 §8, 任务 12, CONTRACTS §1 §3 | A11 的 `RAGPipeline` |
| **A13** | 设计文档 §5.14, 任务 14, CONTRACTS §1 §9 | A12 的 API (运行时), A4 的 storage |
| **A14** | 所有任务文件, CONTRACTS 全文 | A1-A13 全部 |

### 4.2 Agent 产出契约

| Agent | 必须产出的关键类/函数/端点 |
|-------|--------------------------|
| **A1** | `main.py`: `app`, `serve()`, `admin()` 命令; `pyproject.toml` 完整依赖; 全部 `__init__.py` |
| **A2** | `Settings` (含全部子模型 + `load()`); `get_logger()`; 全部 15 个异常类 |
| **A3** | `create_engine()`, `create_session_factory()`, 8 个 ORM 模型, 7 个 Repository, Alembic 首次迁移 |
| **A4** | `StorageBackend` ABC, `LocalFileBackend`, `MinIOBackend`, `TempFileCleaner`, `get_storage_backend()`, `build_storage_key()` |
| **A5** | `EmbeddingService` (单例, `encode()` + `encode_query()` + `dimension`) |
| **A6** | `VectorStore` (`add_documents()`, `search()`, `delete_by_document()`, `count()`) |
| **A7** | `BaseLoader` ABC + 5 个实现, `LoaderFactory`, 3 个 Chunker, `TableExtractor`, `IngestionPipeline` |
| **A8** | `DenseRetriever`, `BM25Retriever`, `rrf_fusion()`, `RerankerService`, `HybridRetriever` |
| **A9** | `LLMClient` ABC + 3 个实现, `LLMFactory`, `PromptManager` |
| **A10** | `Tool`, `ToolEngine`, `ToolRegistry`, `RAG_TOOLS` (retrieve_docs, query_database) |
| **A11** | `RAGPipeline` (`query()` + `query_stream()`) |
| **A12** | 20 个 API 端点, 全局异常处理器, `api/deps.py` 全部依赖注入函数 |
| **A13** | `AdminAPIClient`, `admin/app.py`, 8 个页面文件, 3 个组件文件 |
| **A14** | `conftest.py` (含所有 fixtures), 20+ 测试文件, `Dockerfile`, `Makefile` |

---

## 5. 共享上下文

每个 Agent 启动时必须加载以下环境：

### 5.1 全局常量（不应重复定义）

```python
# 版本号 (A1 定义，其他引用)
__version__ = "0.1.0"

# 基类路径 (A1 定义)
PACKAGE_ROOT = "src/compact_rag"

# 支持的文档格式 (A2 的 IngestionSettings 中定义)
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md", ".html"]

# ChromaDB metadata 字段名 (A6 定义常量化)
CHROMA_DOC_ID = "doc_id"
CHROMA_CHUNK_INDEX = "chunk_index"
CHROMA_PAGE_NUMBER = "page_number"
CHROMA_FILENAME = "filename"
CHROMA_COLLECTION_NAME = "collection_name"
CHROMA_IS_TABLE = "is_table"
CHROMA_TOKEN_COUNT = "token_count"
```

### 5.2 每个 Agent 的启动 Prompt 模板

```
你正在实施 compact-rag 的 [模块名]。

必须参考的文件:
1. docs/design/DESIGN.md — 核心设计文档 (重点关注 §X.X)
2. docs/task/XX-xxx.md — 你负责的任务文件
3. docs/design/CONTRACTS.md — 接口/数据/异常契约 (重点关注 §X)

前置条件:
- [依赖模块] 已由 [上游 Agent] 完成
- 你可直接 import: [列出可用模块]

产出要求:
- [列出文件路径]
- 验收标准: [列出]

代码规范:
- 遵循现有代码风格 (PEP 8, async/await)
- 所有公开方法要有 docstring
- 使用 loguru logger (from compact_rag.common.logger import get_logger)
- 异常使用 compact_rag.common.exceptions 中定义的类型
- 配置从 Settings 注入，不从环境变量直接读取
- 使用 Python 3.11+ 语法 (T | None, list[Type] 等)
```

---

## 6. 验证门禁

每个 Phase 完成后必须通过以下门禁才能进入下一阶段：

### 6.1 Phase 1 门禁

```bash
# 1. 安装依赖
pip install -e ".[dev]" && echo "PASS: install"

# 2. 导入验证
python -c "
from compact_rag.config.settings import Settings
from compact_rag.common.logger import get_logger
from compact_rag.common.exceptions import CompactRAGException
print('PASS: imports')
"

# 3. 启动验证
compact-rag serve &
sleep 3
curl -s http://localhost:8000/v1/health | python -c "import sys,json; d=json.load(sys.stdin); assert 'api' in d; print('PASS: health')"
curl -s http://localhost:8000/v1/info | python -c "import sys,json; d=json.load(sys.stdin); assert 'version' in d; print('PASS: info')"
kill %1
```

### 6.2 Phase 2 门禁

```bash
# 1. 首次迁移
alembic upgrade head && echo "PASS: migrate"

# 2. 8 张表验证 (SQLite)
python -c "
from compact_rag.storage.db.engine import create_engine, create_session_factory
from compact_rag.storage.db.models import Base
import asyncio
async def check():
    from compact_rag.config.settings import Settings
    s = Settings()
    engine = create_engine(s.database)
    async with engine.begin() as conn:
        tables = await conn.run_sync(lambda c: list(Base.metadata.tables.keys()))
    expected = ['collections','documents','document_chunks','conversations',
                'messages','ingestion_jobs','api_keys','storage_files']
    assert all(t in tables for t in expected), f'Missing: {set(expected)-set(tables)}'
    print(f'PASS: all 8 tables exist: {tables}')
asyncio.run(check())
"
```

### 6.3 Phase 3 门禁

```bash
python -c "
import asyncio
from compact_rag.ingestion.pipeline import IngestionPipeline
# ... 组装 pipeline ...
async def main():
    result = await pipeline.ingest_file('tests/fixtures/sample.pdf', 'default')
    assert result.status == 'completed', f'FAIL: {result.error_message}'
    assert result.chunk_count > 0, 'FAIL: no chunks'
    print(f'PASS: ingested {result.filename} → {result.chunk_count} chunks')
asyncio.run(main())
"
```

### 6.4 Phase 4 门禁

```bash
# 各模块独立可导入
python -c "
from compact_rag.retrieval.retriever import HybridRetriever
from compact_rag.generation.llm import LLMFactory
from compact_rag.tool.engine import ToolEngine
print('PASS: all retrieval/generation/tool modules importable')
"
```

### 6.5 Phase 5 门禁

```bash
# 端到端问答
curl -s -X POST :8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"collection":"default"}' \
  | python -c "import sys,json; d=json.load(sys.stdin); assert 'choices' in d; assert d['choices'][0]['message']['content']; print('PASS: RAG chat works')"
```

### 6.6 Phase 6 门禁

```bash
# Streamlit 导入验证
python -c "
from compact_rag.admin.client import AdminAPIClient
from compact_rag.admin.app import main
print('PASS: admin modules importable')
"
```

### 6.7 Phase 7 门禁

```bash
pytest --cov=src/compact_rag --cov-report=term --cov-fail-under=85
```

---

## 7. 回滚与修复策略

### 7.1 单 Agent 失败

```
┌─ Agent 执行
│   ├─ 成功 → 提交产出，通知下游
│   └─ 失败 → 分析原因
│       ├─ 上游产出有问题 → 回滚到上游 Agent，修复后重跑
│       ├─ 自身实现有 bug → 在 Agent 内修复，重新运行
│       └─ 设计有歧义 → 暂停，先更新设计文档/任务文件，再重跑
```

### 7.2 Phase 级故障

如果 Phase 门禁不通过：
1. 确定失败的 Agent
2. 检查该 Agent 的输入依赖是否完整
3. 运行该 Agent 的独立验证步骤
4. 修复后重新运行该 Agent → 重新运行 Phase 门禁

### 7.3 跨 Phase 不兼容

如果后续 Phase 发现前置 Phase 的接口设计有缺陷：
1. 不修改前置 Phase 的代码（避免级联修改）
2. 在后续 Phase 中增加适配层
3. 记录为设计债务，在 Phase 7 统一重构

---

## 附录 A: 快速启动命令

```bash
# 一键执行 Phase 1-5
# (在确认设计文档和任务文件无变更后)

# 创建并激活虚拟环境
python -m venv .venv && source .venv/bin/activate

# Phase 1: A1 + A2 + A4 (骨架 + 配置 + 公共 + 文件存储)
# 各 Agent 按顺序或并行执行

# Phase 2: A3 (数据库)
# 执行数据库迁移

# Phase 3: A5 + A6 + A7 (摄入能力)

# Phase 4: A8 + A9 + A10 (检索 + 生成 + Tool)

# Phase 5: A11 + A12 (RAG + API)

# Phase 6: A13 (管理后台)

# Phase 7: A14 (测试 + 部署)

# 全流程验证
pytest && echo "ALL PASS"
```

## 附录 B: Agent 依赖矩阵

```
         A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14
A1 (骨架)  -  →  →  →                                      (产出供 A2/A3/A4/A12 使用)
A2 (配置)  ←  -  →  →  →  →  →     →  →   →   →    →     (被几乎所有人依赖)
A3 (DB)    ←  ←  -           →              →   →          (供 A7/A11/A12)
A4 (存储)  ←  ←     -                 →        →   →      (供 A7/A12/A13)
A5 (向量)  ←             -  →  →                           (供 A6)
A6 (VecSt) ←                ←  -  →  →                     (供 A7/A8)
A7 (摄入)  ←  ←  ←  ←  ←  ←  -          →                (供 A11)
A8 (检索)  ←              ←     -  →                       (供 A11)
A9 (LLM)   ←                       -  →                   (供 A11)
A10(Tool)  ←                          -  →                (供 A11)
A11(RAG)   ←  ←        ←  ←  ←  ←  ←  -  →               (供 A12)
A12(API)   ←  ←  ←     ←                 ←  -  →          (供 A13)
A13(Admin) ←        ←                          ←  -       (供 A14)
A14(Test)  ←  ←  ←  ←  ←  ←  ←  ←  ←  ←   ←   ←   ←  -   (依赖所有人)
```

```
符号: → = 产出被 ... 消费; ← = 依赖 ... 的产出
```

---

> **基于**: [设计文档 v1.2](../design/DESIGN.md) | [任务索引](../task/00-README.md) | [设计契约](../design/CONTRACTS.md) | [任务文件](../task/)
