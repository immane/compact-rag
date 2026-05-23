# compact-rag 企业 RAG 系统设计文档

> **版本**: v1.1 | **日期**: 2026-05-23 | **状态**: 设计阶段

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [技术选型总览](#3-技术选型总览)
4. [项目目录结构](#4-项目目录结构)
5. [模块详细设计](#5-模块详细设计)
   - 5.1 [配置管理](#51-配置管理)
   - 5.2 [公共基础设施](#52-公共基础设施)
   - 5.3 [关系型数据库层](#53-关系型数据库层)
   - 5.4 [文档摄入管道](#54-文档摄入管道)
   - 5.5 [表格提取子系统](#55-表格提取子系统)
   - 5.6 [向量化服务](#56-向量化服务)
   - 5.7 [向量存储层（ChromaDB）](#57-向量存储层chromadb)
   - 5.8 [混合检索层](#58-混合检索层)
   - 5.9 [生成层（LLM 抽象）](#59-生成层llm-抽象)
   - 5.10 [Tool Calling 子系统](#510-tool-calling-子系统)
   - 5.11 [RAG 管线编排](#511-rag-管线编排)
    - 5.12 [API 层](#512-api-层)
    - 5.13 [文件存储子系统](#513-文件存储子系统)
6. [数据流](#6-数据流)
7. [数据库设计](#7-数据库设计)
   - 7.1 [关系型数据库（SQLAlchemy + Alembic）](#71-关系型数据库sqlalchemy--alembic)
   - 7.2 [向量数据库（ChromaDB）](#72-向量数据库chromadb)
8. [API 接口设计](#8-api-接口设计)
9. [错误处理与异常体系](#9-错误处理与异常体系)
10. [测试策略](#10-测试策略)
11. [性能优化策略](#11-性能优化策略)
12. [部署方案](#12-部署方案)
13. [实施路线图](#13-实施路线图)

---

## 1. 项目概述

### 1.1 项目定位

**compact-rag** 是一个面向企业的、轻量级但功能完备的 Retrieval-Augmented Generation（RAG）系统，采用 Python 实现。核心目标是在保持低资源消耗的前提下，提供可投入生产的文档检索与智能问答能力。

### 1.2 核心能力

| 能力 | 描述 |
|------|------|
| **多格式文档摄入** | 支持 PDF、DOCX、TXT、Markdown、HTML，自动提取文本和表格 |
| **表格智能处理** | 从 PDF/HTML 中提取表格并转为 Markdown，保留结构化关系 |
| **混合检索** | 密集向量检索（Embedding） + 稀疏检索（BM25）+ Cross-Encoder 重排序 |
| **LLM 生成** | 抽象 LLM 接口，支持 OpenAI / Anthropic / Ollama 多种后端 |
| **Tool Calling** | 轻量级工具调用框架，使 LLM 可执行数据库查询、文档检索等操作 |
| **对话记忆** | 完整记录对话历史，支持上下文感知的多轮问答 |
| **REST API** | 兼容 OpenAI API 格式的 HTTP 接口，支持流式输出 |
| **双数据库** | ChromaDB（向量）+ MySQL/SQLite（结构化元数据） |
| **文件存储** | 统一 StorageBackend 抽象，支持本地/MinIO/OSS/S3 多后端切换 |

### 1.3 核心约束

| 约束 | 说明 |
|------|------|
| **低资源运行** | CPU-only 模式可运行，无需 GPU |
| **本地优先** | 核心检索和嵌入能力可完全本地化 |
| **渐进式增强** | 从最简单到最复杂的能力逐级开启 |
| **生产可部署** | 开发用 SQLite，生产用 MySQL，一键切换 |
| **最小依赖** | 每个模块不引入不必要的重量级框架 |

---

## 2. 系统架构

### 2.1 架构总览

```
                              ┌─────────────────────────────────┐
                              │          API Layer              │
                              │     (FastAPI + Pydantic v2)     │
                              │   /v1/chat/completions          │
                              │   /v1/documents                 │
                              │   /v1/collections               │
                              └──────────────┬──────────────────┘
                                             │
                              ┌───────────────▼──────────────────┐
                              │       RAG Pipeline               │
                              │  query → retrieve → rerank →     │
                              │  context → generate → citations  │
                              └───────────────┬──────────────────┘
                      ┌───────────────────────┼───────────────────────┐
                      │                       │                       │
            ┌─────────▼─────────┐   ┌────────▼────────┐  ┌───────────▼──────────┐
            │   Retrieval Layer  │   │  Generation Layer│  │   Tool Calling Layer │
            │                    │   │                  │  │                      │
            │ ┌───────────────┐ │   │ ┌──────────────┐ │  │ ┌──────────────────┐ │
            │ │ Dense Search  │ │   │ │  LLM Abstract │ │  │ │  Tool Registry   │ │
            │ │ (ChromaDB)    │ │   │ │  OpenAI       │ │  │ │  Tool Engine     │ │
            │ ├───────────────┤ │   │ │  Anthropic    │ │  │ │  RAG Tools       │ │
            │ │ Sparse Search │ │   │ │  Ollama       │ │  │ │  - retrieve_docs │ │
            │ │ (BM25)        │ │   │ └──────────────┘ │  │ │  - query_database│ │
            │ ├───────────────┤ │   │ ┌──────────────┐ │  │ └──────────────────┘ │
            │ │ RRF Fusion    │ │   │ │  Prompt Mgr   │ │  │                      │
            │ ├───────────────┤ │   │ │  (Jinja2)     │ │  │                      │
            │ │ Cross-Encoder │ │   │ └──────────────┘ │  │                      │
            │ └───────────────┘ │   └─────────────────┘  └───────────────────────┘
            └─────────┬─────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
┌──────────┐  ┌──────────────┐  ┌──────────────────┐
│ ChromaDB │  │  SQLAlchemy   │  │  Embedding Service│
│ (Vector  │  │  (Relational) │  │  (sentence-       │
│  Store)  │  │              │  │   transformers)    │
└──────────┘  ├──────────────┤  └──────────────────┘
              │ MySQL (prod) │
              │ SQLite (dev) │
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │ StorageBackend│
              │ (文件抽象层)   │
              ├──────────────┤
              │ Local/MinIO   │
              │ OSS/Kodo/S3   │
              └──────────────┘
```

### 2.2 核心数据流

```
文档摄入流程：
  File → StorageBackend.upload → Loader → Chunker → Embedding → ChromaDB + 元数据写入 MySQL/SQLite

检索问答流程：
  Query → QueryTransform → [Hybrid Retrieval] → Rerank → Context Build → LLM → Answer + Citations

Tool Calling 流程：
  Query → LLM ↔ ToolEngine(execute tool) → Context Enrich → LLM → Answer
```

### 2.3 设计原则

1. **关注点分离** —— 每个模块职责单一，通过接口解耦
2. **配置驱动** —— 所有行为参数化，环境通过配置切换
3. **异步优先** —— 使用 `async/await`，SQLAlchemy async，httpx async
4. **优雅降级** —— 每个模块可独立工作，部分失败不崩溃
5. **可观测性** —— 结构化日志，关键路径埋点

---

## 3. 技术选型总览

### 3.1 完整技术栈

| 类别 | 技术 | 版本/型号 | 选择理由 |
|------|------|-----------|----------|
| **语言** | Python | >= 3.11 | async/await 成熟，类型提示完善 |
| **Web 框架** | FastAPI | latest | 高性能异步，OpenAPI 自动生成 |
| **数据验证** | Pydantic | v2 | FastAPI 原生支持，性能优异 |
| **API 客户端** | httpx | latest | 异步 HTTP，支持 HTTP/2 |
| **关系数据库** | MySQL (prod) / SQLite (dev) | — | 成熟稳定，团队熟悉 |
| **ORM** | SQLAlchemy | 2.0+ (async) | 最成熟 Python ORM，异步支持 |
| **数据库迁移** | Alembic | latest | SQLAlchemy 官方迁移工具 |
| **向量数据库** | ChromaDB | latest | 轻量嵌入式，Python 原生 |
| **Embedding 模型** | BGE-small / all-MiniLM-L6-v2 | 22-24M 参数 | CPU 可运行，精度/速度平衡好 |
| **稀疏检索** | rank_bm25 | latest | 零依赖 BM25 实现 |
| **嵌入推理** | sentence-transformers | latest | 社区标准 Embedding 库 |
| **重排序** | cross-encoder | ms-marco-MiniLM-L-6-v2 | 轻量 Cross-Encoder |
| **LLM 抽象** | openai / anthropic / ollama SDK | latest | 原生 SDK，避免框架封装 |
| **文档解析** | pypdf / python-docx | latest | 轻量级，单一职责 |
| **PDF 表格** | Camelot + pdfplumber | latest | 最高精度组合 |
| **HTML 处理** | markdownify | latest | 极轻量 HTML→Markdown |
| **配置管理** | pydantic-settings | latest | 类型安全的环境变量+文件加载 |
| **日志** | loguru | latest | 开箱即用的结构化日志 |
| **依赖管理** | uv / pip | latest | 极速安装 |
| **测试** | pytest + pytest-asyncio | latest | 异步测试支持 |
| **构建工具** | pyproject.toml (hatchling) | — | Python 标准构建方式 |
| **文件存储** | Local / MinIO / OSS / Kodo / S3 | — | 统一抽象层，配置切换后端 |
| **文件存储 SDK** | minio-py / oss2 / qiniu / boto3 | latest | 按需安装，策略模式 |

### 3.2 模型选型决策矩阵

#### Embedding 模型

| 场景 | 模型 | 维度 | 内存 | 推荐理由 |
|------|------|------|------|----------|
| 极低内存 (<256MB) | all-MiniLM-L6-v2 | 384 | ~90MB | 最小最快 |
| 均衡 (推荐) | BGE-small-en-v1.5 / BGE-small-zh-v1.5 | 384 | ~95MB | 中英文兼顾 |
| 高精度 | BGE-base-en-v1.5 | 768 | ~440MB | 精度最高 |

#### Cross-Encoder 重排模型

| 场景 | 模型 | CPU 延迟 |
|------|------|----------|
| 轻量 | cross-encoder/ms-marco-MiniLM-L-6-v2 | ~10ms |
| 均衡 | cross-encoder/ms-marco-MiniLM-L-12-v2 | ~20ms |
| 中文优化 | BAAI/bge-reranker-base | ~50ms |

#### LLM 模型

| 场景 | 模型 | 部署方式 |
|------|------|----------|
| 云端高质量 | gpt-4o / claude-sonnet-4 | OpenAI / Anthropic API |
| 云端低成本 | gpt-4o-mini | OpenAI API |
| 本地开发 | llama3.1 / qwen2.5 | Ollama |
| 本地高质量 | llama3.3-70b | Ollama |

---

## 4. 项目目录结构

```
compact-rag/
├── src/
│   └── compact_rag/
│       ├── __init__.py                # 包标识，版本号
│       ├── main.py                     # CLI 入口 & uvicorn 启动
│       │
│       ├── config/                     # 配置管理
│       │   ├── __init__.py
│       │   └── settings.py            # pydantic-settings，YAML 加载
│       │
│       ├── common/                     # 公共基础设施
│       │   ├── __init__.py
│       │   ├── logger.py              # loguru 结构化日志配置
│       │   └── exceptions.py          # 统一异常类定义
│       │
│       ├── storage/                    # 数据存储层
│       │   ├── __init__.py
│       │   ├── schema.py              # Pydantic 数据模型
│       │   ├── vector_store.py        # ChromaDB CRUD 封装
│       │   ├── file_storage.py        # 文件存储抽象层 + 后端实现
│       │   └── db/                    # 关系型数据库
│       │       ├── __init__.py
│       │       ├── engine.py          # SQLAlchemy async engine + session 工厂
│       │       ├── models.py          # ORM 模型定义（7 张表）
│       │       ├── repository/        # Repository 模式
│       │       │   ├── __init__.py
│       │       │   ├── document.py    # Document CRUD
│       │       │   ├── collection.py  # Collection CRUD
│       │       │   └── conversation.py # Conversation + Messages CRUD
│       │       └── migrations/        # Alembic 迁移
│       │           ├── alembic.ini
│       │           ├── env.py
│       │           └── versions/
│       │
│       ├── ingestion/                  # 文档摄入管道
│       │   ├── __init__.py
│       │   ├── loader.py              # 多格式文档加载器
│       │   ├── chunker.py             # 分块策略
│       │   ├── table_extractor.py     # 表格提取与 Markdown 转换
│       │   └── pipeline.py            # 摄入流程编排
│       │
│       ├── embedding/                  # 向量化服务
│       │   ├── __init__.py
│       │   └── service.py             # sentence-transformers 封装
│       │
│       ├── retrieval/                  # 检索层
│       │   ├── __init__.py
│       │   ├── dense.py               # ChromaDB 向量检索
│       │   ├── sparse.py              # BM25 关键词检索
│       │   ├── fusion.py              # RRF / RSF 融合
│       │   ├── reranker.py            # Cross-Encoder 重排序
│       │   ├── query_transformer.py   # 查询改写
│       │   └── retriever.py           # 混合检索编排
│       │
│       ├── generation/                 # 生成层
│       │   ├── __init__.py
│       │   ├── llm.py                 # LLM 客户端抽象层
│       │   └── prompt.py              # Jinja2 提示词模板管理
│       │
│       ├── tool/                       # Tool Calling 子系统
│       │   ├── __init__.py
│       │   ├── schema.py              # Tool 定义与 JSON Schema 生成
│       │   ├── engine.py              # 工具执行引擎（含重试）
│       │   └── builtin.py             # 内置 RAG 工具
│       │
│       ├── rag/                        # RAG 管线编排
│       │   ├── __init__.py
│       │   └── pipeline.py            # 完整 RAG 问答流程
│       │
│       └── api/                        # REST API
│           ├── __init__.py
│           ├── deps.py                # FastAPI 依赖注入
│           ├── router.py              # 路由注册
│           └── schemas.py             # 请求/响应 Pydantic 模型
│
├── config/
│   ├── default.yaml                   # 默认配置
│   ├── production.yaml                # 生产配置覆盖
│   └── storage.yaml                   # 文件存储配置
│
├── data/                              # 运行时数据（.gitignore）
│   ├── chromadb/                      # ChromaDB 持久化目录
│   └── documents/                     # 示例文档目录
│
├── tests/
│   ├── conftest.py                    # pytest fixtures
│   ├── fixtures/                      # 测试数据
│   ├── test_ingestion/
│   │   ├── test_loader.py
│   │   ├── test_chunker.py
│   │   └── test_table_extractor.py
│   ├── test_embedding/
│   │   └── test_service.py
│   ├── test_storage/
│   │   ├── test_vector_store.py
│   │   ├── test_file_storage.py
│   │   └── test_db_models.py
│   ├── test_retrieval/
│   │   ├── test_dense.py
│   │   ├── test_sparse.py
│   │   ├── test_fusion.py
│   │   └── test_reranker.py
│   ├── test_generation/
│   │   └── test_llm.py
│   ├── test_tool/
│   │   └── test_engine.py
│   ├── test_rag/
│   │   └── test_pipeline.py
│   └── test_api/
│       └── test_router.py
│
├── pyproject.toml                     # 项目元数据与依赖
├── alembic.ini                        # Alembic 配置
├── Makefile                           # 常用命令
├── .env.example                       # 环境变量示例
└── .gitignore
```

---

## 5. 模块详细设计

### 5.1 配置管理

**文件**: `src/compact_rag/config/settings.py`

```python
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from typing import Optional, Literal
import yaml

class DatabaseSettings(BaseModel):
    # dev: sqlite+aiosqlite:///data/compact-rag.db
    # prod: mysql+asyncmy://user:pass@host:3306/compact_rag
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
    api_key: Optional[str] = None           # 留空则读环境变量
    api_base: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 60

class IngestionSettings(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunking_strategy: Literal["recursive", "semantic"] = "recursive"
    supported_extensions: list[str] = [".pdf", ".docx", ".txt", ".md", ".html"]

class Settings(BaseSettings):
    database: DatabaseSettings = DatabaseSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    chromadb: ChromaDBSettings = ChromaDBSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    llm: LLMSettings = LLMSettings()
    ingestion: IngestionSettings = IngestionSettings()
    log_level: str = "INFO"

    @classmethod
    def load(cls, config_path: str = None) -> "Settings":
        """加载 YAML 配置并与环境变量合并"""
        ...
```

**设计要点**：
- 默认配置在 `config/default.yaml`，生产覆盖在 `config/production.yaml`
- 环境变量优先级最高（如 `LLM_API_KEY`）
- `dev` 环境用 SQLite，`prod` 环境用 MySQL
- 通过 `COMPACT_RAG_CONFIG=/path/to/config.yaml` 指定配置文件

---

### 5.2 公共基础设施

#### 日志 (`common/logger.py`)

- 使用 **loguru**，结构化 JSON 日志
- 开发环境: 彩色控制台输出
- 生产环境: JSON 格式，可接入 ELK / Grafana
- 关键路径: 摄入开始/结束、检索耗时、LLM调用耗时

#### 异常 (`common/exceptions.py`)

```python
class CompactRAGException(Exception):
    """基础异常"""
    ...

class DocumentLoadError(CompactRAGException):
    """文档加载异常"""
    ...

class IngestionError(CompactRAGException):
    """摄入流程异常"""
    ...

class RetrievalError(CompactRAGException):
    """检索异常"""
    ...

class GenerationError(CompactRAGException):
    """LLM 生成异常"""
    ...

class ConfigurationError(CompactRAGException):
    """配置异常"""
    ...
```

---

### 5.3 关系型数据库层

#### Engine 工厂 (`storage/db/engine.py`)

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

def create_engine(settings):
    """根据配置创建 async engine"""
    engine = create_async_engine(
        settings.database.url,
        echo=settings.database.echo,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
    )
    return engine

def create_session_factory(engine):
    """创建 async session 工厂"""
    return async_sessionmaker(engine, expire_on_commit=False)
```

**关键设计**：
- `sqlite+aiosqlite:///` → 文件数据库，零配置开发
- `mysql+asyncmy://` → 生产数据库
- 两者共享同一套 ORM 模型，无需代码修改
- 异步 Session，与 FastAPI 异步模型匹配

#### Repository 模式 (`storage/db/repository/`)

每个 Repository 提供与特定表交互的方法，封装 SQLAlchemy 查询：

```python
class DocumentRepository:
    async def create(self, session, **kwargs) -> Document
    async def get_by_id(self, session, doc_id) -> Document | None
    async def list_by_collection(self, session, collection_id) -> list[Document]
    async def update_status(self, session, doc_id, status)
    async def delete(self, session, doc_id)
```

---

### 5.4 文档摄入管道

#### 文件加载器 (`ingestion/loader.py`)

**职责**: 根据文件扩展名自动选择解析器

| 格式 | 解析器 | 依赖 |
|------|--------|------|
| `.pdf` | pypdf | `pypdf` |
| `.docx` | python-docx | `python-docx` |
| `.txt` | 直接读取 | 零依赖 |
| `.md` | 直接读取 | 零依赖 |
| `.html` | BeautifulSoup + markdownify | `beautifulsoup4`, `markdownify` |

元数据提取：
- `filename`: 原始文件名
- `file_type`: 文件格式
- `file_size`: 文件大小
- `page_count`: 页数（PDF/DOCX）
- `hash`: 文件 SHA256（去重用）
- `created_at`: 摄入时间
- `table_count`: 检测到的表格数量

接口设计：
```python
class BaseLoader(ABC):
    @abstractmethod
    async def load(self, file_path: str) -> list[DocumentChunk]:
        """加载并解析文件，返回文本块列表"""
        ...

class LoaderFactory:
    @staticmethod
    def get_loader(file_path: str) -> BaseLoader:
        """根据文件扩展名返回对应 Loader"""
        ...
```

#### 分块策略 (`ingestion/chunker.py`)

三种分块策略：

1. **递归字符分割 (RecursiveCharacterTextSplitter)**
   - 使用分隔符 `["\n\n", "\n", "。", ".", "，", ",", " ", ""]` 逐级分割
   - 默认 `chunk_size=500`, `chunk_overlap=50`
   - 适用于大多数文档

2. **语义分割 (SemanticChunker)**
   - 基于 embedding 相似度阈值检测断点
   - 当相邻句子的余弦相似度低于阈值时分段
   - 适用于长文档和非结构文本

3. **表格感知分割**
   - 检测 Markdown 表格边界，保持表格整体完整性
   - 表格前后各保留一行纯文本作为上下文
   - 超大表格（>50行）拆分为表头 + 数据行组

#### 摄入流程编排 (`ingestion/pipeline.py`)

```python
class IngestionPipeline:
    async def ingest_file(self, file_path: str, collection_name: str) -> IngestionResult:
        """
        完整摄入流程:
        1. 文件类型检测 → 选择 Loader
        2. 加载文档 → 提取文本和表格
        3. 表格转为 Markdown
        4. 分块 (Chunking)
        5. 生成 Embedding
        6. 写入 ChromaDB（向量）
        7. 写入 MySQL/SQLite（元数据）
        返回: IngestionResult(doc_id, chunk_count, status)
        """

    async def ingest_directory(self, dir_path: str, collection_name: str) -> list[IngestionResult]:
        """批量摄入目录下所有支持的文件"""
```

**增量更新机制**：
- 计算文件 SHA256 哈希，已存在且未修改的跳过
- 支持强制重新摄入 (`force=True`)

---

### 5.5 表格提取子系统

**文件**: `ingestion/table_extractor.py`

**设计原则**：分层后备（Fallback）策略，保证最高成功率。

```
                  ┌──────────────┐
                  │  PDF 文件     │
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │ 是否为扫描件？│
                  └──┬───────┬──┘
               是    │       │    否
        ┌───────────▼┐  ┌───▼──────────┐
        │ PaddleOCR  │  │ Camelot      │ (优先 Lattice 模式)
        │ (按需启用)  │  │ 成功?        │
        └────────────┘  └───┬──────┬───┘
                      是 │      │ 否
                         │      └──────────┐
                         ▼                 ▼
                  ┌──────────────┐  ┌──────────────┐
                  │ 输出 Markdown │  │ pdfplumber   │ (后备方案)
                  └──────────────┘  └──────────────┘
```

**路线 B**：HTML / Word 文档中的表格，使用 `markdownify`、`Pandoc` 处理。

**质量评估函数**：
```python
def evaluate_table_quality(markdown_table: str) -> dict:
    """检查：行数 ≥ 2、分隔行有效、列数一致"""
    ...
```

**级别定义**：

| 优先级 | 方案 | 适用场景 |
|--------|------|----------|
| P0 | Camelot + pdfplumber | 80%+ 数字 PDF 表格 |
| P1 | markdownify | HTML 内嵌表格 |
| P2 | Pandoc | Word/HTML 格式转换 |
| P3 | PaddleOCR | 扫描版 PDF（按需，需 GPU） |

---

### 5.6 向量化服务

**文件**: `embedding/service.py`

```python
class EmbeddingService:
    def __init__(self, settings: EmbeddingSettings):
        self.model = SentenceTransformer(
            settings.model_name,
            device=settings.device,
        )
        if settings.use_onnx:
            self.model = self._convert_to_onnx()

    async def encode(self, texts: list[str]) -> np.ndarray:
        """批量编码文本为向量"""
        ...

    async def encode_query(self, query: str) -> np.ndarray:
        """编码单条查询"""
        ...

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
```

**加速方案**（按需启用）：

| 方案 | 加速比 | 适用条件 |
|------|--------|----------|
| ONNX Runtime | 2-3x | CPU-only |
| OpenVINO | 3x | Intel CPU |
| int8 量化 | 4x 内存节省 | 内存紧张 |
| Matryoshka (截断) | 2-4x 加速 | 需支持模型 |

---

### 5.7 向量存储层（ChromaDB）

**文件**: `storage/vector_store.py`

```python
class VectorStore:
    def __init__(self, settings: ChromaDBSettings, embedding_service: EmbeddingService):
        self.client = chromadb.PersistentClient(path=settings.persist_directory)
        self.collection_name = settings.collection_name
        self.embedding_service = embedding_service
        self._ensure_collection()

    async def add_documents(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> list[str]:
        """批量添加文档块到向量存储，返回 chroma_ids"""
        ...

    async def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """向量相似度搜索"""
        query_vec = await self.embedding_service.encode_query(query)
        results = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ...

    async def delete_by_document(self, doc_id: str):
        """按文档 ID 删除所有相关块"""
        ...

    async def list_collections(self) -> list[str]:
        ...
```

**存储元数据结构**：
```json
{
  "doc_id": "uuid",
  "chunk_index": 0,
  "page_number": 3,
  "filename": "report.pdf",
  "is_table": false,
  "collection_name": "finance-2024"
}
```

---

### 5.8 混合检索层

**文件**: `retrieval/`

#### 5.8.1 基于研究结论的设计决策

| 层 | 选型 | 理由 |
|---|------|------|
| Dense 检索 | ChromaDB 向量查询 | 项目统一向量存储 |
| Sparse 检索 | rank_bm25 (纯 Python) | 零依赖，< 5 万条数据足够 |
| 融合策略 | RRF (Reciprocal Rank Fusion) | 不需要归一化，鲁棒性最高 |
| 重排序 | Cross-Encoder MiniLM-L-6-v2 | 10ms 延迟，效果显著 |

#### 5.8.2 BM25 检索器 (`retrieval/sparse.py`)

```python
class BM25Retriever:
    """基于 rank_bm25 的稀疏检索器"""

    def __init__(self):
        self.bm25 = None
        self.documents = []           # 纯文本
        self.doc_ids = []             # 对应的 chroma_id

    def index(self, documents: list[str], doc_ids: list[str]):
        """构建 BM25 索引"""
        tokenized = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
        self.documents = documents
        self.doc_ids = doc_ids

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """返回 (doc_id, bm25_score) 列表"""
        ...

    def _tokenize(self, text: str) -> list[str]:
        """中文分词用 jieba，英文直接 split"""
        ...
```

#### 5.8.3 融合层 (`retrieval/fusion.py`)

```python
def rrf_fusion(
    dense_results: list[SearchResult],    # 含分数和排名
    sparse_results: list[SearchResult],
    k: int = 60,
    top_k: int = 50,
) -> list[SearchResult]:
    """
    Reciprocal Rank Fusion
    score(d) = Σ 1 / (k + rank_i(d))
    """
    scores = {}
    for ranking in [dense_results, sparse_results]:
        for rank, result in enumerate(ranking, start=1):
            if result.id not in scores:
                scores[result.id] = 0
            scores[result.id] += 1 / (k + rank)

    # 按融合分数排序，返回 top_k
    ...
```

#### 5.8.4 重排序 (`retrieval/reranker.py`)

```python
class RerankerService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    async def rerank(
        self, query: str, candidates: list[SearchResult]
    ) -> list[SearchResult]:
        """
        对候选结果用 Cross-Encoder 精细打分重排
        返回按新分数排序的结果
        """
        pairs = [(query, c.content) for c in candidates]
        scores = self.model.predict(pairs)
        # 按分数降序重排
        ...
```

#### 5.8.5 查询转换 (`retrieval/query_transformer.py`)

可选增强，按需启用：
- **HyDE**（Hypothetical Document Embeddings）：先让 LLM 生成假设答案，再用假设答案去检索
- **多查询扩展**：将用户问题改写为多个变体，分别检索后合并

#### 5.8.6 检索编排器 (`retrieval/retriever.py`)

```python
class HybridRetriever:
    def __init__(self, vector_store, bm25_retriever, reranker, settings):
        ...

    async def retrieve(
        self, query: str, top_k: int = 10
    ) -> list[SearchResult]:
        # 1. 查询改写（可选）
        # 2. 密集检索（取 top_k * 2）
        dense_results = await self.vector_store.search(query, top_k=settings.dense_top_k)

        # 3. 稀疏检索（取 top_k * 2）
        sparse_results = self.bm25_retriever.search(query, top_k=settings.sparse_top_k)

        # 4. RRF / RSF 融合
        fused = rrf_fusion(dense_results, sparse_results, top_k=settings.fusion_top_k)

        # 5. Cross-Encoder 重排序
        reranked = await self.reranker.rerank(query, fused)

        return reranked[:top_k]
```

**性能基准参考**（8 万条文档）：

| 配置 | 检索延迟 | 内存 | Recall@10 |
|------|---------|------|----------|
| BM25 only | 15ms | 120MB | 0.72 |
| Dense only (MiniLM+ONNX) | 8ms | 180MB | 0.81 |
| **Hybrid (RRF)** | 20ms | 220MB | **0.87** |
| **Hybrid + Cross-Encoder** | 35ms | 320MB | **0.91** |

---

### 5.9 生成层（LLM 抽象）

**文件**: `generation/llm.py`

```python
from enum import Enum
from abc import ABC, abstractmethod

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

class LLMClient(ABC):
    """LLM 客户端抽象基类"""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        temperature: float = 0.1,
    ) -> ChatResponse:
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        temperature: float = 0.1,
    ) -> AsyncGenerator[str, None]:
        ...

class OpenAIClient(LLMClient):
    """OpenAI API 实现"""
    def __init__(self, model: str, api_key: str, api_base: str | None):
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)

class AnthropicClient(LLMClient):
    """Anthropic API 实现"""

class OllamaClient(LLMClient):
    """Ollama 本地实现"""
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.client = ollama.AsyncClient(host=host)

class LLMFactory:
    @staticmethod
    def create(settings: LLMSettings) -> LLMClient:
        """工厂方法，根据配置返回对应客户端"""
        ...
```

#### 提示词管理 (`generation/prompt.py`)

```python
from jinja2 import Template

# 默认 RAG 系统提示词
SYSTEM_PROMPT = Template("""
你是一个智能知识库助手，基于提供的文档内容回答用户问题。

规则：
1. 仅基于提供的文档内容回答，不编造信息
2. 如果文档中没有相关信息，诚实告知用户
3. 回答要简洁准确，在末尾标注引用的文档来源
4. 当文档中包含表格时，保留 Markdown 表格格式
5. 当用户问及数据时，可调用相关工具获取精确信息

可用集合：{{ collections }}
""")

# RAG 上下文模板
RAG_CONTEXT = Template("""
{% for doc in documents %}
---
[来源 {{ loop.index }}] 文件: {{ doc.filename }}
页码: {{ doc.page_number }}

{{ doc.content }}
{% endfor %}
""")
```

---

### 5.10 Tool Calling 子系统

**设计来源**: 基于研究结论，采用自制轻量框架，约 80 行核心代码，不依赖 LangChain。

#### 文件: `tool/schema.py`

```python
import inspect
import json
from typing import Callable, get_type_hints

class Tool:
    """工具封装：Python 函数 → JSON Schema"""
    def __init__(self, fn: Callable):
        self.fn = fn
        self.name = fn.__name__
        self.description = fn.__doc__ or ""
        self.schema = self._build_schema()

    def _build_schema(self) -> dict:
        """从函数签名自动生成 JSON Schema"""
        sig = inspect.signature(self.fn)
        hints = get_type_hints(self.fn)

        type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}

        properties = {}
        required = []
        for name, param in sig.parameters.items():
            param_type = hints.get(name, str)
            prop = {"type": type_map.get(param_type, "string")}
            if param.default is inspect.Parameter.empty:
                required.append(name)
            properties[name] = prop

        schema = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            }
        }

    def execute(self, **kwargs):
        return self.fn(**kwargs)
```

#### 文件: `tool/engine.py`

```python
class ToolEngine:
    """工具执行引擎：路由、执行、错误处理、重试"""

    def __init__(self, tools: list[Tool], max_retries: int = 2):
        self._tool_map = {t.name: t for t in tools}
        self.max_retries = max_retries

    def get_openai_tools(self) -> list[dict]:
        return [t.to_openai_tool() for t in self._tool_map.values()]

    def execute_tool_call(self, tool_call: dict) -> dict:
        """
        执行单个 tool_call，返回:
        {"role": "tool", "name": str, "content": str, "tool_call_id": str}
        含错误处理和重试
        """
        ...

    async def run_loop(
        self, llm_client, messages: list[dict], tools: list[dict], max_rounds: int = 5
    ) -> str:
        """
        完整的 Tool Calling 循环:
        1. 发送 messages + tools 给 LLM
        2. 如果有 tool_calls → 执行 → 将结果追加到 messages → 回到步骤 1
        3. 如果无 tool_calls → 返回 LLM 的文本回复
        """
        ...
```

#### 文件: `tool/builtin.py` — 内置 RAG 工具

```python
def retrieve_docs(query: str, top_k: int = 3) -> str:
    """从知识库中检索相关文档"""
    # 实际调用 hybrid_retriever.retrieve()

def query_database(sql: str) -> str:
    """执行数据库查询（仅允许 SELECT）"""
    # 实际调用 SQLAlchemy session

RAG_TOOLS = [Tool(retrieve_docs), Tool(query_database)]
```

#### 工具调用路线图

```
Phase 1: retrieve_docs (连接 RAG 检索)
Phase 2: query_database (报表查询)
Phase 3: 自定义工具注册接口
Phase 4: 并行工具调用
```

---

### 5.11 RAG 管线编排

**文件**: `rag/pipeline.py`

```python
class RAGPipeline:
    """
    完整的 RAG 问答流程编排
    """
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: LLMClient,
        tool_engine: ToolEngine | None = None,
    ):
        ...

    async def query(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
        stream: bool = False,
    ) -> RAGResponse:
        """
        执行完整 RAG 流程:
        1. 构建 messages（system + history + question）
        2. 如果有 Tool Calling → 进入 tool loop
        3. 混合检索 → 获取相关文档
        4. 构建 RAG 上下文
        5. 调用 LLM 生成回答
        6. 解析引用标注
        7. 保存对话记录到 MySQL/SQLite
        返回: Answer + Sources + Citations
        """

    async def query_stream(
        self, question: str, conversation_history: list[dict] | None = None
    ) -> AsyncGenerator[str, None]:
        """流式 RAG 问答"""
        ...
```

**RAGResponse 结构**:
```python
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
```

---

### 5.12 API 层

**文件**: `api/router.py`, `api/schemas.py`

#### 端点列表

| 方法 | 路径 | 描述 |
|------|------|------|
| `POST` | `/v1/chat/completions` | 核心问答（兼容 OpenAI API 格式） |
| `POST` | `/v1/documents/ingest` | 上传文档并摄入 |
| `POST` | `/v1/documents/ingest-url` | 从 URL 摄入文档 |
| `GET` | `/v1/documents` | 列出已摄入的文档 |
| `GET` | `/v1/documents/{doc_id}` | 获取文档详情 |
| `DELETE` | `/v1/documents/{doc_id}` | 删除文档及向量 |
| `GET` | `/v1/collections` | 列出所有集合 |
| `POST` | `/v1/collections` | 创建集合 |
| `DELETE` | `/v1/collections/{name}` | 删除集合 |
| `GET` | `/v1/conversations` | 列出对话历史 |
| `GET` | `/v1/conversations/{id}` | 获取对话详情 |
| `GET` | `/v1/health` | 健康检查 |
| `GET` | `/v1/info` | 系统信息（模型、集合数、文档数） |

#### 核心请求/响应格式

**问答请求** (`POST /v1/chat/completions`):

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

**问答响应**:

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

#### 流式响应（SSE）

当 `stream: true` 时，响应为 Server-Sent Events 格式：

```
data: {"id":"rag-xxx","choices":[{"delta":{"role":"assistant"},"index":0}]}
data: {"id":"rag-xxx","choices":[{"delta":{"content":"根据"},"index":0}]}
data: {"id":"rag-xxx","choices":[{"delta":{"content":"公司"},"index":0}]}
...
data: {"id":"rag-xxx","choices":[{"delta":{"content":"财报"},"index":0}]}
data: {"id":"rag-xxx","choices":[{"delta":{},"finish_reason":"stop","citations":[...]},"index":0]}
data: [DONE]
```

#### 依赖注入 (`api/deps.py`)

```python
async def get_rag_pipeline(
    settings: Settings = Depends(get_settings),
) -> RAGPipeline:
    """通过 FastAPI 依赖注入组装 RAGPipeline 单例"""
    ...
```

---

### 5.13 文件存储子系统

**文件**: `storage/file_storage.py`

**设计来源**: 基于调研，支持本地/MinIO/OSS/Kodo/S3 等多种后端，通过策略模式 + 抽象接口实现一站式切换。

#### 5.13.1 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 抽象方式 | `StorageBackend` ABC 抽象接口 | 与 LLM 抽象保持一致，策略模式 |
| 后端切换 | 配置驱动 (`storage.backend: local/minio/oss/...`) | 零代码切换环境 |
| 开发环境 | MinIO (Docker) 或 本地文件系统 | 零成本，S3 API 兼容 |
| 国内生产 | 七牛云 Kodo 或 阿里云 OSS | 流量成本低，合规 |
| 海外生产 | AWS S3 | 全球最成熟 |
| 文件路径策略 | `{category}/{collection_id}/{date}/{hash}{ext}` | 按时间分片，防碰撞，可追溯 |

#### 5.13.2 StorageBackend 抽象接口

```python
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

class StorageBackend(ABC):
    """统一的文件存储后端抽象接口 —— 策略模式核心"""

    @abstractmethod
    async def upload_file(self, local_path: str, remote_key: str) -> str:
        """上传文件，返回访问 URL"""
        ...

    @abstractmethod
    async def upload_bytes(self, data: bytes, remote_key: str,
                           content_type: str = "") -> str:
        """上传字节数据到存储后端。返回文件的访问 URL。"""
        ...

    @abstractmethod
    async def download_file(self, remote_key: str, local_path: str) -> str:
        """下载文件到本地路径。返回本地路径。"""
        ...

    @abstractmethod
    async def download_bytes(self, remote_key: str) -> bytes:
        """读取文件为字节数据。"""
        ...

    @abstractmethod
    async def delete(self, remote_key: str) -> bool:
        """删除文件。"""
        ...

    @abstractmethod
    async def list(self, prefix: str = "") -> List[str]:
        """列出指定前缀下的所有文件键值。"""
        ...

    @abstractmethod
    async def get_url(self, remote_key: str, expires: int = 3600) -> str:
        """获取文件访问 URL（支持预签名 / CDN 加速）。"""
        ...

    @abstractmethod
    async def exists(self, remote_key: str) -> bool:
        """检查文件是否存在。"""
        ...
```

#### 5.13.3 后端实现

**本地文件系统** (`LocalFileBackend`):
- 零依赖，直接操作文件系统
- 适用于开发测试和单机部署
- 路径策略: `{root_dir}/{remote_key}`
- `get_url()` 返回基于 `base_url` 拼接的路径

**MinIO** (`MinIOBackend`):
- Docker 一行命令: `docker run -p 9000:9000 -p 9001:9001 quay.io/minio/minio server /data --console-address ":9001"`
- Python SDK: `pip install minio`
- 完全兼容 S3 API，支持 bucket 自动创建
- 预签名 URL 支持过期时间控制
- 开发/测试环境推荐（生产级私有化部署也适用）

**七牛云 Kodo** (`KodoBackend`):
- Python SDK: `pip install qiniu`
- 外网流量最低 (0.26 元/GB)，CDN 深度集成
- 接口: `put_file` / `put_data` / `private_download_url` / `BucketManager`
- 免费额度: 10GB 存储 + 10GB CDN 回源流量/月

**阿里云 OSS** (`OSSBackend`):
- Python SDK: `pip install oss2`
- 国内功能最全面，支持 STS 临时授权（生产推荐）
- 支持断点续传、批量操作、预签名 URL
- 免费额度: 5GB 存储 + 5GB 下行流量/月

**AWS S3** (`S3Backend`):
- Python SDK: `pip install boto3`
- 全球最成熟，生态最丰富
- 支持多区域、版本控制、生命周期管理
- 海外部署首选

#### 5.13.4 国内云存储价格速查

| 服务商 | 标准存储 (元/GB/月) | 外网流量 (元/GB) | 免费额度 | CDN 集成 |
|--------|-------------------|-----------------|---------|---------|
| **七牛云 Kodo** | 0.115 | **0.26** (最低) | 10GB | 深度集成 |
| 阿里云 OSS | 0.12 | 0.50 | 5GB | 支持 |
| 腾讯云 COS | 0.118 | 0.50 | **50GB** | 支持 |
| AWS S3 | $0.023 | $0.09 | 5GB | 支持(CloudFront) |

#### 5.13.5 配置示例 (`config/storage.yaml`)

```yaml
storage:
  backend: minio          # local | minio | oss | kodo | cos | s3

  local:
    root_dir: ./data/storage
    base_url: http://localhost:8000/files

  minio:
    endpoint: localhost:9000
    access_key: ${MINIO_ACCESS_KEY}
    secret_key: ${MINIO_SECRET_KEY}
    bucket: compact-rag
    secure: false

  oss:
    access_key_id: ${OSS_ACCESS_KEY_ID}
    access_key_secret: ${OSS_ACCESS_KEY_SECRET}
    endpoint: oss-cn-hangzhou.aliyuncs.com
    bucket: compact-rag

  kodo:
    access_key: ${QINIU_ACCESS_KEY}
    secret_key: ${QINIU_SECRET_KEY}
    bucket: compact-rag
    domain: https://cdn.yourdomain.com

  s3:
    region: us-east-1
    access_key_id: ${AWS_ACCESS_KEY_ID}
    secret_access_key: ${AWS_SECRET_ACCESS_KEY}
    bucket: compact-rag
```

#### 5.13.6 工厂函数

```python
from functools import lru_cache

@lru_cache()
def get_storage_backend(settings: StorageSettings) -> StorageBackend:
    """根据配置获取存储后端实例（单例缓存）"""
    backend_type = settings.backend
    if backend_type == "local":
        return LocalFileBackend(
            root_dir=settings.local.root_dir,
            base_url=settings.local.base_url,
        )
    elif backend_type == "minio":
        return MinIOBackend(
            endpoint=settings.minio.endpoint,
            access_key=settings.minio.access_key,
            secret_key=settings.minio.secret_key,
            bucket=settings.minio.bucket,
            secure=settings.minio.secure,
        )
    elif backend_type == "oss":
        return OSSBackend(
            access_key_id=settings.oss.access_key_id,
            access_key_secret=settings.oss.access_key_secret,
            endpoint=settings.oss.endpoint,
            bucket=settings.oss.bucket,
        )
    # elif backend_type == "kodo": ...
    # elif backend_type == "s3": ...
    raise ValueError(f"Unknown storage backend: {backend_type}")
```

#### 5.13.7 在 RAG 系统中的集成

**文档摄入时的文件生命周期**:

```
用户上传 → StorageBackend.upload(temp/{session_id}/{filename})
           │
           ▼ (解析完成)
    原始文件持久化: StorageBackend.upload(docs/{collection_id}/{date}/{hash}{ext})
           │
           ▼ (不再活跃访问)
    归档/清理: 临时文件 TTL 自动清理 (cron 任务, 默认 1 小时)
```

**文件路径策略**:

```python
def build_storage_key(collection_id: str, filename: str, content: bytes = None) -> str:
    """构建持久化存储路径: docs/{collection}/{year}/{month}/{day}/{hash}{ext}"""
    now = datetime.utcnow()
    date_path = f"{now.year}/{now.month:02d}/{now.day:02d}"
    file_hash = hashlib.sha256(content or filename.encode()).hexdigest()[:16]
    ext = Path(filename).suffix
    return f"docs/{collection_id}/{date_path}/{file_hash}{ext}"
```

**临时文件清理** (`TempFileCleaner`):

```python
class TempFileCleaner:
    def __init__(self, backend: StorageBackend, ttl_hours: int = 1):
        self.backend = backend
        self.ttl = timedelta(hours=ttl_hours)

    async def clean_expired(self):
        """清理过期临时文件（按路径中的时间戳判断）"""
        for key in self.backend.list(prefix="temp/"):
            # 解析时间戳 → 判断是否过期 → 删除
            ...
```

#### 5.13.8 后端选择决策流

```
文件存储需求
    │
    ├── 开发/测试环境 ──→ MinIO (Docker) 或 LocalFile
    │
    ├── 中国大陆生产 ──→ 有 CDN 需求? ──是──→ 七牛云 Kodo
    │                    │
    │                    否
    │                    │
    │                    └──→ 阿里云 OSS
    │
    ├── 海外生产 ──→ AWS S3
    │
    └── 私有化部署 ──→ MinIO (K8s/Docker)
```

---

## 6. 数据流

### 6.1 文档摄入完整流程

```
 ┌──────────────────────────────────────────────────────────┐
 │                    Ingestion Pipeline                     │
 │                                                          │
│  ① 文件上传                                               │
│     │                                                    │
│     ▼                                                    │
│  ② 存入 StorageBackend (temp/ 临时区)                      │
│     │  (Local/MinIO/OSS/Kodo/S3)                          │
│     │                                                    │
│     ▼                                                    │
│  ③ 格式检测 → 选择 Loader                                 │
│     │                                                    │
│     ▼                                                    │
│  ④ 文档解析 → 提取文本 + 表格                              │
│     │         (PDF: pypdf, DOCX: python-docx, ...)       │
│     │         (表格: Camelot → pdfplumber 后备,            │
│     │          HTML 表格: markdownify)                    │
│     │                                                    │
│     ▼                                                    │
│  ⑤ 表格 → Markdown 转换                                   │
│     │                                                    │
│     ▼                                                    │
│  ⑥ 统一文本流 → Chunking                                  │
│     │  (Recursive / Semantic / Table-Aware)               │
│     │                                                    │
│     ▼                                                    │
│  ⑦ 生成 Chunk 列表 [DocumentChunk, ...]                   │
│     │                                                    │
│     ├──────────────────────────────────┐                 │
│     ▼                                  ▼                 │
│  ⑧a Embedding Service              ⑧b DB Repo           │
│     │ (sentence-transformers)          │ (SQLAlchemy)     │
│     ▼                                  ▼                 │
│  ⑨a ChromaDB.add()                  ⑨b INSERT INTO       │
│     │ (向量 + 元数据)                   documents,         │
│     │                                  document_chunks    │
│     │                                                    │
│     ├──────────────────────────────────┐                 │
│     ▼                                  ▼                 │
│  ⑩a 持久化原始文件 (StorageBackend)    ⑩b 临时文件清理     │
│     │ docs/{collection}/{date}/         (StorageBackend)  │
│     │ {hash}{ext}                                          │
│     │                                                    │
│     └────────────────┬─────────────────┘                 │
│                      ▼                                    │
│              ⑪ 更新 IngestionJob (completed)              │
 └──────────────────────────────────────────────────────────┘
```

### 6.2 问答检索流程

```
 ┌──────────────────────────────────────────────────────────┐
 │                     RAG Pipeline                          │
 │                                                          │
 │  ① 用户提问 "公司今年的营收目标？"                          │
 │     │                                                    │
 │     ▼                                                    │
 │  ② 加载对话历史 + 构建 System Prompt                       │
 │     │                                                    │
 │     ▼                                                    │
 │  ③ [可选] Query Transformer (HyDE / 多查询)               │
 │     │                                                    │
 │     ▼                                                    │
 │  ④ 混合检索 (HybridRetriever)                             │
 │     │                                                    │
 │     ├──── Dense: ChromaDB.similarity_search(query_vec)    │
 │     │                                                │    │
 │     ├──── Sparse: BM25.search(query_tokens)           │   │
 │     │                                                │    │
 │     ├──── Fusion: RRF(dense_results, sparse_results)  │    │
 │     │                                                │    │
 │     └──── Rerank: CrossEncoder(query, candidates)     │    │
 │                                                      │    │
 │     ▼                                                │    │
 │  ⑤ 构建上下文 [DocumentChunk × 5]                      │    │
 │     │                                                │    │
 │     ▼                                                │    │
 │  ⑥ [可选] Tool Calling Loop                           │    │
 │     │  ┌─ LLM 判断是否需要调用工具                     │    │
 │     │  ├─ 需要 → 执行工具 → 结果回传 → 继续            │    │
 │     │  └─ 不需要 → 生成答案                           │    │
 │     ▼                                                │    │
 │  ⑦ LLM 生成回答 (with 上下文 + citations)              │    │
 │     │                                                │    │
 │     ▼                                                │    │
 │  ⑧ 解析引用标注 → 构建 RAGResponse                     │    │
 │     │                                                │    │
 │     ▼                                                │    │
 │  ⑨ 保存对话记录 → messages 表                          │    │
 │     │                                                │    │
 │     ▼                                                │    │
 │  ⑩ 返回 {answer, citations, usage}                    │    │
 └──────────────────────────────────────────────────────────┘
```

---

## 7. 数据库设计

### 7.1 关系型数据库（SQLAlchemy + Alembic）

#### 表: `collections` — 文档集合

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | 集合唯一标识 |
| `name` | VARCHAR(255) | UNIQUE, NOT NULL | 集合名称（如 "finance-2024"） |
| `description` | TEXT | NULLABLE | 集合描述 |
| `embedding_model` | VARCHAR(255) | NOT NULL | 使用的 embedding 模型名 |
| `chunk_size` | INTEGER | DEFAULT 500 | 分块大小 |
| `chunk_overlap` | INTEGER | DEFAULT 50 | 分块重叠 |
| `document_count` | INTEGER | DEFAULT 0 | 文档数量（冗余计数） |
| `created_at` | DATETIME | NOT NULL | 创建时间 |
| `updated_at` | DATETIME | NOT NULL | 更新时间 |

#### 表: `documents` — 文档元数据

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | 文档唯一标识 |
| `collection_id` | UUID | FK → collections.id | 所属集合 |
| `filename` | VARCHAR(500) | NOT NULL | 原始文件名 |
| `file_type` | VARCHAR(20) | NOT NULL | pdf / docx / txt / md / html |
| `file_size` | INTEGER | NOT NULL | 文件大小（字节） |
| `file_hash` | VARCHAR(64) | NOT NULL | SHA256 哈希（去重） |
| `page_count` | INTEGER | NULLABLE | 页数 |
| `chunk_count` | INTEGER | DEFAULT 0 | 分块数量 |
| `table_count` | INTEGER | DEFAULT 0 | 检测到的表格数 |
| `status` | VARCHAR(20) | DEFAULT 'pending' | pending / processing / completed / failed |
| `error_message` | TEXT | NULLABLE | 失败原因 |
| `metadata` | JSON | NULLABLE | 扩展元数据 |
| `created_at` | DATETIME | NOT NULL | |
| `updated_at` | DATETIME | NOT NULL | |

#### 表: `document_chunks` — Chunk 索引映射

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | |
| `document_id` | UUID | FK → documents.id, CASCADE | |
| `chroma_id` | VARCHAR(255) | NOT NULL | ChromaDB 中的对应 ID |
| `chunk_index` | INTEGER | NOT NULL | 本文档内的块序号 |
| `page_number` | INTEGER | NULLABLE | 所在页码 |
| `is_table` | BOOLEAN | DEFAULT FALSE | 是否为表格块 |
| `token_count` | INTEGER | NULLABLE | Token 估算数 |
| `content_hash` | VARCHAR(64) | NULLABLE | 内容哈希（更新检测） |
| `created_at` | DATETIME | NOT NULL | |

#### 表: `conversations` — 对话会话

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | |
| `collection_id` | UUID | FK → collections.id, NULLABLE | 关联集合 |
| `title` | VARCHAR(500) | DEFAULT '新对话' | 对话标题 |
| `model` | VARCHAR(100) | NOT NULL | 使用的 LLM 模型 |
| `message_count` | INTEGER | DEFAULT 0 | 消息数 |
| `created_at` | DATETIME | NOT NULL | |
| `updated_at` | DATETIME | NOT NULL | |

#### 表: `messages` — 对话消息

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | |
| `conversation_id` | UUID | FK → conversations.id, CASCADE | |
| `role` | VARCHAR(20) | NOT NULL | system / user / assistant / tool |
| `content` | TEXT | NOT NULL | 消息内容 |
| `tool_calls` | JSON | NULLABLE | 工具调用记录 |
| `sources` | JSON | NULLABLE | 引用的文档来源 |
| `token_count` | INTEGER | NULLABLE | Token 消耗 |
| `latency_ms` | INTEGER | NULLABLE | 响应延迟（毫秒） |
| `created_at` | DATETIME | NOT NULL | |

#### 表: `ingestion_jobs` — 摄入任务跟踪

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | |
| `collection_id` | UUID | FK → collections.id | |
| `status` | VARCHAR(20) | DEFAULT 'pending' | pending / running / completed / failed |
| `total_files` | INTEGER | DEFAULT 0 | |
| `processed_files` | INTEGER | DEFAULT 0 | |
| `total_chunks` | INTEGER | DEFAULT 0 | |
| `errors` | JSON | NULLABLE | 错误汇总 |
| `started_at` | DATETIME | NULLABLE | |
| `completed_at` | DATETIME | NULLABLE | |
| `created_at` | DATETIME | NOT NULL | |

#### 表: `api_keys` — API 认证（可选）

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | |
| `name` | VARCHAR(255) | NOT NULL | 密钥名称/备注 |
| `key_hash` | VARCHAR(255) | UNIQUE, NOT NULL | API Key 的哈希值 |
| `permissions` | JSON | DEFAULT '["read"]' | 权限列表 |
| `is_active` | BOOLEAN | DEFAULT TRUE | |
| `expires_at` | DATETIME | NULLABLE | |
| `created_at` | DATETIME | NOT NULL | |

#### 表: `storage_files` — 文件存储记录

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | UUID | PK | |
| `document_id` | UUID | FK → documents.id, NULLABLE | 关联的文档（可为空） |
| `storage_backend` | VARCHAR(50) | NOT NULL | 存储后端类型 (local/minio/oss/kodo/s3) |
| `storage_key` | VARCHAR(1000) | NOT NULL | 存储后端中的键路径 |
| `filename` | VARCHAR(500) | NOT NULL | 原始文件名 |
| `file_size` | INTEGER | NOT NULL | 文件大小（字节） |
| `content_type` | VARCHAR(100) | NULLABLE | MIME 类型 |
| `storage_type` | VARCHAR(20) | DEFAULT 'persistent' | temp（临时）/ persistent（持久化）/ archive（归档） |
| `expires_at` | DATETIME | NULLABLE | 过期时间（临时文件 TTL） |
| `created_at` | DATETIME | NOT NULL | |

---

### 7.2 向量数据库（ChromaDB）

#### Collection 设计

每个文档集合对应一个 ChromaDB Collection：

```
Collection Name: {collection_name}
  ↓
  Documents: [{id: chroma_id, embedding: [384 floats], metadata: {...}, document: chunk_text}]
```

#### Metadata 字段（每个 chunk）

| 字段 | 类型 | 说明 |
|------|------|------|
| `doc_id` | str | 关联的 relations.documents.id |
| `chroma_id` | str | ChromaDB 自动生成的 ID |
| `chunk_index` | int | 块序号 |
| `page_number` | int/None | 页码 |
| `filename` | str | 源文件名 |
| `collection_name` | str | 所属集合 |
| `is_table` | bool | 是否表格块 |
| `token_count` | int | 估算 token 数 |

#### 与关系数据库的同步

- 每次 `ChromaDB.add()` 后，同步写入 `document_chunks` 表
- 删除文档时，先删 ChromaDB（`collection.delete(ids=[...])`），再删关系表
- 两套存储通过 `chroma_id` / `doc_id` 关联

---

## 8. API 接口设计

### 8.1 接口分类

#### 第一组：问答核心接口

```
POST   /v1/chat/completions        # 问答（RAG + Tool Calling）
```

#### 第二组：文档管理接口

```
POST   /v1/documents/ingest        # 上传文件（multipart/form-data）
POST   /v1/documents/ingest-url    # 从 URL 摄入
GET    /v1/documents               # 列出文档（支持分页、集合过滤）
GET    /v1/documents/{doc_id}      # 文档详情
DELETE /v1/documents/{doc_id}      # 删除文档（同时删 ChromaDB 和元数据）
```

#### 第三组：集合管理接口

```
GET    /v1/collections             # 列出集合
POST   /v1/collections             # 创建集合
DELETE /v1/collections/{name}      # 删除集合（含所有文档）
```

#### 第四组：对话记录接口

```
GET    /v1/conversations           # 对话列表
GET    /v1/conversations/{id}      # 对话详情 + 消息历史
DELETE /v1/conversations/{id}      # 删除对话
```

#### 第五组：系统接口

```
GET    /v1/health                  # 健康检查（含 DB/ChromaDB/Storage 连通性）
GET    /v1/info                    # 系统信息（模型、版本、统计）
GET    /v1/files/{storage_key}     # 文件下载/预览（通过 StorageBackend）
```

### 8.2 分页规范

```json
// 响应
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

### 8.3 错误响应格式

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

---

## 9. 错误处理与异常体系

### 9.1 异常层级

```
CompactRAGException
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
│   └── EmptyResultError        # 空结果（非错误，降级处理）
├── GenerationError             # LLM 生成错误
│   ├── LLMTimeoutError         # 超时
│   ├── LLMAuthError            # 认证失败
│   └── LLMRateLimitError       # 速率限制
└── ToolExecutionError          # 工具执行错误
```

### 9.2 降级策略

| 场景 | 降级行为 |
|------|----------|
| BM25 索引为空 | 仅用 Dense 检索 |
| Embedding 服务不可用 | 仅用 BM25 |
| Cross-Encoder 加载失败 | 跳过重排序 |
| LLM API 超时 | 重试 2 次，之后返回错误 |
| 表格提取失败 | 保留原始文本，标记为未解析 |
| 云存储后端不可用 | 降级到本地文件存储 |

---

## 10. 测试策略

### 10.1 测试层级

| 层级 | 范围 | 框架 | 目标覆盖率 |
|------|------|------|-----------|
| 单元测试 | 单个函数/类 | pytest | 85%+ |
| 集成测试 | 模块间交互 | pytest + pytest-asyncio | 70%+ |
| API 测试 | HTTP 接口 | httpx + TestClient | 90%+ |
| 端到端测试 | 完整问答流程 | pytest | 关键路径 |

### 10.2 测试重点

| 模块 | 测试重点 |
|------|---------|
| **Loader** | 各格式正确解析；元数据提取；损坏文件容错；空文件处理 |
| **Chunker** | 分块大小一致性；重叠正确性；表格完整性保留；边界条件（空文本、超长文本） |
| **Table Extractor** | Camelot/pdfplumber 后备逻辑；Markdown 输出正确性；质量评估函数 |
| **Embedding** | 向量维度正确；批量/单条一致性；ONNX 模式可用 |
| **VectorStore** | 写入读取一致；按元数据过滤；删除正确性；集合隔离 |
| **FileStorage** | LocalFile/MinIO CRUD 正确性；预签名 URL；工厂函数切换后端；TTL 清理 |
| **BM25** | 中文分词正确；排序合理性；与空查询的边缘情况 |
| **Fusion (RRF)** | 融合后排序合理性；参数 k 敏感度 |
| **Reranker** | 重排后精度提升；与融合结果的兼容性 |
| **LLM Client** | 各 provider 实例化；消息格式兼容；流式输出正常；超时处理 |
| **Tool Engine** | 参数解析；工具路由；错误恢复；重试逻辑 |
| **RAG Pipeline** | 端到端一致性；引用标注正确性；对话历史持久化 |
| **API** | 请求校验；流式 SSE 正确；错误码规范；并发安全性 |

### 10.3 Fixtures 设计 (`tests/conftest.py`)

```python
@pytest.fixture
async def test_db():
    """创建临时 SQLite 数据库，测试后自动清理"""

@pytest.fixture
async def test_chromadb():
    """创建临时 ChromaDB 实例"""

@pytest.fixture
async def test_documents():
    """标准测试文档集（PDF/TXT/MD 各若干）"""

@pytest.fixture
def mock_llm_client():
    """模拟 LLM 客户端，返回固定答案"""

@pytest.fixture
async def test_rag_pipeline(test_db, test_chromadb, mock_llm_client):
    """组装完整的测试 RAG Pipeline"""
```

---

## 11. 性能优化策略

### 11.1 Embedding 优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| 批量编码 | `batch_size=64` | 2-3x 吞吐提升 |
| ONNX Runtime | `backend="onnx"` | 2-3x 推理加速 |
| int8 量化 | `(emb * 127).astype(np.int8)` | 4x 内存节省 |
| max_seq_length | 192 取代 512 | 1.5-2x 加速 |
| Matryoshka 截断 | `truncate_dim=128` | 2-3x 检索加速 |
| 模型缓存 | 单例模式，进程级缓存 | 避免重复加载 |

### 11.2 ChromaDB 优化

| 优化项 | 方法 |
|--------|------|
| 持久化路径 | SSD 存储 |
| 批量写入 | `collection.add(documents=[...], embeddings=[...])` |
| 元数据索引 | 仅在过滤字段上建索引 |
| 定期清理 | 删除过期 collection |

### 11.3 关系数据库优化

| 优化项 | 方法 |
|--------|------|
| 连接池 | SQLAlchemy async pool, pool_size=5 |
| 索引 | `documents.collection_id`, `documents.file_hash`, `messages.conversation_id` |
| 批量插入 | `session.add_all([...])` |
| 分页查询 | `.limit().offset()` |

### 11.4 API 优化

### 11.5 文件存储优化

| 优化项 | 方法 |
|--------|------|
| CDN 加速 | 七牛云 Kodo 原生 CDN；阿里云 OSS + CDN 回源 |
| 大文件分片 | 断点续传（OSS: `resumable_upload`，S3: `multipart_threshold`） |
| 临时文件清理 | TTL 定时任务，自动清理 `temp/` 目录 |
| 预签名 URL | 避免直接暴露存储地址，防止盗链 |
| 就近上传 | 选择离用户最近的云存储区域 |

| 优化项 | 方法 |
|--------|------|
| 流式响应 | SSE 分块传输 |
| 请求限流 | `slowapi` 令牌桶 |
| 响应缓存 | 相同 query 短期缓存（可选） |
| 并发处理 | FastAPI async + `asyncio.gather` |

---

## 12. 部署方案

### 12.1 开发环境

```bash
# 依赖安装
pip install -e ".[dev]"

# 数据库初始化（SQLite，自动创建）
alembic upgrade head

# 启动文件存储（MinIO Docker，可选）
docker run -d -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  quay.io/minio/minio server /data --console-address ":9001"

# 启动开发服务器
compact-rag serve --config config/default.yaml

# API 访问
curl http://localhost:8000/v1/health
```

### 12.2 生产环境

```bash
# 使用 MySQL
export COMPACT_RAG_CONFIG=config/production.yaml

# production.yaml 内容：
# database:
#   url: "mysql+asyncmy://user:pass@db-host:3306/compact_rag"
# llm:
#   provider: openai
#   model: gpt-4o
#   api_key: ${OPENAI_API_KEY}

# 运行数据库迁移
alembic upgrade head

# 启动（多 worker）
uvicorn compact_rag.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 12.3 Docker 部署

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[production]"
CMD ["uvicorn", "compact_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 12.4 配置切换对照

| 环境变量 | 说明 | 示例 |
|----------|------|------|
| `COMPACT_RAG_CONFIG` | 配置文件路径 | `config/production.yaml` |
| `DATABASE_URL` | 数据库连接（覆盖配置文件） | `mysql+asyncmy://...` |
| `OPENAI_API_KEY` | OpenAI API Key | `sk-xxx` |
| `ANTHROPIC_API_KEY` | Anthropic API Key | `sk-ant-xxx` |
| `OLLAMA_HOST` | Ollama 服务地址 | `http://localhost:11434` |
| `LOG_LEVEL` | 日志级别 | `INFO` / `DEBUG` |
| `STORAGE_BACKEND` | 文件存储后端 | `local` / `minio` / `oss` / `kodo` / `s3` |
| `MINIO_ENDPOINT` | MinIO 服务地址 | `localhost:9000` |
| `OSS_ACCESS_KEY_ID` | 阿里云 OSS AccessKey | `xxx` |
| `QINIU_ACCESS_KEY` | 七牛云 AccessKey | `xxx` |

---

## 13. 实施路线图

### Phase 1 — 项目骨架（目标：可运行的空服务）

| 任务 | 产出 | 优先级 |
|------|------|--------|
| pyproject.toml + 依赖声明 | 可安装的 Python 包 | P0 |
| 配置系统（pydantic-settings + YAML） | 多环境配置加载 | P0 |
| `common/logger.py` | 结构化日志 | P0 |
| `common/exceptions.py` | 异常体系 | P0 |
| `storage/db/engine.py` + `models.py` | SQLAlchemy 异步引擎 + 8 张表 | P0 |
| `storage/file_storage.py` (LocalFileBackend + MinIOBackend) | 文件存储抽象层 | P0 |
| Alembic 初始化 + 首次迁移 | 数据库版本管理 | P0 |
| `main.py` CLI 入口 | `compact-rag serve` | P0 |
| `api/router.py` 健康检查端点 | `/v1/health` 可访问 | P0 |

### Phase 2 — 文档摄入

| 任务 | 产出 | 优先级 |
|------|------|--------|
| `ingestion/loader.py` | 多格式文件加载 | P0 |
| `ingestion/table_extractor.py` | 表格提取 + Markdown 转换 | P0 |
| `ingestion/chunker.py` | 递归/语义分块 | P0 |
| `embedding/service.py` | sentence-transformers 封装 | P0 |
| `storage/vector_store.py` | ChromaDB CRUD | P0 |
| `ingestion/pipeline.py` | 摄入流程编排 | P0 |
| Repository 层（Document/Collection） | DB 读写 | P0 |
| `/v1/documents/ingest` + `/v1/documents` API | 文档上传和列表 | P0 |

### Phase 3 — 检索 + 生成

| 任务 | 产出 | 优先级 |
|------|------|--------|
| `retrieval/dense.py` | ChromaDB 向量搜索 | P0 |
| `retrieval/sparse.py` | BM25 检索 | P0 |
| `retrieval/fusion.py` | RRF 融合 | P0 |
| `retrieval/reranker.py` | Cross-Encoder 重排序 | P1 |
| `retrieval/retriever.py` | 混合检索编排 | P0 |
| `generation/llm.py` | LLM 客户端抽象 | P0 |
| `generation/prompt.py` | 提示词模板 | P0 |

### Phase 4 — RAG 管线 + 对话

| 任务 | 产出 | 优先级 |
|------|------|--------|
| `rag/pipeline.py` | 检索→生成完整流程 | P0 |
| Repository 层（Conversation/Message） | 对话持久化 | P0 |
| `/v1/chat/completions` API | 问答端点 | P0 |
| `/v1/conversations` API | 对话历史 | P1 |
| 流式响应（SSE）支持 | streaming=True | P1 |

### Phase 5 — Tool Calling + 完善

| 任务 | 产出 | 优先级 |
|------|------|--------|
| `tool/schema.py` + `tool/engine.py` | 轻量 Tool Calling 框架 | P1 |
| `tool/builtin.py` | 内置 RAG 工具 | P1 |
| Tool Calling 与 RAG Pipeline 集成 | 工具增强检索 | P1 |
| 错误处理 + 重试策略完善 | 鲁棒性增强 | P1 |
| 查询转换（HyDE / Multi-Query） | 检索质量提升 | P2 |
| OSSBackend / KodoBackend / S3Backend | 云存储后端 | P2 |
| TempFileCleaner 定时清理 | 临时文件管理 | P1 |

### Phase 6 — 测试 + 文档

| 任务 | 产出 | 优先级 |
|------|------|--------|
| 单元测试（全部模块） | > 85% 覆盖率 | P0 |
| 集成测试 | 关键路径覆盖 | P0 |
| API 测试 | 所有端点 | P0 |
| Docker 部署配置 | 容器化 | P2 |
| API 文档完善 | OpenAPI 文档 | P1 |

---

## 附录 A: 与外部 LLM 框架的关系

本项目在设计上**有意避免**依赖 LangChain / LlamaIndex 等重量级框架，理由：

1. **过度封装** — 这些框架对简单操作引入了过多抽象层
2. **版本不稳定** — 频繁破坏性变更
3. **调试困难** — 调用链深，问题定位成本高
4. **功能冗余** — 本系统只需要核心 RAG 能力

但不排除在以下场景选择性引入其子模块：
- `langchain-text-splitters` — 仅用其文本分割功能（已在本设计中使用）
- `langchain-community` 的 `BM25Retriever` — 如果 rank_bm25 性能不满足

### 依赖清单（`pyproject.toml` 核心依赖）

```toml
[project]
name = "compact-rag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "sqlalchemy[asyncio]>=2.0",
    "aiosqlite>=0.20",
    "asyncmy>=0.2",          # MySQL async driver (prod only)
    "alembic>=1.13",
    "chromadb>=0.5",
    "sentence-transformers>=2.7",
    "rank-bm25>=0.2",
    "httpx>=0.27",
    "python-multipart>=0.0.9",
    "pyyaml>=6.0",
    "loguru>=0.7",
    "jinja2>=3.1",
    # 文档处理
    "pypdf>=4.0",
    "python-docx>=1.1",
    "camelot-py[cv]>=0.12",
    "pdfplumber>=0.11",
    "markdownify>=0.12",
    "beautifulsoup4>=4.12",
    "jieba>=0.42",            # 中文分词
    # LLM 客户端
    "openai>=1.30",
    "anthropic>=0.25",
    "ollama>=0.4",
    # 文件存储
    "minio>=7.2",
    "oss2>=2.18",
    "qiniu>=7.13",
    "boto3>=1.34",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "httpx>=0.27",
]
production = [
    "asyncmy>=0.2",
]
```

## 附录 B: 设计决策记录

| 决策编号 | 决策 | 理由 | 替代方案 |
|----------|------|------|----------|
| D-001 | 使用 ChromaDB 而非 Qdrant | 更轻量，Python 原生，无需独立服务 | Qdrant, Weaviate, Milvus |
| D-002 | 关系数据库用 MySQL/SQLite 而非 PostgreSQL | 团队更熟悉，部署简便 | PostgreSQL + pgvector |
| D-003 | 不使用 LangChain 核心库 | 过度封装，调试困难 | LangChain, LlamaIndex |
| D-004 | Tool Calling 自制而非用 LangChain Agent | ~80 行代码即可，减少依赖 | LangChain Agent |
| D-005 | Hybrid 融合用 RRF | 不需要归一化，鲁棒性高 | RSF, Weighted Sum |
| D-006 | Sparse 用 rank_bm25 | 零依赖，< 5 万条足够 | Tantivy, Elasticsearch |
| D-007 | 异步贯穿全栈 | FastAPI + SQLAlchemy async + httpx async | 同步 |
| D-008 | 分块策略默认 recursive | 通用性好，中文优化 | Semantic, Fixed-size |
| D-009 | 文件存储使用抽象接口 + 策略模式 | 与 LLM 抽象保持一致，配置切换后端 | 直接耦合 S3 SDK |
| D-010 | 开发环境默认 MinIO | S3 兼容，零成本，部署简单 | 本地文件系统 |
| D-011 | 国内生产推荐七牛云 Kodo | 外网流量费最低 (0.26元/GB)，CDN深度集成 | 阿里云 OSS |

---

> **文档状态**: 设计完成
> **下一步**: 进入 Phase 1 实施 — 搭建项目骨架
>
> ---
>
> ## 附录 C: 引用资料
>
> 本设计基于以下研究报告：
> - [表格转 Markdown 方案调研](../research/table-to-markdown.md) — Camelot + pdfplumber 等 6 种方案对比
> - [本地低配混合检索搭建指南](../research/hybrid-retrieval.md) — Dense + Sparse + RRF + Rerank
> - [低门槛 Tool Calling 实现方案](../research/tool-calling.md) — 轻量框架，不依赖 LangChain
> - [企业级文件存储方案调研](../research/file-storage.md) — 本地 & 云存储对比
