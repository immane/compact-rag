# 🔍 compact-rag

> 企业级 RAG 系统 — 轻量级、可直接投产的文档检索与智能问答引擎

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-145_passed-brightgreen.svg)](.)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)

**compact-rag** 是一个面向企业的检索增强生成（RAG）系统——纯 CPU 运行、本地优先、全链路异步、零外部服务依赖即可投入生产。

---

## ✨ 核心能力

| 能力 | 描述 |
|---|---|
| **多格式文档摄入** | PDF、DOCX、TXT、Markdown、HTML，自动提取文本和表格 |
| **表格智能处理** | 从 PDF/HTML 中提取表格并转为 Markdown，保留结构化关系 |
| **混合检索** | 密集向量检索（Embedding）+ 稀疏检索（BM25）+ Cross-Encoder 重排序 |
| **LLM 抽象** | 统一接口，支持 OpenAI / Anthropic / Ollama |
| **Tool Calling** | 轻量级 ~80 行工具调用框架，自动生成 JSON Schema |
| **对话记忆** | 完整对话历史，支持上下文感知的多轮问答 |
| **REST API** | 兼容 OpenAI API 格式，支持 SSE 流式输出 |
| **双数据库** | ChromaDB（向量）+ MySQL/SQLite（结构元数据） |
| **文件存储** | 统一 `StorageBackend` 抽象 — Local / MinIO / OSS / S3 多后端 |
| **管理后台** | Streamlit — 8 个页面，零前端代码 |

## 🚀 快速开始

```bash
# 克隆项目
git clone https://github.com/immane/compact-rag
cd compact-rag
python -m venv .venv && source .venv/bin/activate

# 安装
pip install -e ".[dev]"

# 数据库迁移
alembic -c alembic.ini upgrade head

# 启动服务
compact-rag serve

# 验证
curl http://127.0.0.1:8000/v1/health
# → {"api":"ok","database":"ok","chromadb":"ok","storage":"ok"}
```

📖 **[QUICKSTART.md](QUICKSTART.md)** — 完整上手指南，含示例。

## 🏗 系统架构

```
                           ┌─────────────────────────────────┐
                           │          API 层                 │
                           │  FastAPI + Pydantic v2          │
                           │  /v1/chat/completions ...       │
                           └──────────────┬──────────────────┘
                                          │
                           ┌──────────────▼───────────────────┐
                           │       RAG 管线编排               │
                           │  查询 → 检索 → 重排 →            │
                           │  上下文 → 生成 → 引文标注        │
                           └──────────────┬───────────────────┘
              ┌───────────────────────────┼───────────────────────┐
              │                           │                       │
    ┌─────────▼─────────┐   ┌─────────────▼──────────┐  ┌────────▼────────┐
    │   检索层           │   │   生成层               │  │   Tool Calling  │
    │  密集 + 稀疏       │   │  OpenAI/Anthropic/     │  │   引擎 +        │
    │  RRF + CrossEnc    │   │  Ollama • 提示词管理   │  │   内置工具      │
    └─────────┬─────────┘   └────────────────────────┘  └─────────────────┘
              │
    ┌─────────┼─────────┬──────────────┐
    ▼         ▼         ▼              ▼
 ChromaDB  SQLite/MySQL  Embedding    文件存储
 (向量)    (元数据)      服务          (Local/MinIO/OSS/S3)
```

## 📦 安装选项

```bash
pip install -e "."              # 核心依赖
pip install -e ".[dev]"         # + pytest, coverage
pip install -e ".[admin]"       # + Streamlit 管理后台
pip install -e ".[minio]"       # + MinIO 存储
pip install -e ".[oss]"         # + 阿里云 OSS 存储
pip install -e ".[s3]"          # + AWS S3 存储
pip install -e ".[all]"         # 全部依赖
```

## 🔧 CLI 命令

```bash
compact-rag serve              # 启动 API 服务 (127.0.0.1:8000)
compact-rag serve --port 8080  # 自定义端口
compact-rag serve --reload     # 开发模式，代码热重载
compact-rag admin              # 启动管理后台 (127.0.0.1:8501)
compact-rag version            # 查看版本
```

## 📡 API 端点（共 20 个）

| 分组 | 方法 | 路径 | 说明 |
|------|------|------|------|
| **问答** | `POST` | `/v1/chat/completions` | 核心问答（兼容 OpenAI API，支持 SSE 流式） |
| **文档** | `POST` | `/v1/documents/ingest` | 上传文件并摄入 |
| | `POST` | `/v1/documents/ingest-url` | 从 URL 摄入 |
| | `GET` | `/v1/documents` | 文档列表 |
| | `GET` | `/v1/documents/{id}` | 文档详情 |
| | `DELETE` | `/v1/documents/{id}` | 删除文档及向量 |
| **集合** | `GET` | `/v1/collections` | 集合列表 |
| | `POST` | `/v1/collections` | 创建集合 |
| | `DELETE` | `/v1/collections/{name}` | 删除集合 |
| **对话** | `GET` | `/v1/conversations` | 对话列表 |
| | `GET` | `/v1/conversations/{id}` | 对话详情+消息 |
| | `DELETE` | `/v1/conversations/{id}` | 删除对话 |
| **摄入** | `GET` | `/v1/ingestion-jobs` | 任务列表 |
| | `GET` | `/v1/ingestion-jobs/{id}` | 任务详情 |
| **密钥** | `GET` | `/v1/api-keys` | 密钥列表 |
| | `POST` | `/v1/api-keys` | 创建密钥 |
| | `PATCH` | `/v1/api-keys/{id}` | 激活/停用 |
| | `DELETE` | `/v1/api-keys/{id}` | 删除密钥 |
| **系统** | `GET` | `/v1/health` | 健康检查 |
| | `GET` | `/v1/info` | 系统信息 |

OpenAI 兼容的问答请求示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "公司今年的营收目标是多少？"}],
    "collection": "finance-2024",
    "stream": false
  }'
```

完整 API 文档：`http://127.0.0.1:8000/docs`（Swagger）/ `http://127.0.0.1:8000/redoc`

## 🖥 管理后台

启动基于 Streamlit 的管理后台（8 个页面）：

```bash
pip install -e ".[admin]"
compact-rag admin
# → 打开 http://127.0.0.1:8501
```

**设置密码（生产环境）：**
```bash
export ADMIN_PASSWORD="your-password"
compact-rag admin
```

页面：仪表盘、集合管理、文档管理、摄入监控、对话浏览、问答调试台、API 密钥、文件存储。

---

## 🛠 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| **语言** | Python 3.11+ | async/await、类型提示 |
| **Web 框架** | FastAPI + Pydantic v2 | 高性能异步 API |
| **数据库** | SQLAlchemy 2.0 (async) + Alembic | MySQL（生产）/ SQLite（开发） |
| **向量库** | ChromaDB | 嵌入式向量存储 |
| **嵌入模型** | sentence-transformers (BGE-small) | CPU 友好，384 维 |
| **稀疏检索** | rank_bm25 + jieba | 中英文关键词检索 |
| **重排序** | cross-encoder (MiniLM-L-6-v2) | 精度提升 |
| **LLM** | openai / anthropic / ollama SDK | 策略模式，可替换 |
| **提示词** | Jinja2 | 模板化管理 |
| **日志** | loguru | 结构化 JSON 日志 |
| **文件存储** | Local / MinIO / OSS / Kodo / S3 | 策略模式 |
| **管理后台** | Streamlit | Python 原生仪表盘 |
| **测试** | pytest + pytest-asyncio | 145+ 测试用例 |

## 🚦 配置管理

YAML 优先，环境变量可覆盖：

```yaml
# config/default.yaml
database:
  url: "sqlite+aiosqlite:///data/compact-rag.db"

embedding:
  model_name: "BAAI/bge-small-zh-v1.5"
  device: "cpu"

llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.1

retrieval:
  fusion_method: "rrf"
  dense_top_k: 100
  sparse_top_k: 100
  rerank_top_k: 10
```

通过环境变量覆盖：
```bash
DATABASE_URL=mysql+asyncmy://user:pass@host:3306/compact_rag
OPENAI_API_KEY=sk-xxx
COMPACT_RAG_CONFIG=config/production.yaml
```

详见 [config/storage.yaml](config/storage.yaml) 存储后端配置。

## 📊 性能基准

| 配置 | 检索延迟 | Recall@10 |
|---|---|---|
| 仅 BM25 | ≤ 15ms | 0.72 |
| 仅 Dense (ONNX) | ≤ 10ms | 0.81 |
| **混合 (RRF)** | ≤ 25ms | **0.87** |
| **混合 + Cross-Encoder** | ≤ 50ms | **0.91** |

*基准条件：8 万条文档，MiniLM 嵌入模型，纯 CPU。*

## 🧪 测试

```bash
pytest                          # 全部测试 (145+)
pytest --cov=src/compact_rag    # 含覆盖率报告
pytest -m unit                  # 仅单元测试
pytest -m slow                  # 慢速测试（需实际 LLM/Embedding 服务）
```

## 🐳 Docker

```bash
docker build -t compact-rag .
docker run -p 8000:8000 compact-rag
```

## 📁 项目结构

```
compact-rag/
├── src/compact_rag/
│   ├── config/          # pydantic-settings 配置管理
│   ├── common/          # 日志系统、异常类（15 种）
│   ├── storage/         # 数据库、向量存储、文件存储
│   ├── embedding/       # sentence-transformers 封装
│   ├── ingestion/       # 加载器、分块器、表格提取、摄入管道
│   ├── retrieval/       # 密集、稀疏、融合、重排、编排器
│   ├── generation/      # LLM 抽象（3 种 Provider）+ 提示词
│   ├── tool/            # Tool Calling 框架 + 内置工具
│   ├── rag/             # RAG 管线编排
│   ├── api/             # FastAPI 路由（7 模块，20 端点）
│   ├── admin/           # Streamlit 管理后台（8 页）
│   └── main.py          # CLI 入口（typer）
├── config/              # YAML 配置文件
├── tests/               # pytest 测试套件
├── docs/                # 设计文档、契约、任务分解、研究报告
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## 🤖 LLM Provider 支持

```python
# OpenAI
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"

# Anthropic
llm:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"

# Ollama (本地)
llm:
  provider: "ollama"
  model: "llama3.1"
  api_base: "http://localhost:11434"
```

所有 Provider 共享同一个 `LLMClient` 接口，无需修改代码即可切换。

## 🗄 数据库设计

**8 张表**，SQLAlchemy ORM + Alembic 迁移：

```
collections ──< documents ──< document_chunks [CASCADE]
collections ──< conversations [SET NULL] ──< messages [CASCADE]
collections ──< ingestion_jobs
documents ──< storage_files [SET NULL]
api_keys（独立）
```

开发：SQLite（`sqlite+aiosqlite:///`），零配置。  
生产：MySQL（`mysql+asyncmy://`），一行切换。

## 🌐 存储后端

| 后端 | SDK | 推荐场景 |
|------|-----|---------|
| 本地 | 零依赖 | 开发/单机部署 |
| MinIO | `minio` | 开发/私有化部署 |
| 阿里云 OSS | `oss2` | 中国大陆生产 |
| 七牛云 Kodo | `qiniu` | 中国大陆（CDN 优先） |
| AWS S3 | `boto3` | 全球生产 |

## 🎯 设计原则

1. **关注点分离** —— 每个模块职责单一，通过接口解耦
2. **配置驱动** —— 所有行为通过 YAML + 环境变量参数化
3. **异步优先** —— 全链路 `async/await`
4. **优雅降级** —— 部分故障不导致系统崩溃
5. **可观测性** —— 结构化日志（loguru），关键路径埋点
6. **不用 LangChain** —— 自研组件，依赖最小化

## 📄 开源协议

MIT — 详见 [LICENSE](LICENSE)。

---

**[QUICKSTART.md](QUICKSTART.md)** — 分步上手指南。  
**[README.md](README.md)** — English documentation.  
**[docs/design/DESIGN.md](docs/design/DESIGN.md)** — 完整架构与设计文档。
