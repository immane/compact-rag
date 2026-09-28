# 🔍 compact-rag

> 企业级 RAG 系统 — 轻量级、可直接投产的文档检索与智能问答引擎

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1116_passed-brightgreen.svg)](.)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)

**compact-rag** 是一个面向企业的检索增强生成（RAG）系统——纯 CPU 运行、本地优先、全链路异步、零外部服务依赖即可投入生产。

## 🖥️ 管理后台

| 仪表盘 | 问答调试台 | 动态工具配置 |
|---|---|---|
| <img src="docs/images/dashboard.png" alt="compact-rag 管理后台仪表盘" width="320"> | <img src="docs/images/playground.png" alt="compact-rag 问答调试台" width="320"> | <img src="docs/images/tools.png" alt="compact-rag 动态工具配置" width="320"> |

## 🏗 系统架构

```mermaid
flowchart TB
    User[用户 / API 客户端]
    Admin[Streamlit 管理后台]
    API[FastAPI REST API<br/>兼容 OpenAI + SSE]
    Pipeline[RAG 管线<br/>检索 → 重排 → 上下文 → 生成]

    User --> API
    Admin --> API
    API --> Pipeline

    subgraph Retrieval[混合检索]
        Dense[密集向量检索]
        Sparse[BM25 稀疏检索]
        Fusion[RRF / RSF 融合]
        Rerank[Cross-Encoder 重排序]
        Dense --> Fusion
        Sparse --> Fusion
        Fusion --> Rerank
    end

    subgraph Generation[生成与动态工具]
        LLM[LLM Provider<br/>OpenAI / Anthropic / Ollama]
        Tools[运行时配置工具<br/>HTTP API / 向量检索]
    end

    subgraph Ingestion[文档与 API 数据摄入]
        Sources[文件 / URL / 分页 API]
        Loaders[加载、分块、规范化]
        Embed[Embedding 服务]
        Sources --> Loaders --> Embed
    end

    Pipeline --> Dense
    Pipeline --> LLM
    Pipeline --> Tools
    Chroma[(ChromaDB<br/>向量存储)] --> Dense
    Chroma --> Sparse
    Rerank --> Pipeline
    Embed --> Chroma
    Loaders --> SQL[(SQLite / MySQL<br/>元数据与对话)]
    API --> SQL
    Loaders --> Storage[(Local / MinIO / OSS / S3<br/>文件存储)]
```

---

## ✨ 核心能力

| 能力 | 描述 |
|---|---|
| **多格式文档摄入** | PDF、DOCX、TXT、Markdown、HTML，自动提取文本和表格 |
| **表格智能处理** | 从 PDF/HTML 中提取表格并转为 Markdown，保留结构化关系 |
| **混合检索** | 密集向量检索（Embedding）+ 稀疏检索（BM25）+ Cross-Encoder 重排序 |
| **LLM 抽象** | 统一接口，支持 OpenAI / Anthropic / Ollama |
| **动态工具** | 在管理后台配置 HTTP API 或向量检索工具，支持自定义 JSON Schema 参数、请求模板和凭据掩码 |
| **API 数据源** | 配置带鉴权、分页的 API，将记录同步到集合，支持去重、更新和定时同步 |
| **集合隔离** | 每次问答可选择目标集合，避免不同知识库的检索结果相互干扰 |
| **对话记忆** | 完整对话历史，支持上下文感知的多轮问答 |
| **REST API** | 兼容 OpenAI API 格式，支持 SSE 流式输出 |
| **双数据库** | ChromaDB（向量）+ MySQL/SQLite（结构元数据） |
| **文件存储** | 统一 `StorageBackend` 抽象 — Local / MinIO / OSS / S3 多后端 |
| **管理后台** | Streamlit — 9 个页面，零前端代码 |

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

## 📡 API 端点（共 31 个）

| 分组 | 方法 | 路径 | 说明 |
|------|------|------|------|
| **问答** | `POST` | `/v1/chat/completions` | 核心问答（兼容 OpenAI API、SSE 流式、引用与工具链接） |
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
| | `POST` | `/v1/ingestion/sources/sync` | 同步已启用的外部 API 数据源（可由 cron 调用） |
| **工具** | `GET` / `PUT` | `/v1/config/tools` | 查询/更新运行时工具定义（密钥掩码） |
| | `POST` | `/v1/config/tools/{name}/test` | 测试已配置的工具 |
| **数据源** | `GET` / `PUT` | `/v1/config/sources` | 查询/替换动态 API 数据源（令牌掩码） |
| | `DELETE` | `/v1/config/sources/{name}` | 删除 API 数据源 |
| **文件存储** | `GET` | `/v1/files` | 文件列表 |
| | `GET` / `DELETE` | `/v1/files/{storage_key}` | 下载或删除文件 |
| | `POST` | `/v1/files/clean-temp` | 清理过期临时文件 |
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
    "model": "deepseek-flash",
    "messages": [{"role": "user", "content": "公司今年的营收目标是多少？"}],
    "collection": "finance-2024",
    "stream": false
  }'
```

完整 API 文档：`http://127.0.0.1:8000/docs`（Swagger）/ `http://127.0.0.1:8000/redoc`

请求中的 `collection` 用于限定检索知识库。不同书籍或数据集可放入不同集合，再在 Playground 中选择集合或在 API 请求中指定。问答响应包含 `citations`；动态工具返回的 URL 会放在 `tool_links` 中。

## 🖥 管理后台

启动基于 Streamlit 的管理后台（9 个页面）：

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

页面：仪表盘、集合管理、文档管理、摄入监控、对话浏览、问答调试台、API 密钥、文件存储、工具配置。

### 动态聊天工具

在 **Admin → Tools** 中用 JSON 添加工具。HTTP 工具支持 GET/POST，可在 URL、请求头、query 或 body 中用 `{{参数名}}` 插入模型提供的参数；`vector_search` 工具可查询指定集合。`parameters` 是传给模型的 JSON Schema。请求头凭据存于 `data/runtime_config.json`（文件权限 `0600`），在 Admin 和 API 响应中会被掩码。

HTTP 工具定义示例：

```json
{
  "name": "search_reviews",
  "description": "按产品查询客户评论",
  "kind": "http",
  "enabled": true,
  "parameters": {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"]
  },
  "method": "GET",
  "url": "https://api.example.com/reviews",
  "headers": {"x-auth-token": "YOUR_TOKEN"},
  "query": {"q": "{{query}}", "page_size": 10}
}
```

### 定时同步 API 数据源

在 **Admin → Tools → Dynamic Sources** 中配置 endpoint、鉴权 header/token、分页与响应 envelope 字段、记录 ID、标题模板和要索引的正文。同步后的文档进入所配置的 collection，引用保留来源和记录 ID。先在后台手动同步，再配置每日 cron：

```bash
curl -X POST http://127.0.0.1:8000/v1/ingestion/sources/sync
```

可通过 `?source=reviews` 只同步指定数据源。未变化记录按内容 hash 跳过，已变化记录会替换旧版本。默认支持 `data` + `code` + `paginator.current/last` 格式；所有字段路径均可配置。

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
| **管理后台** | Streamlit | Python 原生管理控制台 |
| **测试** | pytest + pytest-asyncio | 1,116 个通过（3 个跳过） |

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
  model: "deepseek-flash"
  temperature: 0.1

retrieval:
  fusion_method: "rrf"
  dense_top_k: 100
  sparse_top_k: 100
  rerank_top_k: 10
```

通过环境变量覆盖（嵌套配置使用 `COMPACT_RAG_` 前缀）：
```bash
COMPACT_RAG_DATABASE__URL=mysql+asyncmy://user:pass@host:3306/compact_rag
OPENAI_API_KEY=sk-xxx
COMPACT_RAG_CONFIG=config/production.yaml
```

后台管理的工具和 API 数据源定义保存在 `data/runtime_config.json`，无需重新部署即可更新。文件可能包含 API 凭据，创建权限为 `0600`，并已加入 Git 忽略规则。

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
pytest                          # 全部测试
pytest --cov=src/compact_rag    # 含覆盖率报告
pytest -m unit                  # 仅单元测试
pytest -m slow                  # 慢速测试（需实际 LLM/Embedding 服务）
```

### 本地 CI 等效命令

下面的命令用来在本地复现 GitHub Actions 中的 lint 与测试流程：

```bash
make ci-install                 # 根据 CI 安装依赖（pip install -e "[dev]" + ruff）
make ci-lint                    # ruff check + ruff format --check
make ci-test                    # pytest --cov=...（生成 coverage.xml）
make ci                         # ci-lint + ci-test
make github-ci                  # 安装 + lint + tests（完整 CI 流程）
```

## 🐳 Docker

```bash
cp .env.example .env   # 启动前配置 LLM API Key
docker build -t compact-rag .
docker run -d --name compact-rag --restart unless-stopped \
  -p 8000:8000 -p 8501:8501 \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  compact-rag
```

容器同时启动 API（8000）和管理后台（8501），启动时自动执行数据库迁移。数据库、向量、上传文件及运行时工具/数据源配置都保存在 `data/`。管理后台默认无密码；对外开放前请设置 `ADMIN_PASSWORD`。

## 📁 项目结构

```
compact-rag/
├── src/compact_rag/
│   ├── config/          # pydantic-settings 配置管理
│   ├── common/          # 日志系统、异常类（15 种）
│   ├── storage/         # 数据库、向量存储、文件存储
│   ├── embedding/       # sentence-transformers 封装
│   ├── ingestion/       # 加载器、分块器、动态 API 数据源与同步
│   ├── retrieval/       # 密集、稀疏、融合、重排、编排器
│   ├── generation/      # LLM 抽象（3 种 Provider）+ 提示词
│   ├── tool/            # 运行时配置的 HTTP / 向量检索工具
│   ├── rag/             # RAG 管线编排
│   ├── api/             # FastAPI 问答、摄入、工具和数据源路由
│   ├── admin/           # Streamlit 管理后台（9 页）
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
  model: "deepseek-flash"
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
