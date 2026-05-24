# 🔍 compact-rag

> Enterprise-grade RAG system — lightweight, production-ready document retrieval and intelligent Q&A.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-145_passed-brightgreen.svg)](.)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)

**compact-rag** is a Retrieval-Augmented Generation (RAG) system built for the enterprise — CPU-only operation, local-first, async throughout, and production-deployable with zero external service dependencies.

---

## ✨ Features

| Capability | Description |
|---|---|
| **Multi-format Ingestion** | PDF, DOCX, TXT, Markdown, HTML — automatic text and table extraction |
| **Intelligent Table Processing** | PDF/HTML table extraction → Markdown, preserving structural relationships |
| **Hybrid Retrieval** | Dense (Embedding) + Sparse (BM25) + Cross-Encoder Reranking |
| **LLM Abstraction** | Unified interface for OpenAI / Anthropic / Ollama |
| **Tool Calling** | Lightweight ~80-line framework for LLM tool execution |
| **Conversation Memory** | Full history with context-aware multi-turn Q&A |
| **REST API** | OpenAI-compatible HTTP API with SSE streaming |
| **Dual Database** | ChromaDB (vector) + MySQL/SQLite (metadata) |
| **File Storage** | Unified `StorageBackend` — Local / MinIO / OSS / S3 |
| **Admin Dashboard** | Streamlit — 8 pages, zero front-end code |

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/immane/compact-rag
cd compact-rag
python -m venv .venv && source .venv/bin/activate

# Install
pip install -e ".[dev]"

# Database migration
alembic -c alembic.ini upgrade head

# Start server
compact-rag serve

# Verify
curl http://127.0.0.1:8000/v1/health
# → {"api":"ok","database":"ok","chromadb":"ok","storage":"ok"}
```

📖 **[QUICKSTART.md](QUICKSTART.md)** — complete setup guide with examples.

## 🏗 Architecture

```
                           ┌─────────────────────────────────┐
                           │          API Layer              │
                           │  FastAPI + Pydantic v2          │
                           │  /v1/chat/completions ...       │
                           └──────────────┬──────────────────┘
                                          │
                           ┌──────────────▼───────────────────┐
                           │       RAG Pipeline               │
                           │  query → retrieve → rerank →     │
                           │  context → generate → citations  │
                           └──────────────┬───────────────────┘
              ┌───────────────────────────┼───────────────────────┐
              │                           │                       │
    ┌─────────▼─────────┐   ┌─────────────▼──────────┐  ┌────────▼────────┐
    │   Retrieval Layer  │   │   Generation Layer     │  │  Tool Calling   │
    │  Dense + Sparse    │   │  OpenAI/Anthropic/     │  │  Engine +       │
    │  RRF + CrossEnc    │   │  Ollama • Prompt Mgr   │  │  Builtin Tools  │
    └─────────┬─────────┘   └────────────────────────┘  └─────────────────┘
              │
    ┌─────────┼─────────┬──────────────┐
    ▼         ▼         ▼              ▼
 ChromaDB  SQLite/MySQL  Embedding    File Storage
 (Vector)  (Metadata)    Service      (Local/MinIO/OSS/S3)
```

## 📦 Installation Options

```bash
pip install -e "."              # Core only
pip install -e ".[dev]"         # + pytest, coverage
pip install -e ".[admin]"       # + Streamlit dashboard
pip install -e ".[minio]"       # + MinIO storage
pip install -e ".[oss]"         # + Alibaba OSS storage
pip install -e ".[s3]"          # + AWS S3 storage
pip install -e ".[all]"         # Everything
```

## 🔧 CLI

```bash
compact-rag serve              # Start API server (127.0.0.1:8000)
compact-rag serve --port 8080  # Custom port
compact-rag serve --reload     # Dev mode with auto-reload
compact-rag admin              # Start Streamlit admin (127.0.0.1:8501)
compact-rag version            # Show version
```

## 📡 API Endpoints (20 total)

| Group | Method | Path | Description |
|-------|--------|------|-------------|
| **Chat** | `POST` | `/v1/chat/completions` | Core Q&A (OpenAI-compatible, SSE streaming) |
| **Documents** | `POST` | `/v1/documents/ingest` | Upload & ingest file |
| | `POST` | `/v1/documents/ingest-url` | Ingest from URL |
| | `GET` | `/v1/documents` | List documents |
| | `GET` | `/v1/documents/{id}` | Document detail |
| | `DELETE` | `/v1/documents/{id}` | Delete document + vectors |
| **Collections** | `GET` | `/v1/collections` | List collections |
| | `POST` | `/v1/collections` | Create collection |
| | `DELETE` | `/v1/collections/{name}` | Delete collection |
| **Conversations** | `GET` | `/v1/conversations` | List conversations |
| | `GET` | `/v1/conversations/{id}` | Detail + messages |
| | `DELETE` | `/v1/conversations/{id}` | Delete conversation |
| **Ingestion** | `GET` | `/v1/ingestion-jobs` | List jobs |
| | `GET` | `/v1/ingestion-jobs/{id}` | Job detail |
| **API Keys** | `GET` | `/v1/api-keys` | List keys |
| | `POST` | `/v1/api-keys` | Create key |
| | `PATCH` | `/v1/api-keys/{id}` | Toggle activate/deactivate |
| | `DELETE` | `/v1/api-keys/{id}` | Delete key |
| **System** | `GET` | `/v1/health` | Health check |
| | `GET` | `/v1/info` | System info |

OpenAI-compatible chat request:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is the revenue target for 2024?"}],
    "collection": "finance-2024",
    "stream": false
  }'
```

Full API docs: `http://127.0.0.1:8000/docs` (Swagger) / `http://127.0.0.1:8000/redoc`

## 🖥 Admin Dashboard

Start the Streamlit-based admin dashboard (8 pages):

```bash
pip install -e ".[admin]"
compact-rag admin
# → Open http://127.0.0.1:8501
```

**Set password (production):**
```bash
export ADMIN_PASSWORD="your-password"
compact-rag admin
```

Pages: Dashboard, Collections, Documents, Ingestion, Conversations, Playground (interactive RAG), API Keys, Storage.

---

## 🛠 Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.11+ | async/await, type hints |
| **Web** | FastAPI + Pydantic v2 | High-performance async API |
| **Database** | SQLAlchemy 2.0 (async) + Alembic | MySQL (prod) / SQLite (dev) |
| **Vector DB** | ChromaDB | Embedded vector storage |
| **Embedding** | sentence-transformers (BGE-small) | CPU-friendly, 384-dim |
| **Sparse Search** | rank_bm25 + jieba | Chinese/English keyword search |
| **Reranking** | cross-encoder (MiniLM-L-6-v2) | Precision boost |
| **LLM** | openai / anthropic / ollama SDK | Strategy pattern, swappable |
| **Prompting** | Jinja2 | Template-based prompt management |
| **Logging** | loguru | Structured JSON logging |
| **Storage** | Local / MinIO / OSS / Kodo / S3 | Strategy pattern |
| **Admin** | Streamlit | Python-native dashboard |
| **Testing** | pytest + pytest-asyncio | 145+ tests |

## 🚦 Configuration

YAML-first with environment variable override:

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

Override with environment variables:
```bash
DATABASE_URL=mysql+asyncmy://user:pass@host:3306/compact_rag
OPENAI_API_KEY=sk-xxx
COMPACT_RAG_CONFIG=config/production.yaml
```

See [config/storage.yaml](config/storage.yaml) for storage backend configuration.

## 📊 Performance Targets

| Configuration | Latency | Recall@10 |
|---|---|---|
| BM25 only | ≤ 15ms | 0.72 |
| Dense only (ONNX) | ≤ 10ms | 0.81 |
| **Hybrid (RRF)** | ≤ 25ms | **0.87** |
| **Hybrid + Cross-Encoder** | ≤ 50ms | **0.91** |

*Benchmarks with 80K documents, MiniLM embeddings, CPU-only.*

## 🧪 Testing

```bash
pytest                          # All tests (145+)
pytest --cov=src/compact_rag    # With coverage
pytest -m unit                  # Unit tests only
pytest -m slow                  # Slow tests (actual LLM/Embedding calls)
```

### Local CI-equivalent commands

```bash
make ci-install                 # Install deps like GitHub Actions
make ci-lint                    # ruff check + ruff format --check on src/compact_rag/
make ci-test                    # pytest with coverage xml + term report
make ci                         # ci-lint + ci-test

make github-ci                  # GitHub all CI progress
```

## 🐳 Docker

```bash
docker build -t compact-rag .
docker run -p 8000:8000 compact-rag
```

## 📁 Project Structure

```
compact-rag/
├── src/compact_rag/
│   ├── config/          # pydantic-settings configuration
│   ├── common/          # logger, exceptions (15 types)
│   ├── storage/         # DB, vector store, file storage
│   ├── embedding/       # sentence-transformers service
│   ├── ingestion/       # loaders, chunkers, table extraction, pipeline
│   ├── retrieval/       # dense, sparse, fusion, reranker, retriever
│   ├── generation/      # LLM abstraction (3 providers) + prompts
│   ├── tool/            # Tool calling framework + builtin tools
│   ├── rag/             # RAG pipeline orchestration
│   ├── api/             # FastAPI routes (7 modules, 20 endpoints)
│   ├── admin/           # Streamlit dashboard (8 pages)
│   └── main.py          # CLI entry (typer)
├── config/              # YAML config files
├── tests/               # pytest test suite
├── docs/                # Design, contracts, tasks, research
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## 🤖 LLM Provider Support

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

# Ollama (local)
llm:
  provider: "ollama"
  model: "llama3.1"
  api_base: "http://localhost:11434"
```

All providers share the same `LLMClient` interface — swap without code changes.

## 🗄 Database Design

**8 tables** via SQLAlchemy ORM + Alembic migrations:

```
collections ──< documents ──< document_chunks [CASCADE]
collections ──< conversations [SET NULL] ──< messages [CASCADE]
collections ──< ingestion_jobs
documents ──< storage_files [SET NULL]
api_keys (standalone)
```

Dev: SQLite (`sqlite+aiosqlite:///`), zero-config.  
Prod: MySQL (`mysql+asyncmy://`), one-line switch.

## 🌐 Storage Backends

| Backend | SDK | Best For |
|---------|-----|----------|
| Local | zero-dependency | Dev / single-node |
| MinIO | `minio` | Dev / private cloud |
| OSS | `oss2` | Mainland China production |
| Kodo | `qiniu` | Mainland China (CDN priority) |
| S3 | `boto3` | Global production |

## 🎯 Design Principles

1. **Separation of Concerns** — each module has a single responsibility
2. **Configuration-Driven** — all behavior parameterized via YAML + env vars
3. **Async-First** — `async/await` throughout the stack
4. **Graceful Degradation** — partial failures don't crash the system
5. **Observable** — structured logging (loguru), key-path telemetry
6. **No LangChain** — self-built components, minimal dependencies

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

**[QUICKSTART.md](QUICKSTART.md)** — step-by-step getting started guide.  
**[README.zh-cn.md](README.zh-cn.md)** — 中文文档.  
**[docs/design/DESIGN.md](docs/design/DESIGN.md)** — full architecture & design document.
