# 🔍 compact-rag

> Enterprise-grade RAG system — lightweight, production-ready document retrieval and intelligent Q&A.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1116_passed-brightgreen.svg)](.)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)

**compact-rag** is a Retrieval-Augmented Generation (RAG) system built for the enterprise — CPU-only operation, local-first, async throughout, and production-deployable with zero external service dependencies.

## 🖥️ Admin UI

| Dashboard | Chat Playground | Dynamic Tools |
|---|---|---|
| <img src="docs/images/dashboard.png" alt="compact-rag admin dashboard" width="320"> | <img src="docs/images/playground.png" alt="compact-rag chat playground" width="320"> | <img src="docs/images/tools.png" alt="compact-rag dynamic tools configuration" width="320"> |

## 🏗 Architecture

```mermaid
flowchart TB
    User[User / API Client]
    Admin[Streamlit Admin]
    API[FastAPI REST API<br/>OpenAI-compatible + SSE]
    Pipeline[RAG Pipeline<br/>Retrieve → Rerank → Context → Generate]

    User --> API
    Admin --> API
    API --> Pipeline

    subgraph Retrieval[Hybrid Retrieval]
        Dense[Dense Search]
        Sparse[BM25 Search]
        Fusion[RRF / RSF Fusion]
        Rerank[Cross-Encoder Reranker]
        Dense --> Fusion
        Sparse --> Fusion
        Fusion --> Rerank
    end

    subgraph Generation[Generation and Tools]
        LLM[LLM Provider<br/>OpenAI / Anthropic / Ollama]
        Tools[Runtime-defined Tools<br/>HTTP API / Vector Search]
    end

    subgraph Ingestion[Document and API Data Ingestion]
        Sources[Files / URLs / Paginated APIs]
        Loaders[Load, Chunk, Normalize]
        Embed[Embedding Service]
        Sources --> Loaders --> Embed
    end

    Pipeline --> Dense
    Pipeline --> LLM
    Pipeline --> Tools
    Chroma[(ChromaDB<br/>Vector Store)] --> Dense
    Chroma --> Sparse
    Rerank --> Pipeline
    Embed --> Chroma
    Loaders --> SQL[(SQLite / MySQL<br/>Metadata and Conversations)]
    API --> SQL
    Loaders --> Storage[(Local / MinIO / OSS / S3<br/>File Storage)]
```

---

## ✨ Features

| Capability | Description |
|---|---|
| **Multi-format Ingestion** | PDF, DOCX, TXT, Markdown, HTML — automatic text and table extraction |
| **Intelligent Table Processing** | PDF/HTML table extraction → Markdown, preserving structural relationships |
| **Hybrid Retrieval** | Dense (Embedding) + Sparse (BM25) + Cross-Encoder Reranking |
| **LLM Abstraction** | Unified interface for OpenAI / Anthropic / Ollama |
| **Dynamic Tools** | Configure HTTP API or vector-search tools in Admin, with JSON Schema arguments, templated requests, masked credentials, and structured tool links |
| **API Data Sources** | Configure authenticated paginated APIs, map records into collections, deduplicate/update records, and sync on a schedule |
| **Collection Isolation** | Select a collection per chat so unrelated knowledge bases do not share retrieval results |
| **Conversation Memory** | Full history with context-aware multi-turn Q&A |
| **REST API** | OpenAI-compatible HTTP API with SSE streaming |
| **Dual Database** | ChromaDB (vector) + MySQL/SQLite (metadata) |
| **File Storage** | Unified `StorageBackend` — Local / MinIO / OSS / S3 |
| **Admin Dashboard** | Streamlit — 9 pages for collections, documents, conversations, tools, and storage |

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

## 📡 API Endpoints

| Group | Method | Path | Description |
|-------|--------|------|-------------|
| **Chat** | `POST` | `/v1/chat/completions` | Core Q&A (OpenAI-compatible, SSE streaming, citations and configured tool links) |
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
| | `POST` | `/v1/ingestion/sources/sync` | Sync enabled external API sources (cron-friendly) |
| **Tools** | `GET` / `PUT` | `/v1/config/tools` | Read/update runtime tool definitions (secrets masked) |
| | `POST` | `/v1/config/tools/{name}/test` | Dry-run a configured tool |
| **Sources** | `GET` / `PUT` | `/v1/config/sources` | List/replace dynamic API sources (tokens masked) |
| | `DELETE` | `/v1/config/sources/{name}` | Remove an API source |
| **Storage** | `GET` | `/v1/files` | List stored files |
| | `GET` / `DELETE` | `/v1/files/{storage_key}` | Download or delete a stored file |
| | `POST` | `/v1/files/clean-temp` | Remove expired temporary files |
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
    "model": "deepseek-flash",
    "messages": [{"role": "user", "content": "What is the revenue target for 2024?"}],
    "collection": "finance-2024",
    "stream": false
  }'
```

Full API docs: `http://127.0.0.1:8000/docs` (Swagger) / `http://127.0.0.1:8000/redoc`

The `collection` field scopes retrieval to one knowledge base. Keep unrelated
books or datasets in separate collections, then select the desired collection
in Playground or pass it in the chat request. Responses include `citations`;
URLs returned by configured tools are available in `tool_links`.

## 🖥 Admin Dashboard

Start the Streamlit-based admin dashboard (9 pages):

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

Pages: Dashboard, Collections, Documents, Ingestion, Conversations, Playground (interactive RAG), API Keys, Storage, and Tools. The Tools page defines arbitrary HTTP/vector-search tools and paginated API data sources; tool names and source types are not built into the application.

### Dynamic chat tools

In **Admin → Tools**, add a JSON tool definition. HTTP tools support GET/POST; use `{{argument_name}}` placeholders in URLs, headers, query parameters, or request bodies. `vector_search` tools query a selected collection. The `parameters` field is a JSON Schema object exposed to the model as a function tool. Credentials in request headers are stored in `data/runtime_config.json` (file mode `0600`) and masked in Admin/API responses.

Example HTTP tool:

```json
{
  "name": "search_reviews",
  "description": "Search customer reviews for a product",
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

### Scheduled API data sources

In **Admin → Tools → Dynamic Sources**, define the endpoint, auth header/token, pagination/envelope paths, record ID, title template, and fields to index. Sources sync into their configured collection, and citations retain source/record metadata. Trigger an initial sync manually, then schedule a daily call:

```bash
curl -X POST http://127.0.0.1:8000/v1/ingestion/sources/sync
```

Sync one source with `?source=reviews`. Unchanged records are skipped by content hash; changed records replace their previous indexed version. The default envelope matches APIs shaped like `data` + `code` + `paginator.current/last`; each field is configurable.

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
| **Admin** | Streamlit | Python-native management console |
| **Testing** | pytest + pytest-asyncio | 1,116 passing tests (3 skipped) |

## 🚦 Configuration

YAML-first with environment variable overrides:

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

Override with environment variables:
```bash
COMPACT_RAG_DATABASE__URL=mysql+asyncmy://user:pass@host:3306/compact_rag
OPENAI_API_KEY=sk-xxx
COMPACT_RAG_CONFIG=config/production.yaml
```

Admin-managed tool and API-source definitions are stored in `data/runtime_config.json` so they can be changed without redeploying. This file may contain API credentials, is created with mode `0600`, and is excluded from Git.

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
pytest                          # All tests
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
cp .env.example .env
# Edit .env and replace the placeholder LLM API key before starting.
docker build -t compact-rag .
docker run -d --name compact-rag --restart unless-stopped \
  -p 8000:8000 -p 8501:8501 \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  compact-rag
```

The container runs the API on port `8000` and Admin on `8501`. Database,
vectors, uploaded files, runtime tool definitions, and source sync state persist
under `data/`. The image runs migrations on startup. The Admin UI has no
password by default; set `ADMIN_PASSWORD` before exposing it beyond localhost.

## 📁 Project Structure

```
compact-rag/
├── src/compact_rag/
│   ├── config/          # pydantic-settings configuration
│   ├── common/          # logger, exceptions (15 types)
│   ├── storage/         # DB, vector store, file storage
│   ├── embedding/       # sentence-transformers service
│   ├── ingestion/       # loaders, chunkers, dynamic API sources, sync pipeline
│   ├── retrieval/       # dense, sparse, fusion, reranker, retriever
│   ├── generation/      # LLM abstraction (3 providers) + prompts
│   ├── tool/            # Runtime-defined HTTP and vector-search tools
│   ├── rag/             # RAG pipeline orchestration
│   ├── api/             # FastAPI chat, ingestion, tools, and source routes
│   ├── admin/           # Streamlit dashboard (9 pages)
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
  model: "deepseek-flash"
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
