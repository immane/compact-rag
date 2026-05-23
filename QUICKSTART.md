# 🚀 Quick Start Guide

> Get compact-rag up and running in 5 minutes.

---

## Prerequisites

- Python 3.11+
- `pip` (bundled with Python)
- Optional: `curl` for API testing
- Optional: Ollama for local LLM inference

## 1. Install

```bash
# Clone the repository
git clone https://github.com/compact-rag/compact-rag
cd compact-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install with dev tools
pip install -e ".[dev]"
```

## 2. Initialize Database

```bash
# Create all 8 tables via Alembic migration
alembic -c alembic.ini upgrade head
```

This creates a SQLite database at `data/compact-rag.db` with all required tables.

## 3. Start the Server

```bash
compact-rag serve
```

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## 4. Verify

```bash
# Health check
curl http://127.0.0.1:8000/v1/health

# Expected output:
# {"api":"ok","database":"ok","chromadb":"ok","storage":"ok"}

# System info
curl http://127.0.0.1:8000/v1/info
```

Open interactive API docs: **http://127.0.0.1:8000/docs**

---

## 📄 Ingest Your First Document

### Upload a File

```bash
curl -X POST http://127.0.0.1:8000/v1/documents/ingest \
  -F "file=@/path/to/report.pdf" \
  -F "collection=my-docs"
```

### From a URL

```bash
curl -X POST http://127.0.0.1:8000/v1/documents/ingest-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/document.pdf", "collection": "my-docs"}'
```

### Check Documents

```bash
curl http://127.0.0.1:8000/v1/documents
```

---

## 💬 Ask Questions

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What are the key findings?"}],
    "collection": "my-docs",
    "stream": false
  }'
```

Response includes citations to source documents:

```json
{
  "id": "rag-1700000000",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Based on the documents...",
      "citations": [
        {"filename": "report.pdf", "page_number": 5, "score": 0.92}
      ]
    }
  }],
  "usage": {"total_tokens": 1430}
}
```

---

## 🔄 Streaming (SSE)

Add `"stream": true` for real-time token streaming:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Summarize the document"}],
    "collection": "my-docs",
    "stream": true
  }'
```

---

## 🤖 LLM Configuration

### Option A: OpenAI (cloud)

```bash
export OPENAI_API_KEY="sk-xxx"
```

`config/default.yaml` already defaults to `openai` + `gpt-4o-mini`.

### Option B: Anthropic (cloud)

```bash
export ANTHROPIC_API_KEY="sk-ant-xxx"
```

Then edit `config/default.yaml`:
```yaml
llm:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
```

### Option C: Ollama (local, free)

```bash
# Install and start Ollama
ollama pull llama3.1
ollama serve
```

Edit `config/default.yaml`:
```yaml
llm:
  provider: "ollama"
  model: "llama3.1"
```

---

## 🖥 Streamlit Admin Dashboard

The admin dashboard provides 8 pages for full system management: Dashboard, Collections, Documents, Ingestion, Conversations, Playground (interactive Q&A), API Keys, and Storage.

### Install & Start

```bash
# Install admin dependencies
pip install -e ".[admin]"

# Start admin UI (new terminal)
streamlit run src/compact_rag/admin/app.py --server.port 8501

# Or via CLI:
compact-rag admin
```

Open **http://127.0.0.1:8501**

### Set Admin Password (optional, recommended for production)

By default, the admin dashboard has no password (suitable for local development on `127.0.0.1`). For production or if exposing the dashboard on a network, set a password:

```bash
# Set password via environment variable
export ADMIN_PASSWORD="your-secure-password"

# Then start the admin
compact-rag admin
```

When a password is set, the login page appears before accessing any admin pages:

```
┌─────────────────────────────────────┐
│       🔍 Compact-RAG Admin         │
│       Authentication Required       │
│                                     │
│  Password: [________________]      │
│  [Login]                           │
└─────────────────────────────────────┘
```

**Security notes:**
- The password is stored only in the environment variable, never persisted to disk
- Comparison uses constant-time `hmac.compare_digest` to prevent timing attacks
- Without a password, the dashboard is accessible only on `127.0.0.1` (localhost)
- In production, place the dashboard behind a VPN or reverse proxy with additional auth

---

## 🗄 Switching to MySQL (Production)

```bash
# Set MySQL connection
export DATABASE_URL="mysql+asyncmy://user:pass@host:3306/compact_rag"

# Run migrations
alembic -c alembic.ini upgrade head

# Start with production config
compact-rag serve --config config/production.yaml
```

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t compact-rag .

# Run
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-xxx \
  -v $(pwd)/data:/app/data \
  compact-rag
```

---

## 🧪 Running Tests

```bash
# All tests
pytest

# With coverage report
pytest --cov=src/compact_rag --cov-report=term-missing

# Specific test module
pytest tests/test_retrieval/

# Unit tests only
pytest -m unit

# Skip slow tests (LLM/Embedding)
pytest -m "not slow"
```

---

## 🛠 Common Commands

```bash
make help           # Show all make targets
make migrate        # Run pending migrations
make migrate-create MSG="add index"   # Create new migration
make test           # Run all tests
make test-cov       # Tests with coverage
make clean          # Remove build artifacts
```

---

## 📁 Creating Test Fixtures

Place test documents in `tests/fixtures/`:

```
tests/fixtures/
├── sample.pdf
├── sample.docx
├── sample.txt
├── sample.md
└── sample_table.html
```

---

## 🔍 Troubleshooting

**Server won't start?**
```bash
# Check port availability
lsof -i :8000

# Use a different port
compact-rag serve --port 8080
```

**Embedding model download slow?**
```bash
# Pre-download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"
```

**ChromaDB errors?**
```bash
# Reset ChromaDB
rm -rf data/chromadb/
```

**Database reset?**
```bash
# Drop all tables and recreate
rm -f data/compact-rag.db
alembic -c alembic.ini upgrade head
```

---

## 📚 Next Steps

- **[README.md](README.md)** — full project documentation (English)
- **[README.zh-cn.md](README.zh-cn.md)** — 中文文档
- **[docs/design/DESIGN.md](docs/design/DESIGN.md)** — architecture & design
- **[docs/design/CONTRACTS.md](docs/design/CONTRACTS.md)** — interface contracts
- `http://127.0.0.1:8000/docs` — interactive Swagger API docs
