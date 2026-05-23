FROM python:3.11-slim

LABEL org.opencontainers.image.title="compact-rag"
LABEL org.opencontainers.image.description="Enterprise RAG system"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY config/ config/
COPY src/ src/
COPY alembic.ini ./

# Install dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Create data directories
RUN mkdir -p data/chromadb data/storage data/documents

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

EXPOSE 8000
EXPOSE 8501

# Default command
CMD ["compact-rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
