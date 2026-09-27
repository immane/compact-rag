FROM python:3.11-slim

LABEL org.opencontainers.image.title="compact-rag"
LABEL org.opencontainers.image.description="Enterprise RAG system"

WORKDIR /app

# System dependencies (including camelot-py runtime requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ghostscript \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy project files
COPY pyproject.toml README.md ./
COPY config/ config/
COPY src/ src/
COPY alembic.ini ./

# Install production + admin (Streamlit) dependencies
RUN pip install --no-cache-dir ".[admin]"

# Copy entrypoint
COPY docker-entrypoint.sh ./

# Create data directories and hand off ownership
RUN mkdir -p data/chromadb data/storage data/documents \
    && chmod +x docker-entrypoint.sh \
    && chown -R appuser:appuser /app

VOLUME ["/app/data"]

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

EXPOSE 8000
EXPOSE 8501

ENTRYPOINT ["/app/docker-entrypoint.sh"]
