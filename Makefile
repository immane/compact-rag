.PHONY: help install dev-install serve admin test test-cov lint clean migrate migrate-create

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

dev-install: ## Install with dev dependencies
	pip install -e ".[dev]"

all-install: ## Install all optional dependencies
	pip install -e ".[all]"

serve: ## Start the API server
	compact-rag serve

serve-prod: ## Start API server in production mode
	uvicorn compact_rag.api.router:app --host 0.0.0.0 --port 8000 --workers 4

admin: ## Start Streamlit admin UI
	streamlit run src/compact_rag/admin/app.py --server.port 8501 --server.address 127.0.0.1

test: ## Run all tests
	pytest

test-cov: ## Run tests with coverage report
	pytest --cov=src/compact_rag --cov-report=term-missing

test-unit: ## Run unit tests only
	pytest -m unit

test-integration: ## Run integration tests
	pytest -m integration

test-slow: ## Run slow tests (requires actual LLM/Embedding services)
	pytest -m slow

lint: ## Lint code
	python -m ruff check src/compact_rag/
	python -m ruff format --check src/compact_rag/

lint-fix: ## Auto-fix lint issues
	python -m ruff check --fix src/compact_rag/
	python -m ruff format src/compact_rag/

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-data: ## Clean runtime data (IRREVERSIBLE)
	rm -rf data/chromadb/ data/storage/ data/compact-rag.db
	@echo "Data cleaned. Run 'make migrate' to recreate database."

migrate: ## Run database migrations
	alembic -c alembic.ini upgrade head

migrate-create: ## Create a new migration (usage: make migrate-create MSG="description")
	alembic -c alembic.ini revision --autogenerate -m "$(MSG)"

migrate-down: ## Rollback last migration
	alembic -c alembic.ini downgrade -1

migrate-history: ## Show migration history
	alembic -c alembic.ini history
