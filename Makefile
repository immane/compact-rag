.PHONY: help install dev-install serve admin test test-cov lint clean migrate migrate-create ci-install ci-lint ci-test ci github-ci github-ci-inner devcontainer-check

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

ci-install: ## Install deps like GitHub Actions CI
	python -m pip install --upgrade pip
	pip install -e ".[dev]"
	pip install ruff

ci-lint: ## Run lint exactly like GitHub Actions
	ruff check src/compact_rag/
	ruff format --check src/compact_rag/

ci-test: ## Run tests with coverage exactly like GitHub Actions
	mkdir -p data
	alembic -c alembic.ini upgrade head
	pytest --cov=src/compact_rag --cov-report=term-missing --cov-report=xml -v

ci: ci-lint ci-test ## Run local CI suite (lint + coverage tests)

devcontainer-check:
	@# Accept either a globally installed devcontainer CLI or npx (from npm)
	@if command -v devcontainer >/dev/null 2>&1; then \
		echo "devcontainer binary found"; \
	elif command -v npx >/dev/null 2>&1; then \
		echo "npx found, will use 'npx @devcontainers/cli' as fallback"; \
	else \
		echo "Neither 'devcontainer' nor 'npx' found in PATH."; \
		echo "Install devcontainer CLI globally with: npm install -g @devcontainers/cli"; \
		echo "Or install Node.js / npm so 'npx' is available."; \
		exit 1; \
	fi

github-ci-inner: ci-install ci-lint ci-test ## Run full CI flow inside the devcontainer

github-ci: devcontainer-check ## Start the devcontainer and run the full CI flow inside it
	@if command -v devcontainer >/dev/null 2>&1; then \
		devcontainer up --workspace-folder .; \
		devcontainer exec --workspace-folder . make github-ci-inner; \
	else \
		echo "devcontainer not found in PATH, falling back to npx @devcontainers/cli"; \
		npx @devcontainers/cli up --workspace-folder .; \
		npx @devcontainers/cli exec --workspace-folder . make github-ci-inner; \
	fi

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
