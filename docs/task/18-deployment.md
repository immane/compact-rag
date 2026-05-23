# 任务 18: 部署方案

> **依赖**: 全部模块 | **优先级**: P2 | **预计工时**: 4h

## 目标

提供开发、生产、Docker 三种环境的完整部署方案，以及 CLI 入口实现。

## 产出文件

```
src/compact_rag/
└── main.py                     # CLI 入口 & uvicorn 启动

pyproject.toml                  # 项目元数据 + 依赖
alembic.ini                     # Alembic 配置
Makefile                        # 常用命令封装
Dockerfile                      # Docker 部署镜像
.env.example                    # 环境变量示例
```

## 详细需求

### 1. `main.py` — CLI 入口

```python
import typer

app = typer.Typer()

@app.command()
def serve(config: str = None):
    """
    启动 API 服务器
    等价于: uvicorn compact_rag.api.router:app
    """
    settings = Settings.load(config)
    uvicorn.run(
        "compact_rag.api.router:app",
        host="0.0.0.0", port=8000,
        log_level=settings.log_level.lower(),
        reload=settings.log_level == "DEBUG",
    )

@app.command()
def admin(config: str = None):
    """
    启动 Streamlit 管理后台
    先检测 API 服务是否运行，然后启动 Streamlit
    """
    import subprocess
    subprocess.run([
        "streamlit", "run",
        "src/compact_rag/admin/app.py",
        "--server.port", "8501",
        "--server.address", "127.0.0.1",
    ])

if __name__ == "__main__":
    app()
```

### 2. 开发环境部署

```bash
# 安装依赖（含开发工具）
pip install -e ".[dev]"

# 数据库初始化（SQLite 自动创建）
alembic upgrade head

# 启动文件存储（MinIO Docker，可选）
docker run -d -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  quay.io/minio/minio server /data --console-address ":9001"

# 启动 API 服务
compact-rag serve --config config/default.yaml

# 验证
curl http://localhost:8000/v1/health

# 启动管理后台（可选，需 pip install -e ".[admin]"）
streamlit run src/compact_rag/admin/app.py --server.port 8501
```

### 3. 生产环境部署

```bash
# 环境变量
export COMPACT_RAG_CONFIG=config/production.yaml

# production.yaml 示意:
# database:
#   url: "mysql+asyncmy://user:pass@db-host:3306/compact_rag"
# llm:
#   provider: openai
#   model: gpt-4o
#   api_key: ${OPENAI_API_KEY}
# storage:
#   backend: kodo
#   kodo:
#     access_key: ${QINIU_ACCESS_KEY}
#     ...

# 数据库迁移
alembic upgrade head

# 启动（多 worker）
uvicorn compact_rag.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Docker 部署

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# 安装系统依赖（Camelot 需要）
RUN apt-get update && apt-get install -y \
    ghostscript libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[production]"

# 复制代码
COPY src/ src/
COPY config/ config/
COPY alembic.ini .

# 启动
CMD alembic upgrade head && \
    uvicorn compact_rag.main:app --host 0.0.0.0 --port 8000 --workers 4
```

```bash
docker build -t compact-rag:latest .
docker run -d -p 8000:8000 \
  -e COMPACT_RAG_CONFIG=/app/config/production.yaml \
  -v ./data:/app/data \
  compact-rag:latest
```

### 5. `Makefile` — 常用命令

```makefile
# Makefile
.PHONY: help install dev test lint clean migrate

help:
	@echo "compact-rag project commands"

install:
	pip install -e ".[dev]"

dev:
	pip install -e ".[dev,admin]"

serve:
	compact-rag serve --config config/default.yaml

admin:
	streamlit run src/compact_rag/admin/app.py --server.port 8501

migrate:
	alembic upgrade head

migrate-create:
	alembic revision --autogenerate -m "$(MSG)"

test:
	pytest

test-cov:
	pytest --cov=src/compact_rag --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache *.egg-info
```

### 6. `pyproject.toml` — 项目元数据

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "compact-rag"
version = "0.1.0"
requires-python = ">=3.11"

[project.scripts]
compact-rag = "compact_rag.main:app"

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "pytest-cov>=5.0", "ruff>=0.4"]
production = ["asyncmy>=0.2"]
admin = ["streamlit>=1.35", "pandas>=2.0", "plotly>=5.18"]
```

### 7. 环境变量速查

| 环境变量 | 说明 | 示例 |
|----------|------|------|
| `COMPACT_RAG_CONFIG` | 配置文件路径 | `config/production.yaml` |
| `DATABASE_URL` | 数据库连接串（覆盖配置文件） | `mysql+asyncmy://user:pass@host:3306/compact_rag` |
| `OPENAI_API_KEY` | OpenAI API Key | `sk-xxx` |
| `ANTHROPIC_API_KEY` | Anthropic API Key | `sk-ant-xxx` |
| `OLLAMA_HOST` | Ollama 服务地址 | `http://localhost:11434` |
| `LOG_LEVEL` | 日志级别 | `INFO` / `DEBUG` |
| `STORAGE_BACKEND` | 存储后端 | `local` / `minio` / `oss` / `kodo` / `s3` |
| `ADMIN_PASSWORD` | 管理后台密码 | `xxx` |

## 验收标准

- [ ] `compact-rag serve` 启动 API 服务成功
- [ ] `compact-rag admin` 启动 Streamlit 后台
- [ ] Docker 镜像构建成功并通过健康检查
- [ ] SQLite 开发环境零配置启动
- [ ] MySQL 生产环境一行切换
- [ ] `make test` / `make lint` / `make format` 可执行
- [ ] `.env.example` 文件完整可用
