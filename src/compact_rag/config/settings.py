"""Configuration management using pydantic-settings with YAML + env var support."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    url: str = "sqlite+aiosqlite:///data/compact-rag.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10


class EmbeddingSettings(BaseModel):
    model_name: str = "BAAI/bge-small-zh-v1.5"
    device: str = "cpu"
    normalize: bool = True
    batch_size: int = 64
    use_onnx: bool = False
    max_seq_length: int = 512


class ChromaDBSettings(BaseModel):
    persist_directory: str = "./data/chromadb"
    collection_name: str = "default"


class RetrievalSettings(BaseModel):
    dense_top_k: int = 100
    sparse_top_k: int = 100
    fusion_top_k: int = 50
    rerank_top_k: int = 10
    fusion_method: Literal["rrf", "rsf"] = "rrf"
    fusion_alpha: float = 0.5
    fusion_k: int = 60


class LLMSettings(BaseModel):
    provider: Literal["openai", "anthropic", "ollama"] = "openai"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = 2048
    timeout: int = Field(default=60, gt=0)


class IngestionSettings(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunking_strategy: Literal["recursive", "semantic"] = "recursive"
    supported_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".md", ".html"]
    )

    @model_validator(mode="after")
    def _validate_chunk_constraints(self) -> IngestionSettings:
        if self.chunk_size < self.chunk_overlap:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be >= chunk_overlap ({self.chunk_overlap})"
            )
        return self


class LocalStorageSettings(BaseModel):
    root_dir: str = "./data/storage"
    base_url: str = "http://localhost:8000/files"


class MinIOStorageSettings(BaseModel):
    endpoint: str = "localhost:9000"
    access_key: str = ""
    secret_key: str = ""
    bucket: str = "compact-rag"
    secure: bool = False


class OSSStorageSettings(BaseModel):
    access_key_id: str = ""
    access_key_secret: str = ""
    endpoint: str = "oss-cn-hangzhou.aliyuncs.com"
    bucket: str = "compact-rag"


class KodoStorageSettings(BaseModel):
    access_key: str = ""
    secret_key: str = ""
    bucket: str = "compact-rag"
    domain: str = ""


class S3StorageSettings(BaseModel):
    region: str = "us-east-1"
    access_key_id: str = ""
    secret_access_key: str = ""
    bucket: str = "compact-rag"


class StorageSettings(BaseModel):
    backend: Literal["local", "minio", "oss", "kodo", "s3"] = "local"
    local: LocalStorageSettings = Field(default_factory=LocalStorageSettings)
    minio: MinIOStorageSettings = Field(default_factory=MinIOStorageSettings)
    oss: OSSStorageSettings = Field(default_factory=OSSStorageSettings)
    kodo: KodoStorageSettings = Field(default_factory=KodoStorageSettings)
    s3: S3StorageSettings = Field(default_factory=S3StorageSettings)


class AdminSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8501
    password: str | None = None


_PROJECT_ROOT = Path(
    __file__
).parent.parent.parent.parent  # src/compact_rag/config/ → project root


def _resolve_config(path: str) -> Path:
    """Resolve a config path relative to CWD first, then project root."""
    p = Path(path)
    if p.exists():
        return p.resolve()
    # Fallback: try relative to project root
    p2 = _PROJECT_ROOT / path
    if p2.exists():
        return p2
    # Return the original path for error reporting
    return p.resolve()


class Settings(BaseSettings):
    """Top-level configuration aggregating all subsystem settings."""

    model_config = SettingsConfigDict(
        env_prefix="COMPACT_RAG_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chromadb: ChromaDBSettings = Field(default_factory=ChromaDBSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    admin: AdminSettings = Field(default_factory=AdminSettings)
    log_level: str = "INFO"

    @classmethod
    def load(cls, config_path: str | None = None) -> Settings:
        """Load YAML configuration and merge with environment variables.

        Priority (highest to lowest):
        1. Environment variables (COMPACT_RAG_* prefix)
        2. .env file
        3. Specified config file (--config / COMPACT_RAG_CONFIG)
        4. config/default.yaml
        5. Pydantic model defaults

        To use production config: COMPACT_RAG_CONFIG=config/production.yaml

        Args:
            config_path: Path to YAML config file. If None, reads from
                         COMPACT_RAG_CONFIG env var, then defaults to
                         config/default.yaml.

        Returns:
            Settings instance with merged configuration.
        """
        from compact_rag.common.exceptions import ConfigurationError

        # Load .env into environment so callers that read os.getenv() (e.g. LLMFactory)
        # can pick up keys defined in a .env file. This uses python-dotenv if
        # available; failure to import is non-fatal.
        try:
            from dotenv import load_dotenv

            env_path = _resolve_config(".env")
            if env_path.exists():
                load_dotenv(env_path)
            else:
                # Try default behavior (cwd)
                load_dotenv()
        except Exception:
            # dotenv not installed or failed to load — proceed without raising.
            pass

        # Determine config file path
        if config_path is None:
            config_path = os.environ.get("COMPACT_RAG_CONFIG", "config/default.yaml")

        # Load default config — resolve relative to CWD or project root
        default_path = _resolve_config("config/default.yaml")
        merged: dict = {}
        if default_path.exists():
            with open(default_path) as f:
                merged = yaml.safe_load(f) or {}

        # Load specified config (overrides defaults + production)
        specified_path = _resolve_config(config_path)
        if specified_path.exists():
            with open(specified_path) as f:
                override = yaml.safe_load(f) or {}
            _deep_merge(merged, override)
        elif config_path != "config/default.yaml":
            raise ConfigurationError(f"Config file not found: {config_path}")

        # Environment variables handled automatically by pydantic-settings
        settings = cls(**merged)

        # Fix relative paths to be absolute from project root
        settings._fix_paths()

        return settings

    @classmethod
    def from_yaml(cls, config_path: str) -> Settings:
        """Load settings purely from a YAML file (no env var merging).

        Args:
            config_path: Path to YAML config file.

        Returns:
            Settings instance.
        """
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        settings = cls(**data)
        settings._fix_paths()
        return settings

    def _fix_paths(self) -> None:
        """Make relative data paths absolute based on project root.

        This ensures that paths like ./data/chromadb work regardless of CWD.
        """

        db_url = self.database.url
        if db_url.startswith("sqlite+aiosqlite:///") and not db_url.startswith(
            "sqlite+aiosqlite:////"
        ):
            path = db_url.replace("sqlite+aiosqlite:///", "")
            if not os.path.isabs(path):
                abs_path = str(_PROJECT_ROOT / path)
                self.database.url = f"sqlite+aiosqlite:///{abs_path}"

        chroma_dir = self.chromadb.persist_directory
        if chroma_dir and not os.path.isabs(chroma_dir):
            self.chromadb.persist_directory = str(_PROJECT_ROOT / chroma_dir)

        storage_dir = self.storage.local.root_dir
        if storage_dir and not os.path.isabs(storage_dir):
            self.storage.local.root_dir = str(_PROJECT_ROOT / storage_dir)


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override dict into base dict in place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


@lru_cache()
def get_settings(config_path: str | None = None) -> Settings:
    """Get cached Settings singleton.

    Args:
        config_path: Optional YAML config path. Only used on first call.

    Returns:
        Cached Settings instance.
    """
    return Settings.load(config_path)
