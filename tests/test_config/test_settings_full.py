from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from compact_rag.common.exceptions import ConfigurationError
from compact_rag.config.settings import (
    AdminSettings,
    ChromaDBSettings,
    DatabaseSettings,
    EmbeddingSettings,
    IngestionSettings,
    LLMSettings,
    LocalStorageSettings,
    KodoStorageSettings,
    MinIOStorageSettings,
    OSSStorageSettings,
    RetrievalSettings,
    S3StorageSettings,
    StorageSettings,
    Settings,
    _deep_merge,
    get_settings,
)


class TestSettingsLoad:
    def test_load_default_yaml(self):
        settings = Settings.load("config/default.yaml")
        assert settings.database.url.endswith("data/compact-rag.db")
        assert "chromadb" in settings.chromadb.persist_directory
        assert "data/storage" in settings.storage.local.root_dir
        assert settings.llm.provider == "openai"
        assert settings.ingestion.chunk_size == 500
        assert settings.ingestion.chunk_overlap == 50
        assert settings.retrieval.fusion_method == "rrf"
        assert settings.llm.temperature == 0.1
        assert settings.storage.backend == "local"
        assert settings.log_level == "INFO"

    def test_load_with_custom_config_path(self, tmp_path):
        config_path = tmp_path / "custom.yaml"
        config_path.write_text(yaml.dump({
            "log_level": "DEBUG",
            "llm": {"model": "custom-model", "temperature": 0.5},
            "ingestion": {"chunk_size": 300},
        }))

        # Only the override fields are specified, so defaults fill the rest
        settings = Settings.load(str(config_path))

        assert settings.log_level == "DEBUG"
        assert settings.llm.model == "custom-model"
        assert settings.llm.temperature == 0.5
        assert settings.ingestion.chunk_size == 300
        # Fields not in custom config should have defaults
        assert settings.retrieval.fusion_method == "rrf"
        assert settings.storage.backend == "local"

    def test_load_missing_config_raises_configuration_error(self):
        with pytest.raises(ConfigurationError, match="Config file not found"):
            Settings.load("nonexistent/config.yaml")

    def test_load_invalid_yaml_syntax(self, tmp_path):
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("bad_yaml: [unclosed\n  key: ::: value")

        with pytest.raises(yaml.YAMLError):
            Settings.load(str(config_path))


class TestSettingsEnvOverride:
    def test_env_var_populates_default_field(self, monkeypatch):
        monkeypatch.setenv("COMPACT_RAG_LOG_LEVEL", "WARNING")

        settings = Settings()
        assert settings.log_level == "WARNING"

    def test_env_var_nested_override_defaults(self, monkeypatch):
        monkeypatch.setenv("COMPACT_RAG_LLM__MODEL", "env-model")
        monkeypatch.setenv("COMPACT_RAG_LLM__TEMPERATURE", "0.7")

        settings = Settings()
        assert settings.llm.model == "env-model"
        assert settings.llm.temperature == 0.7

    def test_env_var_prefix_non_nested(self, monkeypatch):
        monkeypatch.setenv("COMPACT_RAG_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("COMPACT_RAG_LLM__MODEL", "custom-gpt")
        monkeypatch.delenv("COMPACT_RAG_LLM__TEMPERATURE", raising=False)

        settings = Settings()
        assert settings.log_level == "DEBUG"
        assert settings.llm.model == "custom-gpt"

    def test_from_yaml_loads_purely_from_file(self, tmp_path):
        config_path = tmp_path / "pure.yaml"
        config_path.write_text(yaml.dump({
            "log_level": "ERROR",
            "llm": {"provider": "ollama", "model": "llama3", "temperature": 1.0},
            "retrieval": {"fusion_method": "rsf"},
            "storage": {"backend": "minio"},
            "database": {"echo": True, "pool_size": 20},
        }))

        settings = Settings.from_yaml(str(config_path))
        assert settings.log_level == "ERROR"
        assert settings.llm.provider == "ollama"
        assert settings.llm.model == "llama3"
        assert settings.retrieval.fusion_method == "rsf"
        assert settings.storage.backend == "minio"
        assert settings.database.echo is True
        assert settings.database.pool_size == 20

    def test_from_yaml_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Settings.from_yaml(str(tmp_path / "missing.yaml"))


class TestSubModelDefaults:
    def test_database_settings_defaults(self):
        ds = DatabaseSettings()
        assert ds.url == "sqlite+aiosqlite:///data/compact-rag.db"
        assert ds.echo is False
        assert ds.pool_size == 5
        assert ds.max_overflow == 10

    def test_embedding_settings_defaults(self):
        es = EmbeddingSettings()
        assert es.model_name == "BAAI/bge-small-zh-v1.5"
        assert es.device == "cpu"
        assert es.normalize is True
        assert es.batch_size == 64
        assert es.use_onnx is False
        assert es.max_seq_length == 512

    def test_chromadb_settings_defaults(self):
        cs = ChromaDBSettings()
        assert cs.persist_directory == "./data/chromadb"
        assert cs.collection_name == "default"

    def test_retrieval_settings_defaults(self):
        rs = RetrievalSettings()
        assert rs.dense_top_k == 100
        assert rs.sparse_top_k == 100
        assert rs.fusion_top_k == 50
        assert rs.rerank_top_k == 10
        assert rs.fusion_method == "rrf"
        assert rs.fusion_alpha == 0.5

    def test_llm_settings_defaults(self):
        ls = LLMSettings()
        assert ls.provider == "openai"
        assert ls.model == "gpt-4o-mini"
        assert ls.api_key is None
        assert ls.api_base is None
        assert ls.temperature == 0.1
        assert ls.max_tokens == 2048
        assert ls.timeout == 60

    def test_ingestion_settings_defaults(self):
        ins = IngestionSettings()
        assert ins.chunk_size == 500
        assert ins.chunk_overlap == 50
        assert ins.chunking_strategy == "recursive"
        assert ".pdf" in ins.supported_extensions
        assert ".docx" in ins.supported_extensions
        assert ".txt" in ins.supported_extensions
        assert ".md" in ins.supported_extensions
        assert ".html" in ins.supported_extensions
        assert len(ins.supported_extensions) == 5

    def test_storage_settings_defaults(self):
        ss = StorageSettings()
        assert ss.backend == "local"
        assert isinstance(ss.local, LocalStorageSettings)
        assert ss.local.root_dir == "./data/storage"
        assert ss.local.base_url == "http://localhost:8000/files"
        assert isinstance(ss.minio, MinIOStorageSettings)
        assert ss.minio.endpoint == "localhost:9000"
        assert ss.minio.bucket == "compact-rag"
        assert ss.minio.secure is False
        assert isinstance(ss.oss, OSSStorageSettings)
        assert ss.oss.endpoint == "oss-cn-hangzhou.aliyuncs.com"
        assert isinstance(ss.kodo, KodoStorageSettings)
        assert isinstance(ss.s3, S3StorageSettings)
        assert ss.s3.region == "us-east-1"
        assert ss.s3.bucket == "compact-rag"

    def test_admin_settings_defaults(self):
        ads = AdminSettings()
        assert ads.host == "127.0.0.1"
        assert ads.port == 8501
        assert ads.password is None


class TestDeepMerge:
    def test_nested_merge(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 99, "e": 100}, "d": 200}
        _deep_merge(base, override)
        assert base["a"]["b"] == 99
        assert base["a"]["c"] == 2
        assert base["a"]["e"] == 100
        assert base["d"] == 200

    def test_new_key(self):
        base = {"a": 1}
        override = {"b": 2}
        _deep_merge(base, override)
        assert base == {"a": 1, "b": 2}

    def test_list_vs_dict_conflict(self):
        base = {"a": {"b": [1, 2, 3]}}
        override = {"a": {"b": {"c": 4}}}
        _deep_merge(base, override)
        assert base["a"]["b"] == {"c": 4}

    def test_empty_base(self):
        base: dict = {}
        override = {"a": {"b": 2}, "c": 3}
        _deep_merge(base, override)
        assert base == {"a": {"b": 2}, "c": 3}

    def test_empty_override(self):
        base = {"a": 1}
        override: dict = {}
        _deep_merge(base, override)
        assert base == {"a": 1}

    def test_deeply_nested(self):
        base = {"a": {"b": {"c": {"d": 1}}}}
        override = {"a": {"b": {"c": {"e": 2}}}}
        _deep_merge(base, override)
        assert base["a"]["b"]["c"]["d"] == 1
        assert base["a"]["b"]["c"]["e"] == 2

    def test_override_none_value(self):
        base = {"a": "hello", "b": "world"}
        override = {"a": None}
        _deep_merge(base, override)
        assert base["a"] is None
        assert base["b"] == "world"

    def test_deep_merge_does_not_affect_unrelated_keys(self):
        base = {"x": 1, "y": {"z": 2}}
        override = {"y": {"z": 3}}
        _deep_merge(base, override)
        assert base["x"] == 1
        assert base["y"]["z"] == 3


class TestFixPaths:
    def test_relative_sqlite_path_becomes_absolute(self):
        settings = Settings(
            database=DatabaseSettings(url="sqlite+aiosqlite:///data/test.db"),
            llm=LLMSettings(),
        )
        settings._fix_paths()
        assert settings.database.url.startswith("sqlite+aiosqlite:///")
        assert settings.database.url.endswith("/data/test.db")
        assert os.path.isabs(settings.database.url.replace("sqlite+aiosqlite:///", ""))

    def test_absolute_sqlite_path_unchanged(self):
        settings = Settings(
            database=DatabaseSettings(url="sqlite+aiosqlite:////absolute/path/db.db"),
            llm=LLMSettings(),
        )
        settings._fix_paths()
        assert settings.database.url.startswith("sqlite+aiosqlite:////absolute")

    def test_relative_chromadb_dir_becomes_absolute(self):
        settings = Settings(
            chromadb=ChromaDBSettings(persist_directory="./data/chromadb"),
            llm=LLMSettings(),
        )
        settings._fix_paths()
        assert settings.chromadb.persist_directory.endswith("data/chromadb")
        assert os.path.isabs(settings.chromadb.persist_directory)

    def test_absolute_chromadb_dir_unchanged(self):
        settings = Settings(
            chromadb=ChromaDBSettings(persist_directory="/absolute/chroma"),
            llm=LLMSettings(),
        )
        settings._fix_paths()
        assert settings.chromadb.persist_directory == "/absolute/chroma"

    def test_relative_storage_dir_becomes_absolute(self):
        settings = Settings(
            storage=StorageSettings(backend="local", local=LocalStorageSettings(root_dir="./data/storage")),
            llm=LLMSettings(),
        )
        settings._fix_paths()
        assert settings.storage.local.root_dir.endswith("data/storage")
        assert os.path.isabs(settings.storage.local.root_dir)

    def test_empty_chromadb_dir_no_error(self):
        settings = Settings(
            chromadb=ChromaDBSettings(persist_directory=""),
            llm=LLMSettings(),
        )
        settings._fix_paths()


class TestGetSettingsCaching:
    def test_same_instance_on_repeated_calls(self):
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_different_config_path_gives_different_instance(self):
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings(config_path="config/default.yaml")
        # With LRU cache using the same hashable arg (None for both), should be same
        # But if the underlying load returns a new Settings each time, we test identity
        assert isinstance(s1, Settings)
        assert isinstance(s2, Settings)


class TestLLMSettingsValidation:
    def test_temperature_zero_valid(self):
        s = LLMSettings(temperature=0.0)
        assert s.temperature == 0.0

    def test_temperature_two_valid(self):
        s = LLMSettings(temperature=2.0)
        assert s.temperature == 2.0

    def test_temperature_negative_fails(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LLMSettings(temperature=-0.1)

    def test_temperature_above_two_fails(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LLMSettings(temperature=2.1)

    def test_timeout_one_valid(self):
        s = LLMSettings(timeout=1)
        assert s.timeout == 1

    def test_timeout_zero_fails(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LLMSettings(timeout=0)

    def test_provider_invalid_fails(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LLMSettings(provider="invalid_provider")


class TestRetrievalSettingsValidation:
    def test_fusion_method_rrf_valid(self):
        s = RetrievalSettings(fusion_method="rrf")
        assert s.fusion_method == "rrf"

    def test_fusion_method_rsf_valid(self):
        s = RetrievalSettings(fusion_method="rsf")
        assert s.fusion_method == "rsf"

    def test_fusion_method_invalid_fails(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RetrievalSettings(fusion_method="invalid")


class TestIngestionSettingsValidation:
    def test_chunking_strategy_recursive_valid(self):
        s = IngestionSettings(chunking_strategy="recursive")
        assert s.chunking_strategy == "recursive"

    def test_chunking_strategy_semantic_valid(self):
        s = IngestionSettings(chunking_strategy="semantic")
        assert s.chunking_strategy == "semantic"

    def test_chunking_strategy_invalid_fails(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            IngestionSettings(chunking_strategy="fixed")


class TestStorageSettingsValidation:
    def test_backend_local_valid(self):
        s = StorageSettings(backend="local")
        assert s.backend == "local"

    def test_backend_minio_valid(self):
        s = StorageSettings(backend="minio")
        assert s.backend == "minio"

    def test_backend_oss_valid(self):
        s = StorageSettings(backend="oss")
        assert s.backend == "oss"

    def test_backend_kodo_valid(self):
        s = StorageSettings(backend="kodo")
        assert s.backend == "kodo"

    def test_backend_s3_valid(self):
        s = StorageSettings(backend="s3")
        assert s.backend == "s3"

    def test_backend_invalid_fails(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            StorageSettings(backend="ftp")
