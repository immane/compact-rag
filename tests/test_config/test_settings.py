from __future__ import annotations



from compact_rag.config.settings import (
    Settings,
    _deep_merge,
    get_settings,
)


class TestSettingsLoad:
    def test_load_default_yaml(self):
        settings = Settings.load("config/default.yaml")
        # Paths are resolved relative to project root
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

    def test_get_settings_caching(self):
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_deep_merge_nested(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 99, "e": 100}, "d": 200}
        _deep_merge(base, override)
        assert base["a"]["b"] == 99
        assert base["a"]["c"] == 2
        assert base["a"]["e"] == 100
        assert base["d"] == 200

    def test_deep_merge_new_key(self):
        base = {"a": 1}
        override = {"b": 2}
        _deep_merge(base, override)
        assert base == {"a": 1, "b": 2}
