"""Tests for runtime (admin-managed) configuration overrides."""

from __future__ import annotations

from compact_rag.config.runtime import (
    get_runtime_config_path,
    load_runtime_config,
    masked_dynamic_tools_view,
    save_dynamic_tools,
    save_runtime_config,
)
from compact_rag.config.settings import (
    LocalStorageSettings,
    Settings,
    StorageSettings,
)


def _settings(tmp_path):
    return Settings(
        storage=StorageSettings(
            local=LocalStorageSettings(root_dir=str(tmp_path / "storage"))
        )
    )


class TestRuntimePaths:
    def test_path_inside_data_dir(self, tmp_path):
        path = get_runtime_config_path(_settings(tmp_path))
        assert path.parent == tmp_path
        assert path.name == "runtime_config.json"

    def test_load_missing_returns_empty(self, tmp_path):
        assert load_runtime_config(_settings(tmp_path)) == {}


class TestSaveLoad:
    def test_roundtrip_and_file_permissions(self, tmp_path):
        settings = _settings(tmp_path)
        saved = save_runtime_config(
            settings, {"tools_enabled": False, "tools": []}
        )
        assert saved["tools_enabled"] is False
        assert load_runtime_config(settings) == saved
        path = get_runtime_config_path(settings)
        assert path.exists()
        assert oct(path.stat().st_mode & 0o777) == "0o600"

    def test_tool_list_patch_replaces_list(self, tmp_path):
        settings = _settings(tmp_path)
        save_runtime_config(settings, {"tools": [{"name": "lookup", "headers": {"x-token": "a"}}]})
        save_runtime_config(settings, {"tools": [{"name": "lookup", "top_k": 7}]})
        loaded = load_runtime_config(settings)
        # Lists are replaced as a whole; credential keep-on-empty is handled by save_dynamic_tools.
        assert loaded["tools"] == [{"name": "lookup", "top_k": 7}]

    def test_unknown_keys_rejected(self, tmp_path):
        import pytest

        with pytest.raises(ValueError, match="[Uu]nknown"):
            save_runtime_config(_settings(tmp_path), {"nope": 1})

    def test_corrupt_file_returns_empty(self, tmp_path):
        settings = _settings(tmp_path)
        path = get_runtime_config_path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        assert load_runtime_config(settings) == {}


class TestDynamicTools:
    def test_masks_header_secrets(self, tmp_path):
        settings = _settings(tmp_path)
        save_dynamic_tools(settings, [{
            "name": "lookup", "description": "Look up", "kind": "http",
            "headers": {"x-auth-token": "shh"},
        }])
        view = masked_dynamic_tools_view(settings)
        assert view[0]["headers"] == {"x-auth-token": ""}
        assert view[0]["headers_configured"] == {"x-auth-token": True}
        assert "shh" not in str(view)

    def test_blank_header_keeps_previous_secret(self, tmp_path):
        settings = _settings(tmp_path)
        save_dynamic_tools(settings, [{
            "name": "lookup", "description": "Look up", "kind": "http",
            "headers": {"x-auth-token": "shh"},
        }])
        save_dynamic_tools(settings, [{
            "name": "lookup", "description": "Look up", "kind": "http",
            "headers": {"x-auth-token": ""},
        }])
        assert masked_dynamic_tools_view(settings)[0]["headers_configured"]["x-auth-token"]
