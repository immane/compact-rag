"""Tests for runtime (admin-managed) configuration overrides."""

from __future__ import annotations

from compact_rag.config.runtime import (
    effective_settings,
    get_runtime_config_path,
    load_runtime_config,
    masked_commerce_view,
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
            settings, {"commerce_enabled": False, "products": {"collection": "meds"}}
        )
        assert saved["commerce_enabled"] is False
        assert load_runtime_config(settings) == saved
        path = get_runtime_config_path(settings)
        assert path.exists()
        assert oct(path.stat().st_mode & 0o777) == "0o600"

    def test_deep_merge(self, tmp_path):
        settings = _settings(tmp_path)
        save_runtime_config(settings, {"products": {"collection": "a"}})
        save_runtime_config(settings, {"products": {"top_k": 7}})
        loaded = load_runtime_config(settings)
        assert loaded["products"] == {"collection": "a", "top_k": 7}

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


class TestEffectiveSettings:
    def test_defaults_without_overrides(self, tmp_path):
        settings = _settings(tmp_path)
        effective, enabled = effective_settings(settings)
        assert enabled is True
        assert effective.products.collection == "products"
        assert effective.order.create_path == "/orders/link"
        # Original untouched.
        assert settings.products.collection == "products"

    def test_overrides_applied(self, tmp_path):
        settings = _settings(tmp_path)
        save_runtime_config(
            settings,
            {
                "commerce_enabled": False,
                "products": {"collection": "meds", "top_k": 3},
                "order": {"link_ttl_minutes": 10},
            },
        )
        effective, enabled = effective_settings(settings)
        assert enabled is False
        assert effective.products.collection == "meds"
        assert effective.products.top_k == 3
        assert effective.order.link_ttl_minutes == 10

    def test_masked_view_hides_secrets(self, tmp_path):
        settings = _settings(tmp_path)
        save_runtime_config(
            settings,
            {"order": {
                "url_template": "https://x/{product_id}",
                "signing_secret": "shh",
            }},
        )
        view = masked_commerce_view(settings)
        assert view["order"]["signing_secret_configured"] is True
        assert "shh" not in str(view)
        assert view["order"]["mode"] == "template"
        assert view["order"]["url_template"] == "https://x/{product_id}"
