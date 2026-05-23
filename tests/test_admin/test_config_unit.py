from __future__ import annotations

import pytest

from compact_rag.admin.config import get_api_base_url, is_auth_required, verify_password
from compact_rag.api.routers.system import _mask_url


@pytest.mark.unit
class TestAdminConfigContract:
    def test_get_api_base_url_default(self):
        assert get_api_base_url() == "http://127.0.0.1:8000"

    def test_get_api_base_url_custom_port(self):
        assert get_api_base_url(9001) == "http://127.0.0.1:9001"

    def test_auth_required_when_password_configured(self, monkeypatch):
        monkeypatch.setattr(
            "compact_rag.admin.config.get_admin_settings",
            lambda: type("Settings", (), {"password": "secret"})(),
        )
        assert is_auth_required() is True

    def test_auth_not_required_when_password_missing(self, monkeypatch):
        monkeypatch.setattr(
            "compact_rag.admin.config.get_admin_settings",
            lambda: type("Settings", (), {"password": None})(),
        )
        assert is_auth_required() is False

    def test_verify_password_when_auth_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "compact_rag.admin.config.get_admin_settings",
            lambda: type("Settings", (), {"password": None})(),
        )
        assert verify_password("") is True
        assert verify_password("anything") is True

    def test_verify_password_rejects_empty_when_auth_enabled(self, monkeypatch):
        monkeypatch.setattr(
            "compact_rag.admin.config.get_admin_settings",
            lambda: type("Settings", (), {"password": "secret"})(),
        )
        assert verify_password("") is False

    def test_verify_password_constant_time_compare(self, monkeypatch):
        monkeypatch.setattr(
            "compact_rag.admin.config.get_admin_settings",
            lambda: type("Settings", (), {"password": "secret"})(),
        )
        assert verify_password("secret") is True
        assert verify_password("wrong") is False


@pytest.mark.unit
class TestSystemMaskUrlContract:
    def test_masks_credentials_for_url_with_auth(self):
        masked = _mask_url("postgresql+asyncpg://alice:pwd@db.example.com:5432/rag")
        assert masked == "postgresql+asyncpg://***@db.example.com:5432/rag"

    def test_leaves_plain_url_unchanged(self):
        plain = "sqlite+aiosqlite:////tmp/test.db"
        assert _mask_url(plain) == plain
