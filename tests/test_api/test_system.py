from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from compact_rag.api.deps import _cached_settings
from compact_rag.api.router import create_app
from compact_rag.storage.file_storage import LocalFileBackend


@pytest.fixture
def client(test_settings, monkeypatch):
    _cached_settings.cache_clear()

    monkeypatch.setattr(
        "compact_rag.api.routers.system._check_database",
        AsyncMock(return_value="ok"),
    )
    monkeypatch.setattr(
        "compact_rag.api.routers.system._check_chromadb",
        AsyncMock(return_value="ok"),
    )
    monkeypatch.setattr(
        "compact_rag.api.routers.system._check_storage",
        AsyncMock(return_value="ok"),
    )

    app = create_app(settings=test_settings)
    return TestClient(app)


class TestSystemEndpoints:
    def test_health_endpoint(self, client):
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "api" in data
        assert "database" in data
        assert "chromadb" in data
        assert "storage" in data
        assert data["api"] == "ok"

    def test_info_endpoint(self, client, test_settings):
        response = client.get("/v1/info")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "database_url" in data
        assert "embedding_model" in data
        assert "embedding_dimension" in data
        assert "llm_provider" in data
        assert "llm_model" in data
        assert "storage_backend" in data
        assert "log_level" in data
        assert data["version"] == "0.1.0"
        assert data["embedding_dimension"] == 384
        assert data["llm_provider"] == test_settings.llm.provider

    def test_file_download_returns_bytes(self, client):
        class FakeBackend:
            async def exists(self, remote_key: str) -> bool:
                return remote_key == "docs/sample.txt"

            async def get_url(self, remote_key: str, expires: int = 3600) -> str:
                return f"file:///tmp/{remote_key}"

            async def download_bytes(self, remote_key: str) -> bytes:
                assert remote_key == "docs/sample.txt"
                return b"hello storage"

        from compact_rag.api.routers import system as system_router

        client.app.dependency_overrides[system_router.get_storage_backend] = lambda: FakeBackend()
        try:
            response = client.get("/v1/files/docs/sample.txt?download=true")
        finally:
            client.app.dependency_overrides.pop(system_router.get_storage_backend, None)

        assert response.status_code == 200
        assert response.content == b"hello storage"
        content_disposition = response.headers["content-disposition"]
        assert 'filename="sample.txt"' in content_disposition
        assert "filename*=UTF-8''sample.txt" in content_disposition

    def test_file_download_supports_unicode_filename(self, client):
        class FakeBackend:
            async def exists(self, remote_key: str) -> bool:
                return remote_key == "temp/session/一场.pdf"

            async def get_url(self, remote_key: str, expires: int = 3600) -> str:
                return f"file:///tmp/{remote_key}"

            async def download_bytes(self, remote_key: str) -> bytes:
                assert remote_key == "temp/session/一场.pdf"
                return b"unicode file"

        from compact_rag.api.routers import system as system_router

        client.app.dependency_overrides[system_router.get_storage_backend] = lambda: FakeBackend()
        try:
            response = client.get("/v1/files/temp/session/%E4%B8%80%E5%9C%BA.pdf?download=true")
        finally:
            client.app.dependency_overrides.pop(system_router.get_storage_backend, None)

        assert response.status_code == 200
        assert response.content == b"unicode file"
        content_disposition = response.headers["content-disposition"]
        assert 'filename="download.pdf"' in content_disposition
        assert "filename*=UTF-8''%E4%B8%80%E5%9C%BA.pdf" in content_disposition

    def test_list_files_includes_size_and_content_type_for_local_backend(self, client):
        from compact_rag.api.routers import system as system_router

        with tempfile.TemporaryDirectory() as tmp:
            backend = LocalFileBackend(root_dir=tmp)
            file_path = Path(tmp) / "temp" / "session" / "sample.pdf"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(b"%PDF-test")

            client.app.dependency_overrides[system_router.get_storage_backend] = lambda: backend
            try:
                response = client.get("/v1/files")
            finally:
                client.app.dependency_overrides.pop(system_router.get_storage_backend, None)

        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data) == 1
        assert data[0]["filename"] == "sample.pdf"
        assert data[0]["storage_type"] == "temp"
        assert data[0]["content_type"] == "application/pdf"
        assert data[0]["file_size"] == len(b"%PDF-test")
