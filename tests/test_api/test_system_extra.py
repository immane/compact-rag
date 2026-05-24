"""Extra system endpoint tests: redirects, file download errors, clean-temp, delete, health errors."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from compact_rag.api.deps import _cached_settings
from compact_rag.api.router import create_app
from compact_rag.storage.file_storage import LocalFileBackend


@pytest.fixture
def client(test_settings):
    _cached_settings.cache_clear()
    app = create_app(settings=test_settings)
    return TestClient(app)


class TestFileRedirect:
    def test_file_download_false_redirects(self, client):
        """When download=false (default), the endpoint redirects to presigned URL."""

        class FakeBackend:
            async def exists(self, remote_key: str) -> bool:
                return True

            async def get_url(self, remote_key: str, expires: int = 3600) -> str:
                return f"https://presigned.example.com/{remote_key}?exp={expires}"

        from compact_rag.api.routers import system as system_router

        client.app.dependency_overrides[system_router.get_storage_backend] = lambda: FakeBackend()
        try:
            response = client.get("/v1/files/docs/sample.pdf", follow_redirects=False)
        finally:
            client.app.dependency_overrides.pop(system_router.get_storage_backend, None)

        assert response.status_code == 307  # RedirectResponse default
        assert "presigned.example.com" in response.headers["location"]

    def test_file_download_nonexistent_raises_404(self, client):
        """Non-existent file returns 404."""
        from compact_rag.api.routers import system as system_router

        class FakeNotFoundBackend:
            async def exists(self, remote_key: str) -> bool:
                return False

        client.app.dependency_overrides[system_router.get_storage_backend] = lambda: FakeNotFoundBackend()
        try:
            response = client.get("/v1/files/docs/ghost.pdf?download=true")
        finally:
            client.app.dependency_overrides.pop(system_router.get_storage_backend, None)

        assert response.status_code == 404


class TestCleanTempFiles:
    def test_clean_temp_files_endpoint(self, client, test_settings):
        """POST /files/clean-temp returns cleaned count."""
        import tempfile
        from pathlib import Path

        from compact_rag.api.routers import system as system_router

        with tempfile.TemporaryDirectory() as tmp:
            backend = LocalFileBackend(root_dir=tmp)
            # Create an expired file
            old_ts = "00000001000000"
            file_path = Path(tmp) / "temp" / "session" / f"{old_ts}_expired.txt"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(b"old content")

            client.app.dependency_overrides[system_router.get_storage_backend] = lambda: backend
            try:
                response = client.post("/v1/files/clean-temp")
            finally:
                client.app.dependency_overrides.pop(system_router.get_storage_backend, None)

        assert response.status_code == 200
        data = response.json()
        assert "cleaned" in data
        assert isinstance(data["cleaned"], int)


class TestDeleteFile:
    def test_delete_file_endpoint_success(self, client):
        from compact_rag.api.routers import system as system_router

        class FakeDeleteBackend:
            async def exists(self, remote_key: str) -> bool:
                return True

            async def delete(self, remote_key: str) -> bool:
                return True

        client.app.dependency_overrides[system_router.get_storage_backend] = lambda: FakeDeleteBackend()
        try:
            response = client.delete("/v1/files/docs/to-delete.txt")
        finally:
            client.app.dependency_overrides.pop(system_router.get_storage_backend, None)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["storage_key"] == "docs/to-delete.txt"

    def test_delete_file_not_found(self, client):
        from compact_rag.api.routers import system as system_router

        class FakeNotFound:
            async def exists(self, remote_key: str) -> bool:
                return False

        client.app.dependency_overrides[system_router.get_storage_backend] = lambda: FakeNotFound()
        try:
            response = client.delete("/v1/files/docs/missing.txt")
        finally:
            client.app.dependency_overrides.pop(system_router.get_storage_backend, None)

        assert response.status_code == 404


class TestListFilesPrefix:
    def test_list_files_with_prefix_filter(self, client):
        from compact_rag.api.routers import system as system_router

        class FakePrefixBackend:
            async def list(self, prefix: str = "") -> list[str]:
                if prefix:
                    return [p for p in ["docs/a.txt", "docs/b.txt", "temp/c.txt"] if p.startswith(prefix)]
                return ["docs/a.txt", "docs/b.txt", "temp/c.txt"]

            async def exists(self, remote_key: str) -> bool:
                return True  # all exist for simplicity

        client.app.dependency_overrides[system_router.get_storage_backend] = lambda: FakePrefixBackend()
        try:
            response = client.get("/v1/files?prefix=docs")
        finally:
            client.app.dependency_overrides.pop(system_router.get_storage_backend, None)

        assert response.status_code == 200
        data = response.json()["data"]
        keys = [f["storage_key"] for f in data]
        assert all(k.startswith("docs") for k in keys)
        assert len(keys) == 2


class TestHealthErrors:
    def test_health_with_error_status_for_components(self, client, monkeypatch):
        monkeypatch.setattr(
            "compact_rag.api.routers.system._check_database",
            AsyncMock(return_value="error"),
        )
        monkeypatch.setattr(
            "compact_rag.api.routers.system._check_chromadb",
            AsyncMock(return_value="error"),
        )
        monkeypatch.setattr(
            "compact_rag.api.routers.system._check_storage",
            AsyncMock(return_value="error"),
        )

        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["api"] == "ok"
        assert data["database"] == "error"
        assert data["chromadb"] == "error"
        assert data["storage"] == "error"
