from __future__ import annotations

import os

import pytest

from compact_rag.common.exceptions import FileNotFoundError, StorageBackendError
from compact_rag.storage.file_storage import (
    LocalFileBackend,
    TempFileCleaner,
    build_storage_key,
    get_storage_backend,
)


class TestLocalFileBackend:
    @pytest.fixture
    def backend(self, test_settings):
        return LocalFileBackend(
            root_dir=test_settings.storage.local.root_dir,
            base_url=test_settings.storage.local.base_url,
        )

    @pytest.mark.asyncio
    async def test_upload_download_file(self, backend, tmp_path):
        src = tmp_path / "upload.txt"
        src.write_text("hello world")

        key = "test/upload.txt"
        url = await backend.upload_file(str(src), key)

        dest = tmp_path / "downloaded.txt"
        result = await backend.download_file(key, str(dest))

        assert result == str(dest)
        assert dest.read_text() == "hello world"
        assert url.startswith("http://localhost:8000/files/test/upload.txt")

    @pytest.mark.asyncio
    async def test_upload_download_bytes(self, backend):
        data = b"binary content here"
        key = "test/binary.bin"

        await backend.upload_bytes(data, key)
        result = await backend.download_bytes(key)

        assert result == data

    @pytest.mark.asyncio
    async def test_delete_file(self, backend, tmp_path):
        src = tmp_path / "del.txt"
        src.write_text("delete me")
        key = "test/delete.txt"

        await backend.upload_file(str(src), key)
        assert await backend.exists(key) is True

        deleted = await backend.delete(key)
        assert deleted is True
        assert await backend.exists(key) is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, backend):
        result = await backend.delete("nonexistent/file.txt")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_files(self, backend, tmp_path):
        src = tmp_path / "list_test.txt"
        src.write_text("content")

        await backend.upload_file(str(src), "a/b/file1.txt")
        await backend.upload_file(str(src), "a/b/file2.txt")
        await backend.upload_file(str(src), "a/c/file3.txt")

        all_files = await backend.list(prefix="")
        assert len(all_files) >= 3

        ab_files = await backend.list(prefix="a/b")
        assert len(ab_files) == 2
        assert all(f.startswith("a/b/") for f in ab_files)

    @pytest.mark.asyncio
    async def test_list_nonexistent_prefix(self, backend):
        files = await backend.list(prefix="nonexistent")
        assert files == []

    @pytest.mark.asyncio
    async def test_exists(self, backend, tmp_path):
        src = tmp_path / "exists_test.txt"
        src.write_text("data")
        key = "test/exists.txt"

        assert await backend.exists(key) is False
        await backend.upload_file(str(src), key)
        assert await backend.exists(key) is True

    @pytest.mark.asyncio
    async def test_get_url(self, backend, tmp_path):
        src = tmp_path / "url_test.txt"
        src.write_text("content")
        key = "test/url_test.txt"

        await backend.upload_file(str(src), key)
        url = await backend.get_url(key)
        assert url == "http://localhost:8000/files/test/url_test.txt"

    @pytest.mark.asyncio
    async def test_file_not_found_error(self, backend):
        with pytest.raises(FileNotFoundError, match="File not found"):
            await backend.download_file("missing/file.txt", "/tmp/out.txt")

        with pytest.raises(FileNotFoundError, match="File not found"):
            await backend.download_bytes("missing/file.txt")


class TestBuildStorageKey:
    def test_format(self):
        key = build_storage_key("col-123", "report.pdf")
        parts = key.split("/")
        assert parts[0] == "docs"
        assert parts[1] == "col-123"
        assert len(parts[2]) == 4  # year
        assert len(parts[3]) == 2  # month
        assert len(parts[4]) == 2  # day
        assert parts[5].endswith(".pdf")

    def test_custom_category(self):
        key = build_storage_key("col-abc", "image.png", category="temp")
        assert key.startswith("temp/")


class TestTempFileCleaner:
    @pytest.fixture
    def backend(self, test_settings):
        return LocalFileBackend(
            root_dir=test_settings.storage.local.root_dir,
            base_url=test_settings.storage.local.base_url,
        )

    @pytest.mark.asyncio
    async def test_clean_no_expired(self, backend):
        cleaner = TempFileCleaner(backend, ttl_hours=24)
        result = await cleaner.clean_expired()
        assert result == 0
