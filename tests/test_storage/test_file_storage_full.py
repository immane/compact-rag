"""Comprehensive file storage tests covering backends, edge cases, and utilities.

Extends tests/test_storage/test_file_storage.py with exhaustive coverage of
get_storage_backend(), LocalFileBackend edge cases, TempFileCleaner, and
build_storage_key().
"""

from __future__ import annotations

import time
from unittest import mock

import pytest

from compact_rag.common.exceptions import FileNotFoundError, StorageBackendError
from compact_rag.storage.file_storage import (
    LocalFileBackend,
    MinIOBackend,
    TempFileCleaner,
    build_storage_key,
    get_storage_backend,
)


def reset_storage_singleton():
    """Reset the global _storage_backend_instance so tests are isolated."""
    import compact_rag.storage.file_storage as fs_mod

    fs_mod._storage_backend_instance = None


# ────────────────────────────────────────────────────────────────
# get_storage_backend() factory function
# ────────────────────────────────────────────────────────────────


class TestGetStorageBackend:
    def test_returns_local_backend(self, test_settings):
        reset_storage_singleton()
        test_settings.storage.backend = "local"
        backend = get_storage_backend(test_settings.storage)
        assert isinstance(backend, LocalFileBackend)

    def test_returns_minio_backend_with_mock(self, test_settings):
        reset_storage_singleton()
        test_settings.storage.backend = "minio"

        with mock.patch.dict("sys.modules", {"minio": mock.MagicMock()}):
            import sys

            sys.modules["minio"] = mock.MagicMock()
            backend = get_storage_backend(test_settings.storage)
            assert isinstance(backend, MinIOBackend)

    def test_raises_for_unknown_backend(self, test_settings):
        reset_storage_singleton()
        test_settings.storage.backend = "nfs"  # type: ignore[arg-type]
        with pytest.raises(StorageBackendError, match="Unknown storage backend"):
            get_storage_backend(test_settings.storage)

    def test_raises_for_empty_backend_string(self, test_settings):
        reset_storage_singleton()
        test_settings.storage.backend = ""  # type: ignore[arg-type]
        with pytest.raises(StorageBackendError, match="Unknown storage backend"):
            get_storage_backend(test_settings.storage)

    def test_caches_instance(self, test_settings):
        """Subsequent calls return the same cached instance."""
        reset_storage_singleton()
        test_settings.storage.backend = "local"
        b1 = get_storage_backend(test_settings.storage)
        b2 = get_storage_backend(test_settings.storage)
        assert b1 is b2


# ────────────────────────────────────────────────────────────────
# LocalFileBackend edge cases
# ────────────────────────────────────────────────────────────────


class TestLocalFileBackendEdgeCases:
    @pytest.fixture
    def backend(self, test_settings):
        return LocalFileBackend(
            root_dir=test_settings.storage.local.root_dir,
            base_url=test_settings.storage.local.base_url,
        )

    @pytest.mark.asyncio
    async def test_upload_bytes_empty(self, backend):
        """upload_bytes() with empty bytes should succeed."""
        key = "test/empty.bin"
        url = await backend.upload_bytes(b"", key)
        assert url is not None
        data = await backend.download_bytes(key)
        assert data == b""

    @pytest.mark.asyncio
    async def test_upload_file_nonexistent_source(self, backend):
        """upload_file() with non-existent source raises StorageBackendError."""
        with pytest.raises(StorageBackendError, match="Failed to upload file"):
            await backend.upload_file("/nonexistent/path/file.txt", "test/bad.txt")

    @pytest.mark.asyncio
    async def test_download_file_creates_parent_directory(self, backend, tmp_path):
        """download_file() auto-creates parent directories."""
        # Upload first
        src = tmp_path / "nested_src.txt"
        src.write_text("nested content")
        key = "a/b/c/nested.txt"
        await backend.upload_file(str(src), key)

        # Download to a path where parent dirs don't exist
        dest = tmp_path / "new" / "deep" / "path" / "nested.txt"
        result = await backend.download_file(key, str(dest))
        assert result == str(dest)
        assert dest.read_text() == "nested content"

    @pytest.mark.asyncio
    async def test_download_bytes_nonexistent(self, backend):
        with pytest.raises(FileNotFoundError, match="File not found"):
            await backend.download_bytes("not/a/real/key.dat")

    @pytest.mark.asyncio
    async def test_get_url_with_custom_expires(self, backend, tmp_path):
        """get_url() ignores expires for local backend but doesn't crash."""
        src = tmp_path / "url_exp.txt"
        src.write_text("expires test")
        key = "test/url_exp.txt"
        await backend.upload_file(str(src), key)

        url_custom = await backend.get_url(key, expires=7200)
        assert url_custom == "http://localhost:8000/files/test/url_exp.txt"

        url_default = await backend.get_url(key, expires=10)
        assert url_default == "http://localhost:8000/files/test/url_exp.txt"

    @pytest.mark.asyncio
    async def test_delete_already_deleted(self, backend, tmp_path):
        """delete() on a file that was already deleted returns False."""
        src = tmp_path / "twice.txt"
        src.write_text("delete twice")
        key = "test/delete_twice.txt"

        await backend.upload_file(str(src), key)
        assert await backend.delete(key) is True
        assert await backend.delete(key) is False  # Already gone

    @pytest.mark.asyncio
    async def test_exists_newly_created(self, backend, tmp_path):
        src = tmp_path / "exists_new.txt"
        src.write_text("fresh")
        key = "test/fresh.txt"

        assert await backend.exists(key) is False
        await backend.upload_file(str(src), key)
        assert await backend.exists(key) is True

    @pytest.mark.asyncio
    async def test_list_many_files(self, backend, tmp_path):
        """list() with many files returns them all."""
        src = tmp_path / "bulk.txt"
        src.write_text("bulk")
        for i in range(50):
            await backend.upload_file(str(src), f"bulk/file_{i:03d}.txt")

        all_files = await backend.list(prefix="bulk")
        assert len(all_files) == 50
        assert all(f.startswith("bulk/") for f in all_files)

    @pytest.mark.asyncio
    async def test_list_empty_directory(self, backend):
        """list() on empty prefix returns empty list."""
        files = await backend.list(prefix="nothing_here")
        assert files == []

    @pytest.mark.asyncio
    async def test_download_file_nonexistent(self, backend):
        with pytest.raises(FileNotFoundError, match="File not found"):
            await backend.download_file("ghost/file.txt", "/tmp/out.txt")

    @pytest.mark.asyncio
    async def test_multiple_uploads_same_key_overwrites(self, backend, tmp_path):
        """Uploading to the same key overwrites previous content."""
        src1 = tmp_path / "v1.txt"
        src1.write_text("version one")
        key = "test/overwrite.txt"

        await backend.upload_file(str(src1), key)
        assert await backend.download_bytes(key) == b"version one"

        src2 = tmp_path / "v2.txt"
        src2.write_text("version two")
        await backend.upload_file(str(src2), key)
        assert await backend.download_bytes(key) == b"version two"

    @pytest.mark.asyncio
    async def test_file_url_no_base_url(self, test_settings):
        """When base_url is empty, returns file:// URL."""
        backend = LocalFileBackend(
            root_dir=test_settings.storage.local.root_dir,
            base_url="",
        )
        url = backend._make_url("path/to/file.pdf")
        assert url.startswith("file://")
        assert "path/to/file.pdf" in url

    @pytest.mark.asyncio
    async def test_upload_bytes_with_content_type(self, backend):
        """content_type is accepted but ignored by local backend (no error)."""
        key = "test/with_type.json"
        await backend.upload_bytes(b'{"x":1}', key, content_type="application/json")
        data = await backend.download_bytes(key)
        assert data == b'{"x":1}'


# ────────────────────────────────────────────────────────────────
# TempFileCleaner
# ────────────────────────────────────────────────────────────────


class TestTempFileCleanerExtended:
    @pytest.fixture
    def backend(self, test_settings):
        return LocalFileBackend(
            root_dir=test_settings.storage.local.root_dir,
            base_url=test_settings.storage.local.base_url,
        )

    @pytest.mark.asyncio
    async def test_clean_expired_with_no_files(self, backend):
        """No temp files — returns 0."""
        cleaner = TempFileCleaner(backend, ttl_hours=1)
        cleaned = await cleaner.clean_expired()
        assert cleaned == 0

    @pytest.mark.asyncio
    async def test_clean_expired_with_expired_files(self, backend, tmp_path):
        """Files with old timestamps are deleted.
        
        TempFileCleaner requires at least 14-digit timestamps in filenames.
        We pad a small unix timestamp (far in the past) to 14 digits.
        """
        # 1000000 = ~Jan 1970, definitely expired
        old_ts = "00000001000000"
        src = tmp_path / "exp.txt"
        src.write_text("expired")

        await backend.upload_file(str(src), f"temp/sess-old/{old_ts}_file1.txt")
        await backend.upload_file(str(src), f"temp/sess-old/{old_ts}_file2.txt")

        cleaner = TempFileCleaner(backend, ttl_hours=1)
        cleaned = await cleaner.clean_expired()
        assert cleaned == 2

        # Files should be gone
        assert await backend.exists(f"temp/sess-old/{old_ts}_file1.txt") is False

    @pytest.mark.asyncio
    async def test_clean_expired_with_no_expired_files(self, backend, tmp_path):
        """Files with recent timestamps (14-digit zero-padded) are NOT deleted."""
        recent_ts = str(int(time.time())).zfill(14)
        src = tmp_path / "active.txt"
        src.write_text("active")

        await backend.upload_file(str(src), f"temp/sess-recent/{recent_ts}_file.txt")

        cleaner = TempFileCleaner(backend, ttl_hours=24)
        cleaned = await cleaner.clean_expired()
        assert cleaned == 0

        # File should still exist
        assert await backend.exists(f"temp/sess-recent/{recent_ts}_file.txt") is True

    @pytest.mark.asyncio
    async def test_clean_expired_mixed(self, backend, tmp_path):
        """Mixed expired and active files — only expired are deleted."""
        old_ts = "00000001000000"  # very small unix ts → definitely expired
        recent_ts = str(int(time.time())).zfill(14)  # now → not expired
        src = tmp_path / "mix.txt"
        src.write_text("data")

        await backend.upload_file(str(src), f"temp/mix/{old_ts}_old1.txt")
        await backend.upload_file(str(src), f"temp/mix/{old_ts}_old2.txt")
        await backend.upload_file(str(src), f"temp/mix/{recent_ts}_fresh.txt")

        cleaner = TempFileCleaner(backend, ttl_hours=1)
        cleaned = await cleaner.clean_expired()
        assert cleaned == 2

        # Old files deleted, fresh file remains
        assert await backend.exists(f"temp/mix/{old_ts}_old1.txt") is False
        assert await backend.exists(f"temp/mix/{old_ts}_old2.txt") is False
        assert await backend.exists(f"temp/mix/{recent_ts}_fresh.txt") is True

    @pytest.mark.asyncio
    async def test_clean_expired_ignores_non_matching_format(self, backend, tmp_path):
        """Files that don't match the timestamp format are ignored."""
        src = tmp_path / "badformat.txt"
        src.write_text("junk")

        # No underscore → no timestamp extracted
        await backend.upload_file(str(src), "temp/bad/nostamp.txt")
        # No leading timestamp digits
        await backend.upload_file(str(src), "temp/bad/not_number_file.txt")

        cleaner = TempFileCleaner(backend, ttl_hours=0)  # TTL of 0 = everything expired
        cleaned = await cleaner.clean_expired()
        # Should be 0 because neither file matches the timestamp format
        assert cleaned == 0

    @pytest.mark.asyncio
    async def test_clean_expired_high_ttl(self, backend, tmp_path):
        """Very high TTL means nothing expires — even a 1000s-old file."""
        old_ts = str(int(time.time()) - 1000).zfill(14)
        src = tmp_path / "ttl_test.txt"
        src.write_text("ttl")

        await backend.upload_file(str(src), f"temp/high/{old_ts}_file.txt")

        cleaner = TempFileCleaner(backend, ttl_hours=100)  # 100 hours
        cleaned = await cleaner.clean_expired()
        assert cleaned == 0

    @pytest.mark.asyncio
    async def test_clean_expired_zero_ttl(self, backend, tmp_path):
        """TTL of 0 deletes ALL matching files — even recent ones."""
        ts = str(int(time.time())).zfill(14)
        src = tmp_path / "zero_ttl.txt"
        src.write_text("all gone")

        await backend.upload_file(str(src), f"temp/zero/{ts}_a.txt")
        await backend.upload_file(str(src), f"temp/zero/{ts}_b.txt")
        await backend.upload_file(str(src), f"temp/zero/{ts}_c.txt")

        cleaner = TempFileCleaner(backend, ttl_hours=0)
        cleaned = await cleaner.clean_expired()
        assert cleaned == 3


# ────────────────────────────────────────────────────────────────
# build_storage_key()
# ────────────────────────────────────────────────────────────────


class TestBuildStorageKeyExtended:
    def test_category_parameter(self):
        key = build_storage_key("col-abc", "data.json", category="temp")
        assert key.startswith("temp/")

        key = build_storage_key("col-abc", "data.json", category="archive")
        assert key.startswith("archive/")

        key = build_storage_key("col-abc", "data.json", category="docs")
        assert key.startswith("docs/")

    def test_special_characters_in_filename(self):
        """Filenames with spaces, Chinese, and symbols are handled."""
        key = build_storage_key("col-1", "报告 (final).pdf")
        parts = key.split("/")
        # The hash part should be consistent
        assert parts[0] == "docs"
        assert parts[1] == "col-1"
        assert parts[5].endswith(".pdf")

    def test_no_extension_in_filename(self):
        """Filename without extension produces key without extension."""
        key = build_storage_key("col-x", "README")
        parts = key.split("/")
        assert not parts[5].startswith(".")  # No leading dot
        # The last part should be just the hash (no extension)
        filename_part = parts[5]
        assert "." not in filename_part

    def test_consistent_hash_for_same_input(self):
        """Same inputs produce identical keys."""
        key1 = build_storage_key("col-42", "document.pdf")
        key2 = build_storage_key("col-42", "document.pdf")
        assert key1[: key1.rfind("/")] == key2[: key2.rfind("/")]
        # The date path differs across calls, but hash part should match
        hash1 = key1.split("/")[-1]
        hash2 = key2.split("/")[-1]
        assert hash1 == hash2

    def test_different_filenames_produce_different_hashes(self):
        key1 = build_storage_key("col-1", "a.pdf")
        key2 = build_storage_key("col-1", "b.pdf")
        assert key1 != key2

    def test_key_structure_parts(self):
        """Verify the full structure: category/collection/year/month/day/hash.ext."""
        key = build_storage_key("my-col", "report.xlsx", category="export")
        parts = key.split("/")
        assert len(parts) == 6
        assert parts[0] == "export"  # category
        assert parts[1] == "my-col"  # collection
        assert len(parts[2]) == 4  # year
        assert len(parts[3]) == 2  # month
        assert len(parts[4]) == 2  # day
        assert parts[5].endswith(".xlsx")  # hash + extension
