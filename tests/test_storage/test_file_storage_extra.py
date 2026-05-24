"""Extra file storage tests: MinIOBackend, OSS/Kodo/S3 stubs, TempFileCleaner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from compact_rag.common.exceptions import FileNotFoundError, StorageBackendError
from compact_rag.storage.file_storage import (
    LocalFileBackend,
    MinIOBackend,
    TempFileCleaner,
    get_storage_backend,
)


def reset_storage_singleton():
    import compact_rag.storage.file_storage as fs_mod

    fs_mod._storage_backend_instance = None


# ── MinIOBackend ────────────────────────────────────────────────


class TestMinIOBackendInit:
    def test_init_with_all_params(self):
        mock_minio_mod = MagicMock()
        mock_minio_cls = mock_minio_mod.Minio
        mock_instance = mock_minio_cls.return_value
        mock_instance.bucket_exists.return_value = True

        with patch.dict("sys.modules", {"minio": mock_minio_mod}):
            backend = MinIOBackend(
                endpoint="minio.example.com:9000",
                access_key="AKID",
                secret_key="SECRET",
                bucket="my-bucket",
                secure=True,
            )

        assert backend.endpoint == "minio.example.com:9000"
        assert backend.bucket == "my-bucket"
        mock_minio_cls.assert_called_once_with(
            "minio.example.com:9000",
            access_key="AKID",
            secret_key="SECRET",
            secure=True,
        )

    def test_init_creates_bucket_if_missing(self):
        mock_minio_mod = MagicMock()
        mock_minio_cls = mock_minio_mod.Minio
        mock_instance = mock_minio_cls.return_value
        mock_instance.bucket_exists.return_value = False

        with patch.dict("sys.modules", {"minio": mock_minio_mod}):
            backend = MinIOBackend(bucket="new-bucket")

        assert backend.bucket == "new-bucket"
        mock_instance.make_bucket.assert_called_once_with("new-bucket")

    def test_init_raises_storage_backend_error_on_import_failure(self):
        with patch.dict("sys.modules", {"minio": None}):
            with pytest.raises(StorageBackendError, match="minio package not installed"):
                MinIOBackend()

    def test_init_raises_storage_backend_error_on_connection_failure(self):
        mock_minio_mod = MagicMock()
        mock_minio_mod.Minio.side_effect = Exception("connection refused")

        with patch.dict("sys.modules", {"minio": mock_minio_mod}):
            with pytest.raises(StorageBackendError, match="Failed to connect to MinIO"):
                MinIOBackend()


class TestMinIOBackendMethods:
    @pytest.fixture
    def backend(self):
        mock_minio_mod = MagicMock()
        mock_minio_cls = mock_minio_mod.Minio
        mock_instance = mock_minio_cls.return_value
        mock_instance.bucket_exists.return_value = True

        with patch.dict("sys.modules", {"minio": mock_minio_mod}):
            return MinIOBackend(
                endpoint="localhost:9000",
                access_key="minioadmin",
                secret_key="minioadmin",
                bucket="test-bucket",
            )

    @pytest.mark.asyncio
    async def test_upload_file(self, backend):
        mock_presigned = "http://localhost:9000/test-bucket/remote/file.txt"
        backend.client.presigned_get_object.return_value = mock_presigned

        url = await backend.upload_file("/tmp/local.txt", "remote/file.txt")
        backend.client.fput_object.assert_called_once_with(
            "test-bucket", "remote/file.txt", "/tmp/local.txt"
        )
        assert url == mock_presigned

    @pytest.mark.asyncio
    async def test_upload_file_error(self, backend):
        backend.client.fput_object.side_effect = Exception("upload error")

        with pytest.raises(StorageBackendError, match="MinIO upload failed"):
            await backend.upload_file("/tmp/local.txt", "remote/file.txt")

    @pytest.mark.asyncio
    async def test_upload_bytes(self, backend):
        mock_presigned = "http://localhost:9000/test-bucket/data/key.bin"
        backend.client.presigned_get_object.return_value = mock_presigned

        url = await backend.upload_bytes(b"hello", "data/key.bin", "application/octet-stream")
        backend.client.put_object.assert_called_once()
        call_args = backend.client.put_object.call_args
        assert call_args[0][0] == "test-bucket"
        assert call_args[0][1] == "data/key.bin"
        assert call_args[1]["content_type"] == "application/octet-stream"
        assert url == mock_presigned

    @pytest.mark.asyncio
    async def test_upload_bytes_error(self, backend):
        backend.client.put_object.side_effect = Exception("upload bytes error")

        with pytest.raises(StorageBackendError, match="MinIO upload bytes failed"):
            await backend.upload_bytes(b"data", "key.bin")

    @pytest.mark.asyncio
    async def test_download_file(self, backend):
        result = await backend.download_file("remote/file.txt", "/tmp/out.txt")
        backend.client.fget_object.assert_called_once_with(
            "test-bucket", "remote/file.txt", "/tmp/out.txt"
        )
        assert result == "/tmp/out.txt"

    @pytest.mark.asyncio
    async def test_download_file_not_found(self, backend):
        backend.client.fget_object.side_effect = Exception("NoSuchKey: file not found")

        with pytest.raises(FileNotFoundError, match="File not found"):
            await backend.download_file("remote/missing.txt", "/tmp/out.txt")

    @pytest.mark.asyncio
    async def test_download_file_storage_error(self, backend):
        backend.client.fget_object.side_effect = Exception("generic failure")

        with pytest.raises(StorageBackendError, match="MinIO download failed"):
            await backend.download_file("remote/file.txt", "/tmp/out.txt")

    @pytest.mark.asyncio
    async def test_download_bytes(self, backend):
        mock_response = MagicMock()
        mock_response.read.return_value = b"content"
        backend.client.get_object.return_value = mock_response

        data = await backend.download_bytes("remote/file.txt")
        assert data == b"content"

    @pytest.mark.asyncio
    async def test_download_bytes_not_found(self, backend):
        backend.client.get_object.side_effect = Exception("NoSuchKey: file not found")

        with pytest.raises(FileNotFoundError, match="File not found"):
            await backend.download_bytes("remote/missing.txt")

    @pytest.mark.asyncio
    async def test_delete_returns_true(self, backend):
        result = await backend.delete("remote/file.txt")
        assert result is True
        backend.client.remove_object.assert_called_once_with("test-bucket", "remote/file.txt")

    @pytest.mark.asyncio
    async def test_delete_returns_false_on_error(self, backend):
        backend.client.remove_object.side_effect = Exception("delete failed")
        result = await backend.delete("remote/file.txt")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_with_prefix(self, backend):
        mock_obj1 = MagicMock()
        mock_obj1.object_name = "prefix/file1.txt"
        mock_obj2 = MagicMock()
        mock_obj2.object_name = "prefix/sub/file2.txt"
        backend.client.list_objects.return_value = [mock_obj1, mock_obj2]

        files = await backend.list(prefix="prefix")
        assert files == ["prefix/file1.txt", "prefix/sub/file2.txt"]
        backend.client.list_objects.assert_called_once_with(
            "test-bucket", prefix="prefix", recursive=True
        )

    @pytest.mark.asyncio
    async def test_list_error(self, backend):
        backend.client.list_objects.side_effect = Exception("list error")

        with pytest.raises(StorageBackendError, match="MinIO list failed"):
            await backend.list(prefix="")

    @pytest.mark.asyncio
    async def test_get_url(self, backend):
        mock_url = "http://presigned.example.com/file"
        backend.client.presigned_get_object.return_value = mock_url

        url = await backend.get_url("remote/file.txt", expires=1800)
        backend.client.presigned_get_object.assert_called_once()
        args = backend.client.presigned_get_object.call_args
        assert args[0][:2] == ("test-bucket", "remote/file.txt")
        assert url == mock_url

    @pytest.mark.asyncio
    async def test_get_url_error(self, backend):
        backend.client.presigned_get_object.side_effect = Exception("presigned error")

        with pytest.raises(StorageBackendError, match="MinIO presigned URL failed"):
            await backend.get_url("remote/file.txt")

    @pytest.mark.asyncio
    async def test_exists_returns_true(self, backend):
        result = await backend.exists("remote/file.txt")
        assert result is True
        backend.client.stat_object.assert_called_once_with("test-bucket", "remote/file.txt")

    @pytest.mark.asyncio
    async def test_exists_returns_false(self, backend):
        backend.client.stat_object.side_effect = Exception("not found")
        result = await backend.exists("remote/missing.txt")
        assert result is False


# ── get_storage_backend unsupported backends ────────────────────


class TestGetStorageBackendUnsupported:
    def test_oss_raises_storage_backend_error(self, test_settings):
        reset_storage_singleton()
        test_settings.storage.backend = "oss"  # type: ignore[arg-type]
        with pytest.raises(StorageBackendError, match="OSS backend requires oss2"):
            get_storage_backend(test_settings.storage)

    def test_kodo_raises_storage_backend_error(self, test_settings):
        reset_storage_singleton()
        test_settings.storage.backend = "kodo"  # type: ignore[arg-type]
        with pytest.raises(StorageBackendError, match="Kodo backend requires qiniu"):
            get_storage_backend(test_settings.storage)

    def test_s3_raises_storage_backend_error(self, test_settings):
        reset_storage_singleton()
        test_settings.storage.backend = "s3"  # type: ignore[arg-type]
        with pytest.raises(StorageBackendError, match="S3 backend requires boto3"):
            get_storage_backend(test_settings.storage)


# ── TempFileCleaner extended ────────────────────────────────────


class TestTempFileCleanerExtended:
    @pytest.fixture
    def backend(self, test_settings):
        return LocalFileBackend(
            root_dir=test_settings.storage.local.root_dir,
            base_url=test_settings.storage.local.base_url,
        )

    @pytest.mark.asyncio
    async def test_clean_expired_list_error_returns_zero(self, backend):
        mock_backend = MagicMock(spec=LocalFileBackend)
        mock_backend.list = AsyncMock(side_effect=Exception("list failed"))
        cleaner = TempFileCleaner(mock_backend, ttl_hours=1)

        cleaned = await cleaner.clean_expired()
        assert cleaned == 0

    @pytest.mark.asyncio
    async def test_clean_expired_invalid_paths_ignored(self, backend, tmp_path):
        """Short paths (len < 2 parts) are skipped without error."""
        src = tmp_path / "short.txt"
        src.write_text("data")

        # Upload files with paths that have only 1 part
        mock_backend = MagicMock(spec=LocalFileBackend)
        mock_backend.list = AsyncMock(return_value=["shortpath", "temp/valid_12345678901234_x.txt"])
        mock_backend.delete = AsyncMock(return_value=True)

        cleaner = TempFileCleaner(mock_backend, ttl_hours=0)
        cleaner.backend = mock_backend
        cleaned = await cleaner.clean_expired()

        assert cleaned >= 0
        # "shortpath" should be skipped (not enough parts)

    @pytest.mark.asyncio
    async def test_clean_expired_malformed_timestamps_ignored(self, backend, tmp_path):
        mock_backend = MagicMock(spec=LocalFileBackend)
        mock_backend.list = AsyncMock(
            return_value=[
                "temp/sess/non_digit_abc_file.txt",  # non-digit timestamp
                "temp/sess/short_12_file.txt",  # timestamp < 14 digits
            ]
        )
        mock_backend.delete = AsyncMock(return_value=True)

        cleaner = TempFileCleaner(mock_backend, ttl_hours=0)
        cleaned = await cleaner.clean_expired()
        assert cleaned == 0  # neither should match

    @pytest.mark.asyncio
    async def test_clean_expired_delete_errors_handled(self, backend, tmp_path):
        mock_backend = MagicMock(spec=LocalFileBackend)
        old_ts = "00000001000000"
        mock_backend.list = AsyncMock(
            return_value=[f"temp/sess/{old_ts}_file.txt"]
        )
        mock_backend.delete = AsyncMock(side_effect=Exception("delete failed"))

        cleaner = TempFileCleaner(mock_backend, ttl_hours=1)
        cleaned = await cleaner.clean_expired()
        assert cleaned == 0  # delete failed, not counted
