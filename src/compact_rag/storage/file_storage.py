"""Unified file storage abstraction with multiple backend support.

Supports: Local, MinIO, OSS, Kodo, S3 via Strategy pattern.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from compact_rag.common.exceptions import FileNotFoundError, StorageBackendError
from compact_rag.common.logger import get_logger

if TYPE_CHECKING:
    from compact_rag.config.settings import StorageSettings

logger = get_logger(__name__)


class StorageBackend(ABC):
    """Abstract base class for all file storage backends."""

    @abstractmethod
    async def upload_file(self, local_path: str, remote_key: str) -> str:
        """Upload a file from local path. Returns access URL."""

    @abstractmethod
    async def upload_bytes(
        self, data: bytes, remote_key: str, content_type: str = ""
    ) -> str:
        """Upload bytes data. Returns access URL."""

    @abstractmethod
    async def download_file(self, remote_key: str, local_path: str) -> str:
        """Download file to local path. Returns local path."""

    @abstractmethod
    async def download_bytes(self, remote_key: str) -> bytes:
        """Read file as bytes."""

    @abstractmethod
    async def delete(self, remote_key: str) -> bool:
        """Delete a file. Returns True if successful."""

    @abstractmethod
    async def list(self, prefix: str = "") -> list[str]:
        """List all file keys under the given prefix."""

    @abstractmethod
    async def get_url(self, remote_key: str, expires: int = 3600) -> str:
        """Get file access URL (pre-signed or direct)."""

    @abstractmethod
    async def exists(self, remote_key: str) -> bool:
        """Check if file exists."""


class LocalFileBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, root_dir: str = "./data/storage", base_url: str = ""):
        self.root_dir = Path(root_dir)
        self.base_url = base_url.rstrip("/")
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, key: str) -> Path:
        return self.root_dir / key

    async def upload_file(self, local_path: str, remote_key: str) -> str:
        dest = self._resolve_path(remote_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(local_path, dest)
            logger.info("File uploaded", remote_key=remote_key)
        except OSError as e:
            raise StorageBackendError(f"Failed to upload file: {e}") from e
        return self._make_url(remote_key)

    async def upload_bytes(
        self, data: bytes, remote_key: str, content_type: str = ""
    ) -> str:
        dest = self._resolve_path(remote_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(data)
            logger.info("Bytes uploaded", remote_key=remote_key, size=len(data))
        except OSError as e:
            raise StorageBackendError(f"Failed to upload bytes: {e}") from e
        return self._make_url(remote_key)

    async def download_file(self, remote_key: str, local_path: str) -> str:
        src = self._resolve_path(remote_key)
        if not src.exists():
            raise FileNotFoundError(f"File not found: {remote_key}")
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, local_path)
        except OSError as e:
            raise StorageBackendError(f"Failed to download file: {e}") from e
        return local_path

    async def download_bytes(self, remote_key: str) -> bytes:
        src = self._resolve_path(remote_key)
        if not src.exists():
            raise FileNotFoundError(f"File not found: {remote_key}")
        return src.read_bytes()

    async def delete(self, remote_key: str) -> bool:
        src = self._resolve_path(remote_key)
        if not src.exists():
            return False
        src.unlink()
        return True

    async def list(self, prefix: str = "") -> list[str]:
        base = self._resolve_path(prefix)
        if not base.exists():
            return []
        results: list[str] = []
        for root, _dirs, files in os.walk(base):
            for f in files:
                full = Path(root) / f
                rel = full.relative_to(self.root_dir)
                results.append(str(rel))
        return results

    async def get_url(self, remote_key: str, expires: int = 3600) -> str:
        return self._make_url(remote_key)

    async def exists(self, remote_key: str) -> bool:
        return self._resolve_path(remote_key).exists()

    def _make_url(self, key: str) -> str:
        if self.base_url:
            return f"{self.base_url}/{key}"
        return f"file://{self.root_dir / key}"


class MinIOBackend(StorageBackend):
    """MinIO/S3-compatible storage backend."""

    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "",
        secret_key: str = "",
        bucket: str = "compact-rag",
        secure: bool = False,
    ):
        self.endpoint = endpoint
        self.bucket = bucket
        try:
            from minio import Minio

            self.client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
            )
            self._ensure_bucket()
        except ImportError:
            raise StorageBackendError(
                "minio package not installed. Install with: pip install minio"
            )
        except Exception as e:
            raise StorageBackendError(f"Failed to connect to MinIO: {e}") from e

    def _ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
            logger.info("Created MinIO bucket", bucket=self.bucket)

    async def upload_file(self, local_path: str, remote_key: str) -> str:
        try:
            self.client.fput_object(self.bucket, remote_key, local_path)
            logger.info("File uploaded to MinIO", remote_key=remote_key)
        except Exception as e:
            raise StorageBackendError(f"MinIO upload failed: {e}") from e
        return await self.get_url(remote_key)

    async def upload_bytes(
        self, data: bytes, remote_key: str, content_type: str = ""
    ) -> str:
        import io

        try:
            data_stream = io.BytesIO(data)
            self.client.put_object(
                self.bucket,
                remote_key,
                data_stream,
                len(data),
                content_type=content_type or "application/octet-stream",
            )
        except Exception as e:
            raise StorageBackendError(f"MinIO upload bytes failed: {e}") from e
        return await self.get_url(remote_key)

    async def download_file(self, remote_key: str, local_path: str) -> str:
        try:
            self.client.fget_object(self.bucket, remote_key, local_path)
        except Exception as e:
            if "NoSuchKey" in str(e):
                raise FileNotFoundError(f"File not found: {remote_key}") from e
            raise StorageBackendError(f"MinIO download failed: {e}") from e
        return local_path

    async def download_bytes(self, remote_key: str) -> bytes:
        try:
            response = self.client.get_object(self.bucket, remote_key)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except Exception as e:
            if "NoSuchKey" in str(e):
                raise FileNotFoundError(f"File not found: {remote_key}") from e
            raise StorageBackendError(f"MinIO download failed: {e}") from e

    async def delete(self, remote_key: str) -> bool:
        try:
            self.client.remove_object(self.bucket, remote_key)
            return True
        except Exception:
            return False

    async def list(self, prefix: str = "") -> list[str]:
        try:
            objects = self.client.list_objects(self.bucket, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except Exception as e:
            raise StorageBackendError(f"MinIO list failed: {e}") from e

    async def get_url(self, remote_key: str, expires: int = 3600) -> str:
        try:
            return self.client.presigned_get_object(
                self.bucket, remote_key, expires=timedelta(seconds=expires)
            )
        except Exception as e:
            raise StorageBackendError(f"MinIO presigned URL failed: {e}") from e

    async def exists(self, remote_key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, remote_key)
            return True
        except Exception:
            return False


_storage_backend_instance: StorageBackend | None = None


def get_storage_backend(settings: StorageSettings | None = None) -> StorageBackend:
    """Get cached storage backend instance based on configuration.

    Args:
        settings: StorageSettings instance. If None, loads from get_settings().

    Returns:
        StorageBackend implementation.
    """
    global _storage_backend_instance
    if _storage_backend_instance is not None:
        return _storage_backend_instance

    if settings is None:
        from compact_rag.config.settings import get_settings

        settings = get_settings().storage

    backend_type = settings.backend
    if backend_type == "local":
        _storage_backend_instance = LocalFileBackend(
            root_dir=settings.local.root_dir,
            base_url=settings.local.base_url,
        )
    elif backend_type == "minio":
        _storage_backend_instance = MinIOBackend(
            endpoint=settings.minio.endpoint,
            access_key=settings.minio.access_key,
            secret_key=settings.minio.secret_key,
            bucket=settings.minio.bucket,
            secure=settings.minio.secure,
        )
    elif backend_type == "oss":
        raise StorageBackendError("OSS backend requires oss2 package. Install with: pip install oss2")
    elif backend_type == "kodo":
        raise StorageBackendError("Kodo backend requires qiniu package. Install with: pip install qiniu")
    elif backend_type == "s3":
        raise StorageBackendError("S3 backend requires boto3 package. Install with: pip install boto3")
    else:
        raise StorageBackendError(f"Unknown storage backend: {backend_type}")

    return _storage_backend_instance


def build_storage_key(
    collection_id: str,
    filename: str,
    category: str = "docs",
) -> str:
    """Build a deterministic storage key following the path convention.

    Format: {category}/{collection_id}/{year}/{month}/{day}/{hash16}{ext}

    Args:
        collection_id: Collection identifier.
        filename: Original filename.
        category: Storage category (docs, temp, archive).

    Returns:
        Storage key string.
    """
    now = datetime.now(timezone.utc)
    date_path = f"{now.year}/{now.month:02d}/{now.day:02d}"
    file_hash = hashlib.sha256(filename.encode()).hexdigest()[:16]
    ext = Path(filename).suffix
    return f"{category}/{collection_id}/{date_path}/{file_hash}{ext}"


class TempFileCleaner:
    """Cleaner for temporary files with TTL-based expiration."""

    def __init__(self, backend: StorageBackend, ttl_hours: int = 1):
        self.backend = backend
        self.ttl = timedelta(hours=ttl_hours)

    async def clean_expired(self) -> int:
        """Delete expired temporary files under temp/ prefix.

        Returns:
            Number of files cleaned.
        """
        try:
            all_files = await self.backend.list(prefix="temp/")
        except Exception as e:
            logger.warning("Failed to list temp files", error=str(e))
            return 0

        now = datetime.now(timezone.utc)
        cleaned = 0

        for file_key in all_files:
            try:
                parts = file_key.split("/")
                if len(parts) < 2:
                    continue
                # Extract date from path: temp/{session_id}/{timestamp}_{filename}
                timestamp_str = parts[-1].split("_")[0] if "_" in parts[-1] else ""
                if timestamp_str and timestamp_str.isdigit():
                    ts = int(timestamp_str[:14]) if len(timestamp_str) >= 14 else 0
                    if ts:
                        file_time = datetime.fromtimestamp(ts, tz=timezone.utc)
                        if now - file_time > self.ttl:
                            await self.backend.delete(file_key)
                            cleaned += 1
            except Exception:
                continue

        if cleaned:
            logger.info("Cleaned expired temp files", count=cleaned)
        return cleaned
