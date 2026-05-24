"""System endpoints: health check, info, file proxy."""

from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, Query
from fastapi.responses import RedirectResponse

from compact_rag import __version__
from compact_rag.api.deps import get_settings, get_storage_backend
from compact_rag.api.schemas import HealthResponse, InfoResponse
from compact_rag.common.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["System"])

_HEALTH_TIMEOUT = 5  # seconds per component check


async def _check_database() -> str:
    try:
        from compact_rag.storage.db.engine import create_engine

        settings = get_settings()
        engine = create_engine(settings.database)
        async with asyncio.timeout(_HEALTH_TIMEOUT):
            async with engine.connect() as conn:
                await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return "ok"
    except Exception as e:
        logger.warning("Database health check failed", error=str(e))
        return "error"


async def _check_chromadb() -> str:
    try:
        import chromadb

        settings = get_settings()
        client = chromadb.PersistentClient(
            path=settings.chromadb.persist_directory,
        )
        async with asyncio.timeout(_HEALTH_TIMEOUT):
            client.list_collections()
        return "ok"
    except Exception as e:
        logger.warning("ChromaDB health check failed", error=str(e))
        return "error"


async def _check_storage() -> str:
    try:
        backend = get_storage_backend()
        async with asyncio.timeout(_HEALTH_TIMEOUT):
            await backend.list(prefix="")
        return "ok"
    except Exception as e:
        logger.warning("Storage health check failed", error=str(e))
        return "error"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check for all system components."""
    db_status, chroma_status, storage_status = await asyncio.gather(
        _check_database(), _check_chromadb(), _check_storage()
    )
    return HealthResponse(
        api="ok",
        database=db_status,
        chromadb=chroma_status,
        storage=storage_status,
    )


@router.get("/info", response_model=InfoResponse)
async def info(settings=Depends(get_settings)) -> InfoResponse:
    """Get system information."""
    return InfoResponse(
        version=__version__,
        database_url=_mask_url(settings.database.url),
        embedding_model=settings.embedding.model_name,
        embedding_dimension=384,
        llm_provider=settings.llm.provider,
        llm_model=settings.llm.model,
        storage_backend=settings.storage.backend,
        log_level=settings.log_level,
    )


@router.get("/files/{storage_key:path}")
async def serve_file(
    storage_key: str,
    download: bool = Query(False),
    expires: int = Query(3600, ge=60, le=86400),
    backend=Depends(get_storage_backend),
):
    """Proxy file download/access through storage backend."""
    from fastapi.responses import StreamingResponse

    from compact_rag.common.exceptions import FileNotFoundError

    if not await backend.exists(storage_key):
        raise FileNotFoundError(f"File not found: {storage_key}")

    if not download:
        url = await backend.get_url(storage_key, expires=expires)
        return RedirectResponse(url=url)

    data = await backend.download_bytes(storage_key)
    filename = storage_key.split("/")[-1]
    ascii_filename = filename.encode("ascii", errors="ignore").decode("ascii")
    if not ascii_filename or ascii_filename.startswith("."):
        ascii_filename = f"download{Path(filename).suffix}"
    content_disposition = (
        f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{quote(filename)}"
    )
    return StreamingResponse(
        iter([data]),
        media_type=mimetypes.guess_type(filename)[0] or "application/octet-stream",
        headers={
            "Content-Disposition": content_disposition,
        },
    )


@router.get("/files")
async def list_files(
    list: bool = Query(default=True),
    prefix: str = Query(default=""),
    backend=Depends(get_storage_backend),
):
    """List all files in the storage backend."""
    keys = await backend.list(prefix=prefix)
    files = []
    for key in keys:
        try:
            exists = await backend.exists(key)
            if exists:
                filename = key.split("/")[-1]
                file_size = 0
                if hasattr(backend, "_resolve_path"):
                    path = Path(backend._resolve_path(key))
                    if path.exists():
                        file_size = path.stat().st_size
                files.append(
                    {
                        "storage_key": key,
                        "filename": filename,
                        "storage_backend": "local",
                        "storage_type": "temp"
                        if key.startswith("temp/")
                        else "persistent",
                        "content_type": mimetypes.guess_type(filename)[0]
                        or "application/octet-stream",
                        "file_size": file_size,
                    }
                )
        except Exception:
            continue
    return {"data": files, "pagination": {"total": len(files)}}


@router.delete("/files/{storage_key:path}")
async def delete_file(
    storage_key: str,
    backend=Depends(get_storage_backend),
):
    """Delete a file from the storage backend."""
    from compact_rag.common.exceptions import FileNotFoundError

    if not await backend.exists(storage_key):
        raise FileNotFoundError(f"File not found: {storage_key}")
    await backend.delete(storage_key)
    return {"status": "deleted", "storage_key": storage_key}


@router.post("/files/clean-temp")
async def clean_temp_files(backend=Depends(get_storage_backend)):
    """Clean expired temporary files."""
    from compact_rag.storage.file_storage import TempFileCleaner

    cleaner = TempFileCleaner(backend, ttl_hours=1)
    cleaned = await cleaner.clean_expired()
    return {"cleaned": cleaned}


def _mask_url(url: str) -> str:
    """Mask sensitive parts of database URLs."""
    if "@" in url:
        parts = url.split("@")
        if len(parts) == 2:
            return f"{parts[0].split('://')[0]}://***@{parts[1]}"
    return url
