from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from compact_rag.common.exceptions import (
    ConfigurationError,
    DocumentLoadError,
    IngestionError,
    UnsupportedFormatError,
)
from compact_rag.common.logger import get_logger
from compact_rag.ingestion.chunker import chunk_documents
from compact_rag.ingestion.loader import LoaderFactory
from compact_rag.ingestion.table_extractor import TableExtractor
from compact_rag.storage.schema import IngestionResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from compact_rag.config.settings import Settings

logger = get_logger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        if settings is None:
            from compact_rag.config.settings import get_settings

            settings = get_settings()
        self._settings = settings
        self._session = session

    async def ingest_file(
        self,
        file_path: str,
        collection_name: str = "default",
        force: bool = False,
    ) -> IngestionResult:
        start_time = time.perf_counter()
        job_id: str | None = None

        path = Path(file_path)
        if not path.exists():
            raise ConfigurationError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in self._settings.ingestion.supported_extensions:
            raise UnsupportedFormatError(
                f"Unsupported file format: '{ext}'. Supported: {self._settings.ingestion.supported_extensions}"
            )

        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        filename = path.name

        session = self._get_session()

        try:
            from compact_rag.storage.db.repository.collection import (
                CollectionRepository,
            )
            from compact_rag.storage.db.repository.document import DocumentRepository

            collection_repo = CollectionRepository()
            document_repo = DocumentRepository()

            collection = await collection_repo.get_by_name(session, collection_name)
            if collection is None:
                collection = await collection_repo.create(
                    session,
                    name=collection_name,
                    embedding_model=self._settings.embedding.model_name,
                    chunk_size=self._settings.ingestion.chunk_size,
                    chunk_overlap=self._settings.ingestion.chunk_overlap,
                )
                logger.info(
                    "Created collection", name=collection_name, id=collection.id
                )

            existing = await document_repo.get_by_hash(
                session, file_hash, collection.id
            )
            if existing and not force:
                duration_ms = (time.perf_counter() - start_time) * 1000
                return IngestionResult(
                    doc_id=existing.id,
                    filename=existing.filename,
                    status="skipped",
                    chunk_count=existing.chunk_count,
                    table_count=existing.table_count,
                    duration_ms=duration_ms,
                )

            from compact_rag.storage.db.repository.ingestion import (
                IngestionJobRepository,
            )

            job_repo = IngestionJobRepository()
            job = await job_repo.create_job(session, collection.id)
            job_id = job.id
            await session.commit()

            storage_backend = self._get_storage_backend()

            pages = await self._load_document(str(path))
            page_count = max((p.page_number for p in pages), default=0)

            file_type = ext.lstrip(".")
            file_size = path.stat().st_size

            table_extractor = TableExtractor()
            all_tables: list[str] = []

            if file_type == "pdf":
                extracted_tables = table_extractor.extract_from_pdf(str(path))
            elif file_type in ("html", "htm"):
                html_content = path.read_text(encoding="utf-8")
                extracted_tables = table_extractor.extract_from_html(html_content)
            elif file_type == "docx":
                extracted_tables = table_extractor.extract_from_docx(str(path))
            else:
                extracted_tables = []

            from compact_rag.ingestion.loader import LoadedPage

            for t in extracted_tables:
                pages.append(
                    LoadedPage(
                        page_number=t.page_number,
                        content=t.markdown,
                        tables=[],
                        metadata={"table_extraction": True, "method": t.method},
                    )
                )
                all_tables.append(t.markdown)

            chunks = chunk_documents(
                pages,
                chunk_size=self._settings.ingestion.chunk_size,
                chunk_overlap=self._settings.ingestion.chunk_overlap,
                strategy=self._settings.ingestion.chunking_strategy,
            )

            embedding_service = self._get_embedding_service()
            texts = [c.content for c in chunks]
            embeddings = embedding_service.encode(texts) if texts else None

            doc = await document_repo.create(
                session,
                collection_id=collection.id,
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                file_hash=file_hash,
                page_count=page_count,
                chunk_count=len(chunks),
                table_count=len(extracted_tables),
                status="processing",
            )
            await session.flush()

            if embeddings is not None and len(embeddings) > 0:
                vector_store = self._get_vector_store(embedding_service)

                for chunk in chunks:
                    chunk.metadata["doc_id"] = doc.id
                    chunk.metadata["filename"] = filename
                    chunk.metadata["collection_name"] = collection_name

                chroma_ids = vector_store.add_documents(chunks, embeddings)

                from compact_rag.storage.db.repository.chunk import ChunkRepository

                chunk_repo = ChunkRepository()
                for chunk, chroma_id in zip(chunks, chroma_ids):
                    await chunk_repo.create(
                        session,
                        document_id=doc.id,
                        chroma_id=chroma_id,
                        chunk_index=chunk.chunk_index,
                        page_number=chunk.page_number,
                        is_table=chunk.is_table,
                        token_count=chunk.token_count,
                        content_hash=chunk.content_hash,
                    )

            await document_repo.update(session, doc.id, status="completed")
            await collection_repo.increment_document_count(session, collection.id, 1)
            await job_repo.update_progress(
                session,
                job_id,
                processed=1,
                chunks=len(chunks),
            )
            await job_repo.complete_job(
                session,
                job_id,
                status="completed",
            )
            await session.commit()

            from compact_rag.storage.file_storage import build_storage_key

            storage_key = build_storage_key(collection.id, filename)
            await storage_backend.upload_file(str(path), storage_key)

            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "File ingested successfully",
                filename=filename,
                chunks=len(chunks),
                tables=len(extracted_tables),
                duration_ms=duration_ms,
            )

            return IngestionResult(
                doc_id=doc.id,
                filename=filename,
                status="completed",
                chunk_count=len(chunks),
                table_count=len(extracted_tables),
                duration_ms=duration_ms,
            )

        except Exception as e:
            await session.rollback()
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"{type(e).__name__}: {e}"

            logger.error(
                "Ingestion failed",
                filename=filename,
                error=error_msg,
            )

            try:
                if job_id:
                    from compact_rag.storage.db.repository.ingestion import (
                        IngestionJobRepository,
                    )

                    job_repo = IngestionJobRepository()
                    await job_repo.complete_job(
                        session,
                        job_id,
                        status="failed",
                        errors={"error": error_msg},
                    )
                    await session.commit()
            except Exception:
                pass

            return IngestionResult(
                doc_id="",
                filename=filename,
                status="failed",
                error_message=error_msg,
                duration_ms=duration_ms,
            )

    async def ingest_directory(
        self,
        dir_path: str,
        collection_name: str = "default",
    ) -> list[IngestionResult]:
        path = Path(dir_path)
        if not path.is_dir():
            raise ConfigurationError(f"Directory not found: {dir_path}")

        results: list[IngestionResult] = []
        supported = self._settings.ingestion.supported_extensions

        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in supported:
                try:
                    result = await self.ingest_file(str(file_path), collection_name)
                    results.append(result)
                except Exception as e:
                    results.append(
                        IngestionResult(
                            doc_id="",
                            filename=file_path.name,
                            status="failed",
                            error_message=str(e),
                        )
                    )

        logger.info(
            "Directory ingested",
            dir=dir_path,
            total=len(results),
            succeeded=sum(1 for r in results if r.status == "completed"),
        )
        return results

    async def ingest_url(
        self,
        url: str,
        collection_name: str = "default",
    ) -> IngestionResult:
        import tempfile

        try:
            import httpx
        except ImportError:
            raise IngestionError("httpx not installed. Install with: pip install httpx")

        url_path = Path(url)
        safe_name = url_path.name or url.split("/")[-1] or "downloaded_document"
        ext = Path(safe_name).suffix.lower()
        if ext and ext not in self._settings.ingestion.supported_extensions:
            raise UnsupportedFormatError(
                f"URL file type '{ext}' is not supported. Supported: {self._settings.ingestion.supported_extensions}"
            )

        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
            except Exception as e:
                raise IngestionError(f"Failed to download URL '{url}': {e}", cause=e)

        content_type = response.headers.get("content-type", "")
        ext_map = {
            "application/pdf": ".pdf",
            "text/html": ".html",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "text/plain": ".txt",
            "text/markdown": ".md",
        }
        detected_ext = ".html"
        for mime, mapped_ext in ext_map.items():
            if mime in content_type:
                detected_ext = mapped_ext
                break

        if not safe_name or "." not in safe_name:
            safe_name = f"download{detected_ext}"

        download_dir = Path(tempfile.gettempdir()) / "compact_rag_downloads"
        download_dir.mkdir(parents=True, exist_ok=True)
        dest = download_dir / safe_name
        dest.write_bytes(response.content)

        try:
            result = await self.ingest_file(str(dest), collection_name)
            return result
        finally:
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass

    async def _load_document(self, file_path: str):
        loader = LoaderFactory.get_loader(file_path)
        try:
            return await loader.load(file_path)
        except DocumentLoadError:
            raise
        except Exception as e:
            raise DocumentLoadError(
                f"Failed to load document '{os.path.basename(file_path)}': {e}",
                cause=e,
            )

    def _get_session(self):
        if self._session is not None:
            return self._session
        from compact_rag.storage.db.engine import create_engine, create_session_factory

        engine = create_engine(self._settings.database)
        factory = create_session_factory(engine)
        return factory()

    def _get_storage_backend(self):
        from compact_rag.storage.file_storage import get_storage_backend

        return get_storage_backend(self._settings.storage)

    def _get_embedding_service(self):
        from compact_rag.embedding.service import EmbeddingService

        return EmbeddingService(self._settings.embedding)

    def _get_vector_store(self, embedding_service):
        from compact_rag.storage.vector_store import VectorStore

        return VectorStore(self._settings.chromadb, embedding_service)
