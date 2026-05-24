from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from compact_rag.common.exceptions import VectorStoreError
from compact_rag.common.logger import get_logger
from compact_rag.storage.schema import DocumentChunk, SearchResult

if TYPE_CHECKING:
    from compact_rag.config.settings import ChromaDBSettings
    from compact_rag.embedding.service import EmbeddingService

logger = get_logger(__name__)


class VectorStore:
    """ChromaDB vector store wrapper for document chunk storage and search."""

    def __init__(
        self,
        settings: ChromaDBSettings | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        from compact_rag.config.settings import ChromaDBSettings

        self._settings = settings if settings is not None else ChromaDBSettings()

        if embedding_service is not None:
            self._embedding_service = embedding_service
        else:
            from compact_rag.embedding.service import EmbeddingService

            self._embedding_service = EmbeddingService()

        self._client = None
        self._collection = None

    @property
    def client(self):
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._ensure_collection()
        return self._collection

    def _create_client(self):
        try:
            import chromadb
        except ImportError:
            raise VectorStoreError(
                "chromadb not installed. Install with: pip install chromadb"
            )
        persist_dir = Path(self._settings.persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(persist_dir))

    def _ensure_collection(self) -> None:
        try:
            self._collection = self.client.get_or_create_collection(
                name=self._settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to get or create collection: {e}", cause=e)

    def add_documents(
        self, chunks: list[DocumentChunk], embeddings: np.ndarray
    ) -> list[str]:
        if not chunks:
            return []
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        embeddings_list: list[list[float]] = []

        for i, chunk in enumerate(chunks):
            chroma_id = str(uuid.uuid4())
            ids.append(chroma_id)
            documents.append(chunk.content)
            metadata = {
                "chroma_id": chroma_id,
                "doc_id": chunk.metadata.get("doc_id", ""),
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number or 0,
                "filename": chunk.metadata.get("filename", ""),
                "collection_name": chunk.metadata.get(
                    "collection_name", self._settings.collection_name
                ),
                "is_table": chunk.is_table,
                "token_count": chunk.token_count,
            }
            metadatas.append(metadata)

            if embeddings.dtype == np.float32:
                embeddings_list.append(embeddings[i].tolist())
            else:
                embeddings_list.append(embeddings[i].astype(np.float32).tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to add documents to vector store: {e}", cause=e
            )

        logger.info("Documents added to vector store", count=len(ids))
        return ids

    def search(
        self,
        query: str,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[SearchResult]:
        if not query:
            return []

        query_embedding = self._embedding_service.encode_query(query)
        if query_embedding.ndim == 2:
            query_vec = query_embedding[0].tolist()
        else:
            query_vec = query_embedding.tolist()

        try:
            results = self.collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise VectorStoreError(f"Vector search failed: {e}", cause=e)

        search_results: list[SearchResult] = []
        if not results["ids"] or not results["ids"][0]:
            return search_results

        for i, chunk_id in enumerate(results["ids"][0]):
            content = results["documents"][0][i] if results.get("documents") else ""
            metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
            distance = results["distances"][0][i] if results.get("distances") else 1.0
            score = max(0.0, 1.0 - float(distance))

            search_results.append(
                SearchResult(
                    id=chunk_id,
                    content=content or "",
                    score=score,
                    metadata=metadata or {},
                )
            )

        return search_results

    def fetch_documents(
        self,
        where: dict | None = None,
        limit: int | None = None,
        batch_size: int = 1000,
    ) -> list[SearchResult]:
        """Fetch stored chunks for downstream lexical indexing.

        Returns SearchResult items with score=0.0 to reuse a shared structure.
        """
        try:
            if limit is not None:
                raw = self.collection.get(
                    where=where,
                    limit=limit,
                    include=["documents", "metadatas"],
                )
                ids = raw.get("ids") or []
                docs = raw.get("documents") or []
                metas = raw.get("metadatas") or []
                return [
                    SearchResult(
                        id=ids[i],
                        content=docs[i] or "",
                        score=0.0,
                        metadata=metas[i] or {},
                    )
                    for i in range(len(ids))
                ]

            results: list[SearchResult] = []
            offset = 0
            while True:
                raw = self.collection.get(
                    where=where,
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                ids = raw.get("ids") or []
                docs = raw.get("documents") or []
                metas = raw.get("metadatas") or []
                if not ids:
                    break

                for i in range(len(ids)):
                    results.append(
                        SearchResult(
                            id=ids[i],
                            content=docs[i] or "",
                            score=0.0,
                            metadata=metas[i] or {},
                        )
                    )

                if len(ids) < batch_size:
                    break
                offset += len(ids)

            return results
        except Exception as e:
            raise VectorStoreError(f"Failed to fetch documents: {e}", cause=e)

    def delete_by_document(self, doc_id: str) -> int:
        try:
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"],
            )
            if not results["ids"]:
                return 0
            self.collection.delete(ids=results["ids"])
            logger.info(
                "Deleted chunks for document", doc_id=doc_id, count=len(results["ids"])
            )
            return len(results["ids"])
        except Exception as e:
            raise VectorStoreError(f"Failed to delete document {doc_id}: {e}", cause=e)

    def delete_by_ids(self, chroma_ids: list[str]) -> int:
        if not chroma_ids:
            return 0
        try:
            self.collection.delete(ids=chroma_ids)
            logger.info("Deleted chunks by IDs", count=len(chroma_ids))
            return len(chroma_ids)
        except Exception as e:
            raise VectorStoreError(f"Failed to delete chunks: {e}", cause=e)

    def list_collections(self) -> list[str]:
        try:
            return [c.name for c in self.client.list_collections()]
        except Exception as e:
            raise VectorStoreError(f"Failed to list collections: {e}", cause=e)

    def count(self, where: dict | None = None) -> int:
        try:
            if where:
                result = self.collection.get(where=where)
                return len(result["ids"]) if result["ids"] else 0
            return self.collection.count()
        except Exception as e:
            raise VectorStoreError(f"Failed to count documents: {e}", cause=e)
