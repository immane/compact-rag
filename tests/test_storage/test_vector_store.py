from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from compact_rag.storage.schema import DocumentChunk, SearchResult
from compact_rag.storage.vector_store import VectorStore


class TestVectorStore:
    def test_add_documents_returns_chroma_ids(self, mock_embedding_service, mock_chromadb_client, sample_chunks):
        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        embeddings = np.random.randn(len(sample_chunks), 384).astype(np.float32)
        ids = vs.add_documents(sample_chunks, embeddings)

        assert len(ids) == len(sample_chunks)
        assert all(isinstance(i, str) for i in ids)
        assert mock_chromadb_client.get_or_create_collection.return_value.add.called

    def test_add_documents_empty_chunks(self, mock_embedding_service, mock_chromadb_client):
        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        ids = vs.add_documents([], np.array([]).reshape(0, 384))
        assert ids == []

    def test_search_returns_search_result_list(self, mock_embedding_service, mock_chromadb_client):
        mock_embedding_service.encode_query = MagicMock(
            return_value=np.random.randn(1, 384).astype(np.float32)
        )
        mock_chromadb_client.get_or_create_collection.return_value.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["result one", "result two"]],
            "metadatas": [[{"key": "a"}, {"key": "b"}]],
            "distances": [[0.1, 0.5]],
        }

        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        results = vs.search("test query", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].id == "id1"
        assert results[0].content == "result one"
        assert isinstance(results[0].score, float)

    def test_search_empty_query(self, mock_embedding_service, mock_chromadb_client):
        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        results = vs.search("")
        assert results == []

    def test_count_returns_correct_count(self, mock_embedding_service, mock_chromadb_client):
        mock_chromadb_client.get_or_create_collection.return_value.count.return_value = 42

        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        count = vs.count()
        assert count == 42

    def test_count_with_where_filter(self, mock_embedding_service, mock_chromadb_client):
        mock_chromadb_client.get_or_create_collection.return_value.get.return_value = {
            "ids": ["a", "b", "c"],
        }

        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        count = vs.count(where={"collection_name": "test"})
        assert count == 3

    def test_delete_by_document_returns_count(self, mock_embedding_service, mock_chromadb_client):
        mock_chromadb_client.get_or_create_collection.return_value.get.return_value = {
            "ids": ["chunk1", "chunk2"],
        }

        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        deleted = vs.delete_by_document("doc-123")
        assert deleted == 2
        mock_chromadb_client.get_or_create_collection.return_value.delete.assert_called_once()

    def test_delete_by_document_not_found(self, mock_embedding_service, mock_chromadb_client):
        mock_chromadb_client.get_or_create_collection.return_value.get.return_value = {
            "ids": [],
        }

        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        deleted = vs.delete_by_document("nonexistent")
        assert deleted == 0

    def test_list_collections(self, mock_embedding_service, mock_chromadb_client):
        mock_col1 = type("MockCol", (), {"name": "coll_a"})()
        mock_col2 = type("MockCol", (), {"name": "coll_b"})()
        mock_chromadb_client.list_collections.return_value = [mock_col1, mock_col2]

        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        collections = vs.list_collections()
        assert collections == ["coll_a", "coll_b"]

    def test_ensure_collection_uses_cosine_metadata(self, mock_embedding_service, mock_chromadb_client):
        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        vs._ensure_collection()
        mock_chromadb_client.get_or_create_collection.assert_called_once()
        call_kwargs = mock_chromadb_client.get_or_create_collection.call_args[1]
        assert call_kwargs["metadata"] == {"hnsw:space": "cosine"}
