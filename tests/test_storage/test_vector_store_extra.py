"""Extra VectorStore tests: fetch_documents, delete_by_ids, search variants, count edge cases."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from compact_rag.common.exceptions import VectorStoreError
from compact_rag.storage.schema import SearchResult
from compact_rag.storage.vector_store import VectorStore


class TestVectorStoreInit:
    def test_init_without_embedding_service(self):
        vs = VectorStore()
        assert vs._settings.collection_name == "default"

    def test_init_with_custom_settings(self):
        from compact_rag.config.settings import ChromaDBSettings

        settings = ChromaDBSettings(
            persist_directory="/tmp/test_chroma",
            collection_name="custom_collection",
        )
        vs = VectorStore(settings=settings)
        assert vs._settings.collection_name == "custom_collection"


class TestVectorStoreSearchVariants:
    @pytest.fixture
    def vs(self, mock_embedding_service, mock_chromadb_client):
        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client
        return vs

    def test_search_with_where_filter(self, mock_embedding_service, mock_chromadb_client):
        mock_embedding_service.encode_query = MagicMock(
            return_value=np.random.randn(1, 384).astype(np.float32)
        )
        mock_chromadb_client.get_or_create_collection.return_value.query.return_value = {
            "ids": [["id-filter"]],
            "documents": [["filtered result"]],
            "metadatas": [[{"collection_name": "test"}]],
            "distances": [[0.05]],
        }

        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        results = vs.search("query", top_k=5, where={"collection_name": "test"})
        assert len(results) == 1
        assert results[0].id == "id-filter"
        assert results[0].content == "filtered result"

        call_kwargs = mock_chromadb_client.get_or_create_collection.return_value.query.call_args[1]
        assert call_kwargs["where"] == {"collection_name": "test"}

    def test_search_returns_empty_for_no_ids(self, mock_embedding_service, mock_chromadb_client):
        mock_embedding_service.encode_query = MagicMock(
            return_value=np.random.randn(384).astype(np.float32)
        )
        mock_chromadb_client.get_or_create_collection.return_value.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        results = vs.search("empty")
        assert results == []

    def test_search_query_ndim_2(self, mock_embedding_service, mock_chromadb_client):
        mock_embedding_service.encode_query = MagicMock(
            return_value=np.random.randn(1, 384).astype(np.float32)
        )
        mock_chromadb_client.get_or_create_collection.return_value.query.return_value = {
            "ids": [["id-2d"]],
            "documents": [["content 2d"]],
            "metadatas": [[{"k": "v"}]],
            "distances": [[0.2]],
        }

        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        results = vs.search("2d query")
        assert len(results) == 1
        assert results[0].content == "content 2d"


class TestFetchDocuments:
    @pytest.fixture
    def vs(self, mock_embedding_service, mock_chromadb_client):
        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client
        return vs

    def test_fetch_documents_with_limit(self, vs, mock_chromadb_client):
        mock_chromadb_client.get_or_create_collection.return_value.get.return_value = {
            "ids": ["d1", "d2"],
            "documents": ["content A", "content B"],
            "metadatas": [{"m": "a"}, {"m": "b"}],
        }

        results = vs.fetch_documents(limit=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].id == "d1"
        assert results[0].content == "content A"
        assert results[0].score == 0.0

    def test_fetch_documents_with_where_filter(self, vs, mock_chromadb_client):
        mock_chromadb_client.get_or_create_collection.return_value.get.return_value = {
            "ids": ["d-filtered"],
            "documents": ["filtered content"],
            "metadatas": [{"collection_name": "custom"}],
        }

        results = vs.fetch_documents(where={"collection_name": "custom"}, limit=5)
        assert len(results) == 1
        assert results[0].metadata == {"collection_name": "custom"}

    def test_fetch_documents_batched(self, vs, mock_chromadb_client):
        call_count = {"count": 0}
        items = [
            {"ids": [f"d{i}" for i in range(10)], "documents": [f"c{i}" for i in range(10)], "metadatas": [{} for _ in range(10)]},
            {"ids": [f"d{i}" for i in range(10, 15)], "documents": [f"c{i}" for i in range(10, 15)], "metadatas": [{} for _ in range(5)]},
            {"ids": [], "documents": [], "metadatas": []},
        ]

        def _get_side_effect(**kwargs):
            result = items[min(call_count["count"], len(items) - 1)]
            call_count["count"] += 1
            return result

        mock_chromadb_client.get_or_create_collection.return_value.get = MagicMock(
            side_effect=_get_side_effect
        )

        results = vs.fetch_documents(batch_size=10)
        assert len(results) == 15

    def test_fetch_documents_empty_result(self, vs, mock_chromadb_client):
        mock_chromadb_client.get_or_create_collection.return_value.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }

        results = vs.fetch_documents()
        assert results == []

    def test_fetch_documents_error(self, vs, mock_chromadb_client):
        mock_chromadb_client.get_or_create_collection.return_value.get.side_effect = Exception("fetch error")

        with pytest.raises(VectorStoreError, match="Failed to fetch documents"):
            vs.fetch_documents()


class TestDeleteByIds:
    @pytest.fixture
    def vs(self, mock_embedding_service, mock_chromadb_client):
        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client
        return vs

    def test_delete_by_ids_returns_count(self, vs, mock_chromadb_client):
        result = vs.delete_by_ids(["id-1", "id-2", "id-3"])
        assert result == 3
        mock_chromadb_client.get_or_create_collection.return_value.delete.assert_called_once_with(
            ids=["id-1", "id-2", "id-3"]
        )

    def test_delete_by_ids_empty_list_returns_zero(self, vs):
        result = vs.delete_by_ids([])
        assert result == 0

    def test_delete_by_ids_error(self, vs, mock_chromadb_client):
        mock_chromadb_client.get_or_create_collection.return_value.delete.side_effect = Exception("delete error")

        with pytest.raises(VectorStoreError, match="Failed to delete chunks"):
            vs.delete_by_ids(["id-1"])


class TestVectorStoreEdgeCases:
    def test_list_collections_error(self, mock_embedding_service, mock_chromadb_client):
        mock_chromadb_client.list_collections.side_effect = Exception("list error")

        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        with pytest.raises(VectorStoreError, match="Failed to list collections"):
            vs.list_collections()

    def test_count_error(self, mock_embedding_service, mock_chromadb_client):
        mock_chromadb_client.get_or_create_collection.return_value.count.side_effect = Exception("count error")

        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        with pytest.raises(VectorStoreError, match="Failed to count documents"):
            vs.count()

    def test_add_documents_mismatch_error(self, mock_embedding_service, mock_chromadb_client, sample_chunks):
        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        embeddings = np.random.randn(len(sample_chunks) - 1, 384).astype(np.float32)
        with pytest.raises(VectorStoreError, match="Mismatch"):
            vs.add_documents(sample_chunks, embeddings)

    def test_add_documents_error(self, mock_embedding_service, mock_chromadb_client, sample_chunks):
        mock_chromadb_client.get_or_create_collection.return_value.add.side_effect = Exception("add error")

        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        embeddings = np.random.randn(len(sample_chunks), 384).astype(np.float32)
        with pytest.raises(VectorStoreError, match="Failed to add documents"):
            vs.add_documents(sample_chunks, embeddings)

    def test_delete_by_document_error(self, mock_embedding_service, mock_chromadb_client):
        mock_chromadb_client.get_or_create_collection.return_value.get.side_effect = Exception("get error")

        vs = VectorStore(embedding_service=mock_embedding_service)
        vs._client = mock_chromadb_client

        with pytest.raises(VectorStoreError, match="Failed to delete document"):
            vs.delete_by_document("doc-id")
