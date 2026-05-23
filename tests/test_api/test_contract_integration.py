from __future__ import annotations

import asyncio
import hashlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from compact_rag.api.deps import _cached_settings
from compact_rag.api.router import create_app
from compact_rag.storage.db.models import (
    ApiKey,
    Base,
    Collection,
    Conversation,
    Document,
    IngestionJob,
    Message,
)


async def _prepare_db(url: str) -> None:
    engine = create_async_engine(url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        collection = Collection(name="integration-default", description="integration test")
        session.add(collection)
        await session.flush()

        session.add(
            Document(
                collection_id=collection.id,
                filename="integration.md",
                file_type="md",
                file_size=128,
                file_hash=hashlib.sha256(b"integration").hexdigest(),
                status="completed",
                chunk_count=3,
                table_count=0,
            )
        )

        conversation = Conversation(
            collection_id=collection.id,
            title="Integration Conversation",
            model="gpt-4o-mini",
            message_count=1,
        )
        session.add(conversation)
        await session.flush()

        session.add(
            Message(
                conversation_id=conversation.id,
                role="user",
                content="hello integration",
                sources=[{"filename": "integration.md", "score": 0.99}],
                token_count=5,
            )
        )

        session.add(
            IngestionJob(
                collection_id=collection.id,
                status="completed",
                total_files=1,
                processed_files=1,
                total_chunks=3,
                errors=[],
            )
        )

        session.add(
            ApiKey(
                name="integration-key",
                key_hash=hashlib.sha256(b"integration-key").hexdigest(),
                permissions=["read"],
                is_active=True,
            )
        )

        await session.commit()

    await engine.dispose()


async def _drop_db(url: str) -> None:
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def integration_client(test_settings, monkeypatch):
    from compact_rag.api.deps import get_settings as deps_get_settings

    _cached_settings.cache_clear()

    monkeypatch.setattr("compact_rag.api.deps._cached_settings", lambda: test_settings)

    asyncio.run(_prepare_db(test_settings.database.url))

    app = create_app(settings=test_settings)
    app.dependency_overrides[deps_get_settings] = lambda: test_settings

    with TestClient(app) as client:
        yield client

    asyncio.run(_drop_db(test_settings.database.url))


def _assert_paginated_contract(data: dict) -> None:
    assert set(data.keys()) == {"data", "pagination"}
    assert isinstance(data["data"], list)
    assert set(data["pagination"].keys()) == {"page", "page_size", "total", "total_pages"}
    assert isinstance(data["pagination"]["page"], int)
    assert isinstance(data["pagination"]["page_size"], int)
    assert isinstance(data["pagination"]["total"], int)
    assert isinstance(data["pagination"]["total_pages"], int)


@pytest.mark.integration
class TestApiContractIntegration:
    def test_collections_documents_conversations_jobs_api_keys_contract(self, integration_client):
        endpoints = [
            "/v1/collections",
            "/v1/documents",
            "/v1/conversations",
            "/v1/ingestion-jobs",
            "/v1/api-keys",
        ]

        for endpoint in endpoints:
            response = integration_client.get(endpoint)
            assert response.status_code == 200, endpoint
            data = response.json()
            _assert_paginated_contract(data)
            assert data["pagination"]["total"] >= 1, endpoint

    def test_collection_crud_round_trip(self, integration_client):
        create_payload = {
            "name": "integration-created",
            "description": "created by integration test",
            "embedding_model": "BAAI/bge-small-zh-v1.5",
            "chunk_size": 600,
            "chunk_overlap": 60,
        }
        create_resp = integration_client.post("/v1/collections", json=create_payload)
        assert create_resp.status_code == 200
        created = create_resp.json()

        assert created["name"] == create_payload["name"]
        assert created["description"] == create_payload["description"]
        assert created["chunk_size"] == create_payload["chunk_size"]
        assert created["chunk_overlap"] == create_payload["chunk_overlap"]

        list_resp = integration_client.get("/v1/collections", params={"page": 1, "page_size": 100})
        assert list_resp.status_code == 200
        names = {item["name"] for item in list_resp.json()["data"]}
        assert "integration-created" in names

        delete_resp = integration_client.delete("/v1/collections/integration-created")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["status"] == "deleted"

    def test_health_info_and_files_contract(self, integration_client):
        health = integration_client.get("/v1/health")
        assert health.status_code == 200
        health_data = health.json()
        assert set(health_data.keys()) == {"api", "database", "chromadb", "storage"}
        assert health_data["api"] == "ok"

        info = integration_client.get("/v1/info")
        assert info.status_code == 200
        info_data = info.json()
        assert set(info_data.keys()) == {
            "version",
            "database_url",
            "embedding_model",
            "embedding_dimension",
            "llm_provider",
            "llm_model",
            "storage_backend",
            "log_level",
        }
        assert info_data["embedding_dimension"] > 0

        files = integration_client.get("/v1/files")
        assert files.status_code == 200
        files_data = files.json()
        assert set(files_data.keys()) == {"data", "pagination"}
        assert "total" in files_data["pagination"]
