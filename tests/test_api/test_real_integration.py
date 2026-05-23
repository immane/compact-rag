from __future__ import annotations

import os

import httpx
import pytest


def _require_real_integration_enabled() -> None:
    enabled = os.getenv("COMPACT_RAG_RUN_REAL_INTEGRATION", "0").strip()
    if enabled != "1":
        pytest.skip("Set COMPACT_RAG_RUN_REAL_INTEGRATION=1 to run real integration tests")


def _base_url() -> str:
    return os.getenv("COMPACT_RAG_REAL_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _assert_paginated_contract(payload: dict) -> None:
    assert set(payload.keys()) == {"data", "pagination"}
    assert isinstance(payload["data"], list)
    assert isinstance(payload["pagination"], dict)
    assert {"page", "page_size", "total", "total_pages"}.issubset(payload["pagination"].keys())


@pytest.mark.real_integration
@pytest.mark.integration
class TestRealApiIntegration:
    def test_health_info_and_core_lists_against_live_server(self):
        _require_real_integration_enabled()

        with httpx.Client(timeout=20.0, trust_env=False) as client:
            base = _base_url()

            health = client.get(f"{base}/v1/health")
            assert health.status_code == 200
            health_data = health.json()
            assert set(health_data.keys()) == {"api", "database", "chromadb", "storage"}
            assert health_data["api"] == "ok"

            info = client.get(f"{base}/v1/info")
            assert info.status_code == 200
            info_data = info.json()
            assert info_data["version"]
            assert info_data["embedding_dimension"] > 0
            assert info_data["llm_provider"]
            assert info_data["llm_model"]

            for endpoint in [
                "/v1/collections",
                "/v1/documents",
                "/v1/conversations",
                "/v1/ingestion-jobs",
                "/v1/api-keys",
            ]:
                response = client.get(f"{base}{endpoint}")
                assert response.status_code == 200, endpoint
                _assert_paginated_contract(response.json())

            files = client.get(f"{base}/v1/files")
            assert files.status_code == 200
            files_data = files.json()
            assert set(files_data.keys()) == {"data", "pagination"}
            assert "total" in files_data["pagination"]
