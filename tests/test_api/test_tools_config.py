"""Tests for commerce/tools configuration endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from compact_rag.api.deps import _cached_settings, get_commerce_tool_engine
from compact_rag.api.router import create_app
from compact_rag.storage.schema import SearchResult


def _client(test_settings):
    _cached_settings.cache_clear()
    app = create_app(settings=test_settings)
    with TestClient(app) as c:
        yield c


class TestCommerceConfigEndpoints:
    def test_get_masked_defaults(self, test_settings):
        for c in _client(test_settings):
            resp = c.get("/v1/config/commerce")
            assert resp.status_code == 200
            body = resp.json()
            assert body["commerce_enabled"] is True
            assert body["products"]["collection"] == "products"
            assert body["products"]["api_key_configured"] is False
            assert body["order"]["mode"] == "unconfigured"
            assert {t["name"] for t in body["tools"]} == {
                "lookup_products",
                "create_order_link",
            }

    def test_put_and_persist(self, test_settings):
        for c in _client(test_settings):
            resp = c.put("/v1/config/commerce", json={
                "products": {"collection": "meds", "top_k": 3},
                "order": {"link_ttl_minutes": 10},
            })
            assert resp.status_code == 200
            body = c.get("/v1/config/commerce").json()
            assert body["products"]["collection"] == "meds"
            assert body["products"]["top_k"] == 3
            assert body["order"]["link_ttl_minutes"] == 10

    def test_put_empty_rejected(self, test_settings):
        for c in _client(test_settings):
            assert c.put("/v1/config/commerce", json={}).status_code == 400

    def test_secret_empty_keeps_stored_value(self, test_settings):
        for c in _client(test_settings):
            c.put("/v1/config/commerce", json={
                "order": {"signing_secret": "keep-me"}
            })
            assert c.get("/v1/config/commerce").json()["order"][
                "signing_secret_configured"
            ] is True
            # Empty/absent secret must not clear the stored one.
            c.put("/v1/config/commerce", json={
                "order": {"url_template": "https://x/{product_id}", "signing_secret": ""}
            })
            body = c.get("/v1/config/commerce").json()["order"]
            assert body["signing_secret_configured"] is True
            assert body["mode"] == "template"

    def test_master_switch_disables_engine(self, test_settings):
        for c in _client(test_settings):
            c.put("/v1/config/commerce", json={"commerce_enabled": False})
            assert c.get("/v1/config/commerce").json()["commerce_enabled"] is False
            assert get_commerce_tool_engine(test_settings) is None
            c.put("/v1/config/commerce", json={"commerce_enabled": True})
            assert get_commerce_tool_engine(test_settings) is not None

    def test_order_link_dry_run_template(self, test_settings):
        for c in _client(test_settings):
            c.put("/v1/config/commerce", json={"order": {
                "url_template": "https://shop.example.com/o?product={product_id}&qty={quantity}&exp={expires}&sig={signature}",
                "signing_secret": "s",
            }})
            resp = c.post("/v1/config/commerce/test-order-link", json={
                "product_id": "P1", "quantity": 2,
            })
            assert resp.status_code == 200
            body = resp.json()
            assert body["url"].startswith(
                "https://shop.example.com/o?product=P1&qty=2"
            )
            assert body["source"] == "template"

    def test_order_link_dry_run_unconfigured(self, test_settings):
        for c in _client(test_settings):
            resp = c.post("/v1/config/commerce/test-order-link", json={
                "product_id": "P1",
            })
            assert resp.status_code == 200
            assert resp.json()["url"] == ""
            assert "error" in resp.json()

    def test_lookup_dry_run_kb(self, test_settings, monkeypatch):
        import compact_rag.tool.commerce as commerce

        class FakeRetriever:
            async def retrieve(self, **kwargs):
                assert kwargs["collection"] == "meds"
                return [SearchResult(
                    id="c1", content="降压药", score=0.9,
                    metadata={"product_id": "M1", "filename": "cat.pdf"},
                )]

        monkeypatch.setattr(
            commerce, "_default_retriever_provider", lambda: FakeRetriever()
        )
        for c in _client(test_settings):
            c.put("/v1/config/commerce", json={"products": {"collection": "meds"}})
            resp = c.post("/v1/config/commerce/test-lookup", json={
                "disease": "高血压",
            })
            assert resp.status_code == 200
            body = resp.json()
            assert body["source"] == "knowledge_base"
            assert body["products"][0]["product_id"] == "M1"
