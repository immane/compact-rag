"""Tests for commerce tools (product lookup + order-link generation)."""

from __future__ import annotations

import json

from compact_rag.config.settings import OrderLinkSettings, ProductLookupSettings
from compact_rag.storage.schema import SearchResult
from compact_rag.tool.commerce import (
    build_commerce_tools,
    sign_order_payload,
    verify_order_signature,
)
from compact_rag.tool.engine import ToolEngine


def _products_cfg(**overrides):
    base = {"collection": "products", "top_k": 5}
    base.update(overrides)
    return ProductLookupSettings(**base)


def _order_cfg(**overrides):
    base = {"create_path": "/orders/link", "link_ttl_minutes": 30}
    base.update(overrides)
    return OrderLinkSettings(**base)


def _engine(products_cfg, order_cfg, retriever_provider=None):
    return ToolEngine(
        build_commerce_tools(products_cfg, order_cfg, retriever_provider),
        max_retries=0,
    )


async def _call(engine, name, arguments, call_id="call_1"):
    return await engine.execute_tool_call({
        "function": {"name": name, "arguments": json.dumps(arguments)},
        "id": call_id,
    })


class TestSignatures:
    def test_sign_verify_roundtrip(self):
        sig = sign_order_payload("secret", "P001", 2, 1700000000)
        assert verify_order_signature("secret", "P001", 2, 1700000000, sig)

    def test_tampered_payload_rejected(self):
        sig = sign_order_payload("secret", "P001", 2, 1700000000)
        assert not verify_order_signature("secret", "P001", 3, 1700000000, sig)
        assert not verify_order_signature("other", "P001", 2, 1700000000, sig)


class TestCreateOrderLinkTemplate:
    async def test_template_mode_generates_verifiable_url(self):
        engine = _engine(
            _products_cfg(),
            _order_cfg(
                url_template="https://shop.example.com/order?product={product_id}&qty={quantity}&exp={expires}&sig={signature}",
                signing_secret="s3cret",
            ),
        )
        result = await _call(
            engine, "create_order_link", {"product_id": "P001", "quantity": 2}
        )
        payload = json.loads(result["content"])
        assert payload["url"].startswith("https://shop.example.com/order?product=P001")
        assert payload["quantity"] == 2
        assert payload["source"] == "template"

    async def test_unconfigured_returns_error_not_exception(self):
        engine = _engine(_products_cfg(), _order_cfg())
        result = await _call(engine, "create_order_link", {"product_id": "P001"})
        payload = json.loads(result["content"])
        assert payload["url"] == ""
        assert "not configured" in payload["error"]

    async def test_invalid_template_returns_error(self):
        engine = _engine(
            _products_cfg(),
            _order_cfg(url_template="https://x/{missing}", signing_secret="s"),
        )
        result = await _call(engine, "create_order_link", {"product_id": "P001"})
        assert "Invalid url_template" in json.loads(result["content"])["error"]

    async def test_quantity_clamped(self):
        engine = _engine(
            _products_cfg(),
            _order_cfg(
                url_template="https://x/{product_id}/{quantity}",
                signing_secret="s",
            ),
        )
        result = await _call(engine, "create_order_link", {"product_id": "P", "quantity": 0})
        assert json.loads(result["content"])["quantity"] == 1


class TestCreateOrderLinkApi:
    async def test_api_mode_posts_and_returns_url(self, monkeypatch):
        import httpx

        posted = {}

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"url": "https://orders.example.com/L123"}

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                posted["url"] = url
                posted["json"] = json
                return FakeResp()

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
        engine = _engine(
            _products_cfg(),
            _order_cfg(api_base="https://orders.example.com", api_key="k"),
        )
        result = await _call(engine, "create_order_link", {"product_id": "P9"})
        payload = json.loads(result["content"])
        assert payload["url"] == "https://orders.example.com/L123"
        assert payload["source"] == "order_api"
        assert posted["url"] == "https://orders.example.com/orders/link"
        assert posted["json"]["product_id"] == "P9"


class TestLookupProducts:
    def _kb_provider(self, results):
        class FakeRetriever:
            async def retrieve(self, **kwargs):
                assert kwargs["collection"] == "products"
                return results

        return lambda: FakeRetriever()

    async def test_kb_hit(self):
        results = [
            SearchResult(
                id="c1",
                content="降压药甲，适用于高血压",
                score=0.91,
                metadata={"product_id": "MED-001", "filename": "catalog.pdf"},
            )
        ]
        engine = _engine(
            _products_cfg(), _order_cfg(), self._kb_provider(results)
        )
        result = await _call(engine, "lookup_products", {"disease": "高血压"})
        payload = json.loads(result["content"])
        assert payload["source"] == "knowledge_base"
        assert payload["products"][0]["product_id"] == "MED-001"
        assert payload["products"][0]["score"] == 0.91

    async def test_kb_miss_without_api_returns_error(self):
        engine = _engine(
            _products_cfg(), _order_cfg(), self._kb_provider([])
        )
        result = await _call(engine, "lookup_products", {"disease": "未知病"})
        payload = json.loads(result["content"])
        assert payload["products"] == []
        assert payload["source"] == "none"
        assert "error" in payload

    async def test_kb_miss_falls_back_to_api(self, monkeypatch):
        import httpx

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"products": [{"id": "API-1", "name": "ApiDrug"}]}

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, params: dict | None = None, headers=None):
                assert params is not None and params["q"] == "偏头痛"
                return FakeResp()

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
        engine = _engine(
            _products_cfg(api_base="https://p.example.com", api_key="k"),
            _order_cfg(),
            self._kb_provider([]),
        )
        result = await _call(engine, "lookup_products", {"disease": "偏头痛"})
        payload = json.loads(result["content"])
        assert payload["source"] == "api"
        assert payload["products"][0]["product_id"] == "API-1"


class TestExtractOrderLinks:
    def test_extracts_valid_links_only(self):
        from compact_rag.rag.pipeline import extract_order_links

        messages = [
            {"role": "tool", "name": "create_order_link", "content": json.dumps({
                "product_id": "P1", "product_name": "Drug A", "quantity": 2,
                "url": "https://x/1", "source": "template",
            })},
            {"role": "tool", "name": "create_order_link", "content": json.dumps({
                "product_id": "P2", "url": "", "error": "not configured",
            })},
            {"role": "tool", "name": "other_tool", "content": "{}"},
            {"role": "assistant", "content": "hi"},
        ]
        links = extract_order_links(messages)
        assert len(links) == 1
        assert links[0].product_id == "P1"
        assert links[0].url == "https://x/1"

    def test_openai_tool_schema_generated(self):
        engine = _engine(_products_cfg(), _order_cfg())
        tools = engine.get_openai_tools()
        assert {t["function"]["name"] for t in tools} == {
            "lookup_products",
            "create_order_link",
        }


class TestCommerceEnvOverrides:
    def test_products_collection_env_override(self, monkeypatch):
        from compact_rag.config.settings import Settings

        monkeypatch.setenv("COMPACT_RAG_PRODUCTS__COLLECTION", "med_products")
        settings = Settings.load("config/default.yaml")
        assert settings.products.collection == "med_products"

    def test_order_secret_env_override(self, monkeypatch):
        from compact_rag.config.settings import Settings

        monkeypatch.setenv("COMPACT_RAG_ORDER__SIGNING_SECRET", "env-secret")
        settings = Settings.load("config/default.yaml")
        assert settings.order.signing_secret == "env-secret"
