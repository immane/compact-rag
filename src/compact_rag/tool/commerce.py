"""Commerce tools: product lookup and signed order-link generation.

``lookup_products`` searches the product knowledge-base collection first and
falls back to an external product API when configured. ``create_order_link``
generates a purchase URL either via an order service API or via a locally
signed URL template (HMAC-SHA256). Both tools return JSON-serialisable dicts
and never raise for "not found / not configured" situations — they return an
``error`` payload instead so the LLM can narrate the outcome honestly.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from compact_rag.common.logger import get_logger
from compact_rag.tool.schema import Tool

logger = get_logger(__name__)

ORDER_TOOL_NAME = "create_order_link"


def sign_order_payload(
    signing_secret: str, product_id: str, quantity: int, expires: int
) -> str:
    """HMAC-SHA256 signature for a template-mode order link."""
    payload = f"{product_id}:{quantity}:{expires}".encode("utf-8")
    return hmac.new(signing_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def verify_order_signature(
    signing_secret: str, product_id: str, quantity: int, expires: int, signature: str
) -> bool:
    """Constant-time verification of a template-mode order-link signature."""
    expected = sign_order_payload(signing_secret, product_id, quantity, expires)
    return hmac.compare_digest(expected, signature)


def _default_retriever_provider():
    from compact_rag.api.deps import get_hybrid_retriever

    return get_hybrid_retriever()


async def _lookup_knowledge_base(
    disease: str,
    collection: str,
    top_k: int,
    retriever_provider: Callable[[], Any] | None,
) -> list[dict]:
    provider = retriever_provider or _default_retriever_provider
    retriever = provider()
    results = await retriever.retrieve(
        query=disease,
        collection=collection,
        top_k=top_k,
        use_hybrid_search=True,
        use_rerank=True,
    )
    products = []
    for r in results:
        metadata = getattr(r, "metadata", {}) or {}
        products.append(
            {
                "product_id": metadata.get(
                    "product_id", metadata.get("doc_id", getattr(r, "id", ""))
                ),
                "name": metadata.get(
                    "product_name", metadata.get("filename", "unknown")
                ),
                "score": round(float(getattr(r, "score", 0.0)), 4),
                "snippet": (getattr(r, "content", "") or "")[:200],
                "source": "knowledge_base",
            }
        )
    return products


async def _lookup_external_api(
    disease: str,
    top_k: int,
    api_base: str,
    api_key: str | None,
    timeout: int,
) -> list[dict]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.get(
                f"{api_base.rstrip('/')}/products/search",
                params={"q": disease, "top_k": top_k},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("External product API lookup failed", error=str(e))
        return []
    items = data.get("products", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    products = []
    for item in items[:top_k]:
        if not isinstance(item, dict):
            continue
        products.append(
            {
                "product_id": str(
                    item.get("product_id", item.get("id", item.get("sku", "")))
                ),
                "name": str(item.get("name", item.get("title", "unknown"))),
                "score": float(item.get("score", 0.0) or 0.0),
                "snippet": str(item.get("description", item.get("snippet", "")))[:200],
                "source": "api",
            }
        )
    return products


def build_commerce_tools(
    products_cfg,
    order_cfg,
    retriever_provider: Callable[[], Any] | None = None,
) -> list[Tool]:
    """Build the commerce Tool list bound to the given settings sections."""

    async def lookup_products(disease: str, top_k: int = 5) -> dict:
        """根据疾病/症状查找适配的产品。先检索产品知识库，若无命中且配置了外部产品 API 则再查询外部接口。返回产品 id、名称、匹配分与摘要。"""
        top_k = max(1, min(int(top_k or 5), 50))
        try:
            kb_products = await _lookup_knowledge_base(
                disease,
                products_cfg.collection,
                min(top_k, products_cfg.top_k),
                retriever_provider,
            )
        except Exception as e:
            logger.warning("Product KB lookup failed", error=str(e))
            kb_products = []
        if kb_products:
            return {
                "disease": disease,
                "products": kb_products[:top_k],
                "source": "knowledge_base",
            }
        if products_cfg.api_base:
            api_products = await _lookup_external_api(
                disease,
                top_k,
                products_cfg.api_base,
                products_cfg.api_key,
                products_cfg.timeout,
            )
            if api_products:
                return {
                    "disease": disease,
                    "products": api_products,
                    "source": "api",
                }
        return {
            "disease": disease,
            "products": [],
            "source": "none",
            "error": f"No products found for '{disease}'.",
        }

    lookup_products.__name__ = "lookup_products"

    async def create_order_link(
        product_id: str, quantity: int = 1, note: str = "", product_name: str = ""
    ) -> dict:
        """为指定产品生成下单链接。quantity 为购买数量，product_name 为展示用的产品名称。返回的 url 即为可直接打开的下单地址，调用者必须原样使用该 url，绝不自行拼接或编造。"""
        try:
            quantity = max(1, int(quantity))
        except (TypeError, ValueError):
            quantity = 1
        expires_at = (
            datetime.fromtimestamp(
                time.time() + order_cfg.link_ttl_minutes * 60, tz=timezone.utc
            ).isoformat()
        )
        if order_cfg.api_base:
            return await _create_link_via_api(
                order_cfg, product_id, quantity, note, product_name, expires_at
            )
        if order_cfg.url_template and order_cfg.signing_secret:
            return _create_link_via_template(
                order_cfg, product_id, quantity, product_name, expires_at
            )
        return {
            "product_id": product_id,
            "product_name": product_name,
            "quantity": quantity,
            "url": "",
            "error": "Order-link generation is not configured. "
            "Set COMPACT_RAG_ORDER__API_BASE or "
            "COMPACT_RAG_ORDER__URL_TEMPLATE + COMPACT_RAG_ORDER__SIGNING_SECRET.",
        }

    create_order_link.__name__ = ORDER_TOOL_NAME

    return [Tool(lookup_products), Tool(create_order_link)]


async def _create_link_via_api(
    order_cfg, product_id: str, quantity: int, note: str, product_name: str,
    expires_at: str,
) -> dict:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if order_cfg.api_key:
        headers["Authorization"] = f"Bearer {order_cfg.api_key}"
    try:
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            resp = await client.post(
                f"{order_cfg.api_base.rstrip('/')}{order_cfg.create_path}",
                json={
                    "product_id": product_id,
                    "product_name": product_name,
                    "quantity": quantity,
                    "note": note,
                    "expires_at": expires_at,
                },
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("Order API link creation failed", error=str(e))
        return {
            "product_id": product_id,
            "product_name": product_name,
            "quantity": quantity,
            "url": "",
            "error": f"Order service call failed: {e}",
        }
    url = data.get("url", "") if isinstance(data, dict) else ""
    if not url:
        return {
            "product_id": product_id,
            "product_name": product_name,
            "quantity": quantity,
            "url": "",
            "error": "Order service did not return a url.",
        }
    return {
        "product_id": product_id,
        "product_name": product_name,
        "quantity": quantity,
        "url": url,
        "expires_at": data.get("expires_at", expires_at),
        "source": "order_api",
    }


def _create_link_via_template(
    order_cfg, product_id: str, quantity: int, product_name: str, expires_at: str
) -> dict:
    expires = int(time.time()) + order_cfg.link_ttl_minutes * 60
    signature = sign_order_payload(
        order_cfg.signing_secret, product_id, quantity, expires
    )
    try:
        url = order_cfg.url_template.format(
            product_id=product_id,
            quantity=quantity,
            expires=expires,
            signature=signature,
        )
    except (KeyError, IndexError, ValueError) as e:
        return {
            "product_id": product_id,
            "product_name": product_name,
            "quantity": quantity,
            "url": "",
            "error": f"Invalid url_template: {e}",
        }
    return {
        "product_id": product_id,
        "product_name": product_name,
        "quantity": quantity,
        "url": url,
        "expires_at": expires_at,
        "source": "template",
    }
