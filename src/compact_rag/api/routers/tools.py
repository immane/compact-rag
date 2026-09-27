"""Commerce/tools configuration endpoints (admin-managed runtime overrides)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from compact_rag.api.deps import get_settings
from compact_rag.api.schemas import (
    CommerceConfigResponse,
    CommerceConfigUpdate,
    CommerceToolInfo,
    TestLookupRequest,
    TestOrderLinkRequest,
)
from compact_rag.common.logger import get_logger
from compact_rag.config.runtime import (
    effective_settings,
    masked_commerce_view,
    save_runtime_config,
)
from compact_rag.config.settings import Settings
from compact_rag.tool.commerce import build_commerce_tools

logger = get_logger(__name__)
router = APIRouter(tags=["Tools"])

_SECRET_FIELDS = {"api_key", "signing_secret"}


def _build_patch(update: CommerceConfigUpdate) -> dict:
    """Convert an update request into a runtime-config patch.

    Secrets (``api_key``/``signing_secret``): absent or empty keeps the stored
    value; a non-empty value overwrites it. All other fields: present values
    overwrite, absent fields are left untouched.
    """
    patch: dict = {}
    if update.commerce_enabled is not None:
        patch["commerce_enabled"] = update.commerce_enabled
    if update.products is not None:
        sub: dict = {}
        for field in ("collection", "top_k", "api_base", "api_key"):
            value = getattr(update.products, field)
            if value is None:
                continue
            if field in _SECRET_FIELDS and value == "":
                continue
            sub[field] = value
        if sub:
            patch["products"] = sub
    if update.order is not None:
        sub = {}
        for field in (
            "api_base",
            "api_key",
            "create_path",
            "url_template",
            "signing_secret",
            "link_ttl_minutes",
        ):
            value = getattr(update.order, field)
            if value is None:
                continue
            if field in _SECRET_FIELDS and value == "":
                continue
            sub[field] = value
        if sub:
            patch["order"] = sub
    return patch


def _tools_info(settings: Settings) -> list[dict]:
    effective, _ = effective_settings(settings)
    return [
        {"name": t.name, "description": t.description}
        for t in build_commerce_tools(effective.products, effective.order)
    ]


@router.get("/config/commerce", response_model=CommerceConfigResponse)
async def get_commerce_config(settings: Settings = Depends(get_settings)):
    """Return the effective commerce/tools configuration (secrets masked)."""
    view = masked_commerce_view(settings)
    return CommerceConfigResponse(
        commerce_enabled=view["commerce_enabled"],
        products=view["products"],
        order=view["order"],
        tools=[CommerceToolInfo(**t) for t in _tools_info(settings)],
    )


@router.put("/config/commerce", response_model=CommerceConfigResponse)
async def update_commerce_config(
    update: CommerceConfigUpdate, settings: Settings = Depends(get_settings)
):
    """Update runtime commerce/tools overrides (persisted to the data dir)."""
    patch = _build_patch(update)
    if not patch:
        raise HTTPException(status_code=400, detail="Empty update: nothing to change")
    try:
        save_runtime_config(settings, patch)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    view = masked_commerce_view(settings)
    return CommerceConfigResponse(
        commerce_enabled=view["commerce_enabled"],
        products=view["products"],
        order=view["order"],
        tools=[CommerceToolInfo(**t) for t in _tools_info(settings)],
    )


@router.post("/config/commerce/test-lookup")
async def test_product_lookup(
    request: TestLookupRequest, settings: Settings = Depends(get_settings)
):
    """Dry-run product lookup against the effective configuration."""
    effective, _ = effective_settings(settings)
    tools = build_commerce_tools(effective.products, effective.order)
    lookup = next((t for t in tools if t.name == "lookup_products"), None)
    if lookup is None:
        raise HTTPException(status_code=500, detail="lookup_products tool missing")
    try:
        result = await lookup.fn(disease=request.disease, top_k=request.top_k)
    except Exception as e:
        logger.warning("Test product lookup failed", error=str(e))
        return {"disease": request.disease, "products": [], "source": "none",
                "error": str(e)}
    return result if isinstance(result, dict) else {"result": result}


@router.post("/config/commerce/test-order-link")
async def test_order_link(
    request: TestOrderLinkRequest, settings: Settings = Depends(get_settings)
):
    """Dry-run order-link generation against the effective configuration."""
    effective, _ = effective_settings(settings)
    tools = build_commerce_tools(effective.products, effective.order)
    creator = next((t for t in tools if t.name == "create_order_link"), None)
    if creator is None:
        raise HTTPException(status_code=500, detail="create_order_link tool missing")
    try:
        result = await creator.fn(
            product_id=request.product_id,
            quantity=request.quantity,
            product_name=request.product_name,
        )
    except Exception as e:
        logger.warning("Test order-link generation failed", error=str(e))
        return {"product_id": request.product_id, "url": "", "error": str(e)}
    return result if isinstance(result, dict) else {"result": result}
