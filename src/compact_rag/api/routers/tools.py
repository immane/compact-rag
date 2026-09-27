"""Commerce/tools configuration endpoints (admin-managed runtime overrides)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from compact_rag.api.deps import get_settings
from compact_rag.common.logger import get_logger
from compact_rag.config.runtime import (
    get_dynamic_tools,
    masked_dynamic_tools_view,
    masked_sources_view,
    load_runtime_config,
    save_runtime_config,
    save_dynamic_tools,
    save_sources,
)
from compact_rag.config.settings import Settings
from compact_rag.ingestion.sources import SourceDefinition
from compact_rag.tool.dynamic import DynamicToolDefinition, build_dynamic_tools

logger = get_logger(__name__)
router = APIRouter(tags=["Tools"])


@router.get("/config/tools")
async def get_dynamic_tools_config(settings: Settings = Depends(get_settings)):
    config = load_runtime_config(settings)
    return {
        "enabled": config.get("tools_enabled", True),
        "tools": masked_dynamic_tools_view(settings),
    }


@router.put("/config/tools")
async def update_dynamic_tools_config(body: dict, settings: Settings = Depends(get_settings)):
    definitions = body.get("tools")
    if not isinstance(definitions, list):
        raise HTTPException(status_code=400, detail="Body must contain tools: [...]")
    names = [item.get("name") for item in definitions if isinstance(item, dict)]
    if len(names) != len(definitions) or len(set(names)) != len(names):
        raise HTTPException(status_code=400, detail="Tools must be objects with unique names")
    enabled = body.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="enabled must be boolean")
    try:
        for definition in definitions:
            DynamicToolDefinition(**definition)
        merged = save_dynamic_tools(settings, definitions)
        # Validate after secret preservation as stored data is what will execute.
        merged = [DynamicToolDefinition(**item).model_dump() for item in merged]
        patch: dict = {"tools": merged}
        if enabled is not None:
            patch["tools_enabled"] = enabled
        save_runtime_config(settings, patch)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return await get_dynamic_tools_config(settings)


@router.post("/config/tools/{tool_name}/test")
async def test_dynamic_tool(tool_name: str, body: dict, settings: Settings = Depends(get_settings)):
    definition = next(
        (tool for tool in get_dynamic_tools(settings) if tool.get("name") == tool_name),
        None,
    )
    if definition is None:
        raise HTTPException(status_code=404, detail=f"Unknown tool: '{tool_name}'")
    try:
        tool = next(
            (t for t in build_dynamic_tools([definition]) if t.name == tool_name), None
        )
        if tool is None:
            raise HTTPException(status_code=400, detail="Tool is disabled")
        result = await tool.fn(**body)
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Dynamic tool test failed", tool=tool_name, error=str(e))
        return {"error": str(e)}

@router.get("/config/sources")
async def list_sources(settings: Settings = Depends(get_settings)):
    """List dynamic API data source definitions (auth tokens masked)."""
    return {"sources": masked_sources_view(settings)}


@router.put("/config/sources")
async def replace_sources(
    body: dict, settings: Settings = Depends(get_settings)
):
    """Replace the source definition list.

    Each entry is validated as a ``SourceDefinition``. An empty
    ``auth_token`` keeps the previously stored token for the same source
    name; server-managed ``last_sync``/``last_result`` are preserved.
    """
    raw_sources = body.get("sources")
    if not isinstance(raw_sources, list):
        raise HTTPException(
            status_code=400, detail="Body must be {'sources': [...]}"
        )
    names = [s.get("name") for s in raw_sources if isinstance(s, dict)]
    if len(set(names)) != len(names):
        raise HTTPException(status_code=400, detail="Duplicate source names")
    try:
        for entry in raw_sources:
            SourceDefinition(**entry)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    merged = save_sources(settings, [dict(s) for s in raw_sources])
    validated = [SourceDefinition(**s).model_dump() for s in merged]
    save_runtime_config(settings, {"sources": validated})
    return {"sources": masked_sources_view(settings)}


@router.delete("/config/sources/{name}")
async def delete_source(name: str, settings: Settings = Depends(get_settings)):
    """Delete a source definition by name."""
    from compact_rag.config.runtime import get_sources

    stored = get_sources(settings)
    remaining = [s for s in stored if s.get("name") != name]
    if len(remaining) == len(stored):
        raise HTTPException(status_code=404, detail=f"Unknown source: '{name}'")
    save_runtime_config(settings, {"sources": remaining})
    return {"sources": masked_sources_view(settings)}
