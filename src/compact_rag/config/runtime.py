"""Runtime (admin-managed) configuration overrides.

File/env/YAML settings are static; the admin console needs to tweak runtime
tool and data-source definitions without redeploying. Overrides are stored as
JSON in the data directory (which is a persisted volume in Docker)::

    <data>/runtime_config.json      # e.g. data/runtime_config.json

Shape::

    {
      "tools_enabled": true,
      "tools": [{"name": ..., "kind": "http" | "vector_search", ...}],
      "sources": [{"name": ..., ...}]
    }

All keys are optional and deep-merged over the static settings. Secrets live
only in this file (mode 0600) or in environment variables — never in YAML.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from compact_rag.common.logger import get_logger

logger = get_logger(__name__)

RUNTIME_FILENAME = "runtime_config.json"

_ALLOWED_TOP_LEVEL_KEYS = {"tools_enabled", "tools", "sources"}

# mtime-keyed in-process cache: {str(path): (mtime_ns, data)}
_CACHE: dict[str, tuple[int, dict]] = {}


def get_runtime_config_path(settings) -> Path:
    """Resolve the runtime config file inside the data directory."""
    storage_root = Path(settings.storage.local.root_dir)
    if not storage_root.is_absolute():
        storage_root = (Path.cwd() / storage_root).resolve()
    return storage_root.parent / RUNTIME_FILENAME


def load_runtime_config(settings) -> dict:
    """Load raw runtime overrides ({} when absent or corrupt)."""
    path = get_runtime_config_path(settings)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return {}
    cached = _CACHE.get(str(path))
    if cached is not None and cached[0] == mtime_ns:
        return dict(cached[1])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("Ignoring corrupt runtime config", path=str(path), error=str(e))
        return {}
    if not isinstance(data, dict):
        return {}
    _CACHE[str(path)] = (mtime_ns, data)
    return dict(data)


def save_runtime_config(settings, patch: dict) -> dict:
    """Deep-merge ``patch`` into the stored overrides and persist them."""
    unknown = set(patch) - _ALLOWED_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"Unknown runtime config keys: {sorted(unknown)}")
    current = load_runtime_config(settings)
    merged = _deep_merge_dicts(current, patch)
    path = get_runtime_config_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    _CACHE.pop(str(path), None)
    return merged


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def get_sources(settings) -> list[dict]:
    """Raw stored source definitions (including secrets — mask before responding)."""
    sources = load_runtime_config(settings).get("sources", [])
    return [dict(s) for s in sources] if isinstance(sources, list) else []


def masked_sources_view(settings) -> list[dict]:
    """Source definitions with auth tokens replaced by a configured flag."""
    view = []
    for source in get_sources(settings):
        masked = dict(source)
        masked["auth_token_configured"] = bool(masked.get("auth_token"))
        masked.pop("auth_token", None)
        view.append(masked)
    return view


def get_dynamic_tools(settings) -> list[dict]:
    tools = load_runtime_config(settings).get("tools", [])
    return [dict(tool) for tool in tools] if isinstance(tools, list) else []


def masked_dynamic_tools_view(settings) -> list[dict]:
    """Return tool definitions without exposing credential values."""
    view = []
    for definition in get_dynamic_tools(settings):
        item = dict(definition)
        headers = item.get("headers")
        if isinstance(headers, dict):
            item["headers_configured"] = {
                key: bool(value) for key, value in headers.items()
            }
            item["headers"] = {key: "" for key in headers}
        view.append(item)
    return view


def save_dynamic_tools(settings, definitions: list[dict]) -> list[dict]:
    """Replace dynamic tools; blank header values keep existing credentials."""
    existing = {tool.get("name"): tool for tool in get_dynamic_tools(settings)}
    merged = []
    for raw in definitions:
        item = dict(raw)
        name = item.get("name")
        if not name:
            raise ValueError("Each tool must have a name")
        old_headers = (existing.get(name) or {}).get("headers", {})
        headers = item.get("headers") or {}
        item["headers"] = {
            key: value if value else old_headers.get(key, "")
            for key, value in headers.items()
        }
        merged.append(item)
    save_runtime_config(settings, {"tools": merged})
    return merged


def save_sources(settings, sources: list[dict]) -> list[dict]:
    """Replace the stored source list.

    Entries are matched by ``name``: an empty ``auth_token`` keeps the
    previously stored token, and server-managed ``last_sync``/``last_result``
    are preserved unless explicitly provided.
    """
    stored = {s.get("name"): s for s in get_sources(settings)}
    merged = []
    for entry in sources:
        name = entry.get("name")
        if not name:
            raise ValueError("Each source must have a name")
        item = dict(entry)
        previous = stored.get(name, {})
        if not item.get("auth_token") and previous.get("auth_token"):
            item["auth_token"] = previous["auth_token"]
        for carried in ("last_sync", "last_result"):
            if carried not in item and carried in previous:
                item[carried] = previous[carried]
        merged.append(item)
    save_runtime_config(settings, {"sources": merged})
    return merged
