"""Runtime (admin-managed) configuration overrides.

File/env/YAML settings are static; the admin console needs to tweak product
lookup and order-link settings without redeploying. Overrides are stored as
JSON in the data directory (which is a persisted volume in Docker)::

    <data>/runtime_config.json      # e.g. data/runtime_config.json

Shape::

    {
      "commerce_enabled": true,
      "products": {"collection": ..., "top_k": ..., "api_base": ..., "api_key": ...},
      "order": {"api_base": ..., "api_key": ..., "create_path": ...,
                "url_template": ..., "signing_secret": ..., "link_ttl_minutes": ...}
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

_ALLOWED_TOP_LEVEL_KEYS = {"commerce_enabled", "products", "order"}

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
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def effective_settings(settings):
    """Return ``(settings_copy, commerce_enabled)`` with runtime overrides applied."""
    overrides = load_runtime_config(settings)
    commerce_enabled = overrides.get("commerce_enabled", True)
    if not isinstance(commerce_enabled, bool):
        commerce_enabled = True
    merged = settings.model_copy(deep=True)
    products_patch = overrides.get("products") or {}
    if products_patch:
        merged.products = type(merged.products)(
            **{**merged.products.model_dump(), **products_patch}
        )
    order_patch = overrides.get("order") or {}
    if order_patch:
        merged.order = type(merged.order)(
            **{**merged.order.model_dump(), **order_patch}
        )
    return merged, commerce_enabled


def masked_commerce_view(settings) -> dict:
    """Public (secret-masked) view of the effective commerce configuration."""
    effective, commerce_enabled = effective_settings(settings)
    return {
        "commerce_enabled": commerce_enabled,
        "products": {
            "collection": effective.products.collection,
            "top_k": effective.products.top_k,
            "api_base": effective.products.api_base or "",
            "api_key_configured": bool(effective.products.api_key),
        },
        "order": {
            "api_base": effective.order.api_base or "",
            "api_key_configured": bool(effective.order.api_key),
            "create_path": effective.order.create_path,
            "url_template": effective.order.url_template or "",
            "signing_secret_configured": bool(effective.order.signing_secret),
            "link_ttl_minutes": effective.order.link_ttl_minutes,
            "mode": "api"
            if effective.order.api_base
            else ("template" if effective.order.url_template else "unconfigured"),
        },
    }
