"""Dynamic API data sources.

A source is a fully declarative paginated HTTP API (e.g. comment or report
APIs) whose records are ETL-synced into a knowledge-base collection on a
schedule. Nothing is hardcoded: new source types are added through
configuration, not code.

Expected envelope (field names configurable)::

    {"data": [...], "code": 0, "message": "SUCCESS",
     "paginator": {"current": 1, "last": 5, ...}}
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from compact_rag.common.exceptions import IngestionError
from compact_rag.common.logger import get_logger

logger = get_logger(__name__)


class SourceDefinition(BaseModel):
    """Declarative definition of one paginated API data source."""

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    collection: str = "default"
    base_url: str
    path: str = "/"
    method: Literal["GET"] = "GET"
    auth_header: str = "x-auth-token"
    auth_token: str = ""
    page_param: str = "page"
    page_size_param: str | None = "numItemsPerPage"
    page_size: int = Field(default=100, ge=1, le=1000)
    page_start: int = Field(default=1, ge=1)
    params: dict[str, Any] = Field(default_factory=dict)
    items_path: str = "data"
    code_path: str = "code"
    success_code: int = 0
    paginator_path: str = "paginator"
    current_page_field: str = "current"
    last_page_field: str = "last"
    id_field: str = "id"
    updated_at_field: str | None = "updated_at"
    title_template: str = "#{id}"
    body_fields: list[str] = Field(default_factory=lambda: ["content"])
    timeout: int = Field(default=30, gt=0)
    max_pages: int = Field(default=10000, gt=0)
    enabled: bool = True
    # Server-managed sync state (not set by admins).
    last_sync: str | None = None
    last_result: dict | None = None


class SourceRecord(BaseModel):
    """One normalized record fetched from a source API."""

    external_id: str
    title: str
    content: str
    metadata: dict = Field(default_factory=dict)
    updated_at: str | None = None


def get_path(obj: Any, dotted: str) -> Any:
    """Resolve a dotted path (``paginator.current``) inside nested dicts."""
    current = obj
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Path '{dotted}' not found in API response")
        current = current[part]
    return current


def render_title(template: str, record: dict) -> str:
    """Render a title template (``str.format``) against a raw record."""
    try:
        return template.format(**record)
    except KeyError as e:
        raise ValueError(
            f"title_template refers to missing field {e}; "
            f"available fields: {sorted(record.keys())}"
        )


def record_to_source_record(
    source_name: str, definition: SourceDefinition, raw: dict
) -> SourceRecord:
    """Normalize one raw API item into a SourceRecord."""
    if definition.id_field not in raw:
        raise ValueError(
            f"Record is missing id_field '{definition.id_field}': {sorted(raw.keys())}"
        )
    external_id = str(raw[definition.id_field])
    title = render_title(definition.title_template, {k: str(v) for k, v in raw.items()})
    body_parts = []
    for field in definition.body_fields:
        if field in raw and raw[field] not in (None, ""):
            body_parts.append(f"{field}: {raw[field]}")
    content = f"# {title}\n\n" + "\n".join(body_parts)
    updated_at = (
        raw.get(definition.updated_at_field) if definition.updated_at_field else None
    )
    metadata = {
        "source": source_name,
        "external_id": external_id,
        "fetched_from": f"{definition.base_url.rstrip('/')}{definition.path}",
    }
    for field in ("author", "author_name", "category", "status"):
        if field in raw:
            metadata[field] = raw[field]
    return SourceRecord(
        external_id=external_id,
        title=title,
        content=content,
        metadata=metadata,
        updated_at=str(updated_at) if updated_at is not None else None,
    )


def record_content_hash(source_name: str, record: SourceRecord) -> str:
    """Stable hash for change detection (title + content + metadata)."""
    canonical = json.dumps(
        {
            "source": source_name,
            "id": record.external_id,
            "title": record.title,
            "content": record.content,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def fetch_source(
    definition: SourceDefinition, since: str | None = None
) -> list[SourceRecord]:
    """Fetch (incrementally when ``since`` is given) all records of a source."""
    headers = {"Accept": "application/json"}
    if definition.auth_token:
        headers[definition.auth_header] = definition.auth_token
    url = f"{definition.base_url.rstrip('/')}{definition.path}"
    records: list[SourceRecord] = []
    page = definition.page_start
    pages_fetched = 0
    async with httpx.AsyncClient(timeout=definition.timeout, trust_env=False) as client:
        while True:
            if pages_fetched >= definition.max_pages:
                logger.warning("Source hit max_pages guard", source=definition.name)
                break
            params = dict(definition.params)
            params[definition.page_param] = page
            if definition.page_size_param:
                params[definition.page_size_param] = definition.page_size
            payload = await _get_page(client, url, params, headers, definition)
            try:
                items = get_path(payload, definition.items_path)
            except KeyError as e:
                raise IngestionError(
                    f"Source '{definition.name}': items_path error: {e}"
                )
            if not isinstance(items, list):
                raise IngestionError(
                    f"Source '{definition.name}': items_path "
                    f"'{definition.items_path}' did not resolve to a list"
                )
            for item in items:
                if not isinstance(item, dict):
                    continue
                record = record_to_source_record(definition.name, definition, item)
                if since and record.updated_at and record.updated_at <= since:
                    continue
                records.append(record)
            pages_fetched += 1
            try:
                paginator = get_path(payload, definition.paginator_path)
                current = int(paginator[definition.current_page_field])
                last = int(paginator[definition.last_page_field])
            except (KeyError, TypeError, ValueError) as e:
                raise IngestionError(
                    f"Source '{definition.name}': unreadable paginator: {e}"
                )
            if current >= last:
                break
            page = current + 1
    logger.info(
        "Source fetched",
        source=definition.name,
        records=len(records),
        pages=pages_fetched,
    )
    return records


async def _get_page(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
    headers: dict,
    definition: SourceDefinition,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            if definition.code_path:
                try:
                    code = get_path(payload, definition.code_path)
                except KeyError:
                    code = None
                if code != definition.success_code:
                    raise IngestionError(
                        f"Source '{definition.name}' returned code={code} "
                        f"(expected {definition.success_code})"
                    )
            return payload
        except IngestionError:
            raise
        except Exception as e:
            last_error = e
            logger.warning(
                "Source page fetch failed, retrying",
                source=definition.name,
                attempt=attempt + 1,
                error=str(e),
            )
            await asyncio.sleep(0.5 * (2**attempt))
    raise IngestionError(
        f"Source '{definition.name}' page fetch failed after 3 attempts: {last_error}"
    )
