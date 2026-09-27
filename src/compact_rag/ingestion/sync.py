"""Scheduled sync of dynamic API data sources into collections."""

from __future__ import annotations

from datetime import datetime, timezone

from compact_rag.common.logger import get_logger
from compact_rag.config.runtime import save_runtime_config
from compact_rag.ingestion.sources import SourceDefinition, fetch_source

logger = get_logger(__name__)


async def sync_source(definition: SourceDefinition, settings, session) -> dict:
    """Fetch one source incrementally and ingest new/changed records."""
    from compact_rag.ingestion.pipeline import IngestionPipeline

    records = await fetch_source(definition, since=definition.last_sync)
    pipeline = IngestionPipeline(settings=settings, session=session)
    results = await pipeline.ingest_records(
        records,
        collection_name=definition.collection,
        source_name=definition.name,
    )
    completed = sum(1 for r in results if r.status == "completed")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    errors = [
        {"filename": r.filename, "error": r.error_message}
        for r in results
        if r.status == "failed" and r.error_message
    ]
    watermark = definition.last_sync
    stamps = [r.updated_at for r in records if r.updated_at]
    if stamps:
        watermark = max(stamps)
    if watermark is None:
        watermark = datetime.now(timezone.utc).isoformat()
    summary = {
        "source": definition.name,
        "collection": definition.collection,
        "fetched": len(records),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
        "last_sync": watermark,
    }
    _record_sync_result(settings, definition.name, watermark, summary)
    logger.info("Source synced", **{k: v for k, v in summary.items() if k != "errors"})
    return summary


async def sync_sources(settings, session, only: str | None = None) -> list[dict]:
    """Sync all enabled sources (or a single named one)."""
    from compact_rag.config.runtime import load_runtime_config

    raw_sources = load_runtime_config(settings).get("sources", [])
    definitions = []
    for raw in raw_sources:
        try:
            definitions.append(SourceDefinition(**raw))
        except Exception as e:
            logger.warning("Skipping invalid source definition", error=str(e))
    if only:
        definitions = [d for d in definitions if d.name == only]
        if not definitions:
            raise ValueError(f"Unknown source: '{only}'")
    summaries = []
    for definition in definitions:
        if not definition.enabled:
            continue
        try:
            summaries.append(await sync_source(definition, settings, session))
        except Exception as e:
            logger.error("Source sync failed", source=definition.name, error=str(e))
            summary = {
                "source": definition.name,
                "collection": definition.collection,
                "fetched": 0,
                "completed": 0,
                "skipped": 0,
                "failed": 0,
                "errors": [{"error": f"{type(e).__name__}: {e}"}],
                "last_sync": definition.last_sync,
            }
            _record_sync_result(
                settings, definition.name, definition.last_sync, summary
            )
            summaries.append(summary)
    return summaries


def _record_sync_result(settings, name: str, watermark: str | None, summary: dict) -> None:
    from compact_rag.config.runtime import load_runtime_config

    sources = load_runtime_config(settings).get("sources", [])
    for entry in sources:
        if entry.get("name") == name:
            entry["last_sync"] = watermark
            entry["last_result"] = {
                k: v for k, v in summary.items() if k not in ("last_sync",)
            }
    save_runtime_config(settings, {"sources": sources})
