"""Tests for source sync orchestration (watermark + summaries)."""

from __future__ import annotations

from unittest.mock import AsyncMock

from compact_rag.config.runtime import load_runtime_config
from compact_rag.ingestion.sources import SourceDefinition, SourceRecord
from compact_rag.storage.schema import IngestionResult


async def test_sync_source_updates_watermark(test_settings, monkeypatch):
    import compact_rag.ingestion.sync as sync_mod

    records = [
        SourceRecord(external_id="1", title="t1", content="c1",
                     updated_at="2026-09-01T00:00:00"),
        SourceRecord(external_id="2", title="t2", content="c2",
                     updated_at="2026-09-02T00:00:00"),
    ]
    monkeypatch.setattr(sync_mod, "fetch_source", AsyncMock(return_value=records))

    results = [
        IngestionResult(doc_id="d1", filename="f1", status="completed", chunk_count=1),
        IngestionResult(doc_id="", filename="f2", status="failed",
                        error_message="E"),
    ]

    class FakePipeline:
        def __init__(self, *a, **k):
            pass

        async def ingest_records(self, *a, **k):
            return results

    monkeypatch.setattr(
        "compact_rag.ingestion.pipeline.IngestionPipeline", FakePipeline
    )

    definition = SourceDefinition(
        name="comments", base_url="https://x", collection="reviews"
    )
    # Seed the source entry so the watermark has somewhere to land.
    from compact_rag.config.runtime import save_runtime_config

    save_runtime_config(test_settings, {"sources": [definition.model_dump()]})
    summary = await sync_mod.sync_source(definition, test_settings, session=None)

    assert summary["fetched"] == 2
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["last_sync"] == "2026-09-02T00:00:00"
    stored = load_runtime_config(test_settings)["sources"][0]
    assert stored["last_sync"] == "2026-09-02T00:00:00"
    assert stored["last_result"]["completed"] == 1


async def test_sync_sources_skips_disabled_and_unknown(test_settings, monkeypatch):
    import compact_rag.ingestion.sync as sync_mod
    from compact_rag.config.runtime import save_runtime_config

    save_runtime_config(test_settings, {"sources": [
        {"name": "off", "base_url": "https://x", "enabled": False},
    ]})
    assert await sync_mod.sync_sources(test_settings, session=None, only=None) == []

    try:
        await sync_mod.sync_sources(test_settings, session=None, only="nope")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "Unknown source" in str(e)
