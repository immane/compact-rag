from __future__ import annotations

from compact_rag.ingestion.loader import _clean_pdf_text


def test_clean_pdf_text_removes_known_noise_phrase():
    text = "这是一段正文。散不轻把成银 这还是正文。"
    cleaned = _clean_pdf_text(text)

    assert "散不轻把成银" not in cleaned
    assert "这是一段正文" in cleaned
    assert "这还是正文" in cleaned


def test_clean_pdf_text_removes_cid_artifacts():
    text = "你好(cid:129) 世界"
    cleaned = _clean_pdf_text(text)

    assert "(cid:129)" not in cleaned
    assert "你好" in cleaned
    assert "世界" in cleaned
