from __future__ import annotations

import pytest

from compact_rag.generation.prompt import PromptManager


class TestPromptManager:
    @pytest.fixture
    def pm(self):
        return PromptManager()

    def test_render_system_prompt_default(self, pm):
        result = pm.render_system_prompt()
        assert "智能知识库助手" in result
        assert "可用集合" in result
        assert result.endswith("集合：")

    def test_render_system_prompt_with_collections(self, pm):
        result = pm.render_system_prompt(collections=["docs", "reports"])
        assert "docs, reports" in result

    def test_render_rag_context(self, pm):
        documents = [
            {
                "filename": "report.pdf",
                "page_number": 1,
                "content": "Annual revenue: $10M",
            },
            {
                "filename": "notes.md",
                "page_number": 3,
                "content": "Key findings: market growth",
            },
        ]
        result = pm.render_rag_context(documents=documents)
        assert "[来源 1]" in result
        assert "[来源 2]" in result
        assert "report.pdf" in result
        assert "Annual revenue: $10M" in result
        assert "notes.md" in result
        assert "Key findings: market growth" in result

    def test_register_custom_template(self, pm):
        pm.register_template("greeting", "Hello, {{ username }}!")
        result = pm.render("greeting", username="World")
        assert result == "Hello, World!"

    def test_render_unknown_template_raises(self, pm):
        with pytest.raises(KeyError):
            pm.render("nonexistent_template")
