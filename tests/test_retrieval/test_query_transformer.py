from __future__ import annotations

import pytest

from compact_rag.retrieval.query_transformer import QueryTransformer, _normalize


@pytest.fixture
def transformer():
    return QueryTransformer()


@pytest.fixture
def mock_llm_client(mocker):
    return mocker.MagicMock()


class TestHydeTransform:
    @pytest.mark.asyncio
    async def test_hyde_transform_with_normal_query(self, transformer, mock_llm_client):
        result = await transformer.hyde_transform("人工智能如何影响医疗行业", mock_llm_client)
        assert "与问题相关的关键事实" in result
        assert "人工智能如何影响医疗行业" in result

    @pytest.mark.asyncio
    async def test_hyde_transform_with_empty_string_returns_original(self, transformer, mock_llm_client):
        result = await transformer.hyde_transform("", mock_llm_client)
        assert result == ""

    @pytest.mark.asyncio
    async def test_hyde_transform_with_whitespace_returns_original(self, transformer, mock_llm_client):
        result = await transformer.hyde_transform("   ", mock_llm_client)
        assert result == "   "

    @pytest.mark.asyncio
    async def test_hyde_transform_with_no_cjk_chars(self, transformer, mock_llm_client):
        result = await transformer.hyde_transform("What is machine learning?", mock_llm_client)
        assert result == "与问题相关的关键事实：What is machine learning"


class TestMultiQueryExpand:
    @pytest.mark.asyncio
    async def test_expand_with_normal_query(self, transformer, mock_llm_client):
        result = await transformer.multi_query_expand("Python编程教程", mock_llm_client)
        assert len(result) >= 2
        assert "Python编程教程" in result[0]
        assert "关键事实" in result[1]

    @pytest.mark.asyncio
    async def test_expand_with_implicit_hint_why(self, transformer, mock_llm_client):
        result = await transformer.multi_query_expand("为什么深度学习需要大量数据", mock_llm_client)
        assert any("相关背景与前提" in r for r in result)
        assert any("直接证据" in r for r in result)

    @pytest.mark.asyncio
    async def test_expand_with_implicit_hint_reason(self, transformer, mock_llm_client):
        result = await transformer.multi_query_expand("深度学习失败的原因", mock_llm_client)
        assert any("相关背景与前提" in r for r in result)

    @pytest.mark.asyncio
    async def test_expand_with_implicit_hint_how(self, transformer, mock_llm_client):
        result = await transformer.multi_query_expand("如何训练Transformer模型", mock_llm_client)
        assert any("相关背景与前提" in r for r in result)

    @pytest.mark.asyncio
    async def test_expand_with_implicit_hint_implicit(self, transformer, mock_llm_client):
        result = await transformer.multi_query_expand("这个问题的隐含意义", mock_llm_client)
        assert any("相关背景与前提" in r for r in result)

    @pytest.mark.asyncio
    async def test_expand_with_implicit_hint_behind(self, transformer, mock_llm_client):
        result = await transformer.multi_query_expand("现象背后的机理", mock_llm_client)
        assert any("相关背景与前提" in r for r in result)

    @pytest.mark.asyncio
    async def test_expand_with_implicit_hint_infer(self, transformer, mock_llm_client):
        result = await transformer.multi_query_expand("从数据中推断结论", mock_llm_client)
        assert any("相关背景与前提" in r for r in result)

    @pytest.mark.asyncio
    async def test_expand_with_implicit_hint_contrast(self, transformer, mock_llm_client):
        result = await transformer.multi_query_expand("CNN和Transformer的对比", mock_llm_client)
        assert any("相关背景与前提" in r for r in result)

    @pytest.mark.asyncio
    async def test_expand_with_implicit_hint_difference(self, transformer, mock_llm_client):
        result = await transformer.multi_query_expand("监督学习和无监督学习的区别", mock_llm_client)
        assert any("相关背景与前提" in r for r in result)

    @pytest.mark.asyncio
    async def test_expand_with_implicit_hint_influence(self, transformer, mock_llm_client):
        result = await transformer.multi_query_expand("AI对就业的影响", mock_llm_client)
        assert any("相关背景与前提" in r for r in result)

    @pytest.mark.asyncio
    async def test_expand_with_question_mark_adds_cleaned_variant(self, transformer, mock_llm_client):
        result = await transformer.multi_query_expand("什么是RAG？", mock_llm_client)
        assert "什么是RAG" in result

    @pytest.mark.asyncio
    async def test_expand_with_english_question_mark_adds_cleaned_variant(self, transformer, mock_llm_client):
        result = await transformer.multi_query_expand("What is RAG?", mock_llm_client)
        assert "What is RAG" in result

    @pytest.mark.asyncio
    async def test_expand_deduplication(self, transformer, mock_llm_client):
        result = await transformer.multi_query_expand("测试  测试", mock_llm_client)
        assert len(result) == len(set(result))

    @pytest.mark.asyncio
    async def test_expand_with_empty_string(self, transformer, mock_llm_client):
        result = await transformer.multi_query_expand("", mock_llm_client)
        assert result == [""]

    @pytest.mark.asyncio
    async def test_expand_with_whitespace(self, transformer, mock_llm_client):
        result = await transformer.multi_query_expand("   ", mock_llm_client)
        assert result == ["   "]


class TestNormalize:
    def test_normalize_collapses_spaces(self):
        assert _normalize("hello   world") == "hello world"

    def test_normalize_strips_leading_trailing_whitespace(self):
        assert _normalize("  hello world  ") == "hello world"

    def test_normalize_removes_trailing_punctuation(self):
        assert _normalize("hello world?") == "hello world"
        assert _normalize("hello world！") == "hello world"
        assert _normalize("hello world。") == "hello world"
        assert _normalize("hello world！") == "hello world"

    def test_normalize_removes_trailing_multiple_punctuation(self):
        assert _normalize("hello world???!") == "hello world"
        assert _normalize("hello world？？？") == "hello world"

    def test_normalize_handles_mixed_chinese_english(self):
        assert _normalize("人工智能AI时代来了!") == "人工智能AI时代来了"

    def test_normalize_keeps_embedded_punctuation(self):
        assert _normalize("hello, world!") == "hello, world"

    def test_normalize_handles_trailing_colon_and_semicolon(self):
        assert _normalize("test；：") == "test"

    def test_normalize_returns_empty_for_whitespace_only(self):
        assert _normalize("   ") == ""

    def test_normalize_returns_empty_for_empty_string(self):
        assert _normalize("") == ""
