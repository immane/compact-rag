from __future__ import annotations

import json


from compact_rag.tool.builtin import (
    RAG_TOOLS,
    ToolRegistry,
    _FORBIDDEN_PATTERN,
    _SELECT_PATTERN,
    query_database,
    retrieve_docs,
)
from compact_rag.tool.engine import ToolEngine
from compact_rag.tool.schema import Tool


class TestRetrieveDocs:
    def test_returns_expected_format(self):
        result = retrieve_docs("人工智能")
        assert "Retrieving documents for:" in result
        assert "人工智能" in result
        assert "top_k=3" in result

    def test_with_top_k_parameter(self):
        result = retrieve_docs("machine learning", top_k=5)
        assert "machine learning" in result
        assert "top_k=5" in result

    def test_default_top_k_is_three(self):
        result = retrieve_docs("test")
        assert "top_k=3" in result


class TestQueryDatabase:
    def test_allows_select_star(self):
        result = query_database("SELECT * FROM users")
        parsed = json.loads(result)
        assert "result" in parsed
        assert "SELECT" in parsed["result"]

    def test_allows_select_with_columns(self):
        result = query_database("SELECT id, name, email FROM users WHERE active = 1")
        parsed = json.loads(result)
        assert "result" in parsed

    def test_allows_select_with_joins(self):
        sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        result = query_database(sql)
        parsed = json.loads(result)
        assert "result" in parsed

    def test_allows_select_with_subquery(self):
        result = query_database("SELECT * FROM (SELECT id FROM users WHERE active = 1) t")
        parsed = json.loads(result)
        assert "result" in parsed

    def test_blocks_insert(self):
        result = query_database("INSERT INTO users (name) VALUES ('test')")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Only SELECT" in parsed["error"]

    def test_blocks_update(self):
        result = query_database("UPDATE users SET name = 'x' WHERE id = 1")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Only SELECT" in parsed["error"]

    def test_blocks_delete(self):
        result = query_database("DELETE FROM users WHERE id = 1")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Only SELECT" in parsed["error"]

    def test_blocks_drop(self):
        result = query_database("DROP TABLE users")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Only SELECT" in parsed["error"]

    def test_blocks_alter(self):
        result = query_database("ALTER TABLE users ADD COLUMN x INTEGER")
        parsed = json.loads(result)
        assert "error" in parsed

    def test_blocks_create(self):
        result = query_database("CREATE TABLE x (id INTEGER)")
        parsed = json.loads(result)
        assert "error" in parsed

    def test_blocks_truncate(self):
        result = query_database("TRUNCATE TABLE users")
        parsed = json.loads(result)
        assert "error" in parsed

    def test_blocks_grant(self):
        result = query_database("GRANT SELECT ON users TO role")
        parsed = json.loads(result)
        assert "error" in parsed

    def test_blocks_revoke(self):
        result = query_database("REVOKE SELECT ON users FROM role")
        parsed = json.loads(result)
        assert "error" in parsed

    def test_blocks_non_select_statement(self):
        result = query_database("SHOW TABLES")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Only SELECT" in parsed["error"]

    def test_case_insensitive_select(self):
        result = query_database("select * from users")
        parsed = json.loads(result)
        assert "result" in parsed

        result2 = query_database("Select id from users")
        parsed2 = json.loads(result2)
        assert "result" in parsed2

    def test_case_insensitive_forbidden(self):
        result = query_database("insert into users values (1)")
        parsed = json.loads(result)
        assert "error" in parsed

        result2 = query_database("Delete from users")
        parsed2 = json.loads(result2)
        assert "error" in parsed2


class TestSelectPatternRegex:
    def test_matches_select_at_start(self):
        assert _SELECT_PATTERN.search("SELECT * FROM users") is not None

    def test_matches_select_with_leading_whitespace(self):
        assert _SELECT_PATTERN.search("  SELECT * FROM users") is not None
        assert _SELECT_PATTERN.search("\tSELECT * FROM users") is not None

    def test_matches_select_with_leading_newline(self):
        assert _SELECT_PATTERN.search("\nSELECT * FROM users") is not None

    def test_does_not_match_select_in_middle(self):
        assert _SELECT_PATTERN.search("NOT SELECT *") is None

    def test_does_not_match_words_containing_select(self):
        assert _SELECT_PATTERN.search("deselect * from users") is None

    def test_case_insensitive(self):
        assert _SELECT_PATTERN.search("select * from users") is not None


class TestForbiddenPatternRegex:
    def test_matches_insert(self):
        assert _FORBIDDEN_PATTERN.search("INSERT INTO t VALUES (1)") is not None

    def test_matches_update(self):
        assert _FORBIDDEN_PATTERN.search("UPDATE t SET x = 1") is not None

    def test_matches_delete(self):
        assert _FORBIDDEN_PATTERN.search("DELETE FROM t") is not None

    def test_matches_drop(self):
        assert _FORBIDDEN_PATTERN.search("DROP TABLE t") is not None

    def test_matches_alter(self):
        assert _FORBIDDEN_PATTERN.search("ALTER TABLE t ADD x") is not None

    def test_matches_create(self):
        assert _FORBIDDEN_PATTERN.search("CREATE TABLE t (x int)") is not None

    def test_matches_truncate(self):
        assert _FORBIDDEN_PATTERN.search("TRUNCATE TABLE t") is not None

    def test_matches_grant(self):
        assert _FORBIDDEN_PATTERN.search("GRANT SELECT ON t TO r") is not None

    def test_matches_revoke(self):
        assert _FORBIDDEN_PATTERN.search("REVOKE SELECT ON t FROM r") is not None

    def test_word_boundary_does_not_match_deletes(self):
        assert _FORBIDDEN_PATTERN.search("deletes") is None

    def test_word_boundary_does_not_match_inserting(self):
        assert _FORBIDDEN_PATTERN.search("inserting") is None

    def test_case_insensitive(self):
        assert _FORBIDDEN_PATTERN.search("insert into t values (1)") is not None
        assert _FORBIDDEN_PATTERN.search("update t set x = 1") is not None


class TestRagTools:
    def test_contains_two_tools(self):
        assert len(RAG_TOOLS) == 2

    def test_tool_names_are_correct(self):
        names = [t.name for t in RAG_TOOLS]
        assert "retrieve_docs" in names
        assert "query_database" in names

    def test_tools_are_tool_instances(self):
        for t in RAG_TOOLS:
            assert isinstance(t, Tool)


class TestToolRegistry:
    def test_empty_initially(self):
        registry = ToolRegistry()
        assert registry.get_all() == []

    def test_get_all_returns_empty_list_initially(self):
        registry = ToolRegistry()
        assert registry.get_all() == []

    def test_register_tool_appears_in_get_all(self):
        registry = ToolRegistry()
        tool = Tool(retrieve_docs)
        registry.register(tool)
        assert tool in registry.get_all()
        assert len(registry.get_all()) == 1

    def test_unregister_removes_tool(self):
        registry = ToolRegistry()
        tool = Tool(retrieve_docs)
        registry.register(tool)
        assert len(registry.get_all()) == 1
        registry.unregister("retrieve_docs")
        assert len(registry.get_all()) == 0

    def test_unregister_nonexistent_silently(self):
        registry = ToolRegistry()
        registry.unregister("nonexistent")
        assert registry.get_all() == []

    def test_register_function_returns_original_fn(self):
        registry = ToolRegistry()

        @registry.register_function
        def _my_func(x: int) -> str:
            """Test."""
            return str(x)

        assert _my_func(5) == "5"

    def test_register_function_adds_to_registry(self):
        registry = ToolRegistry()

        @registry.register_function
        def _add_func(a: int, b: int) -> int:
            return a + b

        tools = registry.get_all()
        assert len(tools) == 1
        assert tools[0].name == "_add_func"

    def test_get_engine_returns_tool_engine(self):
        registry = ToolRegistry()
        registry.register(Tool(retrieve_docs))
        engine = registry.get_engine()
        assert isinstance(engine, ToolEngine)

    def test_get_engine_includes_registered_tools(self):
        registry = ToolRegistry()
        registry.register(Tool(retrieve_docs))
        engine = registry.get_engine()
        tools = engine.get_openai_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "retrieve_docs"

    def test_get_engine_default_max_retries(self):
        registry = ToolRegistry()
        engine = registry.get_engine()
        assert engine.max_retries == 2

    def test_get_engine_custom_max_retries(self):
        registry = ToolRegistry()
        engine = registry.get_engine(max_retries=5)
        assert engine.max_retries == 5

    def test_get_engine_max_retries_zero(self):
        registry = ToolRegistry()
        engine = registry.get_engine(max_retries=0)
        assert engine.max_retries == 0
