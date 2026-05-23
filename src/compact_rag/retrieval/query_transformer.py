from __future__ import annotations


class QueryTransformer:
    async def hyde_transform(self, query: str, llm_client) -> str:
        return query

    async def multi_query_expand(self, query: str, llm_client) -> list[str]:
        return [query]
