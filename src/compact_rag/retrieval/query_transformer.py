from __future__ import annotations

import re

_SPACE = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[\s\?？!！。,.，;；:：]+$")
_IMPLICIT_HINTS = (
    "为什么",
    "原因",
    "隐含",
    "背后",
    "推断",
    "对比",
    "区别",
    "影响",
    "如何",
    "怎么",
)


class QueryTransformer:
    async def hyde_transform(self, query: str, llm_client) -> str:
        normalized = _normalize(query)
        if not normalized:
            return query
        # Keep HYDE deterministic and local to avoid dependency on external calls.
        return f"与问题相关的关键事实：{normalized}"

    async def multi_query_expand(self, query: str, llm_client) -> list[str]:
        normalized = _normalize(query)
        if not normalized:
            return [query]

        expanded = [normalized]
        expanded.append(f"{normalized} 关键事实")

        if any(h in normalized for h in _IMPLICIT_HINTS):
            expanded.append(f"{normalized} 相关背景与前提")
            expanded.append(f"{normalized} 直接证据")

        if "?" in query or "？" in query:
            expanded.append(_TRAILING_PUNCT.sub("", normalized))

        dedup: list[str] = []
        seen = set()
        for q in expanded:
            cleaned = _normalize(q)
            if cleaned and cleaned not in seen:
                dedup.append(cleaned)
                seen.add(cleaned)
        return dedup or [query]


def _normalize(text: str) -> str:
    text = _SPACE.sub(" ", text.strip())
    return _TRAILING_PUNCT.sub("", text)
