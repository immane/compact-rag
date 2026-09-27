from __future__ import annotations

from jinja2 import BaseLoader, Environment, FileSystemLoader

_DEFAULT_SYSTEM_PROMPT = """\
你是一个智能知识库助手，基于提供的文档内容回答用户问题。

规则：
1. 仅基于提供的文档内容回答，不编造信息
2. 如果文档中没有相关信息，诚实告知用户
3. 回答要简洁准确，在末尾标注引用的文档来源
4. 当文档中包含表格时，保留 Markdown 表格格式
5. 当用户问及数据时，可调用相关工具获取精确信息
6. 允许基于多个片段做隐含推断，但必须明确说明推断依据并给出引用
7. 仅当用户询问产品推荐、购买或下单时，才调用 lookup_products / create_order_link 工具；日常问答不要调用
8. 需要下单链接时：先用 lookup_products 按疾病/症状查产品，再用 create_order_link 为每个选定的产品生成链接；工具返回的 url 必须原样呈现，绝不自行拼接、改写或编造；若工具返回无产品或未配置，诚实告知用户，不要给出链接
9. 涉及病人病情等敏感信息时，只引用文档原文作答，不做超出依据的诊断断言；下单链接仅供用户确认，本系统不自动下单

可用集合：{{ collections | join(", ") }}
"""

_DEFAULT_RAG_CONTEXT = """\
{% for doc in documents %}
---
[来源 {{ loop.index }}] 文件: {{ doc.filename }}
页码: {{ doc.page_number }}

{{ doc.content }}
{% endfor %}
"""


class PromptManager:
    def __init__(self, template_dir: str | None = None) -> None:
        loader = FileSystemLoader(template_dir) if template_dir else BaseLoader()
        self._env = Environment(loader=loader, autoescape=True)

        self._env.from_string(_DEFAULT_SYSTEM_PROMPT)
        self._env.from_string(_DEFAULT_RAG_CONTEXT)
        self._templates: dict[str, str] = {
            "system_prompt": _DEFAULT_SYSTEM_PROMPT,
            "rag_context": _DEFAULT_RAG_CONTEXT,
        }

    def register_template(self, name: str, template_str: str) -> None:
        self._templates[name] = template_str

    def render(self, name: str, **kwargs) -> str:
        template = self._env.from_string(self._templates[name])
        return template.render(**kwargs)

    def render_system_prompt(self, collections: list[str] | None = None) -> str:
        return self.render("system_prompt", collections=collections or [])

    def render_rag_context(self, documents: list[dict]) -> str:
        return self.render("rag_context", documents=documents)
