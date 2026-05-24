"""RAG pipeline orchestration — query, retrieve, generate with citations."""

from __future__ import annotations

import time
from typing import Any, AsyncGenerator

from compact_rag.common.logger import get_logger
from compact_rag.generation.llm import LLMClient
from compact_rag.generation.prompt import PromptManager
from compact_rag.retrieval.retriever import HybridRetriever
from compact_rag.storage.schema import RAGCitation, RAGResponse

logger = get_logger(__name__)


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: LLMClient,
        prompt_manager: PromptManager,
        conversation_repo=None,
        message_repo=None,
        tool_engine=None,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.tool_engine = tool_engine

    async def query(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
        collection: str = "default",
        stream: bool = False,
        top_k: int = 10,
        use_hybrid_search: bool = True,
        use_rerank: bool = True,
        conversation_id: str | None = None,
        db_session=None,
    ) -> Any:
        t_start = time.perf_counter()

        messages = self._build_messages(question, conversation_history or [])
        tool_results = None

        if self.tool_engine:
            try:
                tool_messages, tool_results = await self.tool_engine.execute(
                    messages, self.llm_client
                )
                messages.extend(tool_messages)
            except Exception as e:
                logger.warning("Tool execution failed", error=str(e))

        t_ret_start = time.perf_counter()
        retrieved = await self.retriever.retrieve(
            query=question,
            collection=collection,
            top_k=top_k,
            use_hybrid_search=use_hybrid_search,
            use_rerank=use_rerank,
        )
        retrieval_latency = (time.perf_counter() - t_ret_start) * 1000

        context = self._build_context(retrieved)
        augmented_messages = self._augment_messages(messages, context)

        t_gen_start = time.perf_counter()
        if stream:
            result = await self.llm_client.chat(augmented_messages)
        else:
            result = await self.llm_client.chat(augmented_messages)
        generation_latency = (time.perf_counter() - t_gen_start) * 1000

        if hasattr(result, "content"):
            answer = getattr(result, "content", "") or ""
        elif isinstance(result, dict):
            answer = result.get("content", "")
        else:
            answer = str(result)

        citations = self._build_citations(retrieved)
        if hasattr(result, "token_usage"):
            token_usage = getattr(result, "token_usage", {}) or {}
        elif isinstance(result, dict):
            token_usage = result.get("token_usage", {})
        else:
            token_usage = {}

        if (
            self.conversation_repo is not None
            and db_session is not None
            and conversation_id
        ):
            try:
                await self._save_conversation(
                    db_session,
                    conversation_id,
                    question,
                    answer,
                    citations,
                    token_usage,
                    retrieval_latency + generation_latency,
                )
            except Exception as e:
                logger.warning("Failed to save conversation", error=str(e))

        return RAGResponse(
            id=f"rag-{int(t_start * 1000)}",
            answer=answer,
            citations=citations,
            token_usage=token_usage,
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=generation_latency,
        )

    async def query_stream(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
        collection: str = "default",
        top_k: int = 10,
        use_hybrid_search: bool = True,
        use_rerank: bool = True,
        conversation_id: str | None = None,
        db_session=None,
    ) -> AsyncGenerator[str, None]:
        messages = self._build_messages(question, conversation_history or [])

        if self.tool_engine:
            try:
                tool_messages, _ = await self.tool_engine.execute(
                    messages, self.llm_client
                )
                messages.extend(tool_messages)
            except Exception as e:
                logger.warning("Tool execution failed", error=str(e))

        t_ret_start = time.perf_counter()
        retrieved = await self.retriever.retrieve(
            query=question,
            collection=collection,
            top_k=top_k,
            use_hybrid_search=use_hybrid_search,
            use_rerank=use_rerank,
        )
        retrieval_latency = (time.perf_counter() - t_ret_start) * 1000

        context = self._build_context(retrieved)
        augmented_messages = self._augment_messages(messages, context)

        full_answer = ""
        t_gen_start = time.perf_counter()
        async for chunk in self.llm_client.chat_stream(augmented_messages):
            full_answer += chunk
            yield chunk
        generation_latency = (time.perf_counter() - t_gen_start) * 1000

        citations = self._build_citations(retrieved)

        if (
            self.conversation_repo is not None
            and db_session is not None
            and conversation_id
        ):
            try:
                await self._save_conversation(
                    db_session,
                    conversation_id,
                    question,
                    full_answer,
                    citations,
                    {"completion_tokens": len(full_answer.split())},
                    retrieval_latency + generation_latency,
                )
            except Exception as e:
                logger.warning("Failed to save conversation", error=str(e))

    def _build_messages(self, question: str, history: list[dict]) -> list[dict]:
        system_prompt = self.prompt_manager.render_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-20:]:
            messages.append(
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            )
        messages.append({"role": "user", "content": question})
        return messages

    def _build_context(self, results: list) -> str:
        if not results:
            return "No relevant documents found."
        chunks = []
        for i, r in enumerate(results):
            content = getattr(r, "content", str(r))
            filename = getattr(r, "metadata", {}).get("filename", "unknown")
            chunks.append(f"[Document {i + 1}] (Source: {filename})\n{content}")
        return "\n\n---\n\n".join(chunks)

    def _augment_messages(self, messages: list[dict], context: str) -> list[dict]:
        augmented = list(messages)
        rag_instruction = (
            "Use the following retrieved context to answer the user's question. "
            "You may synthesize evidence across multiple documents and make careful implicit inferences "
            "only when grounded in retrieved text. "
            "Cite sources using [Document N] notation when referencing specific information.\n\n"
            f"### Retrieved Context ###\n{context}\n### End Context ###\n\n"
            "Now answer the user's question based on the context above."
        )
        target_index = None
        for i in range(len(augmented) - 1, -1, -1):
            msg = augmented[i]
            if msg["role"] == "user":
                target_index = i
                break

        if target_index is None:
            augmented.append(
                {
                    "role": "user",
                    "content": f"{rag_instruction}\n\nUser Question: ",
                }
            )
            return augmented

        original_question = augmented[target_index]["content"]
        augmented[target_index]["content"] = (
            f"{rag_instruction}\n\nUser Question: {original_question}"
        )
        return augmented

    def _build_citations(self, results: list) -> list[RAGCitation]:
        citations: list[RAGCitation] = []
        for r in results:
            doc_id = getattr(r, "id", "unknown")
            metadata = getattr(r, "metadata", {})
            citations.append(
                RAGCitation(
                    doc_id=doc_id,
                    chunk_index=metadata.get("chunk_index", 0),
                    page_number=metadata.get("page_number"),
                    filename=metadata.get("filename", ""),
                    score=getattr(r, "score", 0.0),
                    content_snippet=getattr(r, "content", "")[:200],
                )
            )
        return citations

    async def _save_conversation(
        self,
        db_session,
        conversation_id: str,
        question: str,
        answer: str,
        citations: list[RAGCitation],
        token_usage: dict,
        latency_ms: float,
    ) -> None:
        sources = [
            {
                "doc_id": c.doc_id,
                "filename": c.filename,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "score": c.score,
                "snippet": c.content_snippet,
            }
            for c in citations
        ]
        try:
            await self.message_repo.create(
                db_session,
                conversation_id=conversation_id,
                role="user",
                content=question,
            )
            await self.message_repo.create(
                db_session,
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                sources=sources,
                token_count=token_usage.get("total_tokens", 0),
                latency_ms=int(latency_ms),
            )
            await self.conversation_repo.increment_message_count(
                db_session, conversation_id, 2
            )
        except Exception as e:
            logger.warning("Failed to persist conversation turn", error=str(e))
