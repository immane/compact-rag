"""Chat completions endpoint — OpenAI-compatible API."""

from __future__ import annotations

import json
import time
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from compact_rag.api.deps import get_db_session, get_rag_pipeline
from compact_rag.api.schemas import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
    UsageInfo,
)
from compact_rag.common.logger import get_logger
from compact_rag.rag.pipeline import RAGPipeline
from compact_rag.storage.db.models import Conversation

logger = get_logger(__name__)
router = APIRouter(tags=["Chat"])


async def _ensure_conversation(
    session: AsyncSession,
    conversation_id: str | None,
    model: str,
    collection: str,
    title: str,
) -> str:
    if conversation_id:
        return conversation_id

    conv = Conversation(
        id=str(uuid4()),
        collection_id=collection if collection != "default" else None,
        title=title,
        model=model,
    )
    session.add(conv)
    await session.flush()
    return conv.id


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    session: AsyncSession = Depends(get_db_session),
):
    """Core Q&A endpoint — compatible with OpenAI Chat Completions API."""
    call_id = f"rag-{int(time.time() * 1000)}"

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    first_user_text = messages[-1]["content"][:100] if messages else "New Conversation"
    conversation_id = await _ensure_conversation(
        session,
        request.conversation_id,
        request.model,
        request.collection,
        first_user_text,
    )

    if request.stream:
        return StreamingResponse(
            _stream_response(
                call_id, request, pipeline, messages, session, conversation_id
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    response = await _query_with_compat(
        pipeline, request, messages, session, conversation_id
    )
    await session.commit()

    return ChatCompletionResponse(
        id=call_id,
        created=int(time.time()),
        model=request.model,
        conversation_id=conversation_id,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessageResponse(
                    role="assistant",
                    content=response.answer,
                    citations=[
                        {
                            "doc_id": c.doc_id,
                            "filename": c.filename,
                            "page_number": c.page_number,
                            "chunk_index": c.chunk_index,
                            "score": c.score,
                            "content_snippet": c.content_snippet,
                        }
                        for c in response.citations
                    ],
                ),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(**response.token_usage),
    )


async def _stream_response(
    call_id: str,
    request: ChatCompletionRequest,
    pipeline: RAGPipeline,
    messages: list[dict],
    session: AsyncSession,
    conversation_id: str,
):
    """Generate SSE stream for chat response."""
    try:
        # Send initial chunk
        yield f"data: {json.dumps({'id': call_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

        async for chunk in _query_stream_with_compat(
            pipeline, request, messages, session, conversation_id
        ):
            yield f"data: {json.dumps({'id': call_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"

        # Send finish with citations
        citations = getattr(pipeline, "_last_stream_citations", [])
        citation_dicts = [
            {
                "doc_id": c.doc_id,
                "filename": c.filename,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "score": c.score,
                "content_snippet": c.content_snippet,
            }
            for c in citations
        ]
        yield f"data: {json.dumps({'id': call_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'citations': citation_dicts}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

        await session.commit()

    except Exception as e:
        await session.rollback()
        logger.exception("Stream error")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def _is_signature_compat_error(e: TypeError) -> bool:
    text = str(e)
    return (
        "unexpected keyword argument 'use_hybrid_search'" in text
        or "unexpected keyword argument 'use_rerank'" in text
    )


async def _query_with_compat(
    pipeline: RAGPipeline,
    request: ChatCompletionRequest,
    messages: list[dict],
    session: AsyncSession,
    conversation_id: str,
):
    question = messages[-1]["content"]
    history = messages[:-1] if len(messages) > 1 else None

    try:
        return await pipeline.query(
            question=question,
            conversation_history=history,
            collection=request.collection,
            top_k=request.retrieval.top_k,
            use_hybrid_search=request.retrieval.hybrid_search,
            use_rerank=request.retrieval.rerank,
            stream=False,
            conversation_id=conversation_id,
            db_session=session,
        )
    except TypeError as e:
        if not _is_signature_compat_error(e):
            raise
        logger.warning(
            "Pipeline query signature fallback",
            error=str(e),
        )
        return await pipeline.query(
            question=question,
            conversation_history=history,
            collection=request.collection,
            top_k=request.retrieval.top_k,
            stream=False,
            conversation_id=conversation_id,
            db_session=session,
        )
    except Exception:
        logger.exception("RAG query failed")
        raise


async def _query_stream_with_compat(
    pipeline: RAGPipeline,
    request: ChatCompletionRequest,
    messages: list[dict],
    session: AsyncSession,
    conversation_id: str,
):
    question = messages[-1]["content"]
    history = messages[:-1] if len(messages) > 1 else None

    try:
        async for chunk in pipeline.query_stream(
            question=question,
            conversation_history=history,
            collection=request.collection,
            top_k=request.retrieval.top_k,
            use_hybrid_search=request.retrieval.hybrid_search,
            use_rerank=request.retrieval.rerank,
            conversation_id=conversation_id,
            db_session=session,
        ):
            yield chunk
    except TypeError as e:
        if not _is_signature_compat_error(e):
            raise
        logger.warning(
            "Pipeline query_stream signature fallback",
            error=str(e),
        )
        async for chunk in pipeline.query_stream(
            question=question,
            conversation_history=history,
            collection=request.collection,
            top_k=request.retrieval.top_k,
            conversation_id=conversation_id,
            db_session=session,
        ):
            yield chunk
