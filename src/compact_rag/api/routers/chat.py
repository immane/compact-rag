"""Chat completions endpoint — OpenAI-compatible API."""

from __future__ import annotations

import json
import time

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

logger = get_logger(__name__)
router = APIRouter(tags=["Chat"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    session: AsyncSession = Depends(get_db_session),
):
    """Core Q&A endpoint — compatible with OpenAI Chat Completions API."""
    call_id = f"rag-{int(time.time() * 1000)}"

    messages = [
        {"role": m.role, "content": m.content} for m in request.messages
    ]

    if request.stream:
        return StreamingResponse(
            _stream_response(call_id, request, pipeline, messages),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    response = await _query_with_compat(pipeline, request, messages)

    return ChatCompletionResponse(
        id=call_id,
        created=int(time.time()),
        model=request.model,
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
):
    """Generate SSE stream for chat response."""
    try:
        # Send initial chunk
        yield f"data: {json.dumps({'id': call_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

        async for chunk in _query_stream_with_compat(pipeline, request, messages):
            yield f"data: {json.dumps({'id': call_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"

        # Send finish
        yield f"data: {json.dumps({'id': call_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
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
        )
    except Exception:
        logger.exception("RAG query failed")
        raise


async def _query_stream_with_compat(
    pipeline: RAGPipeline,
    request: ChatCompletionRequest,
    messages: list[dict],
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
        ):
            yield chunk
