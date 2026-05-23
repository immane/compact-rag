"""FastAPI application factory and route registration."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from compact_rag.common.exceptions import CompactRAGException, get_http_status
from compact_rag.common.logger import get_logger, setup_logging

logger = get_logger(__name__)


def create_app(settings=None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Settings instance. If None, loaded from defaults.

    Returns:
        Configured FastAPI application.
    """
    use_provided_settings = settings is not None

    if settings is None:
        from compact_rag.config.settings import get_settings

        settings = get_settings()

    # Setup logging
    setup_logging(
        log_level=settings.log_level,
        json_format=settings.log_level == "WARNING",
    )

    app = FastAPI(
        title="compact-rag",
        description="Enterprise RAG System API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    if use_provided_settings:
        from compact_rag.api.deps import get_settings as get_settings_dep

        app.dependency_overrides[get_settings_dep] = lambda: settings

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler
    @app.exception_handler(CompactRAGException)
    async def compact_rag_exception_handler(
        request: Request, exc: CompactRAGException
    ):
        status_code = get_http_status(exc)
        logger.warning(
            f"Exception: {exc.__class__.__name__}",
            message=str(exc),
            request_id=exc.request_id,
        )
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": exc.__class__.__name__,
                    "message": str(exc),
                    "details": exc.details,
                    "request_id": exc.request_id,
                }
            },
        )

    # Register routers
    from compact_rag.api.routers.system import router as system_router
    from compact_rag.api.routers.collections import router as collections_router
    from compact_rag.api.routers.documents import router as documents_router
    from compact_rag.api.routers.chat import router as chat_router
    from compact_rag.api.routers.conversations import router as conversations_router
    from compact_rag.api.routers.ingestion import router as ingestion_router
    from compact_rag.api.routers.api_keys import router as api_keys_router

    app.include_router(system_router, prefix="/v1")
    app.include_router(collections_router, prefix="/v1")
    app.include_router(documents_router, prefix="/v1")
    app.include_router(chat_router, prefix="/v1")
    app.include_router(conversations_router, prefix="/v1")
    app.include_router(ingestion_router, prefix="/v1")
    app.include_router(api_keys_router, prefix="/v1")

    return app


# Module-level app for direct uvicorn usage
app = create_app()
