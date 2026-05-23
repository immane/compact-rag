"""Structured logging system based on loguru."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger as _logger


class InterceptHandler(logging.Handler):
    """Intercept standard library logging and redirect to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        _logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _api_key_patcher(record: dict) -> None:
    """Remove API keys from log records in production."""
    if record.get("extra", {}).get("api_key"):
        record["extra"]["api_key"] = "***"


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    json_format: bool = False,
) -> None:
    """Configure loguru global logger.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for log output.
        json_format: If True, output JSON-formatted logs (production mode).
    """
    _logger.remove()

    # Console sink
    if json_format:
        _logger.add(
            sys.stderr,
            level=log_level,
            serialize=True,
            format="{time} {level} {name} {message} {extra}",
            patcher=_api_key_patcher,
        )
    else:
        _logger.add(
            sys.stderr,
            level=log_level,
            colorize=True,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
        )

    # File sink (optional, always JSON)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            log_path,
            level=log_level,
            serialize=True,
            rotation="100 MB",
            retention="7 days",
            compression="gz",
            patcher=_api_key_patcher,
        )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Suppress noisy third-party loggers
    for lib in ("chromadb", "sqlalchemy.engine", "httpx", "urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str = "compact_rag"):
    """Get a logger instance bound with the given module name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        A loguru logger with context binding.
    """
    return _logger.bind(name=name)
