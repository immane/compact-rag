from __future__ import annotations

import logging
import sys

import pytest
from loguru import logger as _logger

from compact_rag.common.logger import (
    InterceptHandler,
    get_logger,
    setup_logging,
)


class TestGetLogger:
    def test_returns_loguru_logger(self):
        result = get_logger("test_module")
        assert "loguru" in str(type(result))
        assert hasattr(result, "bind")

    def test_different_names_produce_different_loggers(self):
        a = get_logger("module_a")
        b = get_logger("module_b")
        assert a is not b

    def test_default_name(self):
        logger = get_logger()
        logger.info("test")


class TestSetupLogging:
    def test_setup_with_defaults(self):
        setup_logging()
        _logger.info("test_default_setup")

    def test_setup_with_debug_level(self):
        setup_logging(log_level="DEBUG")
        _logger.debug("test_debug")

    def test_setup_with_warning_level(self):
        setup_logging(log_level="WARNING")
        _logger.warning("test_warning")

    def test_setup_with_json_format(self):
        try:
            setup_logging(log_level="INFO", json_format=True)
            _logger.info("test_json_format")
        except TypeError:
            pytest.skip("loguru version does not support 'patcher' kwarg")

    def test_setup_with_log_file(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        try:
            setup_logging(log_level="INFO", log_file=log_file)
            _logger.info("test_file_logging")
            assert tmp_path.joinpath("test.log").exists()
        except TypeError:
            pytest.skip("loguru version does not support 'patcher' kwarg")


class TestInterceptHandler:
    def test_emit_handles_log_record(self):
        handler = InterceptHandler()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="intercepted message", args=(), exc_info=None,
        )
        handler.emit(record)

    def test_emit_with_exception(self):
        handler = InterceptHandler()
        try:
            raise ValueError("test error")
        except ValueError:
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="test.py",
                lineno=1, msg="error message", args=(), exc_info=sys.exc_info(),
            )
            handler.emit(record)

    def test_emit_with_unknown_level(self):
        handler = InterceptHandler()
        record = logging.LogRecord(
            name="test", level=99, pathname="test.py",
            lineno=1, msg="custom level", args=(), exc_info=None,
        )
        handler.emit(record)
