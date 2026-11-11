"""
AutoEdit Pro — Structured Logging
Rich console + JSON file logging with structlog.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

from autoedit.config import get_settings


# Install rich traceback handler
install_rich_traceback(show_locals=True, width=120)


def setup_logging(
    level: str | int | None = None,
    log_file: Path | None = None,
    json_format: bool | None = None,
    rich_console: bool | None = None,
) -> structlog.BoundLogger:
    """
    Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (JSON lines)
        json_format: Use JSON format for file output
        rich_console: Use rich formatting for console
    """
    settings = get_settings()
    log_config = settings.logging

    # Resolve parameters
    level = level or log_config.level
    log_file = log_file or log_config.file
    json_format = json_format if json_format is not None else log_config.format == "json"
    rich_console = rich_console if rich_console is not None else log_config.rich_console

    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure standard library logging
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                markup=True,
                show_time=True,
                show_path=False,
            ) if rich_console else logging.StreamHandler(sys.stderr)
        ],
    )

    # Configure file handler for JSON logs
    if json_format and log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_config.max_bytes,
            backupCount=log_config.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO", utc=True),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            _json_renderer if json_format else structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()


def _json_renderer(logger: Any, method_name: str, event_dict: dict[str, Any]) -> str:
    """Custom JSON renderer with orjson for speed."""
    import orjson
    return orjson.dumps(event_dict).decode("utf-8")


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin to add structured logging to any class."""

    @property
    def logger(self) -> structlog.BoundLogger:
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__module__ + "." + self.__class__.__name__)
        return self._logger


# Context managers for structured logging
class LogContext:
    """Context manager for adding structured context to logs."""

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.tokens: list[structlog.contextvars.BoundContextVars] = []

    def __enter__(self) -> LogContext:
        self.tokens.append(structlog.contextvars.bind_contextvars(**self.kwargs))
        return self

    def __exit__(self, *args: Any) -> None:
        for token in self.tokens:
            structlog.contextvars.unbind_contextvars(*self.kwargs.keys())


def log_step(step_name: str, **kwargs: Any):
    """Decorator to log function entry/exit with timing."""
    import functools
    import time

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **func_kwargs):
            logger = get_logger(func.__module__)
            start = time.perf_counter()
            logger.info("step_start", step=step_name, **kwargs)
            try:
                result = func(*args, **func_kwargs)
                duration = time.perf_counter() - start
                logger.info("step_complete", step=step_name, duration_ms=round(duration * 1000, 2))
                return result
            except Exception as e:
                duration = time.perf_counter() - start
                logger.error("step_failed", step=step_name, duration_ms=round(duration * 1000, 2), error=str(e))
                raise
        return wrapper
    return decorator


# Initialize logging on import
_setup_done = False


def ensure_logging_initialized() -> structlog.BoundLogger:
    """Ensure logging is initialized (call once at app startup)."""
    global _setup_done
    if not _setup_done:
        setup_logging()
        _setup_done = True
    return get_logger("autoedit")


# Convenience functions matching standard logging
def debug(msg: str, **kwargs: Any) -> None:
    get_logger().debug(msg, **kwargs)


def info(msg: str, **kwargs: Any) -> None:
    get_logger().info(msg, **kwargs)


def warning(msg: str, **kwargs: Any) -> None:
    get_logger().warning(msg, **kwargs)


def error(msg: str, **kwargs: Any) -> None:
    get_logger().error(msg, **kwargs)


def critical(msg: str, **kwargs: Any) -> None:
    get_logger().critical(msg, **kwargs)


def exception(msg: str, **kwargs: Any) -> None:
    get_logger().exception(msg, **kwargs)