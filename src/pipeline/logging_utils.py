"""Structured logging utility supporting JSON-Lines output and colored console messages."""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

try:
    from rich.console import Console
    from rich.logging import RichHandler
    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False


class JSONLinesFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_entry.update(record.extra)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True,
) -> logging.Logger:
    """Configures root and pipeline logging."""
    num_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(num_level)
    root_logger.handlers.clear()

    # Console handler (Rich or standard stream)
    if HAVE_RICH:
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
    
    console_handler.setLevel(num_level)
    root_logger.addHandler(console_handler)

    # Optional file handler with JSONL formatting
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        if structured:
            file_handler.setFormatter(JSONLinesFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
        file_handler.setLevel(num_level)
        root_logger.addHandler(file_handler)

    return logging.getLogger("pipeline")
