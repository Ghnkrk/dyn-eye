"""
Structured logging for DYN-EYE pipeline.
Provides per-module loggers with console + file output.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_file_handler = logging.FileHandler(
    _LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log",
    encoding="utf-8",
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))

# Windows' default console/redirected-stdout encoding (cp1252) can't encode
# characters some log messages use (checkmarks, box-drawing banners), which
# crashes the handler mid-pipeline-run. Force UTF-8 where supported.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass  # stdout doesn't support reconfigure (e.g. some captured streams)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for the given module name."""
    logger = logging.getLogger(f"dyneye.{name}")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_file_handler)
        logger.addHandler(_console_handler)
        logger.propagate = False
    return logger
