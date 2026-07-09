"""
Logging configuration.

Uses stdlib logging with a structured, single-line formatter so logs are easy
to grep locally and easy to ship to a log aggregator (e.g. CloudWatch, Loki)
in production without changes.
"""
import logging
import sys

REQUEST_ID_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | request_id=%(request_id)s | %(message)s"
)
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class RequestIdFilter(logging.Filter):
    """Injects a default request_id when one isn't bound via the logging adapter."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def configure_logging(log_level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(REQUEST_ID_LOG_FORMAT))
    handler.addFilter(RequestIdFilter())

    # Avoid duplicate handlers on reload
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "faiss", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
