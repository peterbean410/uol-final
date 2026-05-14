"""Structured JSON logging and alerting for Kubeflow pipeline components.

Provides a JSON formatter and logger factory that outputs structured log lines
to stdout, suitable for collection by the cluster logging stack (e.g. Fluent Bit,
Loki, or CloudWatch agent).

Each log line is a single valid JSON object with fields:
  - timestamp: ISO 8601 UTC timestamp
  - level: log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - logger: logger name
  - message: the formatted log message
  - component: pipeline component name (e.g. "data_preparation", "model_training")

When an exception is logged, an additional "exception" field contains the
formatted traceback as a string.

Also provides an alert_webhook function for sending pipeline failure notifications
to configurable webhooks (Slack or PagerDuty).
"""

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON for cluster log collection.

    Handles special characters, unicode, and exception tracebacks without
    breaking JSON validity. The ``json.dumps`` call with ``ensure_ascii=False``
    preserves unicode characters while still producing valid JSON output.
    """

    def __init__(self, component: str = "unknown") -> None:
        """Initialise the formatter with a component name.

        Args:
            component: The pipeline component name to include in every log line.
        """
        super().__init__()
        self._component = component

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A single-line JSON string representing the log entry.
        """
        log_entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "component": getattr(record, "component", self._component),
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def get_logger(name: str, component: str = "unknown") -> logging.Logger:
    """Create and return a logger configured with structured JSON output to stdout.

    If the logger already has handlers (e.g. from a previous call), this function
    returns it as-is to avoid duplicate log lines.

    Args:
        name: The logger name (typically ``__name__`` of the calling module).
        component: The pipeline component name (e.g. "data_preparation").

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls.
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(StructuredJsonFormatter(component=component))

    logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate output.
    logger.propagate = False

    return logger


def alert_webhook(
    run_id: str,
    failed_component: str,
    error_message: str,
    webhook_url: str | None = None,
) -> int:
    """Send an alert notification via a configurable webhook on pipeline failure.

    Posts a JSON payload to the specified webhook URL containing details about
    the pipeline failure. Supports Slack and PagerDuty webhook formats.

    The webhook URL can be provided directly or read from the
    ``ALERT_WEBHOOK_URL`` environment variable.

    Args:
        run_id: The unique identifier of the failed pipeline run.
        failed_component: The name of the component that failed.
        error_message: The error message describing the failure.
        webhook_url: The URL of the webhook endpoint (Slack or PagerDuty).
            If None, reads from the ``ALERT_WEBHOOK_URL`` environment variable.

    Returns:
        The HTTP status code from the webhook endpoint.

    Raises:
        ValueError: If no webhook URL is configured.
        urllib.error.URLError: If the HTTP POST fails due to network issues.
    """
    url = webhook_url or os.environ.get("ALERT_WEBHOOK_URL", "")
    if not url:
        raise ValueError(
            "No webhook URL configured. Set ALERT_WEBHOOK_URL env var or pass webhook_url."
        )

    payload: dict[str, Any] = {
        "run_id": run_id,
        "failed_component": failed_component,
        "error_message": error_message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        return response.status
