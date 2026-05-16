"""DQN pipeline monitoring and alerting utilities.

Provides a DQN-specific alert webhook function for sending pipeline failure
notifications. Wraps the shared alert_webhook infrastructure from the
probabilisticforecaster monitoring module with DQN pipeline context.

The DQN alert webhook sends a JSON payload containing:
  - run_id: The unique identifier of the failed DQN pipeline run
  - failed_component: The DQN component that failed (e.g. "dqn_training", "dqn_backtest")
  - error_message: The error message describing the failure
  - timestamp: ISO 8601 UTC timestamp of the failure event
"""

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any


def dqn_alert_webhook(
    run_id: str,
    failed_component: str,
    error_message: str,
    webhook_url: str | None = None,
) -> int:
    """Send an alert notification via webhook on DQN pipeline failure.

    Posts a JSON payload to the specified webhook URL containing details about
    the DQN pipeline failure. Supports Slack and PagerDuty webhook formats.

    The webhook URL can be provided directly or read from the
    ``ALERT_WEBHOOK_URL`` environment variable.

    Args:
        run_id: The unique identifier of the failed DQN pipeline run.
        failed_component: The name of the DQN component that failed
            (e.g. "dqn_training", "dqn_backtest").
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
