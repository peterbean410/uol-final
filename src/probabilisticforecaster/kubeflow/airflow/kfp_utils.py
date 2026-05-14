"""KFP client utilities with retry logic for Airflow integration.

Provides the KFPTrigger class that wraps the KFP SDK client to submit
pipeline runs, poll for completion, and check for existing retrain runs.
Used by the Airflow DAGs to trigger and monitor Kubeflow pipeline executions.

Requirements: 7.3, 7.4, 7.5
"""

import logging
import time
from typing import Literal

import tenacity
from kfp.client import Client

logger = logging.getLogger(__name__)

# KFP run statuses considered terminal
_TERMINAL_SUCCEEDED = {"Succeeded", "Completed"}
_TERMINAL_FAILED = {"Failed", "Error"}
_TERMINAL_SKIPPED = {"Skipped"}
_TERMINAL_STATUSES = _TERMINAL_SUCCEEDED | _TERMINAL_FAILED | _TERMINAL_SKIPPED

# Status mapping: KFP status → Airflow task state
_STATUS_MAP: dict[str, str] = {
    "Succeeded": "success",
    "Completed": "success",
    "Failed": "failed",
    "Error": "failed",
    "Skipped": "skipped",
}

# Default KFP host (in-cluster service address)
DEFAULT_KFP_HOST = "http://ml-pipeline.kubeflow.svc.cluster.local:8888"

# Polling interval in seconds
_POLL_INTERVAL_SECONDS = 60


class KFPTrigger:
    """Triggers and monitors KFP pipeline runs with retry logic.

    Wraps the KFP SDK client to:
    1. Submit pipeline runs with parameters (symbol, snapshot_date, s3_data_path,
       training_mode)
    2. Retry submission 3 times with exponential backoff (via tenacity)
    3. Poll run status until terminal state, mapping KFP statuses to Airflow
       task states
    4. Check if a scratch retrain run is already running/completed today for a
       given symbol

    Requirements: 7.3, 7.4, 7.5
    """

    def __init__(self, kfp_host: str = DEFAULT_KFP_HOST) -> None:
        """Initialize the KFP trigger with a client connection.

        Args:
            kfp_host: URL of the KFP API server.
        """
        self.client = Client(host=kfp_host)
        self._current_symbol: str = ""

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
        retry=tenacity.retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def submit_run(
        self,
        symbol: str,
        snapshot_date: str,
        s3_data_path: str,
        training_mode: Literal["scratch", "finetune"] = "scratch",
    ) -> str:
        """Submit a KFP pipeline run with the given parameters.

        Retries up to 3 times with exponential backoff if the KFP API is
        unreachable (ConnectionError, TimeoutError, OSError).

        Args:
            symbol: Trading pair symbol (e.g., "USDJPY").
            snapshot_date: ISO date string for the data snapshot (e.g., "2026-05-13").
            s3_data_path: S3 key path to the EOH snapshot parquet file.
            training_mode: Either "scratch" (full retrain) or "finetune" (incremental).

        Returns:
            The KFP run ID string.

        Raises:
            ConnectionError: If the KFP API is unreachable after 3 retries.
            TimeoutError: If the KFP API times out after 3 retries.

        Requirements: 7.3, 7.4
        """
        from probabilisticforecaster.kubeflow.pipeline.pipeline import (
            forecaster_pipeline,
        )

        self._current_symbol = symbol

        logger.info(
            "Submitting KFP pipeline run: symbol=%s, snapshot_date=%s, "
            "training_mode=%s",
            symbol,
            snapshot_date,
            training_mode,
        )

        run = self.client.create_run_from_pipeline_func(
            forecaster_pipeline,
            arguments={
                "symbol": symbol,
                "forecast_horizon": 1,
                "snapshot_date": snapshot_date,
                "s3_data_path": s3_data_path,
                "training_mode": training_mode,
            },
        )

        logger.info("KFP pipeline run submitted: run_id=%s", run.run_id)
        return run.run_id

    def poll_run_status(
        self,
        run_id: str,
        timeout_seconds: int = 28800,
        poll_interval: int = _POLL_INTERVAL_SECONDS,
    ) -> str:
        """Poll KFP API until the run reaches a terminal state.

        Maps KFP run statuses to Airflow task states:
          - "Succeeded" / "Completed" → "success"
          - "Failed" / "Error" → "failed"
          - "Skipped" → "skipped"

        If the timeout is reached before a terminal state, returns "failed".

        Args:
            run_id: The KFP run ID to poll.
            timeout_seconds: Maximum time to wait in seconds (default: 8 hours).
            poll_interval: Seconds between status checks (default: 60).

        Returns:
            Airflow task state string: "success", "failed", or "skipped".

        Requirements: 7.5
        """
        deadline = time.time() + timeout_seconds

        logger.info(
            "Polling KFP run %s (timeout=%ds, interval=%ds)",
            run_id,
            timeout_seconds,
            poll_interval,
        )

        while time.time() < deadline:
            run = self.client.get_run(run_id)
            status = run.run.status

            if status in _TERMINAL_STATUSES:
                airflow_state = _STATUS_MAP.get(status, "failed")
                logger.info(
                    "KFP run %s reached terminal state: %s → airflow=%s",
                    run_id,
                    status,
                    airflow_state,
                )
                return airflow_state

            logger.debug("KFP run %s status: %s (still running)", run_id, status)
            time.sleep(poll_interval)

        logger.warning("KFP run %s timed out after %ds", run_id, timeout_seconds)
        return "failed"

    def has_retrain_run_today(self, date_str: str) -> bool:
        """Check if a scratch retrain run exists for the current symbol today.

        Queries the KFP API for recent pipeline runs with training_mode='scratch'
        matching the current symbol. Returns True if any such run is currently
        running OR was created on the given date (regardless of final status).

        This is used by the weekly fine-tune DAG to skip fine-tuning when a
        full retrain has already been triggered (drift-triggered retrain
        supersedes fine-tuning).

        Args:
            date_str: ISO date string to check (e.g., "2026-05-13").

        Returns:
            True if a scratch retrain run is active or was created today
            for the current symbol, False otherwise.
        """
        try:
            runs = self.client.list_runs(
                filter='run.pipeline_name="forecaster-pipeline"',
                sort_by="created_at desc",
                page_size=20,
            )
        except Exception as e:
            logger.warning(
                "Failed to query KFP runs for retrain check: %s", e
            )
            return False

        for run in runs.runs or []:
            params = {
                p.name: p.value
                for p in (run.pipeline_spec.parameters or [])
            }

            # Only interested in scratch retrain runs
            if params.get("training_mode") != "scratch":
                continue

            # Must match the current symbol
            if params.get("symbol") != self._current_symbol:
                continue

            # Block if run is still active (regardless of creation date)
            if run.run.status in ("Running", "Pending"):
                logger.info(
                    "Active scratch retrain found for %s (run_id=%s, status=%s)",
                    self._current_symbol,
                    run.run_id,
                    run.run.status,
                )
                return True

            # Block if a scratch run was created today (completed or failed)
            created = run.created_at.strftime("%Y-%m-%d")
            if created == date_str:
                logger.info(
                    "Scratch retrain already ran today for %s (run_id=%s, "
                    "created=%s, status=%s)",
                    self._current_symbol,
                    run.run_id,
                    created,
                    run.run.status,
                )
                return True

        return False
