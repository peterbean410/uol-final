"""DQNPF KFP client utilities with retry logic for Airflow integration.

Provides the DqnpfKFPTrigger class that wraps the KFP SDK client to submit
dqnpf-intraday pipeline runs, poll for completion, and check parent model
availability in the Model Registry.

Also provides the `check_parent_models_available` precondition function used
by both DAGs to fail fast when either parent model has no production version.

Used by the dqnpf-intraday Airflow DAGs to trigger and monitor Kubeflow
pipeline executions.

Requirements: 23.5, 23.6, 23.7
"""

import logging
import time

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


class DqnpfKFPTrigger:
    """Triggers and monitors dqnpf-intraday KFP pipeline runs with retry logic.

    Wraps the KFP SDK client to:
    1. Submit dqnpf-intraday pipeline runs with parameters
       (integration_config_yaml, dqn_model_registry_name,
       forecaster_model_registry_name)
    2. Retry submission 3 times with exponential backoff (via tenacity)
    3. Poll run status until terminal state, mapping KFP statuses to Airflow
       task states

    Requirements: 23.5, 23.6, 23.7
    """

    def __init__(self, kfp_host: str = DEFAULT_KFP_HOST) -> None:
        """Initialize the DQNPF KFP trigger with a client connection.

        Args:
            kfp_host: URL of the KFP API server.
        """
        self.client = Client(host=kfp_host)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
        retry=tenacity.retry_if_exception_type(
            (ConnectionError, TimeoutError, OSError)
        ),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def submit_run(
        self,
        integration_config_yaml: str,
        dqn_model_registry_name: str = "deepqnetwork-usdjpy",
        forecaster_model_registry_name: str = "probabilistic-transformer-usdjpy-h1",
        episode_start_ts: int | None = None,
        episode_end_ts: int | None = None,
    ) -> str:
        """Submit a dqnpf-intraday KFP pipeline run with the given parameters.

        Retries up to 3 times with exponential backoff if the KFP API is
        unreachable (ConnectionError, TimeoutError, OSError).

        Args:
            integration_config_yaml: Path to the integration config YAML
                (mounted in the pipeline pod).
            dqn_model_registry_name: Model Registry name for the DQN model.
            forecaster_model_registry_name: Model Registry name for the
                Forecaster model.
            episode_start_ts: Optional episode start timestamp (Unix seconds).
                Used by the revalidate DAG to specify a custom episode range.
            episode_end_ts: Optional episode end timestamp (Unix seconds).
                Used by the revalidate DAG to specify a custom episode range.

        Returns:
            The KFP run ID string.

        Raises:
            ConnectionError: If the KFP API is unreachable after 3 retries.
            TimeoutError: If the KFP API times out after 3 retries.

        Requirements: 23.5, 23.6
        """
        from tradingmodel.intraday.dqnpf.kubeflow.pipeline.dqnpf_pipeline import (
            dqnpf_intraday_pipeline,
        )

        logger.info(
            "Submitting dqnpf-intraday KFP pipeline run: "
            "config=%s, dqn_registry=%s, forecaster_registry=%s",
            integration_config_yaml,
            dqn_model_registry_name,
            forecaster_model_registry_name,
        )

        arguments = {
            "integration_config_yaml": integration_config_yaml,
            "dqn_model_registry_name": dqn_model_registry_name,
            "forecaster_model_registry_name": forecaster_model_registry_name,
        }

        if episode_start_ts is not None:
            arguments["episode_start_ts"] = episode_start_ts
        if episode_end_ts is not None:
            arguments["episode_end_ts"] = episode_end_ts

        run = self.client.create_run_from_pipeline_func(
            dqnpf_intraday_pipeline,
            arguments=arguments,
        )

        logger.info(
            "dqnpf-intraday KFP pipeline run submitted: run_id=%s", run.run_id
        )
        return run.run_id

    def poll_run_status(
        self,
        run_id: str,
        timeout_seconds: int = 14400,
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
            timeout_seconds: Maximum time to wait in seconds (default: 4 hours).
            poll_interval: Seconds between status checks (default: 60).

        Returns:
            Airflow task state string: "success", "failed", or "skipped".

        Requirements: 23.5
        """
        deadline = time.time() + timeout_seconds

        logger.info(
            "Polling dqnpf-intraday KFP run %s (timeout=%ds, interval=%ds)",
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
                    "dqnpf-intraday KFP run %s reached terminal state: "
                    "%s → airflow=%s",
                    run_id,
                    status,
                    airflow_state,
                )
                return airflow_state

            logger.debug(
                "dqnpf-intraday KFP run %s status: %s (still running)",
                run_id,
                status,
            )
            time.sleep(poll_interval)

        logger.warning(
            "dqnpf-intraday KFP run %s timed out after %ds",
            run_id,
            timeout_seconds,
        )
        return "failed"


# ---------------------------------------------------------------------------
# Precondition check (extracted as testable function)
# ---------------------------------------------------------------------------

# Default parent model registry names
DEFAULT_DQN_MODEL_REGISTRY_NAME = "deepqnetwork-usdjpy"
DEFAULT_FORECASTER_MODEL_REGISTRY_NAME = "probabilistic-transformer-usdjpy-h1"


def check_parent_models_available(
    dqn_model_registry_name: str = DEFAULT_DQN_MODEL_REGISTRY_NAME,
    forecaster_model_registry_name: str = DEFAULT_FORECASTER_MODEL_REGISTRY_NAME,
) -> None:
    """Check that both parent models have production-stage versions.

    Queries the Model Registry for the latest production-stage version of
    both the DQN and Forecaster models. Raises RuntimeError if either parent
    has no production version, failing fast before submitting the KFP run.

    Args:
        dqn_model_registry_name: Registry name for the DQN model.
        forecaster_model_registry_name: Registry name for the Forecaster model.

    Raises:
        RuntimeError: If either parent model has no production version.

    Requirements: 23.6
    """
    from tradingmodel.intraday.dqnpf.kubeflow.registry.registry_client import (
        resolve_production_checkpoint,
    )

    errors: list[str] = []

    try:
        resolve_production_checkpoint(dqn_model_registry_name, "production")
    except ValueError as e:
        errors.append(
            f"DQN parent model '{dqn_model_registry_name}' has no production "
            f"version: {e}"
        )

    try:
        resolve_production_checkpoint(
            forecaster_model_registry_name, "production"
        )
    except ValueError as e:
        errors.append(
            f"Forecaster parent model '{forecaster_model_registry_name}' has "
            f"no production version: {e}"
        )

    if errors:
        raise RuntimeError(
            "Precondition failed, parent models not available:\n"
            + "\n".join(f"  - {err}" for err in errors)
        )
