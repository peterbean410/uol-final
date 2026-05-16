"""Airflow DAGs: Trigger Kubeflow dqnpf-intraday Pipeline runs.

Two DAGs orchestrate the dqnpf-intraday ML pipeline via Kubeflow Pipelines:

1. dqnpf_weekly_backtest, Weekly Sunday 03:00 UTC walk-forward backtest.
   No auto-promotion; submits the dqnpf-intraday-pipeline for the configured
   walk-forward window. Precondition: both parent models (DQN and Forecaster)
   must have production-stage versions in the Model Registry.

2. dqnpf_revalidate; No schedule (triggered manually by operator with
   specified episode_start_ts and episode_end_ts). Submits a backtest with
   operator-specified episode ranges. Same precondition as above.

Both DAGs fail fast if either parent model has no production version.

Requirements: 23.5, 23.6, 23.7
"""

import calendar
import logging
from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# KFP API endpoint (in-cluster service address)
_KFP_HOST = "http://ml-pipeline.kubeflow.svc.cluster.local:8888"

# Parent model registry names
_DQN_MODEL_REGISTRY_NAME = "deepqnetwork-usdjpy"
_FORECASTER_MODEL_REGISTRY_NAME = "probabilisticforecaster-usdjpy"

# Walk-forward window: 30 days of episode data for backtest
_WALKFORWARD_DAYS = 30

# Integration config YAML path (mounted in pipeline pod)
_INTEGRATION_CONFIG_YAML = (
    "/etc/dqnpf/config/dqnpf_pipeline_config.yaml"
)

default_args = {
    "owner": "ml-engineering",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ---------------------------------------------------------------------------
# DAG 1: Weekly Walk-Forward Backtest
# ---------------------------------------------------------------------------

with DAG(
    dag_id="dqnpf_weekly_backtest",
    start_date=datetime(2026, 5, 1),
    schedule="0 3 * * 0",  # Every Sunday at 03:00 UTC
    default_args=default_args,
    catchup=False,
    tags=["dqnpf", "kubeflow", "backtest"],
) as weekly_backtest_dag:

    def _check_parents(**context):
        """Fail fast if either parent model has no production version."""
        from tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils import (
            check_parent_models_available,
        )

        check_parent_models_available()

    check_parents = PythonOperator(
        task_id="check_parent_models",
        python_callable=_check_parents,
    )

    def _submit_weekly_backtest(**context):
        """Submit dqnpf-intraday pipeline run for the walk-forward window."""
        from tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils import (
            DqnpfKFPTrigger,
        )

        trigger = DqnpfKFPTrigger(kfp_host=_KFP_HOST)
        run_id = trigger.submit_run(
            integration_config_yaml=_INTEGRATION_CONFIG_YAML,
            dqn_model_registry_name=_DQN_MODEL_REGISTRY_NAME,
            forecaster_model_registry_name=_FORECASTER_MODEL_REGISTRY_NAME,
        )
        status = trigger.poll_run_status(run_id, timeout_seconds=14400)
        if status != "success":
            raise RuntimeError(
                f"dqnpf-intraday KFP backtest run {run_id} ended with "
                f"status: {status}"
            )

    submit_backtest = PythonOperator(
        task_id="submit_weekly_backtest",
        python_callable=_submit_weekly_backtest,
    )

    check_parents >> submit_backtest


# ---------------------------------------------------------------------------
# DAG 2: Manual Revalidation (operator-triggered)
# ---------------------------------------------------------------------------

with DAG(
    dag_id="dqnpf_revalidate",
    start_date=datetime(2026, 5, 1),
    schedule=None,  # Triggered manually by operator
    default_args=default_args,
    catchup=False,
    tags=["dqnpf", "kubeflow", "revalidate"],
    params={
        "episode_start_ts": 0,
        "episode_end_ts": 0,
    },
) as revalidate_dag:

    def _check_parents_revalidate(**context):
        """Fail fast if either parent model has no production version."""
        from tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils import (
            check_parent_models_available,
        )

        check_parent_models_available()

    check_parents_revalidate = PythonOperator(
        task_id="check_parent_models",
        python_callable=_check_parents_revalidate,
    )

    def _submit_revalidate(**context):
        """Submit dqnpf-intraday pipeline run with operator-specified episode range.

        The operator provides episode_start_ts and episode_end_ts as DAG params
        at trigger time. These are forwarded to the KFP pipeline submission.

        Requirements: 23.6
        """
        from tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils import (
            DqnpfKFPTrigger,
        )

        params = context["params"]
        episode_start_ts = params["episode_start_ts"]
        episode_end_ts = params["episode_end_ts"]

        if episode_start_ts <= 0 or episode_end_ts <= 0:
            raise ValueError(
                "episode_start_ts and episode_end_ts must be positive Unix "
                f"timestamps. Got start={episode_start_ts}, end={episode_end_ts}"
            )

        if episode_end_ts <= episode_start_ts:
            raise ValueError(
                "episode_end_ts must be greater than episode_start_ts. "
                f"Got start={episode_start_ts}, end={episode_end_ts}"
            )

        trigger = DqnpfKFPTrigger(kfp_host=_KFP_HOST)
        run_id = trigger.submit_run(
            integration_config_yaml=_INTEGRATION_CONFIG_YAML,
            dqn_model_registry_name=_DQN_MODEL_REGISTRY_NAME,
            forecaster_model_registry_name=_FORECASTER_MODEL_REGISTRY_NAME,
            episode_start_ts=episode_start_ts,
            episode_end_ts=episode_end_ts,
        )
        status = trigger.poll_run_status(run_id, timeout_seconds=14400)
        if status != "success":
            raise RuntimeError(
                f"dqnpf-intraday KFP revalidate run {run_id} ended with "
                f"status: {status}"
            )

    submit_revalidate = PythonOperator(
        task_id="submit_revalidate",
        python_callable=_submit_revalidate,
    )

    check_parents_revalidate >> submit_revalidate
