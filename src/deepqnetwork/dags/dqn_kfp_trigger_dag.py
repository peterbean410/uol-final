"""Airflow DAGs: Trigger Kubeflow DQN Pipeline runs.

Three DAGs orchestrate the DQN ML pipeline via Kubeflow Pipelines:

1. dqn_weekly_retrain, Weekly Saturday 02:00 UTC full retrain from scratch
   on the latest tick data. Skips if a drift-triggered retrain is already
   running or completed for the same day.

2. dqn_drift_retrain; No schedule (triggered externally by performance
   drift detection). Retrains from scratch on the full episode window.

3. dqn_katib_tuning; No schedule (manual trigger). Submits a Katib
   hyperparameter tuning experiment CR to the cluster.

Each training DAG waits for the EOD tick snapshot to complete via
ExternalTaskSensor before submitting the KFP pipeline run.

Requirements: DQN-R21, DQN-R22, DQN-R23
"""

import calendar
from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.sensors.external_task import ExternalTaskSensor


_SYMBOLS = ["USDJPY", "AUDJPY"]

_KFP_HOST = "http://ml-pipeline.kubeflow.svc.cluster.local:8888"

_EPISODE_LOOKBACK_DAYS = 90

_STEP_SIZE_SECONDS = 5

default_args = {
    "owner": "ml-engineering",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _compute_episode_timestamps(logical_date: datetime) -> tuple[int, int]:
    """Compute episode start and end timestamps from the logical date.

    Args:
        logical_date: The Airflow logical date for the DAG run.

    Returns:
        Tuple of (episode_start_ts, episode_end_ts) as Unix timestamps.
        - episode_end_ts: end of the logical_date day (23:59:59 UTC)
        - episode_start_ts: 90 days before episode_end_ts
    """
    end_of_day = logical_date.replace(hour=23, minute=59, second=59, microsecond=0)
    episode_end_ts = int(calendar.timegm(end_of_day.timetuple()))

    start_day = end_of_day - timedelta(days=_EPISODE_LOOKBACK_DAYS)
    episode_start_ts = int(calendar.timegm(start_day.timetuple()))

    return episode_start_ts, episode_end_ts


with DAG(
    dag_id="dqn_weekly_retrain",
    start_date=datetime(2026, 5, 1),
    schedule="0 2 * * 6",
    default_args=default_args,
    catchup=False,
    tags=["dqn", "kubeflow", "retrain"],
) as weekly_retrain_dag:

    def _check_retrain_not_running(**context):
        """Skip weekly retrain if a scratch retrain is already running today.

        Queries the KFP API for any DQN pipeline runs with
        training_mode='scratch' for each configured symbol. If found running
        (any date) or created today, raises AirflowSkipException.
        """
        from airflow.exceptions import AirflowSkipException
        from deepqnetwork.dags.dqn_kfp_utils import DQNKFPTrigger

        trigger = DQNKFPTrigger(kfp_host=_KFP_HOST)
        today = context["logical_date"].strftime("%Y-%m-%d")
        for symbol in _SYMBOLS:
            trigger._current_symbol = symbol
            if trigger.has_retrain_run_today(today):
                raise AirflowSkipException(
                    f"DQN scratch retrain running/completed for {symbol} on "
                    f"{today}; skipping weekly retrain."
                )

    check_no_retrain = PythonOperator(
        task_id="check_retrain_not_running",
        python_callable=_check_retrain_not_running,
    )

    for symbol in _SYMBOLS:
        wait_for_snapshot = ExternalTaskSensor(
            task_id=f"wait_for_eod_tick_snapshot_{symbol.lower()}",
            external_dag_id="create_eod_tick_snapshot",
            external_task_id="create_eod_tick_snapshot",
            execution_date_fn=lambda dt: dt,
            poke_interval=120,
            timeout=3600,
            mode="reschedule",
        )

        def _submit_retrain(symbol=symbol, **context):
            """Submit KFP scratch retrain run for the given symbol."""
            from deepqnetwork.dags.dqn_kfp_utils import DQNKFPTrigger

            logical_date = context["logical_date"]
            episode_start_ts, episode_end_ts = _compute_episode_timestamps(
                logical_date
            )

            trigger = DQNKFPTrigger(kfp_host=_KFP_HOST)
            run_id = trigger.submit_run(
                symbol=symbol,
                episode_start_ts=episode_start_ts,
                episode_end_ts=episode_end_ts,
                step_size_seconds=_STEP_SIZE_SECONDS,
                training_mode="scratch",
            )
            status = trigger.poll_run_status(run_id, timeout_seconds=28800)
            if status != "success":
                raise RuntimeError(
                    f"DQN KFP retrain run {run_id} for {symbol} ended with "
                    f"status: {status}"
                )

        submit_retrain = PythonOperator(
            task_id=f"submit_retrain_{symbol.lower()}",
            python_callable=_submit_retrain,
        )

        check_no_retrain >> wait_for_snapshot >> submit_retrain


with DAG(
    dag_id="dqn_drift_retrain",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    default_args=default_args,
    catchup=False,
    tags=["dqn", "kubeflow", "retrain", "drift"],
) as drift_retrain_dag:

    for symbol in _SYMBOLS:
        wait_for_snapshot = ExternalTaskSensor(
            task_id=f"wait_for_eod_tick_snapshot_{symbol.lower()}",
            external_dag_id="create_eod_tick_snapshot",
            external_task_id="create_eod_tick_snapshot",
            execution_date_fn=lambda dt: dt,
            poke_interval=120,
            timeout=3600,
            mode="reschedule",
        )

        def _submit_drift_retrain(symbol=symbol, **context):
            """Submit KFP full retrain from scratch for the given symbol."""
            from deepqnetwork.dags.dqn_kfp_utils import DQNKFPTrigger

            logical_date = context["logical_date"]
            episode_start_ts, episode_end_ts = _compute_episode_timestamps(
                logical_date
            )

            trigger = DQNKFPTrigger(kfp_host=_KFP_HOST)
            run_id = trigger.submit_run(
                symbol=symbol,
                episode_start_ts=episode_start_ts,
                episode_end_ts=episode_end_ts,
                step_size_seconds=_STEP_SIZE_SECONDS,
                training_mode="scratch",
            )
            status = trigger.poll_run_status(run_id, timeout_seconds=28800)
            if status != "success":
                raise RuntimeError(
                    f"DQN KFP drift retrain run {run_id} for {symbol} ended "
                    f"with status: {status}"
                )

        submit_drift_retrain = PythonOperator(
            task_id=f"submit_retrain_{symbol.lower()}",
            python_callable=_submit_drift_retrain,
        )

        wait_for_snapshot >> submit_drift_retrain


with DAG(
    dag_id="dqn_katib_tuning",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    default_args=default_args,
    catchup=False,
    tags=["dqn", "kubeflow", "katib", "tuning"],
    params={
        "symbol": "USDJPY",
        "max_trials": 20,
        "parallel_trials": 3,
    },
) as katib_dag:

    def _submit_katib_experiment(**context):
        """Apply the DQN Katib Experiment CR and poll until completion.

        Reads symbol and trial budget from DAG params (overridable at trigger
        time). After the experiment completes, logs the best hyperparameters.
        """
        import time

        import yaml
        from kubernetes import client, config as k8s_config

        params = context["params"]
        symbol = params["symbol"]
        max_trials = params["max_trials"]
        parallel_trials = params["parallel_trials"]

        k8s_config.load_incluster_config()
        api = client.CustomObjectsApi()

        experiment_name = (
            f"dqn-tuning-{symbol.lower()}-{context['run_id'][:8]}"
        )

        with open(
            "deepqnetwork/kubeflow/katib/dqn_experiment.yaml"
        ) as f:
            experiment = yaml.safe_load(f)

        experiment["metadata"]["name"] = experiment_name
        experiment["spec"]["maxTrialCount"] = max_trials
        experiment["spec"]["parallelTrialCount"] = parallel_trials

        api.create_namespaced_custom_object(
            group="kubeflow.org",
            version="v1beta1",
            namespace="kubeflow",
            plural="experiments",
            body=experiment,
        )

        deadline = time.time() + 86400
        while time.time() < deadline:
            exp = api.get_namespaced_custom_object(
                group="kubeflow.org",
                version="v1beta1",
                namespace="kubeflow",
                plural="experiments",
                name=experiment_name,
            )
            conditions = exp.get("status", {}).get("conditions", [{}])
            status = conditions[-1].get("type", "") if conditions else ""
            if status == "Succeeded":
                best = exp["status"].get("currentOptimalTrial", {})
                print(f"Best DQN hyperparameters: {best}")
                return best
            if status == "Failed":
                raise RuntimeError(
                    f"DQN Katib experiment {experiment_name} failed"
                )
            time.sleep(120)

        raise RuntimeError(
            f"DQN Katib experiment {experiment_name} timed out after 24h"
        )

    run_katib = PythonOperator(
        task_id="run_katib_experiment",
        python_callable=_submit_katib_experiment,
    )
