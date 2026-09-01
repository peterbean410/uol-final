"""Airflow DAGs: Trigger Kubeflow Forecaster Pipeline runs.

Three DAGs orchestrate the forecaster ML pipeline via Kubeflow Pipelines:

1. forecaster_weekly_finetune, Weekly Sunday 02:00 UTC fine-tune of the
   production model on the latest data. Skips if a drift-triggered retrain
   is already running or completed for the same day. On failure, escalates
   to a full retrain.

2. forecaster_drift_retrain; No schedule (triggered externally by
   model-drift-detection or escalation from failed fine-tune). Retrains
   from scratch on the full rolling window.

3. forecaster_katib_tuning; No schedule (manual trigger). Submits a Katib
   hyperparameter tuning experiment CR to the cluster.

Each training DAG waits for the EOH snapshot to complete via
ExternalTaskSensor before submitting the KFP pipeline run.

Requirements: 7.1, 7.2, 7.3, 7.5, 7.6
"""

from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.sensors.external_task import ExternalTaskSensor


_SYMBOLS = ["USDJPY", "AUDJPY"]

_KFP_HOST = "http://ml-pipeline.kubeflow.svc.cluster.local:8888"

default_args = {
    "owner": "ml-engineering",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="forecaster_weekly_finetune",
    start_date=datetime(2026, 5, 1),
    schedule="0 2 * * 0",
    default_args=default_args,
    catchup=False,
    tags=["forecaster", "kubeflow", "finetune"],
) as finetune_dag:

    def _check_retrain_not_running(**context):
        """Skip fine-tune if a scratch retrain is running or completed today.

        Queries the KFP API for any forecaster pipeline runs with
        training_mode='scratch' for each configured symbol. If found running
        (any date) or created today, raises AirflowSkipException.
        """
        from airflow.exceptions import AirflowSkipException
        from probabilisticforecaster.kubeflow.airflow.kfp_utils import KFPTrigger

        trigger = KFPTrigger(kfp_host=_KFP_HOST)
        today = context["logical_date"].strftime("%Y-%m-%d")
        for symbol in _SYMBOLS:
            trigger._current_symbol = symbol
            if trigger.has_retrain_run_today(today):
                raise AirflowSkipException(
                    f"Scratch retrain running/completed for {symbol} on {today}; "
                    "skipping fine-tune."
                )

    check_no_retrain = PythonOperator(
        task_id="check_retrain_not_running",
        python_callable=_check_retrain_not_running,
    )

    for symbol in _SYMBOLS:
        wait_for_snapshot = ExternalTaskSensor(
            task_id=f"wait_for_eoh_snapshot_{symbol.lower()}",
            external_dag_id="create_eoh_snapshot",
            external_task_id="create_M5_eoi_snapshot",
            execution_date_fn=lambda dt: dt,
            poke_interval=120,
            timeout=3600,
            mode="reschedule",
        )

        def _submit_finetune(symbol=symbol, **context):
            """Submit KFP fine-tune run for the given symbol."""
            from probabilisticforecaster.kubeflow.airflow.kfp_utils import KFPTrigger

            logical_date = context["logical_date"]
            snapshot_date = logical_date.strftime("%Y-%m-%d")
            s3_data_path = (
                f"marketdata/eoh-snapshot/symbol={symbol}/interval=M5"
                f"/year={logical_date.year}/month={logical_date.month:02d}"
                f"/day={logical_date.day:02d}/hour=23"
                f"/{logical_date.strftime('%Y%m%d')}T230000Z.parquet"
            )

            trigger = KFPTrigger(kfp_host=_KFP_HOST)
            run_id = trigger.submit_run(
                symbol=symbol,
                snapshot_date=snapshot_date,
                s3_data_path=s3_data_path,
                training_mode="finetune",
            )
            status = trigger.poll_run_status(run_id, timeout_seconds=28800)
            if status != "success":
                raise RuntimeError(
                    f"KFP finetune run {run_id} for {symbol} ended with "
                    f"status: {status}"
                )

        submit_finetune = PythonOperator(
            task_id=f"submit_finetune_{symbol.lower()}",
            python_callable=_submit_finetune,
        )

        def _escalate_to_retrain(symbol=symbol, **context):
            """Trigger full retrain DAG as escalation on fine-tune failure.

            When the fine-tune pipeline fails (e.g., quality gate rejects the
            model), this task triggers the drift retrain DAG as a fallback to
            retrain from scratch.
            """
            from airflow.api.common.trigger_dag import trigger_dag as airflow_trigger_dag

            logical_date = context["logical_date"]
            airflow_trigger_dag(
                dag_id="forecaster_drift_retrain",
                run_id=f"escalation_{symbol}_{logical_date.strftime('%Y%m%d')}",
                conf={
                    "symbol": symbol,
                    "snapshot_date": logical_date.strftime("%Y-%m-%d"),
                },
                execution_date=logical_date,
            )

        escalate_retrain = PythonOperator(
            task_id=f"escalate_to_retrain_{symbol.lower()}",
            python_callable=_escalate_to_retrain,
            trigger_rule="all_failed",
        )

        check_no_retrain >> wait_for_snapshot >> submit_finetune >> escalate_retrain


with DAG(
    dag_id="forecaster_drift_retrain",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    default_args=default_args,
    catchup=False,
    tags=["forecaster", "kubeflow", "retrain", "drift"],
) as retrain_dag:

    for symbol in _SYMBOLS:
        wait_for_snapshot = ExternalTaskSensor(
            task_id=f"wait_for_eoh_snapshot_{symbol.lower()}",
            external_dag_id="create_eoh_snapshot",
            external_task_id="create_M5_eoi_snapshot",
            execution_date_fn=lambda dt: dt,
            poke_interval=120,
            timeout=3600,
            mode="reschedule",
        )

        def _submit_retrain(symbol=symbol, **context):
            """Submit KFP full retrain from scratch for the given symbol."""
            from probabilisticforecaster.kubeflow.airflow.kfp_utils import KFPTrigger

            logical_date = context["logical_date"]
            snapshot_date = logical_date.strftime("%Y-%m-%d")
            s3_data_path = (
                f"marketdata/eoh-snapshot/symbol={symbol}/interval=M5"
                f"/year={logical_date.year}/month={logical_date.month:02d}"
                f"/day={logical_date.day:02d}/hour=23"
                f"/{logical_date.strftime('%Y%m%d')}T230000Z.parquet"
            )

            trigger = KFPTrigger(kfp_host=_KFP_HOST)
            run_id = trigger.submit_run(
                symbol=symbol,
                snapshot_date=snapshot_date,
                s3_data_path=s3_data_path,
                training_mode="scratch",
            )
            status = trigger.poll_run_status(run_id, timeout_seconds=28800)
            if status != "success":
                raise RuntimeError(
                    f"KFP retrain run {run_id} for {symbol} ended with "
                    f"status: {status}"
                )

        submit_retrain = PythonOperator(
            task_id=f"submit_retrain_{symbol.lower()}",
            python_callable=_submit_retrain,
        )

        wait_for_snapshot >> submit_retrain


with DAG(
    dag_id="forecaster_katib_tuning",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    default_args=default_args,
    catchup=False,
    tags=["forecaster", "kubeflow", "katib", "tuning"],
    params={
        "symbol": "USDJPY",
        "max_trials": 30,
        "parallel_trials": 5,
    },
) as katib_dag:

    def _submit_katib_experiment(**context):
        """Apply the Katib Experiment CR and poll until completion.

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
            f"forecaster-tuning-{symbol.lower()}-{context['run_id'][:8]}"
        )

        with open(
            "probabilisticforecaster/kubeflow/katib/experiment.yaml"
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
                print(f"Best hyperparameters: {best}")
                return best
            if status == "Failed":
                raise RuntimeError(
                    f"Katib experiment {experiment_name} failed"
                )
            time.sleep(120)

        raise RuntimeError(
            f"Katib experiment {experiment_name} timed out after 24h"
        )

    run_katib = PythonOperator(
        task_id="run_katib_experiment",
        python_callable=_submit_katib_experiment,
    )
