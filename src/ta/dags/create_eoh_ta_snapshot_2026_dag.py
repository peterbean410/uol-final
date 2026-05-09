"""DAG: Create end-of-hour TA snapshots (M1, M5, M15) via Kubernetes, 2026.

Runs hourly every day. On Mon–Fri the run waits for the corresponding
price-snapshot task in the create_eoh_snapshot_2026 DAG. On Sat–Sun
the wait is bypassed so the TA snapshot chain stays unbroken.
"""

from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.standard.operators.python import BranchPythonOperator
from airflow.providers.standard.sensors.external_task import ExternalTaskSensor
from airflow.task.trigger_rule import TriggerRule
from kubernetes.client import models as k8s

default_args = {
    "owner": "fintech",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": True,
}

_ECR_IMAGE = (
    "731833471586.dkr.ecr.ap-southeast-1.amazonaws.com"
    "/forex-ta-snapshot:latest"
)

_INTERVALS = ["M1", "M5", "M15"]
_NO_UPSTREAM_WEEKDAYS = {5, 6}


def _make_branch(wait_task_id: str, skip_task_id: str):
    def _branch(data_interval_end, **_):
        if data_interval_end.weekday() in _NO_UPSTREAM_WEEKDAYS:
            return skip_task_id
        return wait_task_id

    return _branch


with DAG(
    dag_id="create_eoh_ta_snapshot_2026",
    default_args=default_args,
    description="Create FX end-of-hour TA snapshots (M1, M5, M15) for the 2026 backfill",
    schedule="0 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=True,
    tags=["ta", "snapshot", "2026"],
) as dag:
    for interval in _INTERVALS:
        wait_id = f"wait_for_{interval}_price_snapshot"
        skip_id = f"skip_wait_{interval}"

        branch = BranchPythonOperator(
            task_id=f"branch_{interval}",
            python_callable=_make_branch(wait_id, skip_id),
        )

        wait_for_price_snapshot = ExternalTaskSensor(
            task_id=wait_id,
            external_dag_id="create_eoh_snapshot_2026",
            external_task_id=f"create_{interval}_eoi_snapshot",
            poke_interval=60,
            timeout=3600,
            mode="reschedule",
        )

        skip_wait = EmptyOperator(task_id=skip_id)

        create_ta_snapshot = KubernetesPodOperator(
            task_id=f"create_{interval}_eoi_ta_snapshot",
            name=f"create-{interval.lower()}-eoi-ta-snapshot-2026",
            namespace="airflow",
            image=_ECR_IMAGE,
            image_pull_policy="IfNotPresent",
            image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
            service_account_name="airflow-worker",
            cmds=["python", "ta/usecases/create-eoi-ta-snapshot.py"],
            env_vars=[
                k8s.V1EnvVar(name="FX_SYMBOL", value="USDJPY"),
                k8s.V1EnvVar(name="INTERVAL", value=interval),
                k8s.V1EnvVar(name="TIME_WINDOW_IN_MINUTES", value="60"),
                k8s.V1EnvVar(name="EXECUTION_TS", value="{{ data_interval_end.to_iso8601_string() }}"),
            ],
            env_from=[
                k8s.V1EnvFromSource(
                    secret_ref=k8s.V1SecretEnvSource(name="marketdata-credentials")
                ),
                k8s.V1EnvFromSource(
                    secret_ref=k8s.V1SecretEnvSource(name="airflow-s3-marketdata")
                ),
            ],
            container_resources={"requests": {"cpu": "100m", "memory": "256Mi"},
                                "limits": {"cpu": "500m", "memory": "512Mi"}},
            is_delete_operator_pod=True,
            get_logs=True,
            trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        )

        branch >> [wait_for_price_snapshot, skip_wait] >> create_ta_snapshot
