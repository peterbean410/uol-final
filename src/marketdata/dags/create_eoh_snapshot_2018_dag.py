"""DAG: Create end-of-interval price snapshots (M1, M5, M15), 2018 backfill.

Runs hourly every day. On Mon–Fri the run waits for the corresponding
download task in the download_price_bars_hourly_2018 DAG. On Sat–Sun
(when no download fires) the wait is bypassed so the snapshot chain
stays unbroken, `create_snapshot` then re-emits the prior snapshot's
contents under the weekend key.
"""

from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.trigger_rule import TriggerRule
from kubernetes.client import models as k8s

default_args = {
    "owner": "fintech",
    "retries": 7,
    "retry_delay": timedelta(minutes=10),
    "depends_on_past": False,
}

_ECR_IMAGE = (
    "731833471586.dkr.ecr.ap-southeast-1.amazonaws.com"
    "/forex-marketdata-download-interval-price-data:latest"
)

_INTERVALS = ["M1", "M5", "M15"]
# download_price_bars_hourly_2018 runs Mon–Fri (cron "0 * * * 1-5").
# Python weekday(): Mon=0…Sun=6. Skip the wait on Sat and Sun.
_NO_UPSTREAM_WEEKDAYS = {5, 6}


def _make_branch(wait_task_id: str, skip_task_id: str):
    def _branch(data_interval_end, **_):
        if data_interval_end.weekday() in _NO_UPSTREAM_WEEKDAYS:
            return skip_task_id
        return wait_task_id
    return _branch


with DAG(
    max_active_runs=1,  # serial, rate-limited catchup; depends_on_past=False so one failure no longer cascades
    dag_id="create_eoh_snapshot_2018",
    default_args=default_args,
    description="Create FX end-of-interval price snapshots (M1, M5, M15), 2018 backfill",
    schedule="0 * * * *",
    start_date=datetime(2018, 1, 1),
    catchup=True,
    tags=["marketdata", "snapshot", "backfill-2018"],
) as dag:

    for interval in _INTERVALS:
        wait_id = f"wait_for_download_{interval}_price_bars"
        skip_id = f"skip_wait_{interval}"

        branch = BranchPythonOperator(
            task_id=f"branch_{interval}",
            python_callable=_make_branch(wait_id, skip_id),
        )

        wait_for_download = ExternalTaskSensor(
            task_id=wait_id,
            external_dag_id="download_price_bars_hourly_2018",
            external_task_id=f"download_{interval}_price_bars",
            poke_interval=60,
            timeout=86400,
            mode="reschedule",
        )

        skip_wait = EmptyOperator(task_id=skip_id)

        create_snapshot = KubernetesPodOperator(
            task_id=f"create_{interval}_eoi_snapshot",
            name=f"create-{interval.lower()}-eoi-snapshot",
            namespace="airflow",
            image=_ECR_IMAGE,
            image_pull_policy="IfNotPresent",
            image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
            service_account_name="airflow-worker",
            cmds=["python", "marketdata/usecases/create-eoi-price-snapshot.py"],
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
            container_resources=k8s.V1ResourceRequirements(
                requests={"cpu": "100m", "memory": "3Gi"},
                limits={"cpu": "2000m", "memory": "12Gi"},
            ),
            is_delete_operator_pod=True,
            get_logs=True,
            trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        )

        branch >> [wait_for_download, skip_wait] >> create_snapshot
