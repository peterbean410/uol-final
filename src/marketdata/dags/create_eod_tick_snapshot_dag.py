"""DAG: Create end-of-day (EOD) tick snapshots via Kubernetes.

Runs Tue–Sat at 00:00 UTC. Each run waits for the final hourly tick
download of the just-completed trading day (Mon–Fri), then runs the
create-eod-tick-snapshot script to merge that day's tick partitions
with the prior day's and upload the cumulative snapshot to S3 under
`marketdata/eod-tick-snapshot/...`.
"""

from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sensors.external_task import ExternalTaskSensor
from kubernetes.client import models as k8s

default_args = {
    "owner": "fintech",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": True,
}

_ECR_IMAGE = (
    "731833471586.dkr.ecr.ap-southeast-1.amazonaws.com"
    "/forex-marketdata-download-interval-price-data:latest"
)


def _last_tick_hourly_logical_date(_dt, **context):
    """Logical date of the hourly tick download covering the final hour of the day.

    `download_tick_data` fires hourly on weekdays with `logical_date` equal
    to `data_interval_start` (one hour before fire time). Our EOD run fires
    at midnight UTC with `data_interval_end` = midnight; the hourly run that
    fires at that same midnight covers 23:00→00:00, so its logical date is
    23:00 of the previous day.
    """
    return context["data_interval_end"] - timedelta(hours=1)


with DAG(
    dag_id="create_eod_tick_snapshot",
    default_args=default_args,
    description="Create FX end-of-day tick snapshots from per-hour tick partitions",
    schedule="0 0 * * 2-6",
    start_date=datetime(2021, 1, 2),
    catchup=True,
    tags=["marketdata", "tickdata", "snapshot", "eod"],
) as dag:

    wait_for_last_tick_download = ExternalTaskSensor(
        task_id="wait_for_last_tick_download",
        external_dag_id="download_tick_data",
        external_task_id="download_tick_data",
        execution_date_fn=_last_tick_hourly_logical_date,
        poke_interval=300,
        timeout=7200,
        mode="reschedule",
    )

    create_snapshot = KubernetesPodOperator(
        task_id="create_eod_tick_snapshot",
        name="create-eod-tick-snapshot",
        namespace="airflow",
        image=_ECR_IMAGE,
        image_pull_policy="IfNotPresent",
        image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
        service_account_name="airflow-worker",
        cmds=["python", "marketdata/usecases/create-eod-tick-snapshot.py"],
        env_vars=[
            k8s.V1EnvVar(name="FX_SYMBOL", value="USDJPY"),
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
            requests={"cpu": "200m", "memory": "512Mi"},
            limits={"cpu": "1000m", "memory": "2Gi"},
        ),
        is_delete_operator_pod=True,
        get_logs=True,
    )

    wait_for_last_tick_download >> create_snapshot
