"""DAG: Create end-of-day (EOD) tick snapshots via Kubernetes, starting 2026-04-20.

Runs daily at 00:00 UTC. On Tue–Sat the run waits for the final hourly
tick download of the just-completed trading day (Mon–Fri). On Sun and
Mon (when no download fires) the wait is bypassed so the snapshot
chain stays unbroken, `create_snapshot` then re-emits the prior
snapshot's contents under the weekend key.
Snapshots land under `marketdata/eod-tick-snapshot/...`.
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
    "/forex-marketdata-download-interval-price-data:latest"
)

# download_tick_data_2026 runs Mon–Fri (cron "0 * * * 1-5"). The sensor looks
# at the hourly download covering 23:00→00:00 of the prior calendar day. For a
# Sun-midnight run that's Sat 23:00 (no download); for Mon-midnight it's
# Sun 23:00 (no download). Python weekday(): Mon=0…Sun=6.
_NO_UPSTREAM_WEEKDAYS = {6, 0}


def _last_tick_hourly_logical_date(_dt, **context):
    """Logical date of the hourly tick download covering the final hour of the day.

    `download_tick_data_2026` fires hourly on weekdays with `logical_date`
    equal to `data_interval_start` (one hour before fire time). Our EOD run
    fires at midnight UTC with `data_interval_end` = midnight; the hourly run
    that fires at that same midnight covers 23:00→00:00, so its logical date is
    23:00 of the previous day.
    """

    return context["data_interval_end"] - timedelta(hours=1)


def _branch(data_interval_end, **_):
    if data_interval_end.weekday() in _NO_UPSTREAM_WEEKDAYS:
        return "skip_wait"
    return "wait_for_last_tick_download"


with DAG(
    max_active_runs=1,  # depends_on_past: serial; =1 prevents max_active_runs starvation deadlock
    dag_id="create_eod_tick_snapshot_2026_04_20",
    default_args=default_args,
    description="Create FX end-of-day tick snapshots starting from 2026-04-20",
    schedule="0 0 * * *",
    start_date=datetime(2026, 4, 20),
    catchup=True,
    tags=["marketdata", "tickdata", "snapshot", "eod", "2026"],
) as dag:

    branch = BranchPythonOperator(
        task_id="branch",
        python_callable=_branch,
    )

    wait_for_last_tick_download = ExternalTaskSensor(
        task_id="wait_for_last_tick_download",
        external_dag_id="download_tick_data_2026",
        external_task_id="download_tick_data",
        execution_date_fn=_last_tick_hourly_logical_date,
        poke_interval=300,
        timeout=86400,
        mode="reschedule",
    )

    skip_wait = EmptyOperator(task_id="skip_wait")

    create_snapshot = KubernetesPodOperator(
        task_id="create_eod_tick_snapshot",
        name="create-eod-tick-snapshot-2026-0420",
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
            requests={"cpu": "200m", "memory": "3Gi"},
            limits={"cpu": "1000m", "memory": "12Gi"},
        ),
        is_delete_operator_pod=True,
        get_logs=True,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    branch >> [wait_for_last_tick_download, skip_wait] >> create_snapshot
