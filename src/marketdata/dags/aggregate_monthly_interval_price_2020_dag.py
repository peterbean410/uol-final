"""DAG: Aggregate one calendar month of D1 bars into MN1, 2020 backfill.

Runs once per calendar month on the 1st at 00:00 UTC. Waits for the last
daily D1 aggregate of the previous month to complete, then runs the
aggregate-interval-price script to produce an MN1 (calendar-month) bar
for the just-completed month.

W1 is handled by `aggregate_weekly_interval_price_2020`; a monthly
cadence doesn't line up with Mon–Sun week boundaries, so weeks that
straddle month-end would land as partial buckets in both months.
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


def _last_d1_logical_date(_dt, **context):
    """Logical date of the D1 aggregate for the last trading day of the window.

    The daily D1 DAG fires Tue–Sat (`0 0 * * 2-6`); under CronTriggerTimetable
    its `logical_date` equals the fire time, not the data day. Each run
    aggregates the prior 24h, so the run that covers a Mon–Fri data day has
    `logical_date = data_day + 1 day` (always Tue–Sat). Walk back from
    `data_interval_end` to the last weekday, then add one day.
    """
    cursor = context["data_interval_end"] - timedelta(days=1)
    while cursor.weekday() >= 5:  # Sat=5, Sun=6
        cursor -= timedelta(days=1)
    return (cursor + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


with DAG(
    max_active_runs=1,  # depends_on_past: serial; =1 prevents max_active_runs starvation deadlock
    dag_id="aggregate_monthly_interval_price_2020",
    default_args=default_args,
    description="Aggregate one calendar month of D1 bars into an MN1 interval-price partition",
    schedule="0 0 1 * *",
    start_date=datetime(2020, 2, 1),
    catchup=True,
    tags=["marketdata", "pricedata", "aggregation", "bardata"],
) as dag:

    wait_for_last_D1 = ExternalTaskSensor(
        task_id="wait_for_last_D1",
        external_dag_id="aggregate_daily_interval_price_2020",
        external_task_id="aggregate_D1_interval_price",
        execution_date_fn=_last_d1_logical_date,
        poke_interval=600,
        timeout=14400,
        mode="reschedule",
    )

    aggregate_MN1 = KubernetesPodOperator(
        task_id="aggregate_MN1_interval_price",
        name="aggregate-mn1-interval-price",
        namespace="airflow",
        image=_ECR_IMAGE,
        image_pull_policy="IfNotPresent",
        image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
        service_account_name="airflow-worker",
        cmds=["python", "marketdata/usecases/aggregate-interval-price.py"],
        env_vars=[
            k8s.V1EnvVar(name="FX_SYMBOL", value="USDJPY"),
            k8s.V1EnvVar(name="SOURCE_INTERVAL", value="D1"),
            k8s.V1EnvVar(name="TARGET_INTERVAL", value="MN1"),
            k8s.V1EnvVar(name="SOURCE_PARTITION_MINUTES", value="1440"),
            k8s.V1EnvVar(name="TIME_WINDOW_IN_MINUTES", value="43200"),
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

    wait_for_last_D1 >> aggregate_MN1
