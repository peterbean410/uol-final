"""DAG: Aggregate one Mon–Sun week of D1 bars into W1, 2012 backfill.

Mirror of ``aggregate_weekly_interval_price_2020`` rolled back to the
first Monday of 2012 (2012-01-02) and capped at 2019-12-30 so it doesn't
overlap with the 2020 sibling.

Runs every Monday at 00:00 UTC. Waits for the Friday D1 aggregate of the
just-completed week (``aggregate_daily_interval_price_2012``, also a
2012 backfill DAG), then rolls 7 days of D1 bars into a single
Monday-anchored W1 bar.

A weekly cadence (rather than monthly) is required for W1: Mon–Sun weeks
straddle month boundaries, so a monthly window would produce partial-week
buckets that split a single ISO week across two output files.
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
    """Logical date of the Friday D1 aggregate covering the last trading day.

    The daily D1 DAG fires Tue–Sat with ``logical_date`` = the Mon–Fri date
    whose M1 data it aggregates. Our ``data_interval_end`` is a Monday at
    00:00 UTC; the most recent weekday before that is the prior Friday.
    """
    cursor = context["data_interval_end"] - timedelta(days=1)
    while cursor.weekday() >= 5:  # Sat=5, Sun=6
        cursor -= timedelta(days=1)
    return cursor.replace(hour=0, minute=0, second=0, microsecond=0)


with DAG(
    max_active_runs=1,  # depends_on_past: serial; =1 prevents max_active_runs starvation deadlock
    dag_id="aggregate_weekly_interval_price_2012",
    default_args=default_args,
    description=(
        "Aggregate one Mon–Sun week of D1 bars into a W1 interval-price "
        "partition, 2012-01-02 through 2019-12-30 backfill"
    ),
    schedule="0 0 * * 1",
    start_date=datetime(2012, 1, 2),
    end_date=datetime(2019, 12, 30, 23, 59, 59),
    catchup=True,
    tags=["marketdata", "pricedata", "aggregation", "bardata", "backfill"],
) as dag:

    wait_for_last_D1 = ExternalTaskSensor(
        task_id="wait_for_last_D1",
        external_dag_id="aggregate_daily_interval_price_2012",
        external_task_id="aggregate_D1_interval_price",
        execution_date_fn=_last_d1_logical_date,
        poke_interval=600,
        timeout=14400,
        mode="reschedule",
    )

    aggregate_W1 = KubernetesPodOperator(
        task_id="aggregate_W1_interval_price",
        name="aggregate-w1-interval-price",
        namespace="airflow",
        image=_ECR_IMAGE,
        image_pull_policy="IfNotPresent",
        image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
        service_account_name="airflow-worker",
        cmds=["python", "marketdata/usecases/aggregate-interval-price.py"],
        env_vars=[
            k8s.V1EnvVar(name="FX_SYMBOL", value="USDJPY"),
            k8s.V1EnvVar(name="SOURCE_INTERVAL", value="D1"),
            k8s.V1EnvVar(name="TARGET_INTERVAL", value="W1"),
            k8s.V1EnvVar(name="SOURCE_PARTITION_MINUTES", value="1440"),
            k8s.V1EnvVar(name="TIME_WINDOW_IN_MINUTES", value="10080"),
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

    wait_for_last_D1 >> aggregate_W1
