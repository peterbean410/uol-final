"""DAG: Aggregate daily M1 bars into larger intervals (H1, H4, D1), 2020 backfill.

Runs once per trading day, waits for the final hourly M1 download of
the day to complete, then runs the aggregate-interval-price script in
parallel for each target interval to roll up 24 hours of M1 bars and
publish the aggregated bars to S3.
"""

from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sensors.external_task import ExternalTaskSensor
from kubernetes.client import models as k8s

default_args = {
    "owner": "fintech",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}

_ECR_IMAGE = (
    "731833471586.dkr.ecr.ap-southeast-1.amazonaws.com"
    "/forex-marketdata-download-interval-price-data:latest"
)

_TARGET_INTERVALS = ["H1", "H4", "D1"]


def _last_m1_hourly_logical_date(_dt, **context):
    """Logical date of the hourly M1 download covering the last hour of this run.

    The hourly download DAG's `logical_date` equals `data_interval_start`,
    i.e. one hour before its fire time. The hourly run that fires at our
    `data_interval_end` covers the final hour of our 24h window.
    """
    return context["data_interval_end"] - timedelta(hours=1)


with DAG(
    max_active_runs=1,  # serial, rate-limited catchup; depends_on_past=False so one failure no longer cascades
    dag_id="aggregate_daily_interval_price_2020",
    default_args=default_args,
    description="Aggregate 24h of M1 bars into H1/H4/D1 interval-price partitions using the marketdata Helm chart image",
    schedule="0 0 * * 2-6",
    start_date=datetime(2020, 1, 2),
    catchup=True,
    tags=["marketdata", "pricedata", "aggregation", "bardata"],
) as dag:

    wait_for_last_M1_download = ExternalTaskSensor(
        task_id="wait_for_last_M1_download",
        external_dag_id="download_price_bars_hourly_2020",
        external_task_id="download_M1_price_bars",
        execution_date_fn=_last_m1_hourly_logical_date,
        poke_interval=300,
        timeout=86400,
        mode="reschedule",
    )

    for interval in _TARGET_INTERVALS:
        aggregate_task = KubernetesPodOperator(
            task_id=f"aggregate_{interval}_interval_price",
            name=f"aggregate-{interval.lower()}-interval-price",
            namespace="airflow",
            image=_ECR_IMAGE,
            image_pull_policy="IfNotPresent",
            image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
            service_account_name="airflow-worker",
            cmds=["python", "marketdata/usecases/aggregate-interval-price.py"],
            env_vars=[
                k8s.V1EnvVar(name="FX_SYMBOL", value="USDJPY"),
                k8s.V1EnvVar(name="SOURCE_INTERVAL", value="M1"),
                k8s.V1EnvVar(name="TARGET_INTERVAL", value=interval),
                k8s.V1EnvVar(name="TIME_WINDOW_IN_MINUTES", value="1440"),
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

        wait_for_last_M1_download >> aggregate_task
