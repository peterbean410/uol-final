"""DAG: Create end-of-day (EOD) price snapshots for H1, H4, D1, 2020 backfill.

Runs once per trading day (Tue–Sat at 00:00 UTC), waits for the matching
daily-aggregate task in `aggregate_daily_interval_price_2020`, then runs
the create-eoi-price-snapshot script in parallel for each interval.
Snapshots land under `marketdata/eod-snapshot/...`.
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

_INTERVALS = ["H1", "H4", "D1"]


with DAG(
    dag_id="create_eod_snapshot_2020",
    default_args=default_args,
    description="Create FX end-of-day price snapshots (H1, H4, D1) from daily-aggregated bars",
    schedule="0 0 * * 2-6",
    start_date=datetime(2020, 1, 2),
    catchup=True,
    tags=["marketdata", "pricedata", "snapshot", "eod"],
) as dag:

    for interval in _INTERVALS:
        wait_for_aggregate = ExternalTaskSensor(
            task_id=f"wait_for_aggregate_{interval}",
            external_dag_id="aggregate_daily_interval_price_2020",
            external_task_id=f"aggregate_{interval}_interval_price",
            poke_interval=300,
            timeout=7200,
            mode="reschedule",
        )

        create_snapshot = KubernetesPodOperator(
            task_id=f"create_{interval}_eod_snapshot",
            name=f"create-{interval.lower()}-eod-snapshot",
            namespace="airflow",
            image=_ECR_IMAGE,
            image_pull_policy="IfNotPresent",
            image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
            service_account_name="airflow-worker",
            cmds=["python", "marketdata/usecases/create-eoi-price-snapshot.py"],
            env_vars=[
                k8s.V1EnvVar(name="FX_SYMBOL", value="USDJPY"),
                k8s.V1EnvVar(name="INTERVAL", value=interval),
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
                requests={"cpu": "100m", "memory": "256Mi"},
                limits={"cpu": "500m", "memory": "512Mi"},
            ),
            is_delete_operator_pod=True,
            get_logs=True,
        )

        wait_for_aggregate >> create_snapshot
