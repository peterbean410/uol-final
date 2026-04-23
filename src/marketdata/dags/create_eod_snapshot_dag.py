"""DAG: Create end-of-day price snapshots (H1, H4) via Kubernetes.

For each interval, an ExternalTaskSensor waits for the corresponding
download task in the download_price_bars_daily DAG, then a
KubernetesPodOperator runs the create-eoi-price-snapshot script.
All interval chains run in parallel within a single daily DAG run.
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

_INTERVALS = ["H1", "H4"]

with DAG(
    dag_id="create_eod_snapshot",
    default_args=default_args,
    description="Create FX end-of-day price snapshots (H1, H4) using the marketdata Helm chart image",
    schedule="0 0 * * 1-5",
    start_date=datetime(2012, 1, 1),
    catchup=True,
    tags=["marketdata", "snapshot"],
) as dag:

    for interval in _INTERVALS:
        wait_for_download = ExternalTaskSensor(
            task_id=f"wait_for_download_{interval}_price_bars",
            external_dag_id="download_price_bars_daily",
            external_task_id=f"download_{interval}_price_bars",
            poke_interval=60,
            timeout=3600,
            mode="reschedule",
        )

        create_snapshot = KubernetesPodOperator(
            task_id=f"create_{interval}_eod_snapshot",
            name=f"create-{interval.lower()}-eod-snapshot",
            namespace="airflow",
            image=_ECR_IMAGE,
            image_pull_policy="Always",
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

        wait_for_download >> create_snapshot
