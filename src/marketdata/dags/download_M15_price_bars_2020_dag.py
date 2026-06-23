"""DAG: Download price bars (M1, M5, M15) via Kubernetes.

Runs the marketdata download-interval-price-data script as a
KubernetesPodOperator task for each interval, using the image defined
in the Helm chart.  All interval tasks run in parallel within a single
hourly DAG run.
"""

from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
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

_INTERVALS = ["M1", "M5", "M15"]

with DAG(
    max_active_runs=1,  # serial, rate-limited catchup; depends_on_past=False so one failure no longer cascades
    dag_id="download_price_bars_hourly_2020",
    default_args=default_args,
    description="Download FX price bars (M1, M5, M15) using the marketdata Helm chart image",
    schedule="0 * * * 1-5",
    start_date=datetime(2020, 1, 1),
    catchup=True,
    tags=["marketdata", "pricedata", "bardata", "download"],
) as dag:

    for interval in _INTERVALS:
        KubernetesPodOperator(
            task_id=f"download_{interval}_price_bars",
            name=f"download-{interval.lower()}-price-bars",
            namespace="airflow",
            image=_ECR_IMAGE,
            image_pull_policy="IfNotPresent",
            image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
            service_account_name="airflow-worker",
            cmds=["python", "marketdata/usecases/download-interval-price-data.py"],
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
                requests={"cpu": "100m", "memory": "256Mi"},
                limits={"cpu": "500m", "memory": "512Mi"},
            ),
            is_delete_operator_pod=True,
            get_logs=True,
        )
