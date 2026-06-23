"""DAG: Download tick data (bid/ask) per minute via Kubernetes, 2012-2020 backfill.

Closes the gap between the 2012 historical archive and the live tick DAG
which starts at 2021-01-04. Runs the marketdata
download-interval-price-data script with INTERVAL=ticks hourly across
weekdays for each hour from 2012-01-01 through 2020-12-31, uploading
each hour's ticks to S3 as Parquet.
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

with DAG(
    dag_id="download_tick_data_2012",
    default_args=default_args,
    description="Download FX tick data per hour for the 2012-2020 backfill",
    schedule="0 * * * 1-5",
    start_date=datetime(2012, 1, 1),
    end_date=datetime(2020, 12, 31),
    catchup=True,
    max_active_runs=1,
    tags=["marketdata", "pricedata", "tickdata", "download", "backfill-2012"],
) as dag:

    KubernetesPodOperator(
        task_id="download_tick_data",
        name="download-tick-data-2012",
        namespace="airflow",
        image=_ECR_IMAGE,
        image_pull_policy="Always",
        image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
        service_account_name="airflow-worker",
        cmds=["python", "marketdata/usecases/download-interval-price-data.py"],
        env_vars=[
            k8s.V1EnvVar(name="FX_SYMBOL", value="USDJPY"),
            k8s.V1EnvVar(name="INTERVAL", value="ticks"),
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
