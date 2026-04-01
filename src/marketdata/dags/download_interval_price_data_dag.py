"""DAG: Download interval price data via Kubernetes.

Runs the marketdata download-interval-price-data script as a
KubernetesPodOperator task, using the image defined in the Helm chart.
"""

from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

default_args = {
    "owner": "fintech",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="download_interval_price_data",
    default_args=default_args,
    description="Download FX interval price data using the marketdata Helm chart image",
    schedule="*/15 * * * 1-5",
    start_date=datetime(2026, 4, 1),
    catchup=False,
    tags=["marketdata", "pricedata"],
) as dag:

    download_price_data = KubernetesPodOperator(
        task_id="download_interval_price_data",
        name="download-interval-price-data",
        namespace="default",
        image="731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/forex-marketdata-download-interval-price-data:202604011539",
        image_pull_policy="IfNotPresent",
        cmds=["python", "marketdata/usecases/download-interval-price-data.py"],
        env_vars=[
            k8s.V1EnvVar(name="FX_SYMBOL", value="USDJPY"),
            k8s.V1EnvVar(name="FX_INTERVAL", value="15"),
        ],
        env_from=[
            k8s.V1EnvFromSource(
                secret_ref=k8s.V1SecretEnvSource(name="marketdata-credentials")
            )
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "256Mi"},
            limits={"cpu": "500m", "memory": "512Mi"},
        ),
        is_delete_operator_pod=True,
        get_logs=True,
    )
