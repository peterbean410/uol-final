"""DAG: Download FX news data once per day via Kubernetes.

Runs the marketdata download-interval-news-data script daily at 00:00 UTC,
fetching the previous day's news for each configured currency pair and
uploading to S3 as Parquet under
marketdata/interval-news/symbol=<PAIR>/interval=D1/.
"""

from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

default_args = {
    "owner": "fintech",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "depends_on_past": False,
}

_ECR_IMAGE = (
    "731833471586.dkr.ecr.ap-southeast-1.amazonaws.com"
    "/forex-marketdata-download-interval-price-data:latest"
)

# forexnewsapi.com expects dash-separated pairs (e.g. "USD-JPY").
FX_CURRENCY_PAIRS = ["USD-JPY", "EUR-USD", "XAU-USD"]


def _slug(pair: str) -> str:
    return pair.lower().replace("-", "_")


with DAG(
    dag_id="download_news_data",
    default_args=default_args,
    description="Download FX news data daily for configured currency pairs",
    schedule="0 0 * * *",
    start_date=datetime(2021, 1, 1),
    catchup=True,
    max_active_runs=2,
    tags=["marketdata", "newsdata", "download"],
) as dag:

    for pair in FX_CURRENCY_PAIRS:
        slug = _slug(pair)
        KubernetesPodOperator(
            task_id=f"download_news_{slug}",
            name=f"download-news-{slug.replace('_', '-')}",
            namespace="airflow",
            image=_ECR_IMAGE,
            image_pull_policy="Always",
            image_pull_secrets=[
                k8s.V1LocalObjectReference(name="ecr-registry-credentials")
            ],
            service_account_name="airflow-worker",
            cmds=["python", "marketdata/usecases/download-interval-news-data.py"],
            env_vars=[
                k8s.V1EnvVar(name="FX_CURRENCY_PAIR", value=pair),
                k8s.V1EnvVar(name="INTERVAL", value="D1"),
                k8s.V1EnvVar(
                    name="EXECUTION_TS",
                    value="{{ data_interval_end.to_iso8601_string() }}",
                ),
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
