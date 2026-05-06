"""DAG: Create end-of-day (EOD) news snapshots via Kubernetes.

Runs daily at 00:00 UTC. Waits for the upstream ``download_news_data``
task to complete for the same logical date, then merges that day's raw
news parquet with the previous day's cumulative snapshot.

Unlike the tick snapshot DAG, the news download runs every day (including
weekends), so no weekend-bypass branch is needed.

Snapshots land under ``marketdata/eod-news-snapshot/...``.
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

# forexnewsapi.com uses dash-separated pairs.
FX_PAIRS = ["USD-JPY", "EUR-USD", "XAU-USD"]


def _slug(pair: str) -> str:
    return pair.lower().replace("-", "_")


def _news_download_task_id(pair: str) -> str:
    return f"download_news_{_slug(pair)}"


with DAG(
    dag_id="create_eod_news_snapshot",
    default_args=default_args,
    description="Create FX end-of-day news snapshots",
    schedule="0 0 * * *",
    start_date=datetime(2021, 1, 1),
    catchup=True,
    max_active_runs=2,
    tags=["marketdata", "newsdata", "snapshot", "eod"],
) as dag:

    for pair in FX_PAIRS:
        slug = _slug(pair)
        upstream_task_id = _news_download_task_id(pair)

        wait_for_news_download = ExternalTaskSensor(
            task_id=f"wait_for_news_{slug}",
            external_dag_id="download_news_data",
            external_task_id=upstream_task_id,
            poke_interval=300,
            timeout=7200,
            mode="reschedule",
        )

        create_snapshot = KubernetesPodOperator(
            task_id=f"create_eod_news_snapshot_{slug}",
            name=f"create-eod-news-snapshot-{slug.replace('_', '-')}",
            namespace="airflow",
            image=_ECR_IMAGE,
            image_pull_policy="IfNotPresent",
            image_pull_secrets=[
                k8s.V1LocalObjectReference(name="ecr-registry-credentials")
            ],
            service_account_name="airflow-worker",
            cmds=["python", "marketdata/usecases/create-eod-news-snapshot.py"],
            env_vars=[
                k8s.V1EnvVar(name="FX_PAIR", value=pair),
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
                requests={"cpu": "200m", "memory": "512Mi"},
                limits={"cpu": "1000m", "memory": "2Gi"},
            ),
            is_delete_operator_pod=True,
            get_logs=True,
        )

        wait_for_news_download >> create_snapshot
