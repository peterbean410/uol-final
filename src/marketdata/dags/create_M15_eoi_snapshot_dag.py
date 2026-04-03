"""DAG: Create M15 end-of-interval price snapshot via Kubernetes.

Runs the marketdata create-eoi-price-snapshot script as a
KubernetesPodOperator task, using the image defined in the Helm chart.
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

with DAG(
    dag_id="create_M15_eoi_snapshot",
    default_args=default_args,
    description="Create FX M15 end-of-interval price snapshot using the marketdata Helm chart image",
    schedule="0 * * * 1-5",
    start_date=datetime(2012, 1, 1),
    catchup=True,
    tags=["marketdata", "snapshot"],
) as dag:

    wait_for_download = ExternalTaskSensor(
        task_id="wait_for_download_M15_price_bars",
        external_dag_id="download_M15_price_bars",
        external_task_id="download_M15_price_bars",
        poke_interval=60,
        timeout=3600,
        mode="reschedule",
    )

    create_eoi_snapshot = KubernetesPodOperator(
        task_id="create_M15_eoi_snapshot",
        name="create-m15-eoi-snapshot",
        namespace="airflow",
        image="731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/forex-marketdata-download-interval-price-data:latest",
        image_pull_policy="Always",
        image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
        service_account_name="airflow-worker",
        cmds=["python", "marketdata/usecases/create-eoi-price-snapshot.py"],
        env_vars=[
            k8s.V1EnvVar(name="FX_SYMBOL", value="USDJPY"),
            k8s.V1EnvVar(name="INTERVAL", value="M15"),
            k8s.V1EnvVar(name="TIME_WINDOW_IN_MINUTES", value="60"),
            k8s.V1EnvVar(name="EXECUTION_TS", value="{{ data_interval_end.to_iso8601_string() }}"),
        ],
        env_from=[
            k8s.V1EnvFromSource(
                secret_ref=k8s.V1SecretEnvSource(name="marketdata-credentials")
            ),
            k8s.V1EnvFromSource(
                secret_ref=k8s.V1SecretEnvSource(name="airflow-s3-marketdata")
            )
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "256Mi"},
            limits={"cpu": "500m", "memory": "512Mi"},
        ),
        is_delete_operator_pod=True,
        get_logs=True,
    )

    wait_for_download >> create_eoi_snapshot
