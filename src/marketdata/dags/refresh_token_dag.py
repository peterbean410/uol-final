"""DAG: Refresh cTrader API tokens before expiry.

Runs hourly. Checks whether the access token will expire within
120 * _TOKEN_EXPIRY_BUFFER_SECONDS and refreshes it if so,
persisting new tokens to the Kubernetes secret for downstream DAGs.
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

_REFRESH_SCRIPT = """\
import time
from commons.python.appconfig import AppConfig, _TOKEN_EXPIRY_BUFFER_SECONDS

config = AppConfig()
buffer = 120 * _TOKEN_EXPIRY_BUFFER_SECONDS
now = time.time()

if config.access_token and config.token_expiry > now + buffer:
    remaining = config.token_expiry - now
    print(f"Token still valid for {remaining:.0f}s (buffer={buffer}s). Skipping refresh.")
else:
    print(f"Token expires within {buffer}s buffer. Refreshing...")
    config.token_expiry = 0
    config.refresh_access_token()
    print("Token refreshed and persisted to K8s secret.")
"""

with DAG(
    dag_id="refresh_token",
    default_args=default_args,
    description="Refresh cTrader API tokens before they expire",
    schedule="@hourly",
    start_date=datetime(2026, 4, 1),
    catchup=False,
    tags=["marketdata", "auth"],
) as dag:

    refresh_token = KubernetesPodOperator(
        task_id="refresh_token",
        name="refresh-token",
        namespace="airflow",
        image="731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/forex-marketdata-download-interval-price-data:latest",
        image_pull_policy="IfNotPresent",
        image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
        service_account_name="airflow-worker",
        cmds=["python", "-c", _REFRESH_SCRIPT],
        env_from=[
            k8s.V1EnvFromSource(
                secret_ref=k8s.V1SecretEnvSource(name="marketdata-credentials")
            )
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "128Mi"},
            limits={"cpu": "250m", "memory": "256Mi"},
        ),
        is_delete_operator_pod=True,
        get_logs=True,
    )

    refresh_ctrader_secret = KubernetesPodOperator(
        task_id="refresh_ctrader_secret",
        name="refresh-ctrader-secret",
        namespace="modelenv",
        image="731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/forex-marketdata-download-interval-price-data:latest",
        image_pull_policy="IfNotPresent",
        image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
        service_account_name="ctrader-secret-refresher",
        cmds=["python", "-c", _REFRESH_SCRIPT],
        env_from=[
            k8s.V1EnvFromSource(
                secret_ref=k8s.V1SecretEnvSource(name="ctrader-secrets")
            )
        ],
        env_vars=[
            k8s.V1EnvVar(name="K8S_CREDENTIALS_SECRET", value="ctrader-secrets"),
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "128Mi"},
            limits={"cpu": "250m", "memory": "256Mi"},
        ),
        is_delete_operator_pod=True,
        get_logs=True,
    )
