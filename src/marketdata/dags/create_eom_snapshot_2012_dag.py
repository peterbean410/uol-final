"""DAG: Create end-of-month (EOM) price snapshots for MN1, 2012 backfill.

Mirror of ``create_eom_snapshot_2020`` rolled back to 2012-02-01 (first
run covers January 2012) and capped at 2020-01-01 (last run covers
December 2019) so it doesn't overlap with the 2020 sibling, whose
2020-02-01 run is the first to cover January 2020.

Runs once per calendar month on the 1st at 00:00 UTC, waits for the
MN1 aggregate of the just-completed month
(``aggregate_monthly_interval_price_2012``; the 2012-side monthly
aggregate DAG must exist, same way the 2020 DAG depends on its 2020
sibling), then runs the create-eoi-price-snapshot script.

Snapshots land under ``marketdata/eom-snapshot/...``.
"""

from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sensors.external_task import ExternalTaskSensor
from kubernetes.client import models as k8s

default_args = {
    "owner": "fintech",
    "retries": 12,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(hours=4),
    "depends_on_past": True,
}

_ECR_IMAGE = (
    "731833471586.dkr.ecr.ap-southeast-1.amazonaws.com"
    "/forex-marketdata-download-interval-price-data:latest"
)


with DAG(
    max_active_runs=1,  # depends_on_past serial; =1 prevents max_active_runs starvation deadlock
    dag_id="create_eom_snapshot_2012",
    default_args=default_args,
    description=(
        "Create FX end-of-month price snapshots (MN1) from monthly-aggregated "
        "bars, Jan 2012 through Dec 2019 backfill"
    ),
    schedule="0 0 1 * *",
    start_date=datetime(2012, 2, 1),
    end_date=datetime(2020, 1, 1, 23, 59, 59),
    catchup=True,
    tags=["marketdata", "pricedata", "snapshot", "eom", "backfill"],
) as dag:

    wait_for_aggregate = ExternalTaskSensor(
        task_id="wait_for_aggregate_MN1",
        external_dag_id="aggregate_monthly_interval_price_2012",
        external_task_id="aggregate_MN1_interval_price",
        poke_interval=600,
        timeout=86400,
        mode="reschedule",
    )

    create_snapshot = KubernetesPodOperator(
        task_id="create_MN1_eom_snapshot",
        name="create-mn1-eom-snapshot",
        namespace="airflow",
        image=_ECR_IMAGE,
        image_pull_policy="IfNotPresent",
        image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
        service_account_name="airflow-worker",
        cmds=["python", "marketdata/usecases/create-eoi-price-snapshot.py"],
        env_vars=[
            k8s.V1EnvVar(name="FX_SYMBOL", value="USDJPY"),
            k8s.V1EnvVar(name="INTERVAL", value="MN1"),
            k8s.V1EnvVar(name="TIME_WINDOW_IN_MINUTES", value="43200"),
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
