"""DAG: Create end-of-week (EOW) price snapshots for W1, 2012 backfill.

Mirror of ``create_eow_snapshot_2020`` rolled back to 2012-01-02 (the
first Monday of 2012) and capped at 2019-12-30 (the last Monday before
the 2020 DAG's start_date of 2020-01-06) so the two runs don't overlap.

Runs every Monday at 00:00 UTC, waits for the weekly W1 aggregate of the
just-completed ISO week (``aggregate_weekly_interval_price_2012``; the
2012-side weekly aggregate DAG must exist, same way the 2020 DAG depends
on its 2020 sibling), then runs the create-eoi-price-snapshot script.

Snapshots land under ``marketdata/eow-snapshot/...``.
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


with DAG(
    dag_id="create_eow_snapshot_2012",
    default_args=default_args,
    description=(
        "Create FX end-of-week price snapshots (W1) from weekly-aggregated "
        "bars, 2012-01-02 through 2019-12-30 backfill"
    ),
    schedule="0 0 * * 1",
    start_date=datetime(2012, 1, 2),
    end_date=datetime(2019, 12, 30, 23, 59, 59),
    catchup=True,
    tags=["marketdata", "pricedata", "snapshot", "eow", "backfill"],
) as dag:

    wait_for_aggregate = ExternalTaskSensor(
        task_id="wait_for_aggregate_W1",
        external_dag_id="aggregate_weekly_interval_price_2012",
        external_task_id="aggregate_W1_interval_price",
        poke_interval=600,
        timeout=14400,
        mode="reschedule",
    )

    create_snapshot = KubernetesPodOperator(
        task_id="create_W1_eow_snapshot",
        name="create-w1-eow-snapshot",
        namespace="airflow",
        image=_ECR_IMAGE,
        image_pull_policy="IfNotPresent",
        image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
        service_account_name="airflow-worker",
        cmds=["python", "marketdata/usecases/create-eoi-price-snapshot.py"],
        env_vars=[
            k8s.V1EnvVar(name="FX_SYMBOL", value="USDJPY"),
            k8s.V1EnvVar(name="INTERVAL", value="W1"),
            k8s.V1EnvVar(name="TIME_WINDOW_IN_MINUTES", value="10080"),
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
