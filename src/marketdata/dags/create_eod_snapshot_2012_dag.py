"""DAG: Create end-of-day (EOD) price snapshots for H1, H4, D1, 2012 backfill.

Mirror of ``create_eod_snapshot_2020`` but with ``start_date`` rolled back to
2012-01-02 so the 2012-01-01 .. 2019-12-31 gap in
``s3://prod-fintech-forex-sg-731833471586/marketdata/eod-snapshot/`` can be
backfilled. The 2020 DAG continues to own 2020-01-02 onward, so this DAG
declares ``end_date=2019-12-31`` to avoid double-running the same days.

Runs daily at 00:00 UTC. On Tue–Sat the run waits for the matching daily-
aggregate task in ``aggregate_daily_interval_price_2012`` (the 2012-side
aggregate DAG must exist, same way the 2020 DAG depends on its 2020 sibling).
On Sun and Mon (no aggregate fires) the wait is bypassed so the snapshot
chain stays unbroken, ``create_snapshot`` then re-emits the prior snapshot's
contents under the weekend key.

Snapshots land under ``marketdata/eod-snapshot/...``.
"""

from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.trigger_rule import TriggerRule
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

_INTERVALS = ["H1", "H4", "D1"]
# aggregate_daily_interval_price_2012 runs Tue–Sat (cron "0 0 * * 2-6").
# Python weekday(): Mon=0…Sun=6. Skip the wait on Sun and Mon.
_NO_UPSTREAM_WEEKDAYS = {6, 0}


def _make_branch(wait_task_id: str, skip_task_id: str):
    def _branch(data_interval_end, **_):
        if data_interval_end.weekday() in _NO_UPSTREAM_WEEKDAYS:
            return skip_task_id
        return wait_task_id
    return _branch


with DAG(
    max_active_runs=1,  # depends_on_past serial; =1 prevents max_active_runs starvation deadlock
    dag_id="create_eod_snapshot_2012",
    default_args=default_args,
    description=(
        "Create FX end-of-day price snapshots (H1, H4, D1) from "
        "daily-aggregated bars, 2012-01-02 through 2019-12-31 backfill"
    ),
    schedule="0 0 * * *",
    start_date=datetime(2012, 1, 2),
    end_date=datetime(2021, 12, 31, 23, 59, 59),
    catchup=True,
    tags=["marketdata", "pricedata", "snapshot", "eod", "backfill"],
) as dag:

    for interval in _INTERVALS:
        wait_id = f"wait_for_aggregate_{interval}"
        skip_id = f"skip_wait_{interval}"

        branch = BranchPythonOperator(
            task_id=f"branch_{interval}",
            python_callable=_make_branch(wait_id, skip_id),
        )

        wait_for_aggregate = ExternalTaskSensor(
            task_id=wait_id,
            external_dag_id="aggregate_daily_interval_price_2012",
            external_task_id=f"aggregate_{interval}_interval_price",
            poke_interval=300,
            timeout=86400,
            mode="reschedule",
        )

        skip_wait = EmptyOperator(task_id=skip_id)

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
            trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        )

        branch >> [wait_for_aggregate, skip_wait] >> create_snapshot
