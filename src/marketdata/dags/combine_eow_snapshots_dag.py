"""DAG: Manually combine & deduplicate the latest EOW snapshots across lanes.

Manually triggered (schedule=None). Weekly counterpart of
``combine_eod_snapshots``: consolidates the latest end-of-week W1 snapshots
from the two weekly lanes (``create_eow_snapshot_2012`` (2012-01-02 →
2019-12-30, complete) and ``create_eow_snapshot_2020`` (2020-01-06 → now,
owns the current frontier)) into a single deduplicated series and overwrites
``create_eow_snapshot_2020``'s latest snapshot partition
(``marketdata/eow-snapshot/.../year=/month=/{ts}.parquet``, TIME_WINDOW=10080).

Supports ``dag_run.conf.extra_sources`` for pre-reset accumulation segments,
same as the eod combine (the eod 2012 lane had one such reset; the combine log
spans reveal whether the eow lanes do too).
"""

from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

_ECR_IMAGE = (
    "731833471586.dkr.ecr.ap-southeast-1.amazonaws.com"
    "/forex-marketdata-download-interval-price-data:latest"
)

_TARGET_DAG = "create_eow_snapshot_2020"
_SOURCE_LANES = [
    "create_eow_snapshot_2012",
    "create_eow_snapshot_2020",
]

_LANE_SQL_LIST = ", ".join(f"'{d}'" for d in _SOURCE_LANES)

_RESOLVE_CMD = f"""
set -eu
mkdir -p /airflow/xcom
psql -h airflow-postgresql -U postgres -d postgres -tA -v ON_ERROR_STOP=1 \
  -c "SELECT COALESCE(json_object_agg(dag_id, ts)::text, '{{}}') FROM (
        SELECT dag_id,
               to_char(max(data_interval_end) FILTER (WHERE state='success') AT TIME ZONE 'UTC',
                       'YYYY-MM-DD\\"T\\"HH24:MI:SS') AS ts
        FROM dag_run
        WHERE dag_id IN ({_LANE_SQL_LIST})
        GROUP BY dag_id) x;" > /airflow/xcom/return.json
echo "resolved frontiers:"; cat /airflow/xcom/return.json
"""

with DAG(
    dag_id="combine_eow_snapshots",
    description="Manually combine & dedup latest EOW W1 snapshots from the 2012+2020 lanes into create_eow_snapshot_2020's latest partition",
    schedule=None,
    catchup=False,
    tags=["marketdata", "snapshot", "manual", "maintenance"],
    default_args={"owner": "fintech", "retries": 1},
) as dag:
    resolve_frontiers = KubernetesPodOperator(
        task_id="resolve_frontiers",
        name="combine-eow-resolve-frontiers",
        namespace="airflow",
        image="postgres:16-alpine",
        cmds=["/bin/sh", "-c", _RESOLVE_CMD],
        env_vars=[
            k8s.V1EnvVar(
                name="PGPASSWORD",
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="airflow-postgresql", key="postgres-password"
                    )
                ),
            ),
        ],
        do_xcom_push=True,
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "50m", "memory": "64Mi"},
            limits={"cpu": "250m", "memory": "128Mi"},
        ),
        startup_timeout_seconds=300,
        is_delete_operator_pod=True,
        get_logs=True,
    )

    combine_snapshots = KubernetesPodOperator(
        task_id="combine_snapshots",
        name="combine-eow-snapshots",
        namespace="airflow",
        image=_ECR_IMAGE,
        image_pull_policy="Always",
        image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
        service_account_name="airflow-worker",
        cmds=["python", "marketdata/usecases/combine-eoh-snapshots.py"],
        env_vars=[
            k8s.V1EnvVar(name="FX_SYMBOL", value="USDJPY"),
            k8s.V1EnvVar(name="TIME_WINDOW_IN_MINUTES", value="10080"),
            k8s.V1EnvVar(name="INTERVALS", value="W1"),
            k8s.V1EnvVar(name="TARGET_DAG", value=_TARGET_DAG),
            k8s.V1EnvVar(
                name="SOURCE_FRONTIERS",
                value="{{ ti.xcom_pull(task_ids='resolve_frontiers') | tojson }}",
            ),
            k8s.V1EnvVar(
                name="EXTRA_SOURCES",
                value="{{ dag_run.conf.get('extra_sources', {}) | tojson }}",
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
            limits={"cpu": "500m", "memory": "1Gi"},
        ),
        startup_timeout_seconds=600,
        is_delete_operator_pod=True,
        get_logs=True,
    )

    resolve_frontiers >> combine_snapshots
