"""DAG: Manually combine & deduplicate the latest EOD snapshots across lanes.

Manually triggered (schedule=None). EOD counterpart of ``combine_eoh_snapshots``:
consolidates the latest end-of-day price snapshots (H1, H4, D1) from the two
open-ended daily lanes (``create_eod_snapshot_2012`` (accumulating from
2012-01-02) and ``create_eod_snapshot_2020`` (accumulating from 2020-01-02,
owns the current frontier)) into a single deduplicated series and overwrites
``create_eod_snapshot_2020``'s latest snapshot partition with the result.

Each lane holds only ``[lane_start .. its own frontier]``, so the current
frontier snapshot spans 2020→now while 2012–2019 history sits at the 2012
lane's frontier. Unioning both frontiers and deduping on (Timestamp, Symbol)
rebuilds the complete 2012→now series; the 2020 lane's next daily run then
carries it forward automatically (same accumulate-previous mechanics).

Same two-task shape as the eoh combine: a postgres pod resolves the lanes'
live frontiers from the metadata DB into XCom (the marketdata image has no PG
driver), then the marketdata pod does the S3/pandas combine with
TIME_WINDOW_IN_MINUTES=1440 (day-partitioned keys, eod-snapshot root).
"""

from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

_ECR_IMAGE = (
    "731833471586.dkr.ecr.ap-southeast-1.amazonaws.com"
    "/forex-marketdata-download-interval-price-data:latest"
)

_TARGET_DAG = "create_eod_snapshot_2020"
_SOURCE_LANES = [
    "create_eod_snapshot_2012",
    "create_eod_snapshot_2020",
]

# SQL literal list of the lanes to resolve frontiers for.
_LANE_SQL_LIST = ", ".join(f"'{d}'" for d in _SOURCE_LANES)

# Resolve each lane's latest snapshot timestamp (its latest successful run's
# data_interval_end, rendered as naive-UTC ISO-8601) and emit one JSON object to
# the KPO XCom sidecar path. Lanes with no success come through as JSON null and
# are skipped downstream.
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
    dag_id="combine_eod_snapshots",
    description="Manually combine & dedup latest EOD snapshots (H1,H4,D1) from the 2012+2020 lanes into create_eod_snapshot_2020's latest partition",
    schedule=None,
    catchup=False,
    tags=["marketdata", "snapshot", "manual", "maintenance"],
    default_args={"owner": "fintech", "retries": 1},
) as dag:
    resolve_frontiers = KubernetesPodOperator(
        task_id="resolve_frontiers",
        name="combine-eod-resolve-frontiers",
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
        startup_timeout_seconds=300,  # tolerate node "Too many pods" scheduling delay
        is_delete_operator_pod=True,
        get_logs=True,
    )

    combine_snapshots = KubernetesPodOperator(
        task_id="combine_snapshots",
        name="combine-eod-snapshots",
        namespace="airflow",
        image=_ECR_IMAGE,
        image_pull_policy="Always",
        image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
        service_account_name="airflow-worker",
        cmds=["python", "marketdata/usecases/combine-eoh-snapshots.py"],
        env_vars=[
            k8s.V1EnvVar(name="FX_SYMBOL", value="USDJPY"),
            k8s.V1EnvVar(name="TIME_WINDOW_IN_MINUTES", value="1440"),
            k8s.V1EnvVar(name="INTERVALS", value="H1,H4,D1"),
            k8s.V1EnvVar(name="TARGET_DAG", value=_TARGET_DAG),
            k8s.V1EnvVar(
                name="SOURCE_FRONTIERS",
                value="{{ ti.xcom_pull(task_ids='resolve_frontiers') | tojson }}",
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
            # EOD frames are far smaller than M1's (H1 full history ~100K rows);
            # 2Gi is generous head-room.
            requests={"cpu": "250m", "memory": "512Mi"},
            limits={"cpu": "1", "memory": "2Gi"},
        ),
        # Always-pull of the ~177MB :latest can take minutes on a congested
        # node; the 120s default killed the first eoh combine before it ran.
        startup_timeout_seconds=600,
        is_delete_operator_pod=True,
        get_logs=True,
    )

    resolve_frontiers >> combine_snapshots
