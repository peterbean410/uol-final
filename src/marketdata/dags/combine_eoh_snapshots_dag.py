"""DAG: Manually combine & deduplicate the latest EOH snapshots across lanes.

Manually triggered (schedule=None). Consolidates the latest end-of-hour price
snapshots from every open-ended backfill lane (
`create_eoh_snapshot_{2012,2017,2018,2019,2020,2026}`) into a single
deduplicated series and overwrites the target lane's latest snapshot partition
(`create_eoh_snapshot_2026`) with the result, for the M1, M5 and M15 intervals.

Each lane is open-ended and overlapping and holds only `[lane_start .. its own
frontier]` of accumulated history, so no single lane holds the full series. This
job unions every lane's *current* frontier snapshot and dedups on
(Timestamp, Symbol) to rebuild the complete history.

Two tasks:
  1. resolve_frontiers; a tiny postgres:16-alpine pod queries the Airflow
     metadata DB for each lane's latest *successful* run's data_interval_end
     (== the timestamp its latest snapshot is keyed on) and pushes the mapping
     as XCom JSON. This is done at run time so the job always uses the true
     current frontiers rather than hard-coded timestamps.
  2. combine_snapshots; the marketdata image (pandas/pyarrow/boto3) loads each
     lane's frontier snapshot from S3, concatenates + dedups, and overwrites the
     target lane's latest partition.

The marketdata image ships no Postgres driver, hence the split: the DB read
lives in the postgres pod, the S3/pandas work in the marketdata pod.
"""

from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

_ECR_IMAGE = (
    "731833471586.dkr.ecr.ap-southeast-1.amazonaws.com"
    "/forex-marketdata-download-interval-price-data:latest"
)

_TARGET_DAG = "create_eoh_snapshot_2026"
_SOURCE_LANES = [
    "create_eoh_snapshot_2012",
    "create_eoh_snapshot_2017",
    "create_eoh_snapshot_2018",
    "create_eoh_snapshot_2019",
    "create_eoh_snapshot_2020",
    "create_eoh_snapshot_2026",
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
    dag_id="combine_eoh_snapshots",
    description="Manually combine & dedup latest EOH snapshots across lanes into create_eoh_snapshot_2026's latest partition",
    schedule=None,
    catchup=False,
    tags=["marketdata", "snapshot", "manual", "maintenance"],
    default_args={"owner": "fintech", "retries": 1},
) as dag:
    resolve_frontiers = KubernetesPodOperator(
        task_id="resolve_frontiers",
        name="combine-eoh-resolve-frontiers",
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
        name="combine-eoh-snapshots",
        namespace="airflow",
        image=_ECR_IMAGE,
        image_pull_policy="Always",
        image_pull_secrets=[k8s.V1LocalObjectReference(name="ecr-registry-credentials")],
        service_account_name="airflow-worker",
        cmds=["python", "marketdata/usecases/combine-eoh-snapshots.py"],
        env_vars=[
            k8s.V1EnvVar(name="FX_SYMBOL", value="USDJPY"),
            k8s.V1EnvVar(name="INTERVALS", value="M1,M5,M15"),
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
            requests={"cpu": "250m", "memory": "1Gi"},
            limits={"cpu": "1", "memory": "8Gi"},
        ),
        startup_timeout_seconds=600,
        is_delete_operator_pod=True,
        get_logs=True,
    )

    resolve_frontiers >> combine_snapshots
