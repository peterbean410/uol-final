"""Anti-starvation guard for serial catchup DAGs.

On a serial DAG (``max_active_runs=1`` + ``depends_on_past=True``), if two
non-success runs are ever in flight at once, e.g. after a manual clear/re-link
leaves more than one run queued; the scheduler can leave the **later** run
holding the single ``max_active_runs`` slot while it is blocked by
``depends_on_past`` on an **earlier** run it depends on. The earlier run then
stays ``queued`` forever and can never get the slot: a starvation deadlock.

It is invisible to the usual failure scans because **no task fails**; every task
in the wrongly-active later run just sits in ``NONE`` (blocked), and the earlier
run's tasks never schedule. We hit this repeatedly on the 2012 aggregate/snapshot
backfill chains after clearing runs.

This guard runs every 10 minutes and heals it deterministically with one atomic
SQL statement: for any unpaused ``max_active_runs=1`` DAG whose running run is
NOT its earliest non-success run, it demotes the later run(s) back to ``queued``
and activates the earliest, scoping the activation to only the DAGs it actually
fixed. In steady state (one run in flight) it is a no-op.

Implemented as a KubernetesPodOperator running ``psql`` (the executor pattern the
rest of this pipeline uses) rather than a TaskFlow ``@task``, whose worker pod
fails to load the DAG bundle in this KubernetesExecutor + GitDagBundle setup.
"""

import pendulum
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

# One atomic statement. The CTE demotes every 'running' run on a serial unpaused
# DAG that is LATER than that DAG's earliest non-success run; the outer UPDATE
# then activates the earliest non-success queued run, but only for DAGs that the
# CTE just demoted (so idle DAGs are never touched).
_HEAL_SQL = (
    "WITH demoted AS ("
    " UPDATE dag_run dr SET state='queued' FROM dag d"
    " WHERE d.dag_id=dr.dag_id AND d.is_paused=false AND d.max_active_runs=1"
    " AND dr.state='running'"
    " AND dr.logical_date > (SELECT min(e.logical_date) FROM dag_run e"
    " WHERE e.dag_id=dr.dag_id AND e.state<>'success')"
    " RETURNING dr.dag_id)"
    " UPDATE dag_run dr SET state='running', start_date=COALESCE(dr.start_date, now())"
    " FROM dag d"
    " WHERE d.dag_id=dr.dag_id AND dr.dag_id IN (SELECT DISTINCT dag_id FROM demoted)"
    " AND dr.state='queued'"
    " AND dr.logical_date=(SELECT min(e.logical_date) FROM dag_run e"
    " WHERE e.dag_id=dr.dag_id AND e.state<>'success')"
    " AND NOT EXISTS (SELECT 1 FROM dag_run x WHERE x.dag_id=dr.dag_id AND x.state='running');"
)

with DAG(
    dag_id="anti_starvation_guard",
    schedule="*/10 * * * *",
    start_date=pendulum.datetime(2026, 6, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["maintenance", "scheduler"],
) as dag:
    KubernetesPodOperator(
        task_id="heal",
        name="anti-starvation-heal",
        namespace="airflow",
        image="postgres:16-alpine",
        image_pull_policy="IfNotPresent",
        cmds=["sh", "-c"],
        arguments=[
            'PGPASSWORD="$POSTGRES_PASSWORD" psql -h airflow-postgresql -U postgres '
            '-d postgres -v ON_ERROR_STOP=1 -c "' + _HEAL_SQL + '"'
        ],
        env_vars=[
            k8s.V1EnvVar(
                name="POSTGRES_PASSWORD",
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="airflow-postgresql", key="postgres-password"
                    )
                ),
            )
        ],
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "50m", "memory": "64Mi"},
            limits={"cpu": "250m", "memory": "128Mi"},
        ),
        is_delete_operator_pod=True,
        get_logs=True,
    )
