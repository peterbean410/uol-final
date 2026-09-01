"""Airflow DAG: manually warm the modelenv-cache PVC in peterbean namespace.

dqn_training and dqn_backtest pods both mount the ``modelenv-cache`` RWX
PVC at ``/tmp/modelenv-cache`` (see ``platform/peterbean/modelenv-cache-pvc.yaml``
and ``deepqnetwork/kubeflow/pipeline/dqn_pipeline.py``). The first pod to
touch the PVC after it's created still has to download the full market
data from S3; a multi-minute cold pull that the production training
pipeline shouldn't have to absorb interactively.

This DAG submits a one-off ``Job`` into ``peterbean`` that runs the same
``modelenv-server`` binary as the training sidecar against the same PVC,
watches its log for the ``Training market data preload complete`` marker,
then sends it SIGTERM and exits cleanly. After it succeeds, the PVC holds
a hot cache and all subsequent dqn-training pods skip the long S3 pull.

Manual trigger only. Override symbol / s3_prefix / price_snapshot_ts via
the DAG-run config in the Airflow UI.

Requirements: DQN-R22, DQN-R23
"""

from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator


_NAMESPACE = "peterbean"
_PVC_NAME = "modelenv-cache"
_PVC_MOUNT_PATH = "/tmp/modelenv-cache"
_DEFAULT_IMAGE = (
    "731833471586.dkr.ecr.ap-southeast-1.amazonaws.com/dqn/training:latest"
)
_IMAGE_PULL_SECRET = "peterbean-ecr-credentials"
_SERVICE_ACCOUNT = "default-editor"
_S3_SECRET_NAME = "s3-secrets"

_WARMUP_SCRIPT = r"""
set -eu

LOG=/tmp/modelenv-warmup.log
: > "$LOG"

CMD="modelenv-server --mode training --addr 0.0.0.0:50051 \
  --symbol $SYMBOL --s3-prefix $S3_PREFIX \
  --local-cache-dir /tmp/modelenv-cache"
if [ -n "${PRICE_SNAPSHOT_TS:-}" ]; then
  CMD="$CMD --price-snapshot-ts $PRICE_SNAPSHOT_TS"
fi

echo "Launching: $CMD"
$CMD >"$LOG" 2>&1 &
PID=$!

DEADLINE=$(( $(date +%s) + ${WARMUP_TIMEOUT_SECONDS:-3600} ))

while kill -0 "$PID" 2>/dev/null; do
  if grep -q "Training market data preload complete" "$LOG"; then
    echo "Preload complete, stopping modelenv-server (pid $PID)"
    kill -TERM "$PID" 2>/dev/null || true
    sleep 5
    kill -KILL "$PID" 2>/dev/null || true
    echo "--- modelenv-server log (last 40 lines) ---"
    tail -n 40 "$LOG"
    exit 0
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "Warmup deadline exceeded" >&2
    kill -KILL "$PID" 2>/dev/null || true
    tail -n 40 "$LOG"
    exit 2
  fi
  sleep 5
done

echo "modelenv-server exited before reaching preload-complete marker" >&2
echo "--- modelenv-server log (last 80 lines) ---" >&2
tail -n 80 "$LOG" >&2
exit 1
"""


def _normalise_snapshot_ts(value: str) -> str:
    """Coerce a user-supplied snapshot timestamp to modelenv's i64-ns form.

    modelenv-server's ``--price-snapshot-ts`` flag expects an integer
    nanosecond Unix timestamp. The DAG param is friendlier to humans,
    accept either:
      - ``""``                              → no flag emitted (latest)
      - ``"1356912000000000000"`` (ns int)  → passed through
      - ``"1356912000"`` (s int, 10 digits) → multiplied by 1e9
      - ``"2012-12-31"``                    → 00:00:00 UTC, → ns
      - ``"2012-12-31T17:00:00"`` or ``"...+00:00"`` → → ns

    Anything else raises so the DAG fails fast with a clear message.
    """
    value = value.strip()
    if not value:
        return ""

    if value.lstrip("-").isdigit():
        as_int = int(value)
        if len(str(abs(as_int))) <= 10:
            as_int *= 1_000_000_000
        elif len(str(abs(as_int))) <= 13:
            as_int *= 1_000_000
        return str(as_int)

    from datetime import datetime, timezone

    iso = value
    if "T" not in iso:
        iso = f"{iso}T00:00:00+00:00"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError as exc:
        raise ValueError(
            f"price_snapshot_ts {value!r} is neither an integer epoch nor an "
            "ISO 8601 date/datetime (e.g. '2012-12-31' or "
            "'2012-12-31T00:00:00+00:00')"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return str(int(dt.timestamp() * 1_000_000_000))


def _warmup_modelenv_cache(**context):
    """Submit and poll a peterbean-namespace Job that warms the PVC."""
    import hashlib
    import time

    from kubernetes import client, config as k8s_config

    params = context["params"]
    symbol = params["symbol"]
    s3_prefix = params["s3_prefix"]
    snapshot_ts = _normalise_snapshot_ts(
        params.get("price_snapshot_ts", "") or ""
    )
    image = params.get("image") or _DEFAULT_IMAGE
    warmup_timeout = int(params.get("warmup_timeout_seconds", 3600))
    poll_timeout = int(params.get("poll_timeout_seconds", warmup_timeout + 600))

    k8s_config.load_incluster_config()
    batch = client.BatchV1Api()
    core = client.CoreV1Api()

    ti = context.get("ti") or context.get("task_instance")
    try_number = getattr(ti, "try_number", 1) if ti is not None else 1
    key = f"{context['run_id']}|{try_number}"
    slug = hashlib.sha1(key.encode()).hexdigest()[:10]
    job_name = f"modelenv-warmup-{symbol.lower()}-{slug}"[:63].rstrip("-")

    container = client.V1Container(
        name="modelenv-warmup",
        image=image,
        image_pull_policy="Always",
        command=["/bin/sh", "-c", _WARMUP_SCRIPT],
        env=[
            client.V1EnvVar(name="SYMBOL", value=symbol),
            client.V1EnvVar(name="S3_PREFIX", value=s3_prefix),
            client.V1EnvVar(name="PRICE_SNAPSHOT_TS", value=snapshot_ts),
            client.V1EnvVar(name="WARMUP_TIMEOUT_SECONDS", value=str(warmup_timeout)),
        ],
        env_from=[
            client.V1EnvFromSource(
                secret_ref=client.V1SecretEnvSource(name=_S3_SECRET_NAME),
            ),
        ],
        volume_mounts=[
            client.V1VolumeMount(name="modelenv-cache", mount_path=_PVC_MOUNT_PATH),
        ],
        resources=client.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "4Gi"},
            limits={"cpu": "2", "memory": "16Gi"},
        ),
    )

    pod_spec = client.V1PodSpec(
        restart_policy="Never",
        service_account_name=_SERVICE_ACCOUNT,
        image_pull_secrets=[
            client.V1LocalObjectReference(name=_IMAGE_PULL_SECRET),
        ],
        containers=[container],
        volumes=[
            client.V1Volume(
                name="modelenv-cache",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=_PVC_NAME,
                ),
            ),
        ],
    )

    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=job_name,
            namespace=_NAMESPACE,
            labels={
                "app.kubernetes.io/name": "modelenv-warmup",
                "app.kubernetes.io/managed-by": "airflow",
                "symbol": symbol.lower(),
            },
        ),
        spec=client.V1JobSpec(
            backoff_limit=1,
            ttl_seconds_after_finished=3600,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={
                        "app.kubernetes.io/name": "modelenv-warmup",
                        "symbol": symbol.lower(),
                    },
                    annotations={
                        "sidecar.istio.io/inject": "false",
                    },
                ),
                spec=pod_spec,
            ),
        ),
    )

    print(
        f"Submitting Job {_NAMESPACE}/{job_name} "
        f"symbol={symbol} s3_prefix={s3_prefix} snapshot_ts={snapshot_ts!r}"
    )
    batch.create_namespaced_job(namespace=_NAMESPACE, body=job)

    deadline = time.time() + poll_timeout
    last_phase = None
    while time.time() < deadline:
        j = batch.read_namespaced_job_status(name=job_name, namespace=_NAMESPACE)
        st = j.status
        if st.succeeded:
            print(f"Job {_NAMESPACE}/{job_name} succeeded")
            _dump_pod_logs(core, job_name)
            return
        if st.failed and st.failed >= ((j.spec.backoff_limit or 0) + 1):
            _dump_pod_logs(core, job_name)
            raise RuntimeError(
                f"Job {_NAMESPACE}/{job_name} failed (failed pod count={st.failed})"
            )
        phase = f"active={st.active} succeeded={st.succeeded} failed={st.failed}"
        if phase != last_phase:
            print(f"Job {_NAMESPACE}/{job_name} status: {phase}")
            last_phase = phase
        time.sleep(15)

    _dump_pod_logs(core, job_name)
    raise RuntimeError(
        f"Job {_NAMESPACE}/{job_name} timed out after {poll_timeout}s"
    )


def _dump_pod_logs(core, job_name: str) -> None:
    """Best-effort dump of the most recent warmup pod's logs into Airflow."""
    try:
        pods = core.list_namespaced_pod(
            namespace=_NAMESPACE, label_selector=f"job-name={job_name}"
        )
        for p in pods.items[:2]:
            try:
                log = core.read_namespaced_pod_log(
                    name=p.metadata.name,
                    namespace=_NAMESPACE,
                    container="modelenv-warmup",
                    tail_lines=200,
                )
            except Exception as e:  # noqa: BLE001
                log = f"<unable to read logs: {e}>"
            print(f"--- pod {p.metadata.name} log ---\n{log}")
    except Exception as e:  # noqa: BLE001
        print(f"<unable to list pods for job {job_name}: {e}>")


default_args = {
    "owner": "ml-engineering",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="dqn_warmup_modelenv_cache",
    start_date=datetime(2026, 5, 21),
    schedule=None,
    default_args=default_args,
    catchup=False,
    tags=["dqn", "kubernetes", "warmup", "cache"],
    params={
        "symbol": "USDJPY",
        "s3_prefix": "s3://prod-fintech-forex-sg-731833471586",
        "price_snapshot_ts": "",
        "image": _DEFAULT_IMAGE,
        "warmup_timeout_seconds": 3600,
        "poll_timeout_seconds": 4200,
    },
) as dag:
    run_warmup = PythonOperator(
        task_id="warmup_modelenv_cache",
        python_callable=_warmup_modelenv_cache,
    )
