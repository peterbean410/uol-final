"""DQN KFP Pipeline Definition.

Defines the dqn_pipeline using KFP v2 SDK with @dsl.container_component
decorators. Wires dqn_training → dqn_backtest as a 2-step DAG with
retry(2) on training and retry(1) on backtest.

Training component runs with modelenv sidecar (1 GPU, 16Gi memory).
Backtest component runs CPU-only (4Gi memory).

step_size_seconds, num_episodes_per_range, batch_size, learning_rate, training_mode,
checkpoint (for finetune).

Integrates DQNPipelineConfig loading and validation at pipeline start.

Requirements: DQN-R11, DQN-R12
"""

import json
import os
from typing import NamedTuple

from kfp import dsl, kubernetes
from kfp.dsl import Input, Metrics, Model, Output

from deepqnetwork.kubeflow.pipeline.config_schema import DQNPipelineConfig


SPARK_NODE_HOSTNAME: str = os.getenv(
    "DQN_SPARK_NODE_HOSTNAME", "spark-5790"
).strip()

SPARK_TAINTS = (
    ("workload", "ml"),
    ("arch", "arm64"),
)
SPARK_NODE_SELECTOR = (
    ("kubernetes.io/arch", "arm64"),
    ("nvidia.com/gpu.present", "true"),
)


ECR_BASE = "731833471586.dkr.ecr.ap-southeast-1.amazonaws.com"

MODELENV_CACHE_PVC = "modelenv-cache"
MODELENV_CACHE_MOUNT = "/tmp/modelenv-cache"


def _mount_minio_creds(task) -> None:
    """Inject MinIO accesskey/secretkey from the mlpipeline-minio-artifact secret.

    `dqn_model_registration` reads the backtest_metrics artifact via boto3 and
    needs MINIO_ACCESS_KEY / MINIO_SECRET_KEY to reach the in-cluster MinIO when
    the artifact URI is `minio://...`. The Secret is created by the KFP install
    in the same namespace as each pipeline run.
    """
    kubernetes.use_secret_as_env(
        task,
        secret_name="mlpipeline-minio-artifact",
        secret_key_to_env={
            "accesskey": "MINIO_ACCESS_KEY",
            "secretkey": "MINIO_SECRET_KEY",
        },
    )


def _pin_to_spark(task) -> None:
    """Pin a task onto the designated spark GPU node (``SPARK_NODE_HOSTNAME``).

    Adds a toleration for each spark
    NoExecute taint plus a hostname nodeSelector when ``SPARK_NODE_HOSTNAME``
    is set (the default, spark-5790, keeps training/backtest off spark-4214
    where the gemma LLM predictors live). With an empty hostname it falls back
    to the identifying label set (arch=arm64 ∩ GPU present), making the pod
    eligible on either spark node.

    Requires the multi-arch (arm64) dqn images; an amd64-only image will fail to
    exec on these nodes regardless of scheduling.
    """
    for key, value in SPARK_TAINTS:
        kubernetes.add_toleration(
            task,
            key=key,
            operator="Equal",
            value=value,
            effect="NoExecute",
        )
    if SPARK_NODE_HOSTNAME:
        kubernetes.add_node_selector(
            task,
            label_key="kubernetes.io/hostname",
            label_value=SPARK_NODE_HOSTNAME,
        )
        return
    for label_key, label_value in SPARK_NODE_SELECTOR:
        kubernetes.add_node_selector(
            task,
            label_key=label_key,
            label_value=label_value,
        )


def _enable_gpu(task) -> None:
    """Give a task GPU access.

    On the spark node the containerd default runtime is ``nvidia`` (set
    node-side), so a pod gets the GPU just by carrying the NVIDIA env vars;
    no ``nvidia.com/gpu`` resource request, which lets the single GPU be
    shared across pods (the same pattern the gemma LLM predictors use).

    """
    task.set_env_variable("NVIDIA_VISIBLE_DEVICES", "all")
    task.set_env_variable("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")


@dsl.container_component
def dqn_training(
    symbol: str,
    episode_start_ts: int,
    episode_end_ts: int,
    step_size_seconds: int,
    num_episodes_per_range: int,
    repeats_per_date: int,
    batch_size: int,
    learning_rate: float,
    training_mode: str,
    checkpoint: str,
    config_json: str,
    date_start: str,
    date_end: str,
    hour_of_day_start: int,
    hour_of_day_end: int,
    model_checkpoint: Output[Model],
    price_snapshot_date: str = "",
    reward_action_penalty: str = "",
    reward_holding_penalty: str = "",
    reward_lambda: str = "",
):
    """Train DQN agent against modelenv gRPC sidecar.

    Loads DQNPipelineConfig, launches modelenv as a subprocess sidecar,
    runs training, and outputs a checkpoint artifact to S3.

    Supports scratch (random init, full episodes) and finetune
    (load production checkpoint, reduced LR, fewer episodes) modes.
    reward_* are empty-string-default overrides for the decision-persistence
    experiments (turnover/holding/loss-aversion reward shaping).
    """
    return dsl.ContainerSpec(
        image=f"{ECR_BASE}/dqn/training:latest",
        command=["python", "component.py"],
        args=[
            "--checkpoint-output-path", model_checkpoint.uri,
            "--symbol", symbol,
            "--episode-start-ts", str(episode_start_ts),
            "--episode-end-ts", str(episode_end_ts),
            "--step-size-seconds", str(step_size_seconds),
            "--num-episodes-per-range", str(num_episodes_per_range),
            "--repeats-per-date", str(repeats_per_date),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--training-mode", training_mode,
            "--production-checkpoint-path", checkpoint,
            "--date-start", date_start,
            "--date-end", date_end,
            "--hour-start", str(hour_of_day_start),
            "--hour-end", str(hour_of_day_end),
            "--price-snapshot-date", price_snapshot_date,
            "--reward-action-penalty", reward_action_penalty,
            "--reward-holding-penalty", reward_holding_penalty,
            "--reward-lambda", reward_lambda,
        ],
    )


@dsl.container_component
def dqn_backtest(
    model_checkpoint: Input[Model],
    symbol: str,
    episode_start_ts: int,
    episode_end_ts: int,
    step_size_seconds: int,
    config_json: str,
    date_start: str,
    date_end: str,
    hour_of_day_start: int,
    hour_of_day_end: int,
    backtest_metrics: Output[Metrics],
):
    """Evaluate DQN checkpoint via modelenv sidecar on unseen date ranges.

    Loads checkpoint into DQNAdvisor, runs evaluation episodes, computes
    metrics (cumulative P&L, Sharpe ratio, max drawdown, win rate, average
    episode reward, average episode length), and runs degradation gate.
    """
    return dsl.ContainerSpec(
        image=f"{ECR_BASE}/dqn/backtest:latest",
        command=["python", "component.py"],
        args=[
            "--checkpoint-path", model_checkpoint.uri,
            "--output-path", backtest_metrics.uri,
            "--symbol", symbol,
            "--eval-episode-start-ts", str(episode_start_ts),
            "--eval-episode-end-ts", str(episode_end_ts),
            "--step-size-seconds", str(step_size_seconds),
            "--date-start", date_start,
            "--date-end", date_end,
            "--hour-start", str(hour_of_day_start),
            "--hour-end", str(hour_of_day_end),
        ],
    )


@dsl.component(
    base_image=f"{ECR_BASE}/dqn/training:latest",
    packages_to_install=["pyyaml"],
)
def resolve_dqn_config(
    symbol: str,
    step_size_seconds: int,
    num_episodes_per_range: int,
    repeats_per_date: int,
    batch_size: int,
    learning_rate: float,
    training_mode: str,
    checkpoint: str,
    date_start: str,
    date_end: str,
    hour_of_day_start: int,
    hour_of_day_end: int,
) -> NamedTuple(
    "DQNConfigOutputs",
    [
        ("config_json", str),
        ("effective_num_episodes_per_range", int),
        ("effective_learning_rate", float),
        ("date_start", str),
        ("date_end", str),
        ("hour_of_day_start", int),
        ("hour_of_day_end", int),
    ],
):
    """Resolve DQN pipeline config: apply overrides, validate, and output.

    This lightweight component runs at the start of the pipeline to:
    1. Build DQNPipelineConfig with parameter overrides
    2. Apply training mode adjustments (finetune → reduced LR/episodes)
    3. Validate the configuration
    4. Output resolved config JSON and effective parameters

    Raises:
        ValueError: If configuration validation fails.
    """
    import json
    import sys
    from collections import namedtuple
    from dataclasses import asdict

    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    from deepqnetwork.kubeflow.pipeline.config_schema import DQNPipelineConfig

    config = DQNPipelineConfig.from_yaml(
        "/app/deepqnetwork/kubeflow/config/dqn_pipeline_config.yaml"
    ).override(
        symbol=symbol,
        step_size_seconds=step_size_seconds,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_mode=training_mode,
        date_start=date_start,
        date_end=date_end,
        hour_of_day_start=hour_of_day_start,
        hour_of_day_end=hour_of_day_end,
    )

    if training_mode == "finetune":
        config = config.override(learning_rate=config.finetune_learning_rate)

    if date_start and date_end:
        config = config.override(repeats_per_date=repeats_per_date)
    else:
        episodes = (
            config.finetune_num_episodes_per_range
            if training_mode == "finetune"
            else num_episodes_per_range
        )
        config = config.override(num_episodes_per_range=episodes)

    errors = config.validate()
    if training_mode == "finetune" and not checkpoint:
        errors.append("Finetune mode requires a checkpoint path")
    if errors:
        raise ValueError(
            f"DQN pipeline configuration validation failed: {'; '.join(errors)}"
        )

    effective_lr = config.learning_rate
    effective_episodes = (
        config.num_episodes_per_range
        if config.num_episodes_per_range is not None
        else 0
    )

    config_dict = asdict(config)
    config_dict["effective_learning_rate"] = effective_lr
    config_dict["effective_num_episodes_per_range"] = effective_episodes
    config_json = json.dumps(config_dict)

    DQNConfigOutputs = namedtuple(
        "DQNConfigOutputs",
        [
            "config_json",
            "effective_num_episodes_per_range",
            "effective_learning_rate",
            "date_start",
            "date_end",
            "hour_of_day_start",
            "hour_of_day_end",
        ],
    )
    return DQNConfigOutputs(
        config_json=config_json,
        effective_num_episodes_per_range=effective_episodes,
        effective_learning_rate=effective_lr,
        date_start=date_start,
        date_end=date_end,
        hour_of_day_start=hour_of_day_start,
        hour_of_day_end=hour_of_day_end,
    )


def build_dqn_pipeline_config(
    symbol: str = "USDJPY",
    step_size_seconds: int = 5,
    num_episodes_per_range: int = 3000,
    repeats_per_date: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    training_mode: str = "scratch",
    checkpoint: str = "",
    date_start: str = "",
    date_end: str = "",
    hour_of_day_start: int = 0,
    hour_of_day_end: int = 23,
) -> DQNPipelineConfig:
    """Build and validate a DQNPipelineConfig from pipeline parameters.

    This function is called at pipeline submission time (client-side) to
    validate configuration before the pipeline runs. Provides fail-fast
    validation before submitting to the KFP engine.

    Args:
        symbol: Currency pair (USDJPY or AUDJPY).
        episode_start_ts: Episode start timestamp.
        episode_end_ts: Episode end timestamp.
        step_size_seconds: Step size in seconds for the environment.
        num_episodes_per_range: Number of training episodes.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        training_mode: "scratch" for full training or "finetune" for
            incremental training on production model weights.
        checkpoint: S3 key path for production checkpoint (required for finetune).

    Returns:
        Validated DQNPipelineConfig.

    Raises:
        ValueError: If configuration validation fails.
    """
    config = DQNPipelineConfig()

    config = config.override(
        symbol=symbol,
        step_size_seconds=step_size_seconds,
        batch_size=batch_size,
        learning_rate=learning_rate,
        date_start=date_start,
        date_end=date_end,
        hour_of_day_start=hour_of_day_start,
        hour_of_day_end=hour_of_day_end,
        training_mode=training_mode,
    )

    if training_mode == "finetune":
        config = config.override(learning_rate=config.finetune_learning_rate)

    if date_start and date_end:
        config = config.override(repeats_per_date=repeats_per_date)
    else:
        episodes = (
            config.finetune_num_episodes_per_range
            if training_mode == "finetune"
            else num_episodes_per_range
        )
        config = config.override(num_episodes_per_range=episodes)

    errors = config.validate()

    if training_mode == "finetune" and not checkpoint:
        errors.append("Finetune mode requires a checkpoint path")

    if errors:
        raise ValueError(
            f"DQN pipeline configuration validation failed: {'; '.join(errors)}"
        )

    return config


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3", "model-registry"],
)
def dqn_model_registration(
    model_checkpoint: Input[Model],
    backtest_metrics: Input[Metrics],
    pipeline_run_id: str,
    registry_url: str,
    symbol: str,
    config_json: str,
):
    """Register the trained DQN agent in Model Registry after the backtest gate.

    Reads the backtest_metrics JSON artifact, checks the degradation gate, and
    registers + promotes the checkpoint to the production stage under the name
    ``deepqnetwork-{symbol.lower()}``; the same name the dqnpf-intraday
    pipeline resolves at evaluation time. Mirrors the forecaster
    `model_registration` flow: bootstrap-promotes the first version when no
    production exists, otherwise demotes the incumbent to staging.

    Gracefully degrades on Model Registry SDK unavailability (logs and
    returns); any other registration failure fails the pipeline step so the
    checkpoint stays in S3 for inspection.
    """
    import json
    import logging
    import os
    import sys
    import uuid
    from datetime import datetime, timezone
    from urllib.parse import urlparse

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def _log(msg: str, extra: dict = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "logger": __name__,
            "message": msg,
            "component": "dqn_model_registration",
        }
        if extra:
            entry.update(extra)
        print(json.dumps(entry), file=sys.stdout)

    backtest_metrics_uri = backtest_metrics.uri
    model_checkpoint_uri = model_checkpoint.uri

    _log("Reading backtest metrics", extra={"uri": backtest_metrics_uri})

    import boto3

    def _get_object_bytes(uri: str) -> bytes:
        """URI-aware read: minio://b/k → MinIO; s3://b/k → AWS S3; bare key → AWS S3."""
        parsed = urlparse(uri)
        scheme = (parsed.scheme or "").lower()
        if scheme == "minio":
            client = boto3.client(
                "s3",
                endpoint_url="http://minio-service.kubeflow:9000",
                aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
                aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
                region_name="us-east-1",
            )
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
        elif scheme == "s3":
            client = boto3.client("s3")
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
        else:
            client = boto3.client("s3")
            bucket = "prod-fintech-forex-sg-731833471586"
            key = uri.lstrip("/")
        return client.get_object(Bucket=bucket, Key=key)["Body"].read()

    try:
        bt = json.loads(_get_object_bytes(backtest_metrics_uri).decode("utf-8"))
    except Exception as e:
        _log(
            "Failed to read backtest metrics; failing pipeline step",
            extra={"error": str(e)},
        )
        raise

    gate = bt.get("degradation_gate", {})
    gate_passed = gate.get("gate_passed", False)
    gate_skipped = gate.get("gate_skipped", False)
    gate_reason = gate.get("reason", "")

    _log(
        "Degradation gate result",
        extra={
            "gate_passed": gate_passed,
            "gate_skipped": gate_skipped,
            "reason": gate_reason,
        },
    )

    if not gate_passed:
        _log(
            "DQN did not pass degradation gate; skipping registration. "
            "Checkpoint retained in S3 for manual review.",
            extra={"gate_reason": gate_reason},
        )
        return

    _log(
        "Gate passed; registering DQN in Model Registry",
        extra={"registry_url": registry_url},
    )

    try:
        from urllib.parse import urlsplit

        from model_registry import ModelRegistry

        parsed = urlsplit(registry_url)
        is_secure = parsed.scheme != "http"
        registry = ModelRegistry(
            server_address=f"{parsed.scheme}://{parsed.hostname}",
            port=parsed.port or (443 if is_secure else 80),
            author="dqn-pipeline",
            is_secure=is_secure,
            user_token="",
        )

        model_name = f"deepqnetwork-{symbol.lower()}"
        version_id = str(uuid.uuid4())
        config = json.loads(config_json) if config_json else {}
        bt_metrics = bt.get("backtest_metrics", {})

        existing_production: list = []
        try:
            for v in registry.get_model_versions(model_name):
                if (v.custom_properties or {}).get(
                    "lifecycle_stage"
                ) == "production":
                    existing_production.append(v)
        except Exception:
            pass

        has_production = bool(existing_production)
        if not has_production:
            _log("No production DQN exists; bootstrapping initial deployment")

        custom_properties = {
            "version_id": version_id,
            "symbol": symbol,
            "step_size_seconds": str(config.get("step_size_seconds", "")),
            "training_mode": str(config.get("training_mode", "")),
            "training_timestamp": bt.get("timestamp", ""),
            "cumulative_pnl": str(bt_metrics.get("cumulative_pnl", "")),
            "sharpe_ratio": str(bt_metrics.get("sharpe_ratio", "")),
            "max_drawdown": str(bt_metrics.get("max_drawdown", "")),
            "win_rate": str(bt_metrics.get("win_rate", "")),
            "avg_episode_reward": str(bt_metrics.get("avg_episode_reward", "")),
            "hyperparameters": json.dumps(
                {
                    "gamma": config.get("gamma"),
                    "epsilon_start": config.get("epsilon_start"),
                    "epsilon_end": config.get("epsilon_end"),
                    "epsilon_decay_steps": config.get("epsilon_decay_steps"),
                    "batch_size": config.get("batch_size"),
                    "learning_rate": config.get("learning_rate"),
                    "replay_buffer_size": config.get("replay_buffer_size"),
                    "target_update_freq": config.get("target_update_freq"),
                    "hidden_dims": config.get("hidden_dims"),
                    "num_episodes_per_range": config.get("num_episodes_per_range"),
                }
            ),
            "pipeline_run_id": pipeline_run_id,
            "lifecycle_stage": "production" if not has_production else "staging",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        registry.register_model(
            name=model_name,
            uri=model_checkpoint_uri,
            model_format_name="pytorch",
            model_format_version="2",
            version=version_id,
            description=(
                f"DQN trained on {symbol} step_size={config.get('step_size_seconds')}s "
                f"(run={pipeline_run_id})"
            ),
            version_description=f"version {version_id}",
            metadata=custom_properties,
        )

        _log(
            "DQN registered successfully",
            extra={
                "version_id": version_id,
                "model_name": model_name,
                "sharpe_ratio": bt_metrics.get("sharpe_ratio"),
                "cumulative_pnl": bt_metrics.get("cumulative_pnl"),
            },
        )

        if has_production:
            new_version = registry.get_model_version(model_name, version_id)
            if new_version is not None:
                new_version.custom_properties["lifecycle_stage"] = "production"
                registry.update(new_version)

            for prev in existing_production:
                prev.custom_properties["lifecycle_stage"] = "staging"
                registry.update(prev)
                _log(
                    "Demoted previous production DQN version",
                    extra={
                        "version_id": (prev.custom_properties or {}).get(
                            "version_id"
                        )
                    },
                )

        _log(
            "DQN promoted to production",
            extra={"version_id": version_id, "model_name": model_name},
        )

    except ImportError as e:
        _log(
            "Model Registry SDK not available; skipping registration",
            extra={"error": str(e)},
        )
    except Exception as e:
        _log(
            "DQN registration failed; failing the pipeline step. "
            "Model checkpoint remains in S3.",
            extra={"error": str(e)},
        )
        raise


@dsl.pipeline(
    name="dqn-pipeline",
    description=(
        "Deep Q-Network Training Pipeline: "
        "config validation → DQN training (with modelenv sidecar) "
        "→ DQN backtest (with modelenv sidecar)"
    ),
)
def dqn_pipeline(
    symbol: str = "USDJPY",
    step_size_seconds: int = 5,
    num_episodes_per_range: int = 3000,
    repeats_per_date: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    training_mode: str = "scratch",
    checkpoint: str = "",
    date_start: str = "",
    date_end: str = "",
    eval_date_start: str = "2015-01-05",
    eval_date_end: str = "2015-03-31",
    hour_of_day_start: int = 0,
    hour_of_day_end: int = 23,
    model_registry_url: str = "http://model-registry-service.kubeflow.svc.cluster.local:8080",
    price_snapshot_date: str = "",
    reward_action_penalty: str = "",
    reward_holding_penalty: str = "",
    reward_lambda: str = "",
):
    """DQN Pipeline: config → train → backtest.

    Config loading and validation is performed both:
    - At submission time via build_dqn_pipeline_config() (client-side, fail-fast)
    - At runtime via the resolve_dqn_config component (in-cluster validation)

    The pipeline DAG wires two steps:

    1. dqn_training: trains DQN agent against modelenv sidecar, outputs checkpoint
    2. dqn_backtest: evaluates checkpoint on unseen date ranges, runs degradation gate

    DAG structure:
        resolve_dqn_config → dqn_training → dqn_backtest

    Resource allocation:
    - Training: 1 GPU (nvidia.com/gpu: 1);
      CPU-only; 16Gi memory, modelenv sidecar
    - Backtest: CPU-only, 4Gi memory, modelenv sidecar

    Args:
        symbol: Currency pair (USDJPY or AUDJPY).
        episode_start_ts: Episode start timestamp for training.
        episode_end_ts: Episode end timestamp for training.
        step_size_seconds: Step size in seconds for the environment.
        num_episodes_per_range: Number of training episodes (overridden in finetune mode).
        batch_size: Training batch size.
        learning_rate: Learning rate (overridden in finetune mode).
        training_mode: "scratch" for full training or "finetune" for
            incremental training on production model weights.
        checkpoint: S3 key path for production checkpoint (required for finetune).
    """
    config_task = resolve_dqn_config(
        symbol=symbol,
        step_size_seconds=step_size_seconds,
        num_episodes_per_range=num_episodes_per_range,
        repeats_per_date=repeats_per_date,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_mode=training_mode,
        checkpoint=checkpoint,
        date_start=date_start,
        date_end=date_end,
        hour_of_day_start=hour_of_day_start,
        hour_of_day_end=hour_of_day_end,
    )
    config_task.set_caching_options(enable_caching=False)

    training_task = dqn_training(
        symbol=symbol,
        episode_start_ts=0,
        episode_end_ts=0,
        step_size_seconds=step_size_seconds,
        num_episodes_per_range=num_episodes_per_range,
        repeats_per_date=repeats_per_date,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_mode=training_mode,
        checkpoint=checkpoint,
        config_json=config_task.outputs["config_json"],
        date_start=config_task.outputs["date_start"],
        date_end=config_task.outputs["date_end"],
        hour_of_day_start=config_task.outputs["hour_of_day_start"],
        hour_of_day_end=config_task.outputs["hour_of_day_end"],
        price_snapshot_date=price_snapshot_date,
        reward_action_penalty=reward_action_penalty,
        reward_holding_penalty=reward_holding_penalty,
        reward_lambda=reward_lambda,
    )
    training_task.set_retry(num_retries=2)
    training_task.set_caching_options(enable_caching=False)
    training_task.set_memory_request("96Gi")
    training_task.set_memory_limit("96Gi")
    _enable_gpu(training_task)
    kubernetes.mount_pvc(
        training_task,
        pvc_name=MODELENV_CACHE_PVC,
        mount_path=MODELENV_CACHE_MOUNT,
    )
    _mount_minio_creds(training_task)
    _pin_to_spark(training_task)

    backtest_task = dqn_backtest(
        model_checkpoint=training_task.outputs["model_checkpoint"],
        symbol=symbol,
        episode_start_ts=0,
        episode_end_ts=0,
        step_size_seconds=step_size_seconds,
        config_json=config_task.outputs["config_json"],
        date_start=eval_date_start,
        date_end=eval_date_end,
        hour_of_day_start=config_task.outputs["hour_of_day_start"],
        hour_of_day_end=config_task.outputs["hour_of_day_end"],
    )
    backtest_task.set_retry(num_retries=1)
    backtest_task.set_caching_options(enable_caching=False)
    backtest_task.set_memory_request("24Gi")
    backtest_task.set_memory_limit("24Gi")
    kubernetes.mount_pvc(
        backtest_task,
        pvc_name=MODELENV_CACHE_PVC,
        mount_path=MODELENV_CACHE_MOUNT,
    )
    _mount_minio_creds(backtest_task)
    _pin_to_spark(backtest_task)

    registration_task = dqn_model_registration(
        model_checkpoint=training_task.outputs["model_checkpoint"],
        backtest_metrics=backtest_task.outputs["backtest_metrics"],
        pipeline_run_id=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        registry_url=model_registry_url,
        symbol=symbol,
        config_json=config_task.outputs["config_json"],
    )
    registration_task.set_retry(num_retries=1)
    registration_task.set_caching_options(enable_caching=False)
    _mount_minio_creds(registration_task)
    registration_task.after(backtest_task)
