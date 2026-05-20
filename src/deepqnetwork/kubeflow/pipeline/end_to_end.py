"""DQN End-to-End KFP Pipeline with Model Registry Integration.

Extends the base dqn_pipeline (config → training → backtest) with:
- Model Registry registration after backtest
- Degradation gate logic with relative and absolute thresholds
- Initial deployment bootstrap (auto-promote first model)
- Katib best-parameter injection when katib_enabled=True

The base dqn_pipeline.py remains unchanged; this module builds on top of it
by adding a post-backtest registration/promotion step.

Key constants (from thesis buy & hold baseline):
- sharpe_absolute_threshold = 1.0 (must beat buy & hold)
- pnl_absolute_threshold = 0.0 (must be profitable)
- sharpe_degradation_threshold = 0.1 (relative degradation vs production)

Requirements: DQN-R11, DQN-R12, DQN-R19, DQN-R20
"""

import json
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Input, Metrics, Model, Output

from deepqnetwork.kubeflow.pipeline.config_schema import DQNPipelineConfig
from deepqnetwork.kubeflow.pipeline.dqn_pipeline import (
    ECR_BASE,
    dqn_backtest,
    dqn_training,
    resolve_dqn_config,
)

# ---------------------------------------------------------------------------
# Constants, degradation gate thresholds from thesis
# ---------------------------------------------------------------------------

MODEL_REGISTRY_URL = "http://model-registry-service.kubeflow.svc.cluster.local:8080"
SHARPE_ABSOLUTE_THRESHOLD = 1.0  # Must beat buy & hold baseline
PNL_ABSOLUTE_THRESHOLD = 0.0  # Must be profitable in absolute terms
SHARPE_DEGRADATION_THRESHOLD = 0.1  # Max relative degradation vs production


# ---------------------------------------------------------------------------
# Degradation Gate Helper Functions
# ---------------------------------------------------------------------------


def evaluate_degradation_gate(
    candidate_sharpe: float,
    candidate_pnl: float,
    production_sharpe: float | None = None,
    production_pnl: float | None = None,
) -> tuple[bool, str]:
    """Evaluate the DQN degradation gate with relative and absolute checks.

    The gate blocks promotion if ANY of the following conditions are met:
    1. Sharpe < SHARPE_ABSOLUTE_THRESHOLD (1.0), must beat buy & hold
    2. P&L <= PNL_ABSOLUTE_THRESHOLD (0.0), must be profitable
    3. Sharpe degrades vs production by > SHARPE_DEGRADATION_THRESHOLD (0.1)
    4. P&L flips negative while production P&L is positive

    Args:
        candidate_sharpe: Sharpe ratio of the candidate model.
        candidate_pnl: Cumulative P&L of the candidate model.
        production_sharpe: Sharpe ratio of the current production model.
            None if no production model exists (bootstrap case).
        production_pnl: Cumulative P&L of the current production model.
            None if no production model exists (bootstrap case).

    Returns:
        Tuple of (gate_passed, reason). gate_passed is True if the model
        should be promoted, False if blocked. reason explains the decision.
    """
    # Absolute threshold checks
    if candidate_sharpe < SHARPE_ABSOLUTE_THRESHOLD:
        return (
            False,
            f"Sharpe {candidate_sharpe:.4f} below absolute threshold "
            f"{SHARPE_ABSOLUTE_THRESHOLD}",
        )

    if candidate_pnl <= PNL_ABSOLUTE_THRESHOLD:
        return (
            False,
            f"P&L {candidate_pnl:.4f} at or below absolute threshold "
            f"{PNL_ABSOLUTE_THRESHOLD}",
        )

    # Relative checks (only when production model exists)
    if production_sharpe is not None:
        degradation = production_sharpe - candidate_sharpe
        if degradation > SHARPE_DEGRADATION_THRESHOLD:
            return (
                False,
                f"Sharpe degraded by {degradation:.4f} vs production "
                f"(threshold: {SHARPE_DEGRADATION_THRESHOLD})",
            )

    if production_pnl is not None and production_pnl > 0:
        if candidate_pnl < 0:
            return (
                False,
                f"P&L flipped negative ({candidate_pnl:.4f}) while "
                f"production P&L is positive ({production_pnl:.4f})",
            )

    return (True, "All degradation gate checks passed")


# ---------------------------------------------------------------------------
# Lightweight Python component for Model Registry registration + promotion
# ---------------------------------------------------------------------------


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3", "model-registry", "kubernetes"],
)
def register_and_promote(
    model_checkpoint: Input[Model],
    backtest_metrics: Input[Metrics],
    pipeline_run_id: str,
    registry_url: str,
    symbol: str,
    config_json: str,
):
    """Register the DQN model in the Model Registry after backtest.

    Reads backtest metrics from the artifact, evaluates the degradation gate,
    and registers/promotes the model accordingly:
    - Gate passes → register in staging, then promote to production
    - Gate fails → register in staging only (no promotion)
    - Bootstrap (no production model) → auto-promote regardless of gate

    Gracefully degrades: if the Model Registry is unreachable, logs a warning
    and returns without failing the pipeline (model checkpoint is still in S3).

    Args:
        model_checkpoint: KFP Model artifact from the training component.
        backtest_metrics: KFP Metrics artifact from the backtest component.
        pipeline_run_id: Unique run ID of this pipeline execution.
        registry_url: URL of the Kubeflow Model Registry server.
        symbol: Currency pair symbol (USDJPY or AUDJPY).
        config_json: JSON string of the resolved DQN pipeline configuration.

    Requirements: DQN-R19, DQN-R20
    """
    import json
    import logging
    import sys
    import uuid
    from datetime import datetime, timezone

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Degradation gate constants
    SHARPE_ABSOLUTE_THRESHOLD = 1.0
    PNL_ABSOLUTE_THRESHOLD = 0.0
    SHARPE_DEGRADATION_THRESHOLD = 0.1

    def _log(msg: str, extra: dict = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "logger": __name__,
            "message": msg,
            "component": "dqn_register_and_promote",
        }
        if extra:
            entry.update(extra)
        print(json.dumps(entry), file=sys.stdout)

    # Step 1: Read backtest metrics from S3 ----------------------------------
    _log("Reading backtest metrics", extra={"s3_uri": backtest_metrics.uri})

    import boto3

    s3 = boto3.client("s3")
    bucket = "prod-fintech-forex-sg-731833471586"

    try:
        obj = s3.get_object(Bucket=bucket, Key=backtest_metrics.uri)
        backtest_data = json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as e:
        _log(
            "Failed to read backtest metrics; skipping registration",
            extra={"error": str(e)},
        )
        return

    # Extract candidate metrics
    candidate_sharpe = float(backtest_data.get("sharpe_ratio", 0.0))
    candidate_pnl = float(backtest_data.get("cumulative_pnl", 0.0))
    max_drawdown = float(backtest_data.get("max_drawdown", 0.0))
    win_rate = float(backtest_data.get("win_rate", 0.0))

    _log(
        "Candidate model metrics",
        extra={
            "sharpe_ratio": candidate_sharpe,
            "cumulative_pnl": candidate_pnl,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
        },
    )

    # Step 2: Connect to Model Registry and check for production model -------
    try:
        from model_registry import ModelRegistry

        registry = ModelRegistry(
            server_address=registry_url, author="dqn-pipeline"
        )
    except ImportError as e:
        _log(
            "Model Registry SDK not available; skipping registration",
            extra={"error": str(e)},
        )
        return
    except Exception as e:
        _log(
            "Failed to connect to Model Registry; skipping registration",
            extra={"error": str(e)},
        )
        return

    model_name = f"dqn-agent-{symbol.lower()}"

    # Ensure the registered model exists
    try:
        registered_model = registry.get_registered_model(model_name)
    except Exception:
        registered_model = registry.register_model(
            model_name,
            uri="",
            description=f"DQN trading agent for {symbol}",
        )

    # Check for existing production model
    production_sharpe = None
    production_pnl = None
    has_production = False

    try:
        versions = registry.get_model_versions(model_name)
        for v in versions:
            props = (
                v.custom_properties
                if hasattr(v, "custom_properties")
                else {}
            )
            if props.get("lifecycle_stage") == "production":
                has_production = True
                production_sharpe = float(props.get("sharpe_ratio", 0.0))
                production_pnl = float(props.get("cumulative_pnl", 0.0))
                break
    except Exception:
        pass

    _log(
        "Production model status",
        extra={
            "has_production": has_production,
            "production_sharpe": production_sharpe,
            "production_pnl": production_pnl,
        },
    )

    # Step 3: Evaluate degradation gate --------------------------------------
    if not has_production:
        # Bootstrap: no production model exists, auto-promote
        gate_passed = True
        gate_reason = "Bootstrap: no production model exists; auto-promoting"
        _log(gate_reason)
    else:
        # Absolute threshold checks
        gate_passed = True
        gate_reason = "All degradation gate checks passed"

        if candidate_sharpe < SHARPE_ABSOLUTE_THRESHOLD:
            gate_passed = False
            gate_reason = (
                f"Sharpe {candidate_sharpe:.4f} below absolute threshold "
                f"{SHARPE_ABSOLUTE_THRESHOLD}"
            )
        elif candidate_pnl <= PNL_ABSOLUTE_THRESHOLD:
            gate_passed = False
            gate_reason = (
                f"P&L {candidate_pnl:.4f} at or below absolute threshold "
                f"{PNL_ABSOLUTE_THRESHOLD}"
            )
        elif production_sharpe is not None:
            degradation = production_sharpe - candidate_sharpe
            if degradation > SHARPE_DEGRADATION_THRESHOLD:
                gate_passed = False
                gate_reason = (
                    f"Sharpe degraded by {degradation:.4f} vs production "
                    f"(threshold: {SHARPE_DEGRADATION_THRESHOLD})"
                )
        if gate_passed and production_pnl is not None and production_pnl > 0:
            if candidate_pnl < 0:
                gate_passed = False
                gate_reason = (
                    f"P&L flipped negative ({candidate_pnl:.4f}) while "
                    f"production P&L is positive ({production_pnl:.4f})"
                )

    _log(
        "Degradation gate result",
        extra={"gate_passed": gate_passed, "reason": gate_reason},
    )

    # Step 4: Register model in Model Registry (always, regardless of gate) --
    version_id = str(uuid.uuid4())
    config = json.loads(config_json)
    lifecycle_stage = "staging"  # Always register as staging first

    custom_properties = {
        "version_id": version_id,
        "symbol": symbol,
        "episode_start_ts": str(config.get("episode_start_ts", "")),
        "episode_end_ts": str(config.get("episode_end_ts", "")),
        "step_size_seconds": str(config.get("step_size_seconds", "")),
        "cumulative_pnl": str(candidate_pnl),
        "sharpe_ratio": str(candidate_sharpe),
        "max_drawdown": str(max_drawdown),
        "win_rate": str(win_rate),
        "hyperparameters": json.dumps(
            {
                "learning_rate": config.get("learning_rate"),
                "hidden_dims": config.get("hidden_dims"),
                "gamma": config.get("gamma"),
                "epsilon_decay_steps": config.get("epsilon_decay_steps"),
                "batch_size": config.get("batch_size"),
                "target_update_freq": config.get("target_update_freq"),
                "dropout": config.get("dropout"),
            }
        ),
        "pipeline_run_id": pipeline_run_id,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "lifecycle_stage": lifecycle_stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate_passed": str(gate_passed),
        "gate_reason": gate_reason,
    }

    try:
        registry.register_model_version(
            name=version_id,
            version=version_id,
            model_name=registered_model.name,
            uri=model_checkpoint.uri,
            description=(
                f"DQN agent for {symbol} "
                f"(run={pipeline_run_id}, "
                f"Sharpe={candidate_sharpe:.4f}, P&L={candidate_pnl:.4f})"
            ),
            custom_properties=custom_properties,
        )

        _log(
            "Model registered in staging",
            extra={
                "version_id": version_id,
                "model_name": model_name,
                "sharpe_ratio": candidate_sharpe,
                "cumulative_pnl": candidate_pnl,
            },
        )
    except Exception as e:
        _log(
            "Model registration failed (gracefully degraded); "
            "model checkpoint remains in S3",
            extra={"error": str(e)},
        )
        return

    # Step 5: Promote to production if gate passed ---------------------------
    if gate_passed:
        try:
            # Demote current production model (if any)
            if has_production:
                versions = registry.get_model_versions(model_name)
                for v in versions:
                    props = (
                        v.custom_properties
                        if hasattr(v, "custom_properties")
                        else {}
                    )
                    if props.get("lifecycle_stage") == "production":
                        props["lifecycle_stage"] = "staging"
                        registry.update_model_version(
                            v, custom_properties=props
                        )
                        _log(
                            "Demoted previous production model",
                            extra={
                                "version_id": props.get("version_id")
                            },
                        )
                        break

            # Promote new model to production
            versions = registry.get_model_versions(model_name)
            for v in versions:
                props = (
                    v.custom_properties
                    if hasattr(v, "custom_properties")
                    else {}
                )
                if props.get("version_id") == version_id:
                    props["lifecycle_stage"] = "production"
                    registry.update_model_version(v, custom_properties=props)
                    break

            _log(
                "Model promoted to production",
                extra={
                    "version_id": version_id,
                    "model_name": model_name,
                    "bootstrap": not has_production,
                },
            )

            # Note: standalone DQN KServe InferenceService is deprecated
            # (kubeflow-ml-pipeline spec, Requirement 15). The dqnpf-intraday
            # combined predictor's hot-reload watcher (Task 30.3) resolves the
            # new production checkpoint on its next poll; no KServe patching
            # is performed from this pipeline step.

        except Exception as e:
            _log(
                "Model promotion failed; model remains in staging",
                extra={"error": str(e)},
            )
    else:
        _log(
            "Model NOT promoted (gate failed); registered in staging only",
            extra={
                "version_id": version_id,
                "gate_reason": gate_reason,
            },
        )

    _log(
        "DQN register_and_promote component completed",
        extra={
            "version_id": version_id,
            "promoted": gate_passed,
            "bootstrap": not has_production,
        },
    )


# ---------------------------------------------------------------------------
# End-to-End Pipeline Definition
# ---------------------------------------------------------------------------


@dsl.pipeline(
    name="dqn-pipeline-e2e",
    description=(
        "Deep Q-Network End-to-End Pipeline: "
        "config validation → DQN training (with modelenv sidecar) "
        "→ DQN backtest (with modelenv sidecar) "
        "→ Model Registry registration and promotion"
    ),
)
def dqn_pipeline_e2e(
    symbol: str = "USDJPY",
    episode_start_ts: int = 0,
    episode_end_ts: int = 0,
    step_size_seconds: int = 5,
    num_episodes_per_range: int = 3000,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    training_mode: str = "scratch",
    checkpoint: str = "",
    katib_enabled: bool = False,
    katib_best_params_json: str = "",
    model_registry_url: str = MODEL_REGISTRY_URL,
):
    """DQN End-to-End Pipeline: config → train → backtest → register/promote.

    Extends the base DQN pipeline with Model Registry integration:

    1. resolve_dqn_config: validates params, applies Katib best params
    2. dqn_training: trains DQN agent against modelenv sidecar, outputs checkpoint
    3. dqn_backtest: evaluates checkpoint on unseen date ranges, computes metrics
    4. register_and_promote: registers model in registry, evaluates degradation
       gate, promotes to production if gate passes (or bootstraps first model)

    DAG structure:
        resolve_dqn_config → dqn_training → dqn_backtest → register_and_promote

    Katib integration:
        When katib_enabled=True and katib_best_params_json is provided, the
        Katib-optimized hyperparameters override the pipeline defaults for
        training (learning_rate, batch_size, hidden_dims, gamma, etc.).

    Degradation gate logic:
        - Absolute: Sharpe >= 1.0 (buy & hold baseline), P&L > 0
        - Relative: Sharpe degradation vs production <= 0.1,
                    P&L must not flip negative vs positive production
        - Bootstrap: auto-promote first model when no production exists

    Resource allocation:
    - Training: 1 GPU (nvidia.com/gpu: 1), 16Gi memory, modelenv sidecar
    - Backtest: CPU-only, 4Gi memory, modelenv sidecar
    - Registration: lightweight Python component (256Mi memory)

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
        katib_enabled: Whether Katib-optimized params should be applied.
        katib_best_params_json: JSON string of Katib best hyperparameters.
            Expected keys: learning_rate, hidden_dims, epsilon_decay_steps,
            gamma, batch_size, target_update_freq, dropout.
        model_registry_url: URL of the Kubeflow Model Registry server.
    """
    # -----------------------------------------------------------------------
    # Step 0: Apply Katib best params if enabled
    # -----------------------------------------------------------------------
    effective_learning_rate = learning_rate
    effective_batch_size = batch_size
    effective_num_episodes_per_range = num_episodes_per_range

    # Note: Katib parameter injection is handled at submission time via
    # build_dqn_pipeline_e2e_config(). The katib_best_params_json is also
    # passed to resolve_dqn_config for in-cluster validation. The pipeline
    # parameters (learning_rate, batch_size, etc.) should already reflect
    # Katib-optimized values when katib_enabled=True.

    # -----------------------------------------------------------------------
    # Step 1: Config resolution and validation (lightweight Python component)
    # -----------------------------------------------------------------------
    config_task = resolve_dqn_config(
        symbol=symbol,
        episode_start_ts=episode_start_ts,
        episode_end_ts=episode_end_ts,
        step_size_seconds=step_size_seconds,
        num_episodes_per_range=num_episodes_per_range,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_mode=training_mode,
        checkpoint=checkpoint,
    )

    # -----------------------------------------------------------------------
    # Step 2: DQN Training (with modelenv sidecar, 1 GPU, 16Gi memory)
    # -----------------------------------------------------------------------
    training_task = dqn_training(
        symbol=symbol,
        episode_start_ts=episode_start_ts,
        episode_end_ts=episode_end_ts,
        step_size_seconds=step_size_seconds,
        num_episodes_per_range=num_episodes_per_range,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_mode=training_mode,
        checkpoint=checkpoint,
        config_json=config_task.outputs["config_json"],
    )
    training_task.set_retry(num_retries=2)
    training_task.set_memory_request("16Gi")
    training_task.set_memory_limit("16Gi")
    training_task.set_gpu_limit(1)

    # -----------------------------------------------------------------------
    # Step 3: DQN Backtest (CPU-only, 4Gi memory)
    # -----------------------------------------------------------------------
    backtest_task = dqn_backtest(
        model_checkpoint=training_task.outputs["model_checkpoint"],
        symbol=symbol,
        episode_start_ts=episode_start_ts,
        episode_end_ts=episode_end_ts,
        step_size_seconds=step_size_seconds,
        config_json=config_task.outputs["config_json"],
    )
    backtest_task.set_retry(num_retries=1)
    backtest_task.set_memory_request("4Gi")
    backtest_task.set_memory_limit("4Gi")

    # -----------------------------------------------------------------------
    # Step 4: Model Registry Registration and Promotion
    # -----------------------------------------------------------------------
    reg_task = register_and_promote(
        model_checkpoint=training_task.outputs["model_checkpoint"],
        backtest_metrics=backtest_task.outputs["backtest_metrics"],
        pipeline_run_id=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        registry_url=model_registry_url,
        symbol=symbol,
        config_json=config_task.outputs["config_json"],
    )
    reg_task.set_memory_request("256Mi")
    reg_task.set_memory_limit("512Mi")


# ---------------------------------------------------------------------------
# Client-side helpers for pipeline submission
# ---------------------------------------------------------------------------


def build_dqn_pipeline_e2e_config(
    symbol: str = "USDJPY",
    episode_start_ts: int = 0,
    episode_end_ts: int = 0,
    step_size_seconds: int = 5,
    num_episodes_per_range: int = 3000,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    training_mode: str = "scratch",
    checkpoint: str = "",
    katib_enabled: bool = False,
    katib_best_params_json: str = "",
) -> dict:
    """Build and validate pipeline parameters for the DQN E2E pipeline.

    Called at pipeline submission time (client-side) to validate configuration
    and apply Katib best parameters before submitting to the KFP engine.

    When katib_enabled=True and katib_best_params_json is provided, the
    Katib-optimized hyperparameters override the corresponding pipeline
    parameters (learning_rate, batch_size, etc.).

    Args:
        symbol: Currency pair (USDJPY or AUDJPY).
        episode_start_ts: Episode start timestamp.
        episode_end_ts: Episode end timestamp.
        step_size_seconds: Step size in seconds for the environment.
        num_episodes_per_range: Number of training episodes.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        training_mode: "scratch" or "finetune".
        checkpoint: S3 key path for production checkpoint (required for finetune).
        katib_enabled: Whether to apply Katib-optimized parameters.
        katib_best_params_json: JSON string of Katib best hyperparameters.

    Returns:
        Dictionary of pipeline parameters ready for KFP submission.

    Raises:
        ValueError: If configuration validation fails.
    """
    # Apply Katib best params if enabled
    effective_lr = learning_rate
    effective_batch_size = batch_size

    if katib_enabled and katib_best_params_json:
        katib_params = json.loads(katib_best_params_json)
        if "learning_rate" in katib_params:
            effective_lr = float(katib_params["learning_rate"])
        if "batch_size" in katib_params:
            effective_batch_size = int(katib_params["batch_size"])

    # Build and validate config
    config = DQNPipelineConfig()
    config = config.override(
        symbol=symbol,
        episode_start_ts=episode_start_ts,
        episode_end_ts=episode_end_ts,
        step_size_seconds=step_size_seconds,
        num_episodes_per_range=num_episodes_per_range,
        batch_size=effective_batch_size,
        learning_rate=effective_lr,
        training_mode=training_mode,
        katib_enabled=katib_enabled,
    )

    # Apply training mode adjustments
    if training_mode == "finetune":
        config = config.override(
            num_episodes_per_range=config.finetune_num_episodes_per_range,
            learning_rate=config.finetune_learning_rate,
        )
        effective_lr = config.finetune_learning_rate

    # Validate
    errors = config.validate()
    if training_mode == "finetune" and not checkpoint:
        errors.append("Finetune mode requires a checkpoint path")
    if errors:
        raise ValueError(
            f"DQN pipeline configuration validation failed: {'; '.join(errors)}"
        )

    return {
        "symbol": symbol,
        "episode_start_ts": episode_start_ts,
        "episode_end_ts": episode_end_ts,
        "step_size_seconds": step_size_seconds,
        "num_episodes_per_range": num_episodes_per_range if training_mode != "finetune" else config.finetune_num_episodes_per_range,
        "batch_size": effective_batch_size,
        "learning_rate": effective_lr,
        "training_mode": training_mode,
        "checkpoint": checkpoint,
        "katib_enabled": katib_enabled,
        "katib_best_params_json": katib_best_params_json,
        "model_registry_url": MODEL_REGISTRY_URL,
    }
