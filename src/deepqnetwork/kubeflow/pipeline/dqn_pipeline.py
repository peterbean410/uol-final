"""DQN KFP Pipeline Definition.

Defines the dqn_pipeline using KFP v2 SDK with @dsl.container_component
decorators. Wires dqn_training → dqn_backtest as a 2-step DAG with
retry(2) on training and retry(1) on backtest.

Training component runs with modelenv sidecar (1 GPU, 16Gi memory).
Backtest component runs CPU-only (4Gi memory).

Pipeline accepts parameters: symbol, episode_start_ts, episode_end_ts,
step_size_seconds, num_episodes, batch_size, learning_rate, training_mode,
checkpoint (for finetune).

Integrates DQNPipelineConfig loading and validation at pipeline start.

Requirements: DQN-R11, DQN-R12
"""

import json
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Input, Metrics, Model, Output

from deepqnetwork.kubeflow.pipeline.config_schema import DQNPipelineConfig

# ECR registry base for all component images
ECR_BASE = "731833471586.dkr.ecr.ap-southeast-1.amazonaws.com"


# ---------------------------------------------------------------------------
# Container Components
# ---------------------------------------------------------------------------


@dsl.container_component
def dqn_training(
    symbol: str,
    episode_start_ts: int,
    episode_end_ts: int,
    step_size_seconds: int,
    num_episodes: int,
    batch_size: int,
    learning_rate: float,
    training_mode: str,
    checkpoint: str,
    config_json: str,
    model_checkpoint: Output[Model],
):
    """Train DQN agent against modelenv gRPC sidecar.

    Loads DQNPipelineConfig, launches modelenv as a subprocess sidecar,
    runs training, and outputs a checkpoint artifact to S3.

    Supports scratch (random init, full episodes) and finetune
    (load production checkpoint, reduced LR, fewer episodes) modes.
    """
    return dsl.ContainerSpec(
        image=f"{ECR_BASE}/dqn/training:latest",
        command=["python", "component.py"],
        args=[
            "--checkpoint-output-path", model_checkpoint.uri,
            "--symbol", symbol,
            "--episode-start-ts", str(episode_start_ts),
            "--episode-end-ts", str(episode_end_ts),
            "--num-episodes", str(num_episodes),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--training-mode", training_mode,
            "--production-checkpoint-path", checkpoint,
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
        ],
    )


# ---------------------------------------------------------------------------
# Lightweight Python component for config resolution at runtime
# ---------------------------------------------------------------------------


@dsl.component(base_image="python:3.11-slim", packages_to_install=["pyyaml"])
def resolve_dqn_config(
    symbol: str,
    episode_start_ts: int,
    episode_end_ts: int,
    step_size_seconds: int,
    num_episodes: int,
    batch_size: int,
    learning_rate: float,
    training_mode: str,
    checkpoint: str,
) -> NamedTuple(
    "DQNConfigOutputs",
    [
        ("config_json", str),
        ("effective_num_episodes", int),
        ("effective_learning_rate", float),
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
    from collections import namedtuple
    from dataclasses import asdict, dataclass, field
    from typing import Literal

    @dataclass
    class _DQNConfig:
        """Inline DQNPipelineConfig for the lightweight component."""

        grpc_address: str = "localhost:50051"
        symbol: str = "USDJPY"
        episode_start_ts: int = 0
        episode_end_ts: int = 0
        step_size_seconds: int = 5
        gamma: float = 0.99
        epsilon_start: float = 1.0
        epsilon_end: float = 0.01
        epsilon_decay_steps: int = 50_000
        batch_size: int = 64
        replay_buffer_size: int = 300_000
        target_update_freq: int = 1000
        train_freq: int = 4
        tau: float = 1.0
        hidden_dims: list = field(default_factory=lambda: [256, 256, 128])
        activation: str = "relu"
        layer_norm: bool = True
        dropout: float = 0.0
        dueling: bool = False
        learning_rate: float = 1e-4
        grad_clip_norm: float = 10.0
        weight_decay: float = 0.0
        loss_function: str = "huber"
        num_episodes: int = 3000
        max_steps_per_episode: int = 30_000
        checkpoint_interval: int = 50
        gpu_enabled: bool = True
        num_workers: int = 1
        max_wall_time_hours: int = 8
        training_mode: str = "scratch"
        finetune_learning_rate: float = 1e-5
        finetune_num_episodes: int = 500

    cfg = _DQNConfig(
        symbol=symbol,
        episode_start_ts=episode_start_ts,
        episode_end_ts=episode_end_ts,
        step_size_seconds=step_size_seconds,
        num_episodes=num_episodes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_mode=training_mode,
    )

    # Apply training mode adjustments
    if training_mode == "finetune":
        effective_lr = cfg.finetune_learning_rate
        effective_episodes = cfg.finetune_num_episodes
    else:
        effective_lr = cfg.learning_rate
        effective_episodes = cfg.num_episodes

    # Validate
    errors: list = []
    if cfg.symbol not in ("USDJPY", "AUDJPY"):
        errors.append(f"Invalid symbol: {cfg.symbol}")
    if cfg.step_size_seconds <= 0:
        errors.append(
            f"step_size_seconds must be positive: {cfg.step_size_seconds}"
        )
    if not (1e-6 <= cfg.learning_rate <= 0.01):
        errors.append(
            f"learning_rate out of range [1e-6, 0.01]: {cfg.learning_rate}"
        )
    if cfg.batch_size <= 0:
        errors.append(f"batch_size must be positive: {cfg.batch_size}")
    if cfg.num_episodes <= 0:
        errors.append(f"num_episodes must be positive: {cfg.num_episodes}")
    if training_mode not in ("scratch", "finetune"):
        errors.append(f"Invalid training_mode: {training_mode}")
    if training_mode == "finetune" and not checkpoint:
        errors.append(
            "Finetune mode requires a checkpoint path"
        )

    if errors:
        raise ValueError(
            f"DQN pipeline configuration validation failed: {'; '.join(errors)}"
        )

    config_dict = asdict(cfg)
    config_dict["effective_learning_rate"] = effective_lr
    config_dict["effective_num_episodes"] = effective_episodes
    config_json = json.dumps(config_dict)

    DQNConfigOutputs = namedtuple(
        "DQNConfigOutputs",
        ["config_json", "effective_num_episodes", "effective_learning_rate"],
    )
    return DQNConfigOutputs(
        config_json=config_json,
        effective_num_episodes=effective_episodes,
        effective_learning_rate=effective_lr,
    )


# ---------------------------------------------------------------------------
# Config Helpers (client-side, for use at submission time)
# ---------------------------------------------------------------------------


def build_dqn_pipeline_config(
    symbol: str = "USDJPY",
    episode_start_ts: int = 0,
    episode_end_ts: int = 0,
    step_size_seconds: int = 5,
    num_episodes: int = 3000,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    training_mode: str = "scratch",
    checkpoint: str = "",
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
        num_episodes: Number of training episodes.
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

    # Apply pipeline parameter overrides
    config = config.override(
        symbol=symbol,
        episode_start_ts=episode_start_ts,
        episode_end_ts=episode_end_ts,
        step_size_seconds=step_size_seconds,
        num_episodes=num_episodes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_mode=training_mode,
    )

    # Apply training mode adjustments
    if training_mode == "finetune":
        config = config.override(
            num_episodes=config.finetune_num_episodes,
            learning_rate=config.finetune_learning_rate,
        )

    # Validate configuration
    errors = config.validate()

    # Additional pipeline-level validation
    if training_mode == "finetune" and not checkpoint:
        errors.append("Finetune mode requires a checkpoint path")

    if errors:
        raise ValueError(
            f"DQN pipeline configuration validation failed: {'; '.join(errors)}"
        )

    return config


# ---------------------------------------------------------------------------
# Pipeline Definition
# ---------------------------------------------------------------------------


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
    episode_start_ts: int = 0,
    episode_end_ts: int = 0,
    step_size_seconds: int = 5,
    num_episodes: int = 3000,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    training_mode: str = "scratch",
    checkpoint: str = "",
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
    - Training: 1 GPU (nvidia.com/gpu: 1), 16Gi memory, modelenv sidecar
    - Backtest: CPU-only, 4Gi memory, modelenv sidecar

    Args:
        symbol: Currency pair (USDJPY or AUDJPY).
        episode_start_ts: Episode start timestamp for training.
        episode_end_ts: Episode end timestamp for training.
        step_size_seconds: Step size in seconds for the environment.
        num_episodes: Number of training episodes (overridden in finetune mode).
        batch_size: Training batch size.
        learning_rate: Learning rate (overridden in finetune mode).
        training_mode: "scratch" for full training or "finetune" for
            incremental training on production model weights.
        checkpoint: S3 key path for production checkpoint (required for finetune).
    """
    # -----------------------------------------------------------------------
    # Step 0: Config resolution and validation (lightweight Python component)
    # -----------------------------------------------------------------------
    config_task = resolve_dqn_config(
        symbol=symbol,
        episode_start_ts=episode_start_ts,
        episode_end_ts=episode_end_ts,
        step_size_seconds=step_size_seconds,
        num_episodes=num_episodes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_mode=training_mode,
        checkpoint=checkpoint,
    )

    # -----------------------------------------------------------------------
    # Step 1: DQN Training (with modelenv sidecar, 1 GPU, 16Gi memory)
    # -----------------------------------------------------------------------
    training_task = dqn_training(
        symbol=symbol,
        episode_start_ts=episode_start_ts,
        episode_end_ts=episode_end_ts,
        step_size_seconds=step_size_seconds,
        num_episodes=num_episodes,
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
    # Step 2: DQN Backtest (CPU-only, 4Gi memory)
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
