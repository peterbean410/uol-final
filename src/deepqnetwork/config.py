"""Configuration management for the DQN trading agent.

Loads configuration from YAML file with CLI argument overrides.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DQNConfig:
    """Unified configuration for the DQN trading agent."""

    # Environment
    grpc_address: str = "localhost:50051"
    symbol: str = "USDJPY"
    episode_start_ts: int = 0
    episode_end_ts: int = 0
    step_size_seconds: int = 5

    # Date-range episode scheduling (replaces raw episode_start_ts/end_ts)
    date_start: str = ""
    date_end: str = ""
    hour_of_day_start: int = 0
    hour_of_day_end: int = 23

    # Agent
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 50_000
    batch_size: int = 64
    replay_buffer_size: int = 300_000
    target_update_freq: int = 1000
    train_freq: int = 4
    tau: float = 1.0

    # Network
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 128])
    activation: str = "relu"
    layer_norm: bool = True
    dropout: float = 0.0
    dueling: bool = False
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    loss_function: str = "huber"

    # Training
    num_episodes: int = 3000
    max_steps_per_episode: int = 30_000
    checkpoint_interval: int = 50
    checkpoint_dir: str = "deepqnetwork/checkpoints/"
    log_interval: int = 10

    # Mode
    mode: str = "train"
    checkpoint: str | None = None
    device: str = "auto"

    # S3
    s3_checkpoint_prefix: str | None = None
    model_version: str | None = None


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="DQN Trading Agent - Training and Inference"
    )

    # Core CLI arguments
    parser.add_argument(
        "--config",
        type=str,
        default="deepqnetwork/config.yaml",
        help="Path to YAML configuration file (default: deepqnetwork/config.yaml)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "live"],
        default=None,
        help="Operating mode: train or live",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to load",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: auto, cpu, cuda, cuda:0, cuda:1, mps",
    )
    parser.add_argument(
        "--s3-checkpoint-prefix",
        type=str,
        default=None,
        help="S3 prefix for checkpoint uploads",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Model version identifier for S3 path",
    )

    # Additional hyperparameter overrides
    parser.add_argument("--grpc-address", type=str, default=None)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--episode-start-ts", type=int, default=None)
    parser.add_argument("--episode-end-ts", type=int, default=None)
    parser.add_argument("--date-start", type=str, default=None, help="ISO date for first episode (e.g. 2012-01-01)")
    parser.add_argument("--date-end", type=str, default=None, help="ISO date for last episode (e.g. 2022-12-31)")
    parser.add_argument("--hour-start", dest="hour_of_day_start", type=int, default=None, help="Hour of day to start each episode (0-23)")
    parser.add_argument("--hour-end", dest="hour_of_day_end", type=int, default=None, help="Hour of day to end each episode (0-23)")
    parser.add_argument("--step-size-seconds", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-end", type=float, default=None)
    parser.add_argument("--epsilon-decay-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--replay-buffer-size", type=int, default=None)
    parser.add_argument("--target-update-freq", type=int, default=None)
    parser.add_argument("--train-freq", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--dueling", action="store_true", default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--loss-function", type=str, choices=["huber", "mse"], default=None)

    return parser


# Mapping from CLI arg names (with hyphens) to DQNConfig field names (with underscores)
_CLI_TO_CONFIG = {
    "mode": "mode",
    "checkpoint": "checkpoint",
    "device": "device",
    "s3_checkpoint_prefix": "s3_checkpoint_prefix",
    "model_version": "model_version",
    "grpc_address": "grpc_address",
    "symbol": "symbol",
    "episode_start_ts": "episode_start_ts",
    "episode_end_ts": "episode_end_ts",
    "date_start": "date_start",
    "date_end": "date_end",
    "hour_of_day_start": "hour_of_day_start",
    "hour_of_day_end": "hour_of_day_end",
    "step_size_seconds": "step_size_seconds",
    "gamma": "gamma",
    "epsilon_start": "epsilon_start",
    "epsilon_end": "epsilon_end",
    "epsilon_decay_steps": "epsilon_decay_steps",
    "batch_size": "batch_size",
    "replay_buffer_size": "replay_buffer_size",
    "target_update_freq": "target_update_freq",
    "train_freq": "train_freq",
    "tau": "tau",
    "learning_rate": "learning_rate",
    "grad_clip_norm": "grad_clip_norm",
    "weight_decay": "weight_decay",
    "dropout": "dropout",
    "dueling": "dueling",
    "num_episodes": "num_episodes",
    "max_steps_per_episode": "max_steps_per_episode",
    "checkpoint_interval": "checkpoint_interval",
    "checkpoint_dir": "checkpoint_dir",
    "log_interval": "log_interval",
    "loss_function": "loss_function",
}


def _load_yaml(path: str) -> dict:
    """Load configuration from a YAML file.

    Returns an empty dict if the file does not exist.
    """
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_config(args: list[str] | None = None) -> DQNConfig:
    """Load configuration from YAML file with CLI argument overrides.

    Order of precedence (highest to lowest):
    1. CLI arguments (only those explicitly provided)
    2. YAML configuration file values
    3. DQNConfig dataclass defaults

    Args:
        args: Optional list of CLI arguments. If None, uses sys.argv.

    Returns:
        Fully resolved DQNConfig instance.

    Raises:
        ValueError: If mode is 'live' and no checkpoint is provided.
    """
    parser = _build_parser()
    parsed = parser.parse_args(args)

    # Load YAML config
    yaml_data = _load_yaml(parsed.config)

    # Start with YAML values, then override with explicit CLI args
    config_kwargs: dict = {}

    # Apply YAML values first
    for yaml_key, value in yaml_data.items():
        if yaml_key == "betas" and isinstance(value, list):
            config_kwargs["betas"] = tuple(value)
        elif yaml_key == "hidden_dims" and isinstance(value, list):
            config_kwargs["hidden_dims"] = value
        else:
            config_kwargs[yaml_key] = value

    # Override with CLI args that were explicitly provided (not None)
    parsed_dict = vars(parsed)
    for cli_key, config_key in _CLI_TO_CONFIG.items():
        value = parsed_dict.get(cli_key)
        if value is not None:
            config_kwargs[config_key] = value

    # Remove 'config' key if it leaked in (it's not a DQNConfig field)
    config_kwargs.pop("config", None)

    # Build the config
    config = DQNConfig(**config_kwargs)

    # Validation: live mode requires a checkpoint
    if config.mode == "live" and config.checkpoint is None:
        raise ValueError(
            "Live mode requires a --checkpoint argument. "
            "Cannot serve without a trained model."
        )

    return config
