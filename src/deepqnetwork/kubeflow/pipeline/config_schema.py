"""DQN Pipeline configuration schema with YAML loading, validation, and CLI conversion.

Extends the concept of PipelineConfig from the Transformer pipeline with DQN-specific
fields for environment connection, agent hyperparameters, network architecture, training
schedule, infrastructure, Katib tuning, serving, and monitoring.

The configuration is loaded from YAML and validated at pipeline submission time.
It supports `to_cli_args()` to convert fields into CLI arguments for the underlying
`deepqnetwork/train.py` entry point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import yaml


@dataclass
class DQNPipelineConfig:
    """Complete DQN pipeline configuration loaded from YAML.

    Covers all parameters needed to submit a DQN training pipeline run:
    environment connection, agent hyperparameters, network architecture,
    training schedule, infrastructure, Katib tuning, serving, and monitoring.
    """

    # Environment (modelenv gRPC server connection)
    grpc_address: str = "localhost:50051"
    symbol: str = "USDJPY"
    episode_start_ts: int = 0
    episode_end_ts: int = 0
    step_size_seconds: int = 5

    # Date-range episode scheduling; one episode per calendar date
    date_start: str = ""
    date_end: str = ""
    hour_of_day_start: int = 0
    hour_of_day_end: int = 23

    # Agent hyperparameters
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 50_000
    batch_size: int = 64
    replay_buffer_size: int = 300_000
    target_update_freq: int = 1000
    train_freq: int = 4
    tau: float = 1.0  # 1.0 = hard copy; <1.0 = soft Polyak averaging

    # Network architecture
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 128])
    activation: str = "relu"  # relu, leaky_relu, gelu
    layer_norm: bool = True
    dropout: float = 0.0
    dueling: bool = False
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    loss_function: str = "huber"  # huber, mse

    # Training schedule, episode-count knobs are mode-specific and mutually
    # exclusive; both default to None (effective defaults: 3000 fixed / 3 date-range).
    num_episodes_per_range: int | None = None  # fixed-window mode only
    repeats_per_date: int | None = None        # date-range mode only
    max_steps_per_episode: int = 30_000
    checkpoint_interval: int = 50

    # Infrastructure
    gpu_enabled: bool = True
    num_workers: int = 1
    max_wall_time_hours: int = 8

    # Katib hyperparameter tuning
    katib_enabled: bool = False
    katib_max_trials: int = 20
    katib_parallel_trials: int = 3
    katib_trial_timeout_hours: int = 4

    # Serving
    serving_min_replicas: int = 0
    serving_max_replicas: int = 2
    serving_target_concurrency: int = 5

    # Monitoring, absolute thresholds from thesis buy & hold baseline
    alert_webhook_url: str = ""
    sharpe_degradation_threshold: float = 0.1
    sharpe_absolute_threshold: float = 1.0  # Must beat buy & hold baseline
    pnl_absolute_threshold: float = 0.0  # Must be profitable in absolute terms

    # Training mode
    training_mode: Literal["scratch", "finetune"] = "scratch"
    # "scratch": Random init, full training (num_episodes_per_range episodes)
    # "finetune": Load production checkpoint, reduced LR, fewer episodes
    finetune_learning_rate: float = 1e-5  # 10x lower than base LR
    finetune_num_episodes_per_range: int = 500  # Fewer episodes when fine-tuning

    @classmethod
    def from_yaml(cls, path: str) -> "DQNPipelineConfig":
        """Load config from YAML file.

        Only keys matching dataclass fields are loaded; unknown keys are ignored.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A DQNPipelineConfig instance with values from the YAML file.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            return cls()

        # Normalise list/tuple fields from YAML
        kwargs = {}
        for key, value in data.items():
            if key not in cls.__dataclass_fields__:
                continue
            if key == "betas" and isinstance(value, list):
                kwargs[key] = tuple(value)
            elif key == "hidden_dims" and isinstance(value, list):
                kwargs[key] = value
            else:
                kwargs[key] = value

        return cls(**kwargs)

    def override(self, **kwargs: object) -> "DQNPipelineConfig":
        """Return a new config with specified fields overridden.

        Args:
            **kwargs: Field names and their new values.

        Returns:
            A new DQNPipelineConfig with the specified overrides applied.
        """
        current = asdict(self)
        current.update(kwargs)
        # Restore tuple for betas if it was serialised as list
        if isinstance(current.get("betas"), list):
            current["betas"] = tuple(current["betas"])
        return DQNPipelineConfig(**current)

    def validate(self) -> list[str]:
        """Validate all parameters at submission time.

        Returns:
            List of error messages. Empty list means the config is valid.
        """
        errors: list[str] = []

        # Environment
        if not self.grpc_address:
            errors.append("grpc_address must not be empty")
        if self.symbol not in ("USDJPY", "AUDJPY"):
            errors.append(f"Invalid symbol: {self.symbol}")
        if self.step_size_seconds <= 0:
            errors.append(
                f"step_size_seconds must be positive: {self.step_size_seconds}"
            )

        # Date-range episode scheduling
        if self.date_start and self.date_end:
            from datetime import date as _date
            try:
                ds = _date.fromisoformat(self.date_start)
                de = _date.fromisoformat(self.date_end)
                if ds >= de:
                    errors.append(
                        f"date_start ({self.date_start}) must be < date_end ({self.date_end})"
                    )
            except ValueError as e:
                errors.append(f"Invalid date format: {e}")
        if not (0 <= self.hour_of_day_start <= 23):
            errors.append(
                f"hour_of_day_start must be 0-23: {self.hour_of_day_start}"
            )
        if not (0 <= self.hour_of_day_end <= self.hour_of_day_start + 24):
            errors.append(
                f"hour_of_day_end must be 0-{self.hour_of_day_start + 24}: "
                f"{self.hour_of_day_end}"
            )
        if self.hour_of_day_start == self.hour_of_day_end:
            errors.append(
                "hour_of_day_start and hour_of_day_end must differ"
            )

        # Agent
        if not (0.0 <= self.gamma <= 1.0):
            errors.append(f"gamma must be in [0, 1]: {self.gamma}")
        if not (0.0 <= self.epsilon_start <= 1.0):
            errors.append(f"epsilon_start must be in [0, 1]: {self.epsilon_start}")
        if not (0.0 <= self.epsilon_end <= 1.0):
            errors.append(f"epsilon_end must be in [0, 1]: {self.epsilon_end}")
        if self.epsilon_end > self.epsilon_start:
            errors.append(
                f"epsilon_end ({self.epsilon_end}) must be <= epsilon_start ({self.epsilon_start})"
            )
        if self.epsilon_decay_steps <= 0:
            errors.append(
                f"epsilon_decay_steps must be positive: {self.epsilon_decay_steps}"
            )
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive: {self.batch_size}")
        if self.replay_buffer_size <= 0:
            errors.append(
                f"replay_buffer_size must be positive: {self.replay_buffer_size}"
            )
        if self.target_update_freq <= 0:
            errors.append(
                f"target_update_freq must be positive: {self.target_update_freq}"
            )
        if self.train_freq <= 0:
            errors.append(f"train_freq must be positive: {self.train_freq}")
        if not (0.0 < self.tau <= 1.0):
            errors.append(f"tau must be in (0, 1]: {self.tau}")

        # Network
        if not self.hidden_dims:
            errors.append("hidden_dims must not be empty")
        if any(d <= 0 for d in self.hidden_dims):
            errors.append(f"All hidden_dims must be positive: {self.hidden_dims}")
        if self.activation not in ("relu", "leaky_relu", "gelu"):
            errors.append(f"Invalid activation: {self.activation}")
        if not (0.0 <= self.dropout < 1.0):
            errors.append(f"dropout must be in [0, 1): {self.dropout}")
        if not (1e-6 <= self.learning_rate <= 0.01):
            errors.append(f"learning_rate out of range [1e-6, 0.01]: {self.learning_rate}")
        if self.grad_clip_norm <= 0:
            errors.append(f"grad_clip_norm must be positive: {self.grad_clip_norm}")
        if self.weight_decay < 0:
            errors.append(f"weight_decay must be non-negative: {self.weight_decay}")
        if self.loss_function not in ("huber", "mse"):
            errors.append(f"Invalid loss_function: {self.loss_function}")

        # Training
        # Episode-count knobs are mode-specific; reject the one that has no
        # effect in the active mode rather than silently ignoring it.
        is_date_range = bool(self.date_start and self.date_end)
        if is_date_range and self.num_episodes_per_range is not None:
            errors.append(
                "num_episodes_per_range has no effect in date-range mode; "
                "set repeats_per_date instead"
            )
        if not is_date_range and self.repeats_per_date is not None:
            errors.append(
                "repeats_per_date has no effect in fixed-window mode; "
                "set num_episodes_per_range instead"
            )
        if self.num_episodes_per_range is not None and self.num_episodes_per_range <= 0:
            errors.append(f"num_episodes_per_range must be positive: {self.num_episodes_per_range}")
        if self.repeats_per_date is not None and self.repeats_per_date <= 0:
            errors.append(f"repeats_per_date must be positive: {self.repeats_per_date}")
        if self.max_steps_per_episode <= 0:
            errors.append(
                f"max_steps_per_episode must be positive: {self.max_steps_per_episode}"
            )
        if self.checkpoint_interval <= 0:
            errors.append(
                f"checkpoint_interval must be positive: {self.checkpoint_interval}"
            )

        # Infrastructure
        if not (1 <= self.num_workers <= 4):
            errors.append(f"num_workers must be in [1, 4]: {self.num_workers}")
        if self.max_wall_time_hours <= 0:
            errors.append(
                f"max_wall_time_hours must be positive: {self.max_wall_time_hours}"
            )

        # Katib
        if self.katib_enabled:
            if self.katib_max_trials <= 0:
                errors.append(
                    f"katib_max_trials must be positive: {self.katib_max_trials}"
                )
            if self.katib_parallel_trials <= 0:
                errors.append(
                    f"katib_parallel_trials must be positive: {self.katib_parallel_trials}"
                )
            if self.katib_trial_timeout_hours <= 0:
                errors.append(
                    f"katib_trial_timeout_hours must be positive: {self.katib_trial_timeout_hours}"
                )

        # Monitoring
        if self.sharpe_absolute_threshold < 0:
            errors.append(
                f"sharpe_absolute_threshold must be non-negative: {self.sharpe_absolute_threshold}"
            )

        # Finetune
        if self.training_mode == "finetune":
            if not (1e-6 <= self.finetune_learning_rate <= 0.01):
                errors.append(
                    f"finetune_learning_rate out of range [1e-6, 0.01]: {self.finetune_learning_rate}"
                )
            if self.finetune_num_episodes_per_range <= 0:
                errors.append(
                    f"finetune_num_episodes_per_range must be positive: {self.finetune_num_episodes_per_range}"
                )

        return errors

    def to_cli_args(self) -> list[str]:
        """Convert config fields to CLI arguments for `deepqnetwork/train.py`.

        Generates a list of command-line arguments matching the argparse interface
        defined in `deepqnetwork.config.load_config()`. The resulting list can be
        passed directly to `load_config(args=...)`.

        Returns:
            List of CLI argument strings (e.g., ["--symbol", "USDJPY", ...]).
        """
        args: list[str] = []

        # Environment
        args.extend(["--grpc-address", self.grpc_address])
        args.extend(["--symbol", self.symbol])
        args.extend(["--episode-start-ts", str(self.episode_start_ts)])
        args.extend(["--episode-end-ts", str(self.episode_end_ts)])
        args.extend(["--step-size-seconds", str(self.step_size_seconds)])

        # Date-range episode scheduling
        if self.date_start:
            args.extend(["--date-start", self.date_start])
        if self.date_end:
            args.extend(["--date-end", self.date_end])
        args.extend(["--hour-start", str(self.hour_of_day_start)])
        args.extend(["--hour-end", str(self.hour_of_day_end)])

        # Agent
        args.extend(["--gamma", str(self.gamma)])
        args.extend(["--epsilon-start", str(self.epsilon_start)])
        args.extend(["--epsilon-end", str(self.epsilon_end)])
        args.extend(["--epsilon-decay-steps", str(self.epsilon_decay_steps)])
        args.extend(["--batch-size", str(self.batch_size)])
        args.extend(["--replay-buffer-size", str(self.replay_buffer_size)])
        args.extend(["--target-update-freq", str(self.target_update_freq)])
        args.extend(["--train-freq", str(self.train_freq)])
        args.extend(["--tau", str(self.tau)])

        # Network, only fields that have CLI counterparts in train.py
        effective_lr = (
            self.finetune_learning_rate
            if self.training_mode == "finetune"
            else self.learning_rate
        )
        args.extend(["--learning-rate", str(effective_lr)])
        args.extend(["--dropout", str(self.dropout)])
        args.extend(["--grad-clip-norm", str(self.grad_clip_norm)])
        args.extend(["--weight-decay", str(self.weight_decay)])
        args.extend(["--loss-function", self.loss_function])
        if self.dueling:
            args.append("--dueling")

        # Training, emit only the knob that applies to the active mode so the
        # downstream DQNConfig never trips train()'s mis-set guard.
        if self.date_start and self.date_end:
            if self.repeats_per_date is not None:
                args.extend(["--repeats-per-date", str(self.repeats_per_date)])
        else:
            effective_episodes = (
                self.finetune_num_episodes_per_range
                if self.training_mode == "finetune"
                else self.num_episodes_per_range
            )
            if effective_episodes is not None:
                args.extend(["--num-episodes-per-range", str(effective_episodes)])
        args.extend(["--max-steps-per-episode", str(self.max_steps_per_episode)])
        args.extend(["--checkpoint-interval", str(self.checkpoint_interval)])

        # Mode is always train for the pipeline
        args.extend(["--mode", "train"])

        return args
