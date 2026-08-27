"""Pipeline configuration schema with YAML loading, validation, and date resolution.

Extends the existing ForecasterConfig concept with infrastructure parameters
for Kubeflow pipeline orchestration, Katib tuning, KServe serving, and monitoring.
"""

from dataclasses import dataclass, field
from typing import Literal

import yaml


@dataclass
class PipelineConfig:
    """Complete pipeline configuration loaded from YAML."""

    # Model parameters (from ForecasterConfig)
    symbol: str = "USDJPY"
    forecast_horizon: int = 1  # 1, 3, 6, 12
    lookback_window: int = 36
    historical_window: int = 1440
    num_features: int = 16
    num_layers: int = 3
    num_heads: int = 4
    d_ff: int = 64  # Feed-forward hidden dimension in Transformer layers
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 5
    random_seed: int = 42

    # Date ranges (static fallback when no snapshot_date is provided)
    train_start: str = "2012-01-01"
    train_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    test_end: str = "2026-04-30"

    # Rolling-window parameters (used when snapshot_date is provided at runtime)
    # Percentage-based split of total available data (from data_start to snapshot_date)
    data_start: str = "2012-01-01"  # Earliest available data
    train_pct: float = 0.75  # 75% of total history for training
    test_pct: float = 0.25  # 25% of total history for test/evaluation
    # When snapshot_date is provided:
    #   total_days = snapshot_date - data_start
    #   train_days = total_days * train_pct
    #   test_days  = total_days * test_pct
    #   test_end   = snapshot_date
    #   test_start = snapshot_date - test_days
    #   train_end  = test_start - 1 day
    #   train_start = data_start

    # Infrastructure
    num_workers: int = 1  # DDP workers (1-4)
    max_wall_time_hours: int = 8

    # Katib
    katib_max_trials: int = 30
    katib_parallel_trials: int = 5
    katib_trial_timeout_hours: int = 2

    # Serving
    serving_min_replicas: int = 0
    serving_max_replicas: int = 4
    serving_target_concurrency: int = 10
    canary_traffic_percent: int = 0

    # Monitoring
    alert_webhook_url: str = ""
    nll_degradation_threshold: float = 0.1
    da_degradation_threshold: float = 0.05
    nll_absolute_threshold: float = 3.5
    da_absolute_threshold: float = 0.50

    # Scheduling
    schedule_mode: Literal["daily", "weekly", "on-demand", "drift-triggered"] = (
        "drift-triggered"
    )

    # Training mode
    training_mode: Literal["scratch", "finetune"] = "scratch"
    # "scratch": Random init, full training (5 epochs on rolling window)
    # "finetune": Load production model weights, train 1-2 epochs on recent data with reduced LR
    finetune_epochs: int = 2  # Epochs when fine-tuning (fewer than full training)
    finetune_learning_rate: float = 0.0001  # Reduced LR for fine-tuning (10x lower)
    #
    # Scheduling strategy:
    #   - Weekly (@weekly): fine-tune the production model on the latest week's data
    #   - Drift-triggered: retrain from scratch on the full rolling window
    # The Airflow DAG runs two schedules:
    #   1. Weekly fine-tune DAG: training_mode="finetune", snapshot_date=today
    #   2. Drift-triggered retrain DAG: training_mode="scratch", snapshot_date=today
    #      (only fires when model-drift-detection flags degradation)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def override(self, **kwargs) -> "PipelineConfig":
        """Return a new config with specified fields overridden."""
        from dataclasses import asdict

        current = asdict(self)
        current.update(kwargs)
        return PipelineConfig(**current)

    def validate(self) -> list[str]:
        """Validate all parameters. Returns list of error messages (empty = valid)."""
        errors = []
        if self.symbol not in ("USDJPY", "AUDJPY"):
            errors.append(f"Invalid symbol: {self.symbol}")
        if self.forecast_horizon not in (1, 3, 6, 12):
            errors.append(f"Invalid forecast_horizon: {self.forecast_horizon}")
        if not (0.0001 <= self.learning_rate <= 0.01):
            errors.append(f"learning_rate out of range: {self.learning_rate}")
        if self.num_layers not in (2, 3, 4):
            errors.append(f"Invalid num_layers: {self.num_layers}")
        if self.num_heads not in (2, 4, 8):
            errors.append(f"Invalid num_heads: {self.num_heads}")
        if self.d_ff not in (32, 64, 128):
            errors.append(f"Invalid d_ff: {self.d_ff}")
        if not (0.05 <= self.dropout <= 0.3):
            errors.append(f"dropout out of range: {self.dropout}")
        if self.batch_size not in (32, 64, 128):
            errors.append(f"Invalid batch_size: {self.batch_size}")
        if self.lookback_window not in (24, 36, 48):
            errors.append(f"Invalid lookback_window: {self.lookback_window}")
        if not (1 <= self.num_workers <= 4):
            errors.append(f"num_workers out of range: {self.num_workers}")
        if self.max_wall_time_hours <= 0:
            errors.append(f"Invalid max_wall_time_hours: {self.max_wall_time_hours}")
        if not (0.0 < self.train_pct < 1.0):
            errors.append(f"train_pct must be in (0, 1), got {self.train_pct}")
        if not (0.0 < self.test_pct < 1.0):
            errors.append(f"test_pct must be in (0, 1), got {self.test_pct}")
        if self.train_pct + self.test_pct > 1.0:
            errors.append(
                f"train_pct + test_pct must be <= 1.0, got {self.train_pct + self.test_pct}"
            )
        if not (0.0 <= self.nll_absolute_threshold <= 10.0):
            errors.append(
                f"nll_absolute_threshold out of range: {self.nll_absolute_threshold}"
            )
        if not (0.0 <= self.da_absolute_threshold <= 1.0):
            errors.append(
                f"da_absolute_threshold out of range: {self.da_absolute_threshold}"
            )
        return errors

    def resolve_date_ranges(self, snapshot_date: str | None = None) -> "PipelineConfig":
        """Compute train/test date ranges.

        If snapshot_date is provided, computes ranges as a percentage split of
        total available history (data_start to snapshot_date):
            total_days = snapshot_date - data_start
            test_days  = total_days * test_pct
            test_end   = snapshot_date
            test_start = snapshot_date - test_days
            train_end  = test_start - 1 day
            train_start = data_start

        If snapshot_date is None, returns self unchanged (uses static ranges).

        Args:
            snapshot_date: ISO date string (e.g., "2026-05-13").

        Returns:
            New PipelineConfig with resolved date ranges.
        """
        if snapshot_date is None:
            return self

        from datetime import date, timedelta

        end = date.fromisoformat(snapshot_date)
        start = date.fromisoformat(self.data_start)
        total_days = (end - start).days

        test_days = int(total_days * self.test_pct)
        test_end = end
        test_start = end - timedelta(days=test_days)
        train_end = test_start - timedelta(days=1)
        train_start = start

        return self.override(
            train_start=train_start.isoformat(),
            train_end=train_end.isoformat(),
            test_start=test_start.isoformat(),
            test_end=test_end.isoformat(),
        )
