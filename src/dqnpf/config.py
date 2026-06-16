"""Configuration management for the dqnpf integration layer.

Loads configuration from a YAML file with CLI argument overrides.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class IntegrationConfig:
    """Configuration for the integration layer.

    Attributes:
        symbol: Currency pair (e.g. "USDJPY"). Bound at construction.
        variance_threshold: Sigma value (bps) above which opened positions
            consume the risk budget. Default 4.5.
        max_risk_long_units: Hard cap on cumulative long exposure opened
            during high-sigma regimes. Default 2.
        max_risk_short_units: Hard cap on cumulative short exposure opened
            during high-sigma regimes. Default 1.
        directional_disagreement: Whether sign conflict with forecaster
            suppresses the DQN action. Default False.
        directional_tolerance: abs(mu) below which directional check is
            skipped, in bps. Default 1.0.
        forecast_horizon: Forecaster horizon in 5-min bars. Default 1.
        min_bars_warmup: M5 bars required before forecaster is valid.
            Default 1440.
        step_size_seconds: Decision interval. Default 60.
        dqn_checkpoint_path: Path to DQN checkpoint (trained at 60s).
        forecaster_checkpoint_path: Path to forecaster checkpoint.
        device: Device specifier ("auto", "cpu", "cuda", "mps", ...).
        grpc_address: modelenv gRPC endpoint.
        num_episodes: Number of episodes for training or backtesting. Used only
            in legacy fixed-window mode (when ``date_start``/``date_end`` are
            unset); in date-range mode the episode count is one per calendar date.
        episode_start_ts: Unix-seconds start of each episode window (legacy
            fixed-window mode).
        episode_end_ts: Unix-seconds end of each episode window (legacy
            fixed-window mode).
        date_start: ISO date (e.g. "2012-02-01") for the first backtest episode.
            When set with ``date_end``, the backtest runs one hour-bound episode
            per calendar date (mirroring DQN training) instead of the legacy
            fixed window. Default "" (unset → legacy mode).
        date_end: ISO date for the last backtest episode (inclusive).
        hour_of_day_start: Hour (0-23) each episode begins. Default None =
            inherit the value the DQN model was trained with (read from the
            checkpoint), so evaluation aligns with the trained session.
        hour_of_day_end: Hour each episode ends; ``>= 24`` rolls into the next
            day (e.g. 47 = 23:00 next day → a 24 h session from hour 23). Default
            None = inherit from the DQN checkpoint.
        seed: Seed forwarded to modelenv Reset() for reproducibility.
        pip_size: Price increment of one pip for ``symbol``, used to convert
            the environment's raw monetary PnL into pips for reporting.
            Default 0.01 (USDJPY and other JPY quote pairs).
    """

    symbol: str = "USDJPY"
    variance_threshold: float = 4.5
    max_risk_long_units: int = 2
    max_risk_short_units: int = 1
    directional_disagreement: bool = False
    directional_tolerance: float = 1.0
    # Conviction (|mu|/sigma) at/above which the forecaster-only backtest
    # baseline sizes up to 2 units instead of 1.
    forecaster_conviction_threshold: float = 2.0
    forecast_horizon: int = 1
    min_bars_warmup: int = 1440
    step_size_seconds: int = 60
    dqn_checkpoint_path: str | None = None
    forecaster_checkpoint_path: str | None = None
    device: str = "auto"
    grpc_address: str = "localhost:50051"
    num_episodes: int = 1
    episode_start_ts: int = 0
    episode_end_ts: int = 0
    date_start: str = ""
    date_end: str = ""
    hour_of_day_start: int | None = None
    hour_of_day_end: int | None = None
    seed: int = 0
    pip_size: float = 0.01

    def __post_init__(self) -> None:
        if self.pip_size <= 0:
            raise ValueError("pip_size must be positive")
        if self.variance_threshold < 0:
            raise ValueError("variance_threshold must be non-negative")
        if self.max_risk_long_units < 0:
            raise ValueError("max_risk_long_units must be non-negative")
        if self.max_risk_short_units < 0:
            raise ValueError("max_risk_short_units must be non-negative")
        if self.directional_tolerance < 0:
            raise ValueError("directional_tolerance must be non-negative")
        if self.forecast_horizon not in {1, 3, 6, 12}:
            raise ValueError(
                "forecast_horizon must be one of {1, 3, 6, 12}, "
                f"got {self.forecast_horizon}"
            )
        if self.num_episodes < 1:
            raise ValueError("num_episodes must be >= 1")
        if bool(self.date_start) != bool(self.date_end):
            raise ValueError(
                "date_start and date_end must be set together (date-range mode) "
                "or both empty (legacy fixed-window mode)"
            )
        if self.hour_of_day_start is not None and not 0 <= self.hour_of_day_start <= 23:
            raise ValueError("hour_of_day_start must be in [0, 23]")
        if self.hour_of_day_end is not None and not 0 <= self.hour_of_day_end <= 47:
            raise ValueError("hour_of_day_end must be in [0, 47] (>=24 = next-day)")


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


def _str2bool(value: str) -> bool:
    """Parse string to bool for argparse (accepts 'true'/'false'/'1'/'0')."""
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes", "y", "t")


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Every ``IntegrationConfig`` field has a corresponding ``--flag`` so that
    ``DqnpfPipelineConfig.to_cli_args()`` can produce a complete override argv
    that round-trips through ``load_config``.
    """
    parser = argparse.ArgumentParser(
        description="DQN + ProbabilisticForecaster integration layer"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tradingmodel/intraday/dqnpf/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument(
        "--variance-threshold", dest="variance_threshold", type=float, default=None
    )
    parser.add_argument(
        "--max-risk-long-units",
        "--max-risk-long",
        dest="max_risk_long_units",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-risk-short-units",
        "--max-risk-short",
        dest="max_risk_short_units",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--directional-disagreement",
        dest="directional_disagreement",
        type=_str2bool,
        default=None,
    )
    parser.add_argument(
        "--directional-tolerance",
        dest="directional_tolerance",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--forecast-horizon", dest="forecast_horizon", type=int, default=None
    )
    parser.add_argument(
        "--min-bars-warmup", dest="min_bars_warmup", type=int, default=None
    )
    parser.add_argument(
        "--step-size-seconds", dest="step_size_seconds", type=int, default=None
    )
    parser.add_argument(
        "--dqn-checkpoint-path",
        "--dqn-checkpoint",
        dest="dqn_checkpoint_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--forecaster-checkpoint-path",
        "--forecaster-checkpoint",
        dest="forecaster_checkpoint_path",
        type=str,
        default=None,
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--grpc-address", dest="grpc_address", type=str, default=None)
    parser.add_argument("--num-episodes", dest="num_episodes", type=int, default=None)
    parser.add_argument(
        "--episode-start-ts", dest="episode_start_ts", type=int, default=None
    )
    parser.add_argument(
        "--episode-end-ts", dest="episode_end_ts", type=int, default=None
    )
    parser.add_argument("--date-start", dest="date_start", type=str, default=None)
    parser.add_argument("--date-end", dest="date_end", type=str, default=None)
    parser.add_argument(
        "--hour-start",
        "--hour-of-day-start",
        dest="hour_of_day_start",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--hour-end",
        "--hour-of-day-end",
        dest="hour_of_day_end",
        type=int,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pip-size", dest="pip_size", type=float, default=None)
    return parser


_CONFIG_FIELDS = {f.name for f in fields(IntegrationConfig)}


def load_config(args: list[str] | None = None) -> IntegrationConfig:
    """Load configuration from YAML with CLI argument overrides.

    Order of precedence (highest to lowest):
    1. CLI arguments (only those explicitly provided)
    2. YAML configuration file values
    3. IntegrationConfig dataclass defaults
    """
    parser = _build_parser()
    parsed = parser.parse_args(args)

    yaml_data = _load_yaml(parsed.config)

    config_kwargs: dict = {}
    for key, value in yaml_data.items():
        if key in _CONFIG_FIELDS:
            config_kwargs[key] = value

    parsed_dict = vars(parsed)
    for key, value in parsed_dict.items():
        if key == "config" or value is None:
            continue
        if key in _CONFIG_FIELDS:
            config_kwargs[key] = value

    return IntegrationConfig(**config_kwargs)
