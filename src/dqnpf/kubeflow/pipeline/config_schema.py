"""Pipeline configuration for the dqnpf-intraday KFP pipeline.

Wraps the existing :class:`IntegrationConfig` with pipeline-level fields
(parent model registry references, infrastructure limits) and exposes
``from_yaml`` / ``to_yaml`` / ``to_cli_args`` / ``validate`` helpers.

The dataclass is intentionally flat: every ``IntegrationConfig`` field is
mirrored as a top-level attribute so a single YAML document round-trips
through ``yaml.safe_dump`` / ``yaml.safe_load`` without nested-dataclass
gymnastics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from tradingmodel.intraday.dqnpf.config import IntegrationConfig


_INTEGRATION_FIELDS = {f.name for f in fields(IntegrationConfig)}


@dataclass
class DqnpfPipelineConfig:
    """Configuration for the dqnpf-intraday KFP pipeline.

    Combines screening parameters (mirrored from :class:`IntegrationConfig`)
    with pipeline-level fields used by the KFP backtest component and the
    Model Registry resolver.
    """

    # --- Mirrored IntegrationConfig fields (kept in sync) ---
    symbol: str = "USDJPY"
    variance_threshold: float = 4.5
    max_risk_long_units: int = 2
    max_risk_short_units: int = 1
    forecaster_risk_aversion: float = 0.1
    forecaster_position_size: float = 10_000_000.0
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
    screen_profit_window_sessions: int = 7

    # --- Pipeline-level fields ---
    dqn_model_registry_name: str = "deepqnetwork-usdjpy"
    forecaster_model_registry_name: str = "probabilistic-transformer-usdjpy-h1"
    dqn_lifecycle_stage: str = "production"
    forecaster_lifecycle_stage: str = "production"
    max_wall_time_hours: int = 4

    # ------------------------------------------------------------------ #
    # Construction / serialisation
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DqnpfPipelineConfig":
        """Load a config from a YAML file, ignoring unknown keys."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def to_yaml(self, path: str | Path) -> None:
        """Write this config to a YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=True)

    def to_integration_config(self) -> IntegrationConfig:
        """Extract the embedded :class:`IntegrationConfig`.

        Raises:
            ValueError: If any mirrored field is invalid (relayed from
                :meth:`IntegrationConfig.__post_init__`).
        """
        return IntegrationConfig(
            **{name: getattr(self, name) for name in _INTEGRATION_FIELDS}
        )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> list[str]:
        """Return a list of error messages (empty if config is valid)."""
        errors: list[str] = []
        try:
            self.to_integration_config()
        except ValueError as exc:
            errors.append(str(exc))
        if self.max_wall_time_hours <= 0:
            errors.append(
                f"max_wall_time_hours must be > 0, got {self.max_wall_time_hours}"
            )
        if self.dqn_lifecycle_stage not in {"staging", "production", "archived"}:
            errors.append(
                f"dqn_lifecycle_stage must be staging|production|archived, "
                f"got {self.dqn_lifecycle_stage!r}"
            )
        if self.forecaster_lifecycle_stage not in {
            "staging",
            "production",
            "archived",
        }:
            errors.append(
                f"forecaster_lifecycle_stage must be staging|production|archived, "
                f"got {self.forecaster_lifecycle_stage!r}"
            )
        return errors

    # ------------------------------------------------------------------ #
    # CLI argv for the integration-layer entry points
    # ------------------------------------------------------------------ #

    def to_cli_args(self) -> list[str]:
        """Emit ``--flag value`` argv covering every ``IntegrationConfig`` field.

        The output is consumable by
        :func:`tradingmodel.intraday.dqnpf.config.load_config` and reproduces
        the same ``IntegrationConfig`` when parsed back. ``None`` values are
        omitted (load_config falls back to YAML/defaults).
        """
        args: list[str] = []
        for name in sorted(_INTEGRATION_FIELDS):
            value = getattr(self, name)
            if value is None:
                continue
            flag = "--" + name.replace("_", "-")
            if isinstance(value, bool):
                args.extend([flag, "true" if value else "false"])
            else:
                args.extend([flag, str(value)])
        return args
