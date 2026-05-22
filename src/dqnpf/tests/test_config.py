"""Unit tests for IntegrationConfig defaults and YAML round-trip."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path

import pytest
import yaml

from tradingmodel.intraday.dqnpf.config import IntegrationConfig, load_config


# ---------------------------------------------------------------------------
# Default values (Req 11.1)
# ---------------------------------------------------------------------------


def test_defaults_match_spec() -> None:
    cfg = IntegrationConfig()
    assert cfg.symbol == "USDJPY"
    assert cfg.variance_threshold == 4.5
    assert cfg.max_risk_long_units == 2
    assert cfg.max_risk_short_units == 1
    assert cfg.directional_disagreement is False
    assert cfg.directional_tolerance == 1.0
    assert cfg.forecast_horizon == 1
    assert cfg.min_bars_warmup == 1440
    assert cfg.step_size_seconds == 60


# ---------------------------------------------------------------------------
# Explicit validation (Req 5.3, 5.4, 11.2)
# ---------------------------------------------------------------------------


def test_negative_variance_threshold_raises() -> None:
    with pytest.raises(ValueError, match="variance_threshold"):
        IntegrationConfig(variance_threshold=-0.1)


def test_negative_max_risk_long_raises() -> None:
    with pytest.raises(ValueError, match="max_risk_long_units"):
        IntegrationConfig(max_risk_long_units=-1)


def test_negative_max_risk_short_raises() -> None:
    with pytest.raises(ValueError, match="max_risk_short_units"):
        IntegrationConfig(max_risk_short_units=-1)


def test_negative_directional_tolerance_raises() -> None:
    with pytest.raises(ValueError, match="directional_tolerance"):
        IntegrationConfig(directional_tolerance=-0.5)


@pytest.mark.parametrize("invalid_horizon", [0, 2, 4, 5, 7, 13, -1])
def test_invalid_forecast_horizon_raises(invalid_horizon: int) -> None:
    with pytest.raises(ValueError, match="forecast_horizon"):
        IntegrationConfig(forecast_horizon=invalid_horizon)


@pytest.mark.parametrize("valid_horizon", [1, 3, 6, 12])
def test_valid_forecast_horizons_accepted(valid_horizon: int) -> None:
    cfg = IntegrationConfig(forecast_horizon=valid_horizon)
    assert cfg.forecast_horizon == valid_horizon


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


def test_yaml_round_trip_preserves_fields(tmp_path: Path) -> None:
    original = IntegrationConfig(
        symbol="EURUSD",
        variance_threshold=3.1,
        max_risk_long_units=4,
        max_risk_short_units=2,
        directional_disagreement=True,
        directional_tolerance=0.5,
        forecast_horizon=3,
        min_bars_warmup=2000,
        step_size_seconds=60,
        dqn_checkpoint_path="/tmp/dqn.pt",
        forecaster_checkpoint_path="/tmp/fc.pt",
        device="cpu",
        grpc_address="localhost:1234",
        num_episodes=5,
        episode_start_ts=1_700_000_000,
        episode_end_ts=1_800_000_000,
        seed=42,
    )

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml.safe_dump(asdict(original)))

    reloaded = load_config(["--config", str(yaml_path)])

    for f in fields(IntegrationConfig):
        assert getattr(reloaded, f.name) == getattr(original, f.name), f.name


def test_yaml_round_trip_with_default_config_file() -> None:
    """The shipped config.yaml should load without error and match field types."""
    default_path = (
        Path(__file__).resolve().parents[1] / "config.yaml"
    )
    assert default_path.exists()
    cfg = load_config(["--config", str(default_path)])
    assert cfg.symbol == "USDJPY"
    assert cfg.variance_threshold == 4.5


# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------


def test_cli_overrides_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml.safe_dump({"symbol": "USDJPY", "variance_threshold": 4.5}))

    cfg = load_config(
        [
            "--config",
            str(yaml_path),
            "--symbol",
            "EURUSD",
            "--variance-threshold",
            "7.0",
            "--max-risk-long",
            "3",
        ]
    )
    assert cfg.symbol == "EURUSD"
    assert cfg.variance_threshold == 7.0
    assert cfg.max_risk_long_units == 3
