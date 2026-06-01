"""Property-based tests for DqnpfPipelineConfig.

Feature: kubeflow-ml-pipeline (dqnpf-intraday section)

Properties:
- DQNPF-CFG-1: YAML round-trip
- DQNPF-CFG-2: CLI argv generation
- DQNPF-CFG-3: Invalid configs surface errors
"""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from tradingmodel.intraday.dqnpf.config import IntegrationConfig, load_config
from tradingmodel.intraday.dqnpf.kubeflow.pipeline.config_schema import (
    DqnpfPipelineConfig,
)


_INTEGRATION_FIELDS = {f.name for f in fields(IntegrationConfig)}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_valid_config = st.builds(
    DqnpfPipelineConfig,
    symbol=st.sampled_from(["USDJPY", "AUDJPY", "EURUSD"]),
    variance_threshold=st.floats(min_value=0.0, max_value=50.0, allow_nan=False),
    max_risk_long_units=st.integers(min_value=0, max_value=10),
    max_risk_short_units=st.integers(min_value=0, max_value=10),
    directional_disagreement=st.booleans(),
    directional_tolerance=st.floats(min_value=0.0, max_value=20.0, allow_nan=False),
    forecast_horizon=st.sampled_from([1, 3, 6, 12]),
    min_bars_warmup=st.integers(min_value=36, max_value=5000),
    step_size_seconds=st.sampled_from([5, 30, 60, 300]),
    num_episodes=st.integers(min_value=1, max_value=100),
    episode_start_ts=st.integers(min_value=0, max_value=2_000_000_000),
    episode_end_ts=st.integers(min_value=0, max_value=2_000_000_000),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    max_wall_time_hours=st.integers(min_value=1, max_value=48),
)


# ---------------------------------------------------------------------------
# DQNPF-CFG-1: YAML round-trip
# ---------------------------------------------------------------------------


@given(cfg=_valid_config)
def test_yaml_round_trip_preserves_all_fields(
    cfg: DqnpfPipelineConfig, tmp_path_factory: pytest.TempPathFactory
) -> None:
    tmp_path = tmp_path_factory.mktemp("dqnpf_cfg_yaml")
    yaml_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(yaml_path)

    reloaded = DqnpfPipelineConfig.from_yaml(yaml_path)

    assert asdict(reloaded) == asdict(cfg)


def test_default_yaml_loads_cleanly() -> None:
    """The shipped default YAML resolves to a valid DqnpfPipelineConfig."""
    default_path = (
        Path(__file__).resolve().parents[1] / "config" / "dqnpf_pipeline_config.yaml"
    )
    assert default_path.exists()
    cfg = DqnpfPipelineConfig.from_yaml(default_path)
    assert cfg.validate() == []
    assert cfg.symbol == "USDJPY"
    assert cfg.dqn_model_registry_name == "deepqnetwork-usdjpy"


# ---------------------------------------------------------------------------
# DQNPF-CFG-2: CLI argv generation
# ---------------------------------------------------------------------------


@given(cfg=_valid_config)
def test_cli_argv_round_trips_via_load_config(
    cfg: DqnpfPipelineConfig, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """to_cli_args produces argv that load_config parses back to the same IntegrationConfig."""
    tmp_path = tmp_path_factory.mktemp("dqnpf_cli_argv")
    # Point load_config at a non-existent config file so YAML defaults don't
    # contaminate the round-trip; CLI flags are the sole source of truth.
    empty_config_path = tmp_path / "missing.yaml"

    argv = ["--config", str(empty_config_path)] + cfg.to_cli_args()
    parsed = load_config(argv)
    expected = cfg.to_integration_config()

    for f in fields(IntegrationConfig):
        assert getattr(parsed, f.name) == getattr(expected, f.name), f.name


@given(cfg=_valid_config)
def test_cli_argv_only_covers_integration_fields(cfg: DqnpfPipelineConfig) -> None:
    """to_cli_args emits one flag per IntegrationConfig field (excluding Nones)."""
    argv = cfg.to_cli_args()
    flags = {arg for arg in argv if arg.startswith("--")}
    expected_flags = {
        "--" + name.replace("_", "-")
        for name in _INTEGRATION_FIELDS
        if getattr(cfg, name) is not None
    }
    assert flags == expected_flags


# ---------------------------------------------------------------------------
# Mirror integrity: DqnpfPipelineConfig must declare every IntegrationConfig
# field, or to_integration_config() raises AttributeError at runtime (it
# iterates the live IntegrationConfig field set and getattr()s each off self).
# ---------------------------------------------------------------------------


def test_pipeline_config_mirrors_every_integration_field() -> None:
    """Every IntegrationConfig field is mirrored on DqnpfPipelineConfig."""
    pipeline_fields = {f.name for f in fields(DqnpfPipelineConfig)}
    missing = _INTEGRATION_FIELDS - pipeline_fields
    assert not missing, (
        f"DqnpfPipelineConfig is missing mirrored IntegrationConfig field(s): "
        f"{sorted(missing)}. Add them (with matching defaults) and a --flag in "
        f"config._build_parser, else to_integration_config()/to_cli_args break."
    )


def test_default_pipeline_config_builds_integration_config() -> None:
    """The default config produces a valid IntegrationConfig (pod startup path)."""
    ic = DqnpfPipelineConfig().to_integration_config()
    for name in _INTEGRATION_FIELDS:
        assert getattr(ic, name) == getattr(DqnpfPipelineConfig(), name), name


# ---------------------------------------------------------------------------
# DQNPF-CFG-3: Invalid configs surface errors
# ---------------------------------------------------------------------------


@given(threshold=st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False))
def test_negative_variance_threshold_surfaces_error(threshold: float) -> None:
    cfg = DqnpfPipelineConfig(variance_threshold=threshold)
    errors = cfg.validate()
    assert errors
    assert any("variance_threshold" in e for e in errors)


@given(horizon=st.integers().filter(lambda v: v not in {1, 3, 6, 12}))
def test_invalid_forecast_horizon_surfaces_error(horizon: int) -> None:
    cfg = DqnpfPipelineConfig(forecast_horizon=horizon)
    errors = cfg.validate()
    assert errors
    assert any("forecast_horizon" in e for e in errors)


@given(hours=st.integers(max_value=0))
def test_non_positive_wall_time_surfaces_error(hours: int) -> None:
    cfg = DqnpfPipelineConfig(max_wall_time_hours=hours)
    errors = cfg.validate()
    assert errors
    assert any("max_wall_time_hours" in e for e in errors)


@pytest.mark.parametrize(
    "field,bad_value,expected_token",
    [
        ("max_risk_long_units", -1, "max_risk_long_units"),
        ("max_risk_short_units", -1, "max_risk_short_units"),
        ("directional_tolerance", -0.5, "directional_tolerance"),
        ("dqn_lifecycle_stage", "deployed", "dqn_lifecycle_stage"),
        ("forecaster_lifecycle_stage", "deployed", "forecaster_lifecycle_stage"),
    ],
)
def test_invalid_field_surfaces_error(
    field: str, bad_value, expected_token: str
) -> None:
    cfg = DqnpfPipelineConfig(**{field: bad_value})
    errors = cfg.validate()
    assert errors
    assert any(expected_token in e for e in errors)


def test_valid_default_config_has_no_errors() -> None:
    assert DqnpfPipelineConfig().validate() == []
