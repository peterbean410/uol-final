"""Property-based tests for IntegrationConfig validation.

Feature: dqnpf, Property 12, Config validation
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from tradingmodel.intraday.dqnpf.config import IntegrationConfig


@given(
    variance_threshold=st.floats(min_value=0, max_value=100, allow_nan=False),
    max_risk_long_units=st.integers(min_value=0, max_value=20),
    max_risk_short_units=st.integers(min_value=0, max_value=20),
    directional_disagreement=st.booleans(),
    directional_tolerance=st.floats(min_value=0, max_value=50, allow_nan=False),
    forecast_horizon=st.sampled_from([1, 3, 6, 12]),
)
def test_valid_fields_construct_successfully(
    variance_threshold: float,
    max_risk_long_units: int,
    max_risk_short_units: int,
    directional_disagreement: bool,
    directional_tolerance: float,
    forecast_horizon: int,
) -> None:
    cfg = IntegrationConfig(
        symbol="USDJPY",
        variance_threshold=variance_threshold,
        max_risk_long_units=max_risk_long_units,
        max_risk_short_units=max_risk_short_units,
        directional_disagreement=directional_disagreement,
        directional_tolerance=directional_tolerance,
        forecast_horizon=forecast_horizon,
    )
    assert cfg.variance_threshold == variance_threshold
    assert cfg.max_risk_long_units == max_risk_long_units
    assert cfg.max_risk_short_units == max_risk_short_units
    assert cfg.directional_disagreement == directional_disagreement
    assert cfg.directional_tolerance == directional_tolerance
    assert cfg.forecast_horizon == forecast_horizon


@given(
    variance_threshold=st.floats(
        max_value=-1e-9, allow_nan=False, allow_infinity=False
    ),
)
def test_negative_variance_threshold_rejected(variance_threshold: float) -> None:
    with pytest.raises(ValueError):
        IntegrationConfig(variance_threshold=variance_threshold)


@given(max_risk_long_units=st.integers(max_value=-1))
def test_negative_max_risk_long_rejected(max_risk_long_units: int) -> None:
    with pytest.raises(ValueError):
        IntegrationConfig(max_risk_long_units=max_risk_long_units)


@given(max_risk_short_units=st.integers(max_value=-1))
def test_negative_max_risk_short_rejected(max_risk_short_units: int) -> None:
    with pytest.raises(ValueError):
        IntegrationConfig(max_risk_short_units=max_risk_short_units)


@given(
    directional_tolerance=st.floats(
        max_value=-1e-9, allow_nan=False, allow_infinity=False
    ),
)
def test_negative_directional_tolerance_rejected(directional_tolerance: float) -> None:
    with pytest.raises(ValueError):
        IntegrationConfig(directional_tolerance=directional_tolerance)


@given(forecast_horizon=st.integers().filter(lambda v: v not in {1, 3, 6, 12}))
def test_invalid_forecast_horizon_rejected(forecast_horizon: int) -> None:
    with pytest.raises(ValueError):
        IntegrationConfig(forecast_horizon=forecast_horizon)
