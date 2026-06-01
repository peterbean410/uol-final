"""Unit tests for the IntegrationLayer (budget, HOLD bypass, boundaries)."""

from __future__ import annotations

import math

import pytest

from tradingmodel.intraday.dqnpf.integration import ScreenedAction
from tradingmodel.intraday.dqnpf.tests.conftest import FakeActionResult, make_layer


def _a(idx: int) -> FakeActionResult:
    return FakeActionResult(action=idx, action_name=f"action_{idx}")


def test_buy_1_increments_long_by_one_on_high_sigma() -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_long_units=5)
    layer.screen(_a(1), mu=0.0, sigma=5.0)
    assert layer.risk_long_used == 1
    assert layer.risk_short_used == 0


def test_buy_2_increments_long_by_two_on_high_sigma() -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_long_units=5)
    layer.screen(_a(2), mu=0.0, sigma=5.0)
    assert layer.risk_long_used == 2


def test_sell_1_increments_short_by_one_on_high_sigma() -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_short_units=5)
    layer.screen(_a(3), mu=0.0, sigma=5.0)
    assert layer.risk_short_used == 1
    assert layer.risk_long_used == 0


def test_sell_2_increments_short_by_two_on_high_sigma() -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_short_units=5)
    layer.screen(_a(4), mu=0.0, sigma=5.0)
    assert layer.risk_short_used == 2


def test_hold_never_increments_budget_regardless_of_sigma() -> None:
    layer = make_layer(variance_threshold=1.0)
    layer.screen(_a(0), mu=0.0, sigma=5.0)
    layer.screen(_a(0), mu=0.0, sigma=0.1)
    assert layer.risk_long_used == 0
    assert layer.risk_short_used == 0


def test_low_sigma_does_not_increment_budget() -> None:
    layer = make_layer(variance_threshold=4.5, max_risk_long_units=5)
    layer.screen(_a(2), mu=0.0, sigma=4.5)  # boundary: sigma == threshold
    assert layer.risk_long_used == 0


def test_on_position_closed_buy_decrements_long() -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_long_units=5)
    layer.screen(_a(2), mu=0.0, sigma=5.0)  # long=2
    layer.on_position_closed("buy", 1)
    assert layer.risk_long_used == 1


def test_on_position_closed_sell_decrements_short() -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_short_units=5)
    layer.screen(_a(4), mu=0.0, sigma=5.0)  # short=2
    layer.on_position_closed("sell", 2)
    assert layer.risk_short_used == 0


def test_on_position_closed_clamps_to_zero() -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_long_units=5)
    layer.screen(_a(1), mu=0.0, sigma=5.0)  # long=1
    layer.on_position_closed("buy", 99)
    assert layer.risk_long_used == 0


def test_on_position_closed_unknown_side_is_noop(caplog: pytest.LogCaptureFixture) -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_long_units=5)
    layer.screen(_a(1), mu=0.0, sigma=5.0)
    with caplog.at_level("WARNING"):
        layer.on_position_closed("hold", 1)
    assert layer.risk_long_used == 1
    assert any("unknown side" in record.message for record in caplog.records)


def test_screened_action_fields_on_pass() -> None:
    layer = make_layer(variance_threshold=1.0)
    result = layer.screen(_a(1), mu=0.0, sigma=5.0)
    assert isinstance(result, ScreenedAction)
    assert result.action == 1
    assert result.action_name == "action_1"
    assert result.screened is False
    assert result.reason == "pass"
    assert result.sigma == 5.0
    assert result.risk_long_used == 1
    assert result.risk_short_used == 0


def test_screened_action_fields_on_budget_exhausted() -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_long_units=1)
    layer.screen(_a(1), mu=0.0, sigma=5.0)  # long=1
    result = layer.screen(_a(1), mu=0.0, sigma=5.0)
    assert result.action == 0
    assert result.action_name == "HOLD"
    assert result.screened is True
    assert result.reason == "budget_exhausted"
    assert result.risk_long_used == 1
    assert result.risk_short_used == 0


def test_directional_conflict_at_tolerance_boundary_is_skipped() -> None:
    # abs(mu) == tolerance → rule is skipped (strict inequality)
    layer = make_layer(
        variance_threshold=10.0,
        directional_disagreement=True,
        directional_tolerance=1.0,
    )
    result = layer.screen(_a(1), mu=-1.0, sigma=0.1)  # LONG with mu < 0
    assert result.reason == "pass"
    assert result.action == 1


def test_directional_conflict_just_above_tolerance_triggers() -> None:
    layer = make_layer(
        variance_threshold=10.0,
        directional_disagreement=True,
        directional_tolerance=1.0,
    )
    epsilon = math.nextafter(1.0, math.inf) - 1.0
    result = layer.screen(_a(1), mu=-(1.0 + 1e-6), sigma=0.1)
    assert result.action == 0
    assert result.reason == "directional_conflict"
    # And just above tolerance by one ULP also triggers
    layer2 = make_layer(
        variance_threshold=10.0,
        directional_disagreement=True,
        directional_tolerance=1.0,
    )
    result2 = layer2.screen(_a(1), mu=-(1.0 + epsilon), sigma=0.1)
    assert result2.reason == "directional_conflict"


def test_variance_threshold_at_boundary_is_low_sigma_path() -> None:
    # sigma == threshold → no budget consumption (strict inequality)
    layer = make_layer(variance_threshold=4.5, max_risk_long_units=5)
    layer.screen(_a(1), mu=0.0, sigma=4.5)
    assert layer.risk_long_used == 0


def test_variance_threshold_just_above_is_high_sigma_path() -> None:
    layer = make_layer(variance_threshold=4.5, max_risk_long_units=5)
    epsilon = math.nextafter(4.5, math.inf) - 4.5
    layer.screen(_a(1), mu=0.0, sigma=4.5 + epsilon)
    assert layer.risk_long_used == 1


# ---------------------------------------------------------------------------
# Per-UTC-day budget reset (timestamp_ns)
# ---------------------------------------------------------------------------

_NANOS_PER_DAY = 86_400_000_000_000


def test_budget_resets_on_utc_day_boundary() -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_long_units=2)
    day0 = 10 * _NANOS_PER_DAY
    # Exhaust the day-0 budget (2 units), then a third open is blocked.
    assert layer.screen(_a(1), 0.0, 5.0, timestamp_ns=day0).reason == "pass"
    assert layer.screen(_a(1), 0.0, 5.0, timestamp_ns=day0 + 1).reason == "pass"
    assert (
        layer.screen(_a(1), 0.0, 5.0, timestamp_ns=day0 + 2).reason
        == "budget_exhausted"
    )
    # Crossing into day 1 resets the budget -> opens allowed again.
    day1 = 11 * _NANOS_PER_DAY
    assert layer.screen(_a(1), 0.0, 5.0, timestamp_ns=day1).reason == "pass"
    assert layer.risk_long_used == 1


def test_budget_does_not_reset_within_same_day() -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_long_units=2)
    day0 = 10 * _NANOS_PER_DAY
    layer.screen(_a(1), 0.0, 5.0, timestamp_ns=day0)
    # Later the same UTC day (just before midnight) must not reset.
    later = day0 + _NANOS_PER_DAY - 1
    layer.screen(_a(1), 0.0, 5.0, timestamp_ns=later)
    assert layer.risk_long_used == 2
    assert (
        layer.screen(_a(1), 0.0, 5.0, timestamp_ns=later).reason
        == "budget_exhausted"
    )


def test_budget_never_resets_without_timestamp() -> None:
    # Legacy behaviour: omitting timestamp_ns keeps the lifetime cap.
    layer = make_layer(variance_threshold=1.0, max_risk_long_units=2)
    layer.screen(_a(1), 0.0, 5.0)
    layer.screen(_a(1), 0.0, 5.0)
    assert layer.screen(_a(1), 0.0, 5.0).reason == "budget_exhausted"
    assert layer.risk_long_used == 2
