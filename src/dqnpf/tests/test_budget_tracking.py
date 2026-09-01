"""Unit tests for budget tracking (Req 3 acceptance criteria)."""

from __future__ import annotations

import pytest

from dqnpf.tests.conftest import FakeActionResult, make_layer


@pytest.mark.parametrize(
    "action,expected_long,expected_short",
    [
        (1, 1, 0),
        (2, 2, 0),
        (3, 0, 1),
        (4, 0, 2),
    ],
)
def test_action_increments_only_correct_counter_on_high_sigma(
    action: int, expected_long: int, expected_short: int
) -> None:
    layer = make_layer(
        variance_threshold=1.0, max_risk_long_units=5, max_risk_short_units=5
    )
    layer.screen(FakeActionResult(action=action), mu=0.0, sigma=5.0)
    assert layer.risk_long_used == expected_long
    assert layer.risk_short_used == expected_short


@pytest.mark.parametrize("action", [1, 2, 3, 4])
def test_low_sigma_never_increments_budget(action: int) -> None:
    """Req 3.5: low-sigma path must not consume budget for any non-HOLD action."""
    layer = make_layer(
        variance_threshold=4.5,
        max_risk_long_units=10,
        max_risk_short_units=10,
    )
    layer.screen(FakeActionResult(action=action), mu=0.0, sigma=4.5)
    assert layer.risk_long_used == 0
    assert layer.risk_short_used == 0


@pytest.mark.parametrize("sigma", [0.1, 4.5, 9.9])
def test_hold_never_increments_budget(sigma: float) -> None:
    """Req 3.5/3.6: HOLD bypasses the budget regardless of sigma."""
    layer = make_layer(variance_threshold=4.5)
    layer.screen(FakeActionResult(action=0), mu=0.0, sigma=sigma)
    assert layer.risk_long_used == 0
    assert layer.risk_short_used == 0


def test_on_position_closed_buy_decrements_long_by_n() -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_long_units=5)
    layer.screen(FakeActionResult(action=2), mu=0.0, sigma=5.0)
    layer.on_position_closed("buy", 1)
    assert layer.risk_long_used == 1


def test_on_position_closed_sell_decrements_short_by_n() -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_short_units=5)
    layer.screen(FakeActionResult(action=4), mu=0.0, sigma=5.0)
    layer.on_position_closed("sell", 2)
    assert layer.risk_short_used == 0


@pytest.mark.parametrize("side,initial_action", [("buy", 1), ("sell", 3)])
def test_on_position_closed_clamps_to_zero(side: str, initial_action: int) -> None:
    layer = make_layer(
        variance_threshold=1.0, max_risk_long_units=5, max_risk_short_units=5
    )
    layer.screen(FakeActionResult(action=initial_action), mu=0.0, sigma=5.0)
    layer.on_position_closed(side, 999)
    assert layer.risk_long_used == 0
    assert layer.risk_short_used == 0


@pytest.mark.parametrize("bad_side", ["", "buy_1", "long", "BUY", "sell ", " buy"])
def test_on_position_closed_unknown_side_is_noop(
    bad_side: str, caplog: pytest.LogCaptureFixture
) -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_long_units=5)
    layer.screen(FakeActionResult(action=1), mu=0.0, sigma=5.0)
    with caplog.at_level("WARNING"):
        layer.on_position_closed(bad_side, 1)
    assert layer.risk_long_used == 1
    assert any("unknown side" in record.message for record in caplog.records)


def test_budget_release_unblocks_previously_blocked_action() -> None:
    """Req 3.7 + Property 7: release decrements; a blocked action then passes."""
    layer = make_layer(variance_threshold=1.0, max_risk_long_units=2)
    layer.screen(FakeActionResult(action=2), mu=0.0, sigma=5.0)
    blocked = layer.screen(FakeActionResult(action=1), mu=0.0, sigma=5.0)
    assert blocked.reason == "budget_exhausted"

    layer.on_position_closed("buy", 1)
    after = layer.screen(FakeActionResult(action=1), mu=0.0, sigma=5.0)
    assert after.reason == "pass"
    assert after.action == 1
