"""Unit tests for the PF-screen profitability gate (T-1.2-06).

The gate keeps the screen active only while it has been net money-saving over a
trailing window of ``screen_profit_window_sessions`` sessions, measured by a
SHADOW next-bar counterfactual on the trades the screen would suppress. Below a
full window the screen defaults active.
"""

from __future__ import annotations

from dqnpf.tests.conftest import FakeActionResult, make_layer


def _a(idx: int) -> FakeActionResult:
    return FakeActionResult(action=idx, action_name=f"action_{idx}")


def _blocking_layer(window: int) -> object:
    return make_layer(
        variance_threshold=1.0,
        max_risk_long_units=0,
        max_risk_short_units=0,
        screen_profit_window_sessions=window,
    )


def test_screen_active_until_window_fills() -> None:
    layer = _blocking_layer(window=2)
    layer.begin_session()
    r = layer.screen(_a(1), mu=0.0, sigma=5.0, price=100.0)
    assert r.reason == "budget_exhausted" and r.gate_active is True


def test_blocking_a_winner_costs_the_screen_and_deactivates_it() -> None:
    layer = _blocking_layer(window=1)
    layer.begin_session()
    layer.screen(_a(1), mu=0.0, sigma=5.0, price=100.0)
    layer.screen(_a(0), mu=0.0, sigma=5.0, price=101.0)
    layer.begin_session()
    assert layer.gate_active is False
    r = layer.screen(_a(1), mu=0.0, sigma=5.0, price=100.0)
    assert r.reason == "gate_bypassed" and r.action == 1


def test_blocking_a_loser_keeps_the_screen_active() -> None:
    layer = _blocking_layer(window=1)
    layer.begin_session()
    layer.screen(_a(1), mu=0.0, sigma=5.0, price=100.0)
    layer.screen(_a(0), mu=0.0, sigma=5.0, price=99.0)
    layer.begin_session()
    assert layer.gate_active is True
    r = layer.screen(_a(1), mu=0.0, sigma=5.0, price=100.0)
    assert r.reason == "budget_exhausted" and r.action == 0


def test_counterfactual_uses_units_and_direction() -> None:
    layer = _blocking_layer(window=1)
    layer.begin_session()
    layer.screen(_a(4), mu=0.0, sigma=5.0, price=100.0)
    layer.screen(_a(0), mu=0.0, sigma=5.0, price=102.0)
    assert layer._session_cf == 4.0
    layer.begin_session()
    assert layer.gate_active is True


def test_shadow_measures_even_while_bypassed() -> None:
    layer = _blocking_layer(window=1)
    layer.begin_session()
    layer.screen(_a(1), mu=0.0, sigma=5.0, price=100.0)
    layer.screen(_a(0), mu=0.0, sigma=5.0, price=101.0)
    layer.begin_session()
    assert layer.gate_active is False
    layer.screen(_a(1), mu=0.0, sigma=5.0, price=100.0)
    layer.screen(_a(0), mu=0.0, sigma=5.0, price=98.0)
    assert layer._session_cf == 2.0
    layer.begin_session()
    assert layer.gate_active is True


def test_no_pending_mark_without_a_blocked_trade() -> None:
    layer = _blocking_layer(window=2)
    layer.begin_session()
    layer.screen(_a(0), mu=0.0, sigma=5.0, price=100.0)
    layer.screen(_a(0), mu=0.0, sigma=5.0, price=200.0)
    assert layer._session_cf == 0.0
