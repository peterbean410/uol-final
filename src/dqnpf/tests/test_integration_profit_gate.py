"""Unit tests for the PF-screen profitability gate (T-1.2-06).

The gate keeps the screen active only while it has been net money-saving over a
trailing window of ``screen_profit_window_sessions`` sessions, measured by a
SHADOW next-bar counterfactual on the trades the screen would suppress. Below a
full window the screen defaults active.
"""

from __future__ import annotations

from tradingmodel.intraday.dqnpf.tests.conftest import FakeActionResult, make_layer


def _a(idx: int) -> FakeActionResult:
    return FakeActionResult(action=idx, action_name=f"action_{idx}")


# --- legacy parity: gate disabled => identical to pre-gate behaviour ----------


def test_gate_disabled_preserves_legacy_suppression() -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_long_units=0)
    # high-sigma long with zero budget -> suppressed (budget_exhausted), as before
    r = layer.screen(_a(1), mu=0.0, sigma=5.0)
    assert r.reason == "budget_exhausted"
    assert r.action == 0  # HOLD
    assert r.gate_active is True  # always-active sentinel when gate off


def test_gate_disabled_ignores_price_arg() -> None:
    layer = make_layer(variance_threshold=1.0, max_risk_long_units=5)
    r = layer.screen(_a(1), mu=0.0, sigma=5.0, price=100.0)
    assert r.reason == "pass" and r.action == 1
    assert layer.risk_long_used == 1  # budget still consumed on pass


# --- gate enabled: activation / deactivation by trailing counterfactual --------


def _blocking_layer(window: int) -> object:
    # zero budget => every high-sigma open is suppressed by the screen
    return make_layer(
        variance_threshold=1.0,
        max_risk_long_units=0,
        max_risk_short_units=0,
        screen_profit_gate_enabled=True,
        screen_profit_window_sessions=window,
    )


def test_screen_active_until_window_fills() -> None:
    layer = _blocking_layer(window=2)
    layer.begin_session()
    r = layer.screen(_a(1), mu=0.0, sigma=5.0, price=100.0)
    # history empty (< window) -> screen defaults ACTIVE -> still suppresses
    assert r.reason == "budget_exhausted" and r.gate_active is True


def test_blocking_a_winner_costs_the_screen_and_deactivates_it() -> None:
    layer = _blocking_layer(window=1)
    layer.begin_session()  # session 1
    # block a long at 100; next bar price rises to 101 => blocked a WINNER
    layer.screen(_a(1), mu=0.0, sigma=5.0, price=100.0)
    layer.screen(_a(0), mu=0.0, sigma=5.0, price=101.0)  # marks pending
    layer.begin_session()  # roll session-1 cf (= -1) into history (full window=1)
    assert layer.gate_active is False  # sum(history) = -1 <= 0 -> screen OFF
    # now the same blocking situation passes through unscreened
    r = layer.screen(_a(1), mu=0.0, sigma=5.0, price=100.0)
    assert r.reason == "gate_bypassed" and r.action == 1


def test_blocking_a_loser_keeps_the_screen_active() -> None:
    layer = _blocking_layer(window=1)
    layer.begin_session()  # session 1
    # block a long at 100; next bar price FALLS to 99 => blocked a LOSER (screen saved money)
    layer.screen(_a(1), mu=0.0, sigma=5.0, price=100.0)
    layer.screen(_a(0), mu=0.0, sigma=5.0, price=99.0)
    layer.begin_session()  # roll session-1 cf (= +1) into history
    assert layer.gate_active is True  # sum(history) = +1 > 0 -> screen stays ON
    r = layer.screen(_a(1), mu=0.0, sigma=5.0, price=100.0)
    assert r.reason == "budget_exhausted" and r.action == 0


def test_counterfactual_uses_units_and_direction() -> None:
    # SELL-2 blocked at 100; price rises to 102 -> short would have LOST 2*2=4,
    # so the screen SAVED +4 (cf positive).
    layer = _blocking_layer(window=1)
    layer.begin_session()
    layer.screen(_a(4), mu=0.0, sigma=5.0, price=100.0)  # SELL2 blocked
    layer.screen(_a(0), mu=0.0, sigma=5.0, price=102.0)  # mark
    assert layer._session_cf == 4.0
    layer.begin_session()
    assert layer.gate_active is True


def test_shadow_measures_even_while_bypassed() -> None:
    # Once deactivated, the gate must KEEP measuring so it can switch back on.
    layer = _blocking_layer(window=1)
    layer.begin_session()
    layer.screen(_a(1), mu=0.0, sigma=5.0, price=100.0)  # block long
    layer.screen(_a(0), mu=0.0, sigma=5.0, price=101.0)  # winner -> cf -1
    layer.begin_session()
    assert layer.gate_active is False  # bypassed now
    # bypassed session: the screen would block a LOSER (price falls) -> cf +1
    layer.screen(_a(1), mu=0.0, sigma=5.0, price=100.0)  # bypassed, but shadow records
    layer.screen(_a(0), mu=0.0, sigma=5.0, price=98.0)
    assert layer._session_cf == 2.0  # +2 saved this session
    layer.begin_session()  # history=[+2] -> screen reactivates
    assert layer.gate_active is True


def test_no_pending_mark_without_a_blocked_trade() -> None:
    layer = _blocking_layer(window=2)
    layer.begin_session()
    # a HOLD action is never "blocked", so no counterfactual is registered
    layer.screen(_a(0), mu=0.0, sigma=5.0, price=100.0)
    layer.screen(_a(0), mu=0.0, sigma=5.0, price=200.0)
    assert layer._session_cf == 0.0
