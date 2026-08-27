"""Property-based tests for the integration layer.

Feature: dqnpf, Properties 1, 3, 4, 5, 6, 7, 8, 9, 13
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings, strategies as st

from tradingmodel.intraday.dqnpf.action_mapper import ACTION_MAP, Direction, map_action
from tradingmodel.intraday.dqnpf.tests.conftest import FakeActionResult, make_layer


_VALID_REASONS = {"pass", "budget_exhausted", "directional_conflict"}
_VALID_ACTIONS = {0, 1, 2, 3, 4}

_action_index = st.integers(min_value=0, max_value=4)
_finite_float = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
)
_positive_sigma = st.floats(
    min_value=1e-6, max_value=100.0, allow_nan=False, allow_infinity=False
)


def _action(idx: int) -> FakeActionResult:
    return FakeActionResult(action=idx, action_name=f"action_{idx}")


# ---------------------------------------------------------------------------
# Property 1: Screened action validity
# ---------------------------------------------------------------------------


@given(
    action_index=_action_index,
    mu=_finite_float,
    sigma=_positive_sigma,
    variance_threshold=st.floats(
        min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False
    ),
)
def test_screened_action_validity(
    action_index: int,
    mu: float,
    sigma: float,
    variance_threshold: float,
) -> None:
    layer = make_layer(
        variance_threshold=variance_threshold,
    )
    result = layer.screen(_action(action_index), mu, sigma)
    assert result.action in _VALID_ACTIONS
    assert result.reason in _VALID_REASONS
    assert (result.reason == "pass") == (not result.screened)


# ---------------------------------------------------------------------------
# Property 3: Low-sigma pass-through
# ---------------------------------------------------------------------------


@given(
    action_index=_action_index,
    mu=_finite_float,
    variance_threshold=st.floats(
        min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    sigma_offset=st.floats(
        min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
)
def test_low_sigma_pass_through(
    action_index: int,
    mu: float,
    variance_threshold: float,
    sigma_offset: float,
) -> None:
    # sigma <= threshold
    sigma = max(0.0, variance_threshold - sigma_offset)
    layer = make_layer(variance_threshold=variance_threshold)
    result = layer.screen(_action(action_index), mu, sigma)
    assert result.action == action_index
    assert result.reason == "pass"
    assert result.screened is False
    assert layer.risk_long_used == 0
    assert layer.risk_short_used == 0


# ---------------------------------------------------------------------------
# Property 4: High-sigma budget consumption
# ---------------------------------------------------------------------------


@given(
    action_index=st.sampled_from([1, 2, 3, 4]),
    mu=_finite_float,
    variance_threshold=st.floats(
        min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
    sigma_excess=st.floats(
        min_value=1e-3, max_value=20.0, allow_nan=False, allow_infinity=False
    ),
)
def test_high_sigma_budget_consumption(
    action_index: int,
    mu: float,
    variance_threshold: float,
    sigma_excess: float,
) -> None:
    layer = make_layer(
        variance_threshold=variance_threshold,
        max_risk_long_units=10,
        max_risk_short_units=10,
    )
    sigma = variance_threshold + sigma_excess
    unit = map_action(action_index)
    result = layer.screen(_action(action_index), mu, sigma)

    assert result.action == action_index
    assert result.reason == "pass"
    if unit.direction == Direction.LONG:
        assert layer.risk_long_used == unit.risk_units
        assert layer.risk_short_used == 0
    else:
        assert layer.risk_short_used == unit.risk_units
        assert layer.risk_long_used == 0


# ---------------------------------------------------------------------------
# Property 5: Budget exhaustion
# ---------------------------------------------------------------------------


@given(
    action_index=st.sampled_from([1, 2, 3, 4]),
    mu=_finite_float,
)
def test_budget_exhaustion_triggers_hold(action_index: int, mu: float) -> None:
    unit = map_action(action_index)
    # Pre-load each budget so a single high-sigma BUY/SELL_n exceeds the cap.
    layer = make_layer(
        variance_threshold=1.0,
        max_risk_long_units=unit.risk_units,
        max_risk_short_units=unit.risk_units,
    )
    # Force budget to be (cap - risk_units + 1) so action would overflow.
    if unit.direction == Direction.LONG:
        # Pre-fill long counter to (cap), so any LONG action overflows.
        layer._risk_long_units = unit.risk_units  # type: ignore[attr-defined]
    else:
        layer._risk_short_units = unit.risk_units  # type: ignore[attr-defined]

    sigma = 5.0  # > threshold
    result = layer.screen(_action(action_index), mu, sigma)
    assert result.action == 0
    assert result.action_name == "HOLD"
    assert result.reason == "budget_exhausted"
    assert result.screened is True


# ---------------------------------------------------------------------------
# Property 6: Budget never exceeded
# ---------------------------------------------------------------------------


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(
    actions=st.lists(_action_index, min_size=1, max_size=40),
    sigmas=st.lists(_positive_sigma, min_size=1, max_size=40),
    mu=_finite_float,
    max_long=st.integers(min_value=0, max_value=5),
    max_short=st.integers(min_value=0, max_value=5),
)
def test_budget_never_exceeded(
    actions: list[int],
    sigmas: list[float],
    mu: float,
    max_long: int,
    max_short: int,
) -> None:
    layer = make_layer(
        variance_threshold=0.5,
        max_risk_long_units=max_long,
        max_risk_short_units=max_short,
    )
    # Truncate to common length so we always have both inputs.
    n = min(len(actions), len(sigmas))
    for i in range(n):
        layer.screen(_action(actions[i]), mu, sigmas[i])
        assert layer.risk_long_used <= max_long
        assert layer.risk_short_used <= max_short


# ---------------------------------------------------------------------------
# Property 7: Budget release
# ---------------------------------------------------------------------------


@given(
    side=st.sampled_from(["buy", "sell"]),
    units_to_release=st.integers(min_value=0, max_value=10),
)
def test_budget_release(side: str, units_to_release: int) -> None:
    layer = make_layer(
        variance_threshold=1.0,
        max_risk_long_units=2,
        max_risk_short_units=2,
    )
    # Fill both budgets to the cap via high-sigma actions.
    layer.screen(_action(2), 0.0, 5.0)  # BUY_2 → long=2
    layer.screen(_action(4), 0.0, 5.0)  # SELL_2 → short=2

    # Previously blocked BUY_1 / SELL_1 should HOLD.
    blocked_long = layer.screen(_action(1), 0.0, 5.0)
    assert blocked_long.reason == "budget_exhausted"
    blocked_short = layer.screen(_action(3), 0.0, 5.0)
    assert blocked_short.reason == "budget_exhausted"

    layer.on_position_closed(side, units_to_release)

    if side == "buy":
        assert layer.risk_long_used == max(0, 2 - units_to_release)
        assert layer.risk_short_used == 2
    else:
        assert layer.risk_short_used == max(0, 2 - units_to_release)
        assert layer.risk_long_used == 2

    # If we released at least 1 from the side, the corresponding 1-unit action
    # should now pass through.
    if units_to_release >= 1:
        action_idx = 1 if side == "buy" else 3
        result = layer.screen(_action(action_idx), 0.0, 5.0)
        assert result.reason == "pass"
        assert result.screened is False


@given(
    actions=st.lists(st.sampled_from([1, 2, 3, 4]), min_size=1, max_size=10),
)
def test_single_symbol_isolation(actions: list[int]) -> None:
    layer_a = make_layer(
        symbol="USDJPY",
        variance_threshold=0.5,
        max_risk_long_units=10,
        max_risk_short_units=10,
    )
    layer_b = make_layer(
        symbol="AUDJPY",
        variance_threshold=0.5,
        max_risk_long_units=10,
        max_risk_short_units=10,
    )
    for idx in actions:
        layer_a.screen(_action(idx), 0.0, 5.0)
    assert layer_b.risk_long_used == 0
    assert layer_b.risk_short_used == 0
    assert layer_a.symbol == "USDJPY"
    assert layer_b.symbol == "AUDJPY"
