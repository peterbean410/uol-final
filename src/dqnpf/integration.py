"""Integration layer: screens DQN actions using forecaster signals and risk budgets.

Sits between a DQN action recommendation and the modelenv ``Step()`` call.
Stateful only for cumulative risk-budget counters that track exposure
opened during high-uncertainty regimes (``sigma > variance_threshold``).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

from dqnpf.action_mapper import (
    ACTION_NAMES,
    Direction,
    map_action,
)
from dqnpf.config import IntegrationConfig
from dqnpf.forecaster_bridge import ForecasterBridge
from dqnpf.signal_cache import SignalCache

logger = logging.getLogger(__name__)

_HOLD_ACTION = 0
_HOLD_NAME = "HOLD"
_REASON_PASS = "pass"
_REASON_BUDGET = "budget_exhausted"
# The profit gate decided the screen has NOT been earning its keep over the
# trailing window, so the DQN action passes through unscreened even though a
# screen rule WOULD have suppressed it.
_REASON_GATE_BYPASS = "gate_bypassed"


@dataclass
class _PendingBlock:
    """A DQN action the screen suppressed, awaiting next-bar mark-to-market.

    The counterfactual P&L of the blocked trade is computed one bar later in
    modelenv's money convention ``pnl = (close_next - entry) * volume``, with
    ``volume = units`` and ``sign`` from the action direction.
    """

    sign: float  # +1 long, -1 short
    units: int
    entry_price: float


@dataclass
class ScreenedAction:
    """Result of screening a DQN action.

    Attributes:
        action: Final action index (0-4), potentially overridden to HOLD.
        action_name: Human-readable action name.
        screened: True if a risk rule modified the action.
        reason: One of ``"pass"``, ``"budget_exhausted"``, ``"gate_bypassed"``.
        sigma: Sigma value at decision time, for downstream logging.
        risk_long_used: Cumulative long budget units after this action.
        risk_short_used: Cumulative short budget units after this action.
    """

    action: int
    action_name: str
    screened: bool
    reason: str
    sigma: float
    risk_long_used: int
    risk_short_used: int
    # Whether the profit gate currently has the screen ACTIVE. Always True when
    # the gate is disabled (legacy behaviour). Diagnostic only.
    gate_active: bool = True


class _DQNAdvisorLike(Protocol):
    def recommend_action(self, state: Any) -> Any: ...


class IntegrationLayer:
    """Screen DQN actions using forecaster signals and risk budgets.

    Bound to a single symbol from ``config`` for the instance's lifetime.

    Args:
        dqn: Loaded DQNAdvisor (or any object exposing ``recommend_action``).
        forecaster_bridge: ForecasterBridge for signal computation.
        signal_cache: SignalCache for caching forecaster predictions.
        config: IntegrationConfig with thresholds and budget caps.
    """

    def __init__(
        self,
        dqn: _DQNAdvisorLike,
        forecaster_bridge: ForecasterBridge,
        signal_cache: SignalCache,
        config: IntegrationConfig,
    ) -> None:
        self._dqn = dqn
        self._bridge = forecaster_bridge
        self._cache = signal_cache
        self._config = config
        self._symbol = config.symbol
        self._risk_long_units: int = 0
        self._risk_short_units: int = 0
        # UTC day index of the last screened step; the risk budget resets on
        # each day boundary so high-sigma throttling is per-day rather than a
        # one-shot lifetime cap (which would freeze trading after the first few
        # opens when sigma is persistently above variance_threshold).
        self._current_day: int | None = None

        # --- Profitability gate state (active only when enabled in config) ---
        # The screen's value is measured continuously in SHADOW: each bar, the
        # next-bar counterfactual P&L of any trade the screen WOULD suppress is
        # accrued (as money SAVED) into the current session, then rolled into a
        # trailing window of `screen_profit_window_sessions` sessions at each
        # `begin_session()`. The gate keeps the screen active only while that
        # trailing sum is positive; otherwise the DQN trades unscreened.
        self._gate_window: int = config.screen_profit_window_sessions
        self._cf_history: deque[float] = deque(maxlen=self._gate_window)
        self._session_cf: float = 0.0
        self._pending: _PendingBlock | None = None
        self._gate_active: bool = True
        self._session_started: bool = False

        logger.info(
            "IntegrationLayer initialised: symbol=%s, variance_threshold=%.4f, "
            "max_risk_long=%d, max_risk_short=%d, forecast_horizon=%d, "
            "min_bars_warmup=%d, step_size_seconds=%d",
            self._symbol,
            config.variance_threshold,
            config.max_risk_long_units,
            config.max_risk_short_units,
            config.forecast_horizon,
            config.min_bars_warmup,
            config.step_size_seconds,
        )

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def risk_long_used(self) -> int:
        return self._risk_long_units

    @property
    def risk_short_used(self) -> int:
        return self._risk_short_units

    _NANOS_PER_DAY = 86_400_000_000_000

    def _maybe_reset_budget(self, timestamp_ns: int | None) -> None:
        """Reset the risk budget when ``timestamp_ns`` crosses a UTC day."""
        if timestamp_ns is None:
            return
        day = timestamp_ns // self._NANOS_PER_DAY
        if day != self._current_day:
            if self._current_day is not None:
                logger.debug(
                    "risk budget reset on day boundary: symbol=%s day=%d "
                    "(was long=%d short=%d)",
                    self._symbol,
                    day,
                    self._risk_long_units,
                    self._risk_short_units,
                )
            self._current_day = day
            self._risk_long_units = 0
            self._risk_short_units = 0

    @property
    def gate_active(self) -> bool:
        """Whether the profit gate currently has the screen active.

        Always ``True`` when the gate is disabled in config.
        """
        return self._gate_active

    def begin_session(self) -> None:
        """Mark a session boundary for the profitability gate.

        Rolls the just-finished session's counterfactual P&L into the trailing
        window and recomputes whether the screen stays active for the new
        session. Call once at the start of each trading session/episode (and
        once per pre-seed/warm-up session so the gate is informed from bar 0).
        """
        if self._session_started:
            self._cf_history.append(self._session_cf)
        self._session_cf = 0.0
        self._pending = None
        self._session_started = True
        self._recompute_gate()

    def _recompute_gate(self) -> None:
        """Set ``_gate_active`` from the trailing-window counterfactual sum.

        Until the window has filled (fewer than ``screen_profit_window_sessions``
        completed sessions) the screen defaults to ACTIVE; the conservative
        choice, and the backtest pre-seeds the window so the gate is fully
        informed from the first scored session.
        """
        if len(self._cf_history) < self._gate_window:
            self._gate_active = True
        else:
            self._gate_active = sum(self._cf_history) > 0.0

    def _mark_pending_counterfactual(self, price: float | None) -> None:
        """Mark the previously-blocked trade to ``price`` (its next bar).

        Adds the money the screen SAVED, i.e. minus the blocked trade's
        realised next-bar P&L, to the current session's counterfactual.
        """
        p = self._pending
        if p is None or price is None or p.entry_price <= 0.0:
            return
        blocked_pnl = p.sign * p.units * (price - p.entry_price)
        self._session_cf += -blocked_pnl
        self._pending = None

    def _evaluate_rules(self, unit: Any, mu: float, sigma: float) -> str | None:
        """Run the screen rules; return a hold-reason or ``None`` (pass).

        Mutates the risk budget exactly as the legacy ``screen`` did (consumes
        on high-sigma pass-through opens). This is the SHADOW evaluation; it
        always runs as if the screen were active, so the counterfactual ledger
        measures the screen's true hypothetical value independent of whether the
        gate is currently honouring it.
        """
        high_sigma = sigma > self._config.variance_threshold

        # Rule 1: Risk budget (high-sigma only)
        if high_sigma:
            if unit.direction == Direction.LONG:
                if (
                    self._risk_long_units + unit.risk_units
                    > self._config.max_risk_long_units
                ):
                    return _REASON_BUDGET
            elif unit.direction == Direction.SHORT:
                if (
                    self._risk_short_units + unit.risk_units
                    > self._config.max_risk_short_units
                ):
                    return _REASON_BUDGET

        # Rule 2: Pass-through (consume budget only on high-sigma opens)
        if high_sigma:
            if unit.direction == Direction.LONG:
                self._risk_long_units += unit.risk_units
            elif unit.direction == Direction.SHORT:
                self._risk_short_units += unit.risk_units
        return None



    def screen(
        self,
        dqn_action: Any,
        mu: float,
        sigma: float,
        timestamp_ns: int | None = None,
        price: float | None = None,
    ) -> ScreenedAction:
        """Apply risk rules in priority order: budget, directional, pass-through.

        Args:
            dqn_action: Object exposing ``action`` (int 0-4) and ``action_name``.
            mu: Forecaster mean forward return (bps).
            sigma: Forecaster sigma (bps).
            timestamp_ns: Decision time in UTC nanoseconds. When provided, the
                risk budget resets at each UTC day boundary. When ``None`` the
                budget is never reset (legacy lifetime behaviour).
            price: Latest M5 close at decision time, in quote-currency price
                units. Required for the profitability gate's next-bar
                counterfactual; ignored when the gate is disabled.

        Returns:
            ScreenedAction with the final action and reason.
        """
        self._maybe_reset_budget(timestamp_ns)
        unit = map_action(dqn_action.action)

        # Gate: settle the previous bar's blocked trade against this bar's price
        # BEFORE evaluating the new bar (next-bar mark-to-market).
        self._mark_pending_counterfactual(price)

        # Shadow evaluation of the screen (always run; mutates the hypothetical
        # budget), gives the would-be hold decision regardless of the gate.
        hold_reason = self._evaluate_rules(unit, mu, sigma)

        # Gate: register the trade the screen would block for next-bar marking.
        if (
            hold_reason is not None
            and price is not None
            and unit.direction != Direction.NONE
        ):
            self._pending = _PendingBlock(
                sign=1.0 if unit.direction == Direction.LONG else -1.0,
                units=unit.risk_units,
                entry_price=price,
            )

        # Honour the suppression only while the gate says the screen is
        # currently profitable; otherwise pass the DQN action through.
        if hold_reason is not None and self._gate_active:
            held = self._hold(hold_reason, sigma)
            held.gate_active = self._gate_active
            return held
        if hold_reason is not None:
            return self._passthrough(dqn_action, mu, sigma, _REASON_GATE_BYPASS)
        return self._passthrough(dqn_action, mu, sigma, _REASON_PASS)

    def _passthrough(
        self, dqn_action: Any, mu: float, sigma: float, reason: str
    ) -> ScreenedAction:
        action_name = getattr(dqn_action, "action_name", ACTION_NAMES[dqn_action.action])
        result = ScreenedAction(
            action=dqn_action.action,
            action_name=action_name,
            screened=False,
            reason=reason,
            sigma=sigma,
            risk_long_used=self._risk_long_units,
            risk_short_used=self._risk_short_units,
            gate_active=self._gate_active,
        )
        logger.debug(
            "screen %s: symbol=%s action=%s mu=%.4f sigma=%.4f "
            "risk_long=%d risk_short=%d gate_active=%s",
            reason,
            self._symbol,
            action_name,
            mu,
            sigma,
            self._risk_long_units,
            self._risk_short_units,
            self._gate_active,
        )
        return result

    def on_position_closed(self, side: str, units: int) -> None:
        """Release ``units`` from the appropriate budget counter (clamped at zero).

        Args:
            side: "buy" decrements the long counter; "sell" the short counter.
            units: Number of risk units to release. Excess is clamped to zero.
        """
        if side == "buy":
            self._risk_long_units = max(0, self._risk_long_units - units)
        elif side == "sell":
            self._risk_short_units = max(0, self._risk_short_units - units)
        else:
            logger.warning(
                "on_position_closed: unknown side=%r (units=%d), ignoring",
                side,
                units,
            )

    def _hold(self, reason: str, sigma: float) -> ScreenedAction:
        if reason == _REASON_BUDGET:
            logger.info(
                "budget exhausted: symbol=%s sigma=%.4f risk_long=%d risk_short=%d",
                self._symbol,
                sigma,
                self._risk_long_units,
                self._risk_short_units,
            )
        return ScreenedAction(
            action=_HOLD_ACTION,
            action_name=_HOLD_NAME,
            screened=True,
            reason=reason,
            sigma=sigma,
            risk_long_used=self._risk_long_units,
            risk_short_used=self._risk_short_units,
        )
