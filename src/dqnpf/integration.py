"""Integration layer: screens DQN actions using forecaster signals and risk budgets.

Sits between a DQN action recommendation and the modelenv ``Step()`` call.
Stateful only for cumulative risk-budget counters that track exposure
opened during high-uncertainty regimes (``sigma > variance_threshold``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from tradingmodel.intraday.dqnpf.action_mapper import (
    ACTION_NAMES,
    Direction,
    map_action,
)
from tradingmodel.intraday.dqnpf.config import IntegrationConfig
from tradingmodel.intraday.dqnpf.forecaster_bridge import ForecasterBridge
from tradingmodel.intraday.dqnpf.signal_cache import SignalCache

logger = logging.getLogger(__name__)

_HOLD_ACTION = 0
_HOLD_NAME = "HOLD"
_REASON_PASS = "pass"
_REASON_BUDGET = "budget_exhausted"
_REASON_DIRECTIONAL = "directional_conflict"


@dataclass
class ScreenedAction:
    """Result of screening a DQN action.

    Attributes:
        action: Final action index (0-4), potentially overridden to HOLD.
        action_name: Human-readable action name.
        screened: True if a risk rule modified the action.
        reason: One of ``"pass"``, ``"budget_exhausted"``, ``"directional_conflict"``.
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

        logger.info(
            "IntegrationLayer initialised: symbol=%s, variance_threshold=%.4f, "
            "max_risk_long=%d, max_risk_short=%d, directional_disagreement=%s, "
            "directional_tolerance=%.4f, forecast_horizon=%d, "
            "min_bars_warmup=%d, step_size_seconds=%d",
            self._symbol,
            config.variance_threshold,
            config.max_risk_long_units,
            config.max_risk_short_units,
            config.directional_disagreement,
            config.directional_tolerance,
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

    def screen(
        self,
        dqn_action: Any,
        mu: float,
        sigma: float,
        timestamp_ns: int | None = None,
    ) -> ScreenedAction:
        """Apply risk rules in priority order: budget, directional, pass-through.

        Args:
            dqn_action: Object exposing ``action`` (int 0-4) and ``action_name``.
            mu: Forecaster mean forward return (bps).
            sigma: Forecaster sigma (bps).
            timestamp_ns: Decision time in UTC nanoseconds. When provided, the
                risk budget resets at each UTC day boundary. When ``None`` the
                budget is never reset (legacy lifetime behaviour).

        Returns:
            ScreenedAction with the final action and reason.
        """
        self._maybe_reset_budget(timestamp_ns)
        unit = map_action(dqn_action.action)
        high_sigma = sigma > self._config.variance_threshold

        # Rule 1: Risk budget (high-sigma only)
        if high_sigma:
            if unit.direction == Direction.LONG:
                if (
                    self._risk_long_units + unit.risk_units
                    > self._config.max_risk_long_units
                ):
                    return self._hold(_REASON_BUDGET, sigma)
            elif unit.direction == Direction.SHORT:
                if (
                    self._risk_short_units + unit.risk_units
                    > self._config.max_risk_short_units
                ):
                    return self._hold(_REASON_BUDGET, sigma)

        # Rule 2: Directional conflict
        if (
            self._config.directional_disagreement
            and unit.direction != Direction.NONE
            and abs(mu) > self._config.directional_tolerance
        ):
            mu_direction = Direction.LONG if mu > 0 else Direction.SHORT
            if mu_direction != unit.direction:
                return self._hold(_REASON_DIRECTIONAL, sigma)

        # Rule 3: Pass-through (consume budget only on high-sigma opens)
        if high_sigma:
            if unit.direction == Direction.LONG:
                self._risk_long_units += unit.risk_units
            elif unit.direction == Direction.SHORT:
                self._risk_short_units += unit.risk_units

        action_name = getattr(dqn_action, "action_name", ACTION_NAMES[dqn_action.action])
        result = ScreenedAction(
            action=dqn_action.action,
            action_name=action_name,
            screened=False,
            reason=_REASON_PASS,
            sigma=sigma,
            risk_long_used=self._risk_long_units,
            risk_short_used=self._risk_short_units,
        )
        logger.debug(
            "screen pass: symbol=%s action=%s mu=%.4f sigma=%.4f "
            "risk_long=%d risk_short=%d",
            self._symbol,
            action_name,
            mu,
            sigma,
            self._risk_long_units,
            self._risk_short_units,
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
