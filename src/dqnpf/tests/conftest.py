"""Shared pytest fixtures and helpers for the dqnpf integration layer tests."""

from __future__ import annotations

from dataclasses import dataclass

from tradingmodel.intraday.dqnpf.config import IntegrationConfig
from tradingmodel.intraday.dqnpf.integration import IntegrationLayer


@dataclass
class FakeActionResult:
    """Minimal stand-in for deepqnetwork.advisor.ActionResult.

    The integration layer only reads ``action`` and ``action_name``.
    """

    action: int
    action_name: str = ""


def make_layer(
    *,
    symbol: str = "USDJPY",
    variance_threshold: float = 4.5,
    max_risk_long_units: int = 2,
    max_risk_short_units: int = 1,
    directional_disagreement: bool = False,
    directional_tolerance: float = 1.0,
    screen_profit_gate_enabled: bool = False,
    screen_profit_window_sessions: int = 7,
) -> IntegrationLayer:
    """Build an IntegrationLayer with stubbed collaborators for screen()-only tests."""
    config = IntegrationConfig(
        symbol=symbol,
        variance_threshold=variance_threshold,
        max_risk_long_units=max_risk_long_units,
        max_risk_short_units=max_risk_short_units,
        directional_disagreement=directional_disagreement,
        directional_tolerance=directional_tolerance,
        screen_profit_gate_enabled=screen_profit_gate_enabled,
        screen_profit_window_sessions=screen_profit_window_sessions,
    )
    return IntegrationLayer(
        dqn=None,  # type: ignore[arg-type], unused by screen()
        forecaster_bridge=None,  # type: ignore[arg-type]
        signal_cache=None,  # type: ignore[arg-type]
        config=config,
    )
