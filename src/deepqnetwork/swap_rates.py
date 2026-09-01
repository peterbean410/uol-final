"""Swap-rate resolution mirroring the modelenv Rust environment.

modelenv applies overnight financing (swap) in Training/backtest mode by seeding
a built-in per-symbol default table inside ``Environment::new``
(``modelenv/core/src/environment.rs::default_daily_swap_rates``). The DQN
training and backtest components launch modelenv as a sidecar **without** swap
CLI flags, so the active rates come from that default table unless overridden via
the environment variables modelenv also reads
(``MODELENV_SWAP_RATE_LONG`` / ``MODELENV_SWAP_RATE_SHORT``).

This module reproduces that resolution so the pipeline reports can record the
*actual* (long, short) daily swap rates a run used. Keep it in sync with
``modelenv/core/src/environment.rs`` and ``modelenv/core/src/config.rs``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_DAILY_SWAP_RATES: dict[str, tuple[float, float]] = {
    "USDJPY": (0.1121, -0.2424),
}

SWAP_RATE_UNITS = "price units/day per unit volume"


def default_swap_rate_for(symbol: str) -> tuple[float, float]:
    """Built-in Training-mode (long, short) default for ``symbol``, else (0, 0)."""
    return DEFAULT_DAILY_SWAP_RATES.get(symbol, (0.0, 0.0))


def _non_empty(env: Mapping[str, str], key: str) -> str | None:
    """modelenv's non_empty_env: trimmed value, or None if absent/blank."""
    value = env.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def resolve_swap_rates(
    symbol: str, env: Mapping[str, str] | None = None
) -> dict[str, object]:
    """Resolve the (long, short) daily swap rates a Training/backtest run uses.

    Reproduces ``server/src/main.rs`` precedence for Training mode (the mode the
    DQN training and backtest components run modelenv in): per-side
    ``--swap-rate-*`` overrides filled from the default table, then the built-in
    default table. Only the environment-variable forms of those flags are read
    here, since the components launch modelenv without swap CLI flags.

    Returns a report-ready dict: ``long_per_day``, ``short_per_day``, ``source``
    (``"override"`` | ``"default_table"``), ``symbol``, ``units``.
    """
    env = os.environ if env is None else env
    default_long, default_short = default_swap_rate_for(symbol)

    long_raw = _non_empty(env, "MODELENV_SWAP_RATE_LONG")
    short_raw = _non_empty(env, "MODELENV_SWAP_RATE_SHORT")
    long_override = _parse_float(long_raw)
    short_override = _parse_float(short_raw)
    if long_override is not None or short_override is not None:
        long_rate = default_long if long_override is None else long_override
        short_rate = default_short if short_override is None else short_override
        source = "override"
    else:
        long_rate, short_rate, source = default_long, default_short, "default_table"

    return {
        "symbol": symbol,
        "long_per_day": long_rate,
        "short_per_day": short_rate,
        "source": source,
        "units": SWAP_RATE_UNITS,
    }


def _parse_float(value: str | None) -> float | None:
    """Parse a float like modelenv does; silently ignore unparseable values."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
