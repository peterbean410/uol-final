"""Forecaster signal regimes for the integration-layer prototype.

The integration layer screens DQN actions on ``(mu, sigma)`` expressed in
**basis points** (the same unit convention as ``ForecasterBridge``, which scales
the forecaster's fractional outputs by 10,000). We generate two regimes so the
prototype can show BOTH sides of the honest story:

* ``informative``, sigma tracks EWMA realised volatility and mu tracks a short
  EWMA of returns. Sigma genuinely rises before turbulent bars, so the high-sigma
  screen has something real to react to. This is the *feasibility* case.

* ``collapsed``; a constant sigma (above the variance threshold) and a constant
  mu, reproducing the production forecaster's documented failure mode (near-flat
  sigma -> the screen degenerates into a global on/off switch). This is the
  *honest critique* case that the profitability gate must catch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BPS = 10_000.0  # fractional return -> basis points (matches ForecasterBridge)


@dataclass
class Signals:
    """Per-bar forecaster outputs, aligned 1:1 with the price frame."""

    mu_bps: np.ndarray
    sigma_bps: np.ndarray
    regime: str


def _ewma(x: np.ndarray, halflife: float) -> np.ndarray:
    return (
        pd.Series(x).ewm(halflife=halflife, adjust=False).mean().to_numpy()
    )


def _ewma_vol(x: np.ndarray, halflife: float) -> np.ndarray:
    s = pd.Series(x)
    mean = s.ewm(halflife=halflife, adjust=False).mean()
    var = (s - mean).pow(2).ewm(halflife=halflife, adjust=False).mean()
    return np.sqrt(var.to_numpy())


def one_bar_returns(close: np.ndarray) -> np.ndarray:
    """Fractional 1-bar returns r_t = (c_t - c_{t-1}) / c_{t-1}, r_0 = 0."""
    r = np.zeros_like(close)
    r[1:] = (close[1:] - close[:-1]) / close[:-1]
    return r


def informative_signals(
    close: np.ndarray, *, mu_halflife: float = 12.0, sigma_halflife: float = 24.0
) -> Signals:
    """Sigma = EWMA realised vol (bps); mu = EWMA mean return (bps)."""
    r = one_bar_returns(close)
    sigma = _ewma_vol(r, sigma_halflife) * BPS
    mu = _ewma(r, mu_halflife) * BPS
    # A small floor avoids a degenerate sigma=0 at the very first bars.
    sigma = np.maximum(sigma, 1e-3)
    return Signals(mu_bps=mu, sigma_bps=sigma, regime="informative")


def collapsed_signals(
    close: np.ndarray, *, sigma_const_bps: float = 5.0, mu_const_bps: float = -0.2
) -> Signals:
    """Constant sigma/mu; the production collapse (near-flat sigma).

    ``sigma_const_bps`` is set ABOVE the default variance_threshold (4.5) so that
    every directional open counts as high-sigma, exactly the global on/off
    pathology observed in the production OOS-2016 run.
    """
    n = len(close)
    return Signals(
        mu_bps=np.full(n, mu_const_bps),
        sigma_bps=np.full(n, sigma_const_bps),
        regime="collapsed",
    )
