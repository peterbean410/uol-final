"""Forecaster signal cache.

Caches the most recent (mu, sigma) prediction keyed by the latest completed
M5 bar timestamp. At a 60s decision interval, four out of five steps land
within the same M5 bar and reuse the cached signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class CachedSignal:
    mu: float
    sigma: float
    bar_timestamp: int


class SignalCache:
    """Cache forecaster predictions keyed by the latest M5 bar timestamp."""

    def __init__(self) -> None:
        self._cache: CachedSignal | None = None

    def get_or_compute(
        self,
        latest_bar_ts: int,
        compute_fn: Callable[[], tuple[float, float]],
    ) -> tuple[float, float]:
        """Return cached (mu, sigma) if the bar timestamp matches, else recompute.

        Args:
            latest_bar_ts: Timestamp of the latest completed M5 bar.
            compute_fn: Zero-argument callable that produces (mu, sigma).

        Returns:
            (mu, sigma) tuple.
        """
        if self._cache is not None and self._cache.bar_timestamp == latest_bar_ts:
            logger.debug("signal cache hit at ts=%d", latest_bar_ts)
            return self._cache.mu, self._cache.sigma

        logger.debug("signal cache miss at ts=%d", latest_bar_ts)
        mu, sigma = compute_fn()
        self._cache = CachedSignal(mu=mu, sigma=sigma, bar_timestamp=latest_bar_ts)
        return mu, sigma

    def invalidate(self) -> None:
        """Clear the cache. Called on Reset()."""
        self._cache = None
