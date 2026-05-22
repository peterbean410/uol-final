"""Unit tests for SignalCache."""

from __future__ import annotations

from tradingmodel.intraday.dqnpf.signal_cache import CachedSignal, SignalCache


def _const(mu: float, sigma: float):
    calls = {"n": 0}

    def fn() -> tuple[float, float]:
        calls["n"] += 1
        return mu, sigma

    return fn, calls


def test_empty_cache_invokes_compute_fn() -> None:
    cache = SignalCache()
    fn, calls = _const(0.5, 1.5)

    mu, sigma = cache.get_or_compute(100, fn)

    assert (mu, sigma) == (0.5, 1.5)
    assert calls["n"] == 1


def test_cache_hit_returns_cached_value() -> None:
    cache = SignalCache()
    fn, calls = _const(0.5, 1.5)
    cache.get_or_compute(100, fn)

    fn2, calls2 = _const(99.0, 99.0)
    mu, sigma = cache.get_or_compute(100, fn2)

    assert (mu, sigma) == (0.5, 1.5)
    assert calls2["n"] == 0
    assert calls["n"] == 1


def test_cache_miss_invokes_compute_fn() -> None:
    cache = SignalCache()
    fn1, calls1 = _const(0.5, 1.5)
    cache.get_or_compute(100, fn1)

    fn2, calls2 = _const(0.7, 1.7)
    mu, sigma = cache.get_or_compute(200, fn2)

    assert (mu, sigma) == (0.7, 1.7)
    assert calls2["n"] == 1


def test_invalidate_clears_cache() -> None:
    cache = SignalCache()
    fn1, calls1 = _const(0.5, 1.5)
    cache.get_or_compute(100, fn1)

    cache.invalidate()

    fn2, calls2 = _const(0.9, 2.0)
    mu, sigma = cache.get_or_compute(100, fn2)

    assert (mu, sigma) == (0.9, 2.0)
    assert calls2["n"] == 1


def test_cached_signal_fields() -> None:
    signal = CachedSignal(mu=0.1, sigma=0.2, bar_timestamp=42)
    assert signal.mu == 0.1
    assert signal.sigma == 0.2
    assert signal.bar_timestamp == 42
