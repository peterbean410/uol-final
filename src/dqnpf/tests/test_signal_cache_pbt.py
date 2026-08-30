"""Property-based tests for the signal cache.

Feature: dqnpf, Property 10 (Signal cache hit
Feature: dqnpf, Property 11) Signal cache miss
"""

from __future__ import annotations

from hypothesis import given, strategies as st

from dqnpf.signal_cache import SignalCache


_FINITE_FLOAT = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)


class _Counter:
    """Compute_fn that records invocation count and yields a fixed result."""

    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        self.sigma = sigma
        self.calls = 0

    def __call__(self) -> tuple[float, float]:
        self.calls += 1
        return self.mu, self.sigma


@given(
    latest_bar_ts=st.integers(min_value=0, max_value=2**63 - 1),
    repeat_count=st.integers(min_value=1, max_value=10),
    mu=_FINITE_FLOAT,
    sigma=_FINITE_FLOAT,
)
def test_cache_hit_compute_fn_called_at_most_once(
    latest_bar_ts: int, repeat_count: int, mu: float, sigma: float
) -> None:
    """Property 10: unchanged bar timestamp → compute_fn called at most once."""
    cache = SignalCache()
    compute = _Counter(mu, sigma)

    results = [cache.get_or_compute(latest_bar_ts, compute) for _ in range(repeat_count)]

    assert compute.calls == 1
    assert all(r == (mu, sigma) for r in results)


@given(
    timestamps=st.lists(
        st.integers(min_value=0, max_value=2**63 - 1),
        min_size=2,
        max_size=10,
        unique=True,
    ),
    mu=_FINITE_FLOAT,
    sigma=_FINITE_FLOAT,
)
def test_cache_miss_invokes_compute_and_updates(
    timestamps: list[int], mu: float, sigma: float
) -> None:
    """Property 11: changed bar timestamp → compute_fn invoked, cache updated."""
    cache = SignalCache()

    expected_calls = 0
    for i, ts in enumerate(timestamps):
        # Each unique ts forces a recompute; use mu/sigma derived from ts so
        # we can assert the cache returns the correct value after update.
        unique_mu = mu + i
        unique_sigma = sigma + i
        compute = _Counter(unique_mu, unique_sigma)
        result = cache.get_or_compute(ts, compute)
        expected_calls += 1
        assert compute.calls == 1
        assert result == (unique_mu, unique_sigma)

        # Subsequent call with the same ts must NOT recompute.
        compute2 = _Counter(99.0, 99.0)
        result2 = cache.get_or_compute(ts, compute2)
        assert compute2.calls == 0
        assert result2 == (unique_mu, unique_sigma)
