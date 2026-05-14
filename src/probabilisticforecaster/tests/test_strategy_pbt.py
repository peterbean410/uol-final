"""Property-based tests for the Trading Strategy module (strategy.py).

Uses Hypothesis to verify correctness of trading strategy position sizing
across randomly generated mu and sigma values.
"""

import math

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.strategy import DirectionalStrategy, MeanVarianceStrategy


# ---------------------------------------------------------------------------
# Property 10: Directional Strategy Sign Correctness
# ---------------------------------------------------------------------------


class TestDirectionalStrategySignCorrectness:
    """Property 10: Directional Strategy Sign Correctness.

    For any prediction μ̂ ≠ 0, the directional strategy position SHALL have
    the same sign as μ̂ and absolute value equal to the configured position
    size (10m). For μ̂ = 0, the position should be 0.

    **Validates: Requirements 9.1**
    """

    @given(
        mu=st.floats(allow_nan=False, allow_infinity=False),
        sigma=st.floats(allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_nonzero_mu_position_sign_matches_mu_sign(self, mu: float, sigma: float):
        """For non-zero μ̂, position sign matches μ̂ sign and |position| == position_size.

        **Validates: Requirements 9.1**
        """
        assume(mu != 0.0)

        config = ForecasterConfig()
        strategy = DirectionalStrategy()
        position = strategy.compute_position(mu, sigma, config)

        # Sign of position must match sign of mu
        if mu > 0:
            assert position > 0, (
                f"Expected positive position for mu={mu}, got {position}"
            )
        else:
            assert position < 0, (
                f"Expected negative position for mu={mu}, got {position}"
            )

        # Absolute value must equal position_size
        assert abs(position) == config.position_size, (
            f"Expected |position| == {config.position_size}, "
            f"got |{position}| = {abs(position)}"
        )

    @given(
        sigma=st.floats(allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_zero_mu_returns_zero_position(self, sigma: float):
        """For μ̂ = 0, position should be exactly 0.

        **Validates: Requirements 9.1**
        """
        config = ForecasterConfig()
        strategy = DirectionalStrategy()
        position = strategy.compute_position(0.0, sigma, config)

        assert position == 0.0, (
            f"Expected position == 0 for mu=0, got {position}"
        )


# ---------------------------------------------------------------------------
# Property 11: Mean-Variance Position Formula Correctness
# ---------------------------------------------------------------------------


class TestMeanVariancePositionFormula:
    """Property 11: Mean-Variance Position Formula Correctness.

    For any prediction (μ̂, σ̂) where σ̂ > 0 and risk aversion γ > 0, the
    mean-variance strategy position SHALL equal
    `clip(μ̂ / (σ̂² × γ), -1, 1) × position_size`.

    Also tests the σ̂ = 0 fallback to directional strategy.

    **Validates: Requirements 9.2**
    """

    @given(
        mu=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        sigma=st.floats(min_value=1e-8, max_value=1e4, allow_nan=False, allow_infinity=False),
        gamma=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_position_equals_clipped_formula(self, mu: float, sigma: float, gamma: float):
        """Mean-variance position matches clip(μ̂ / (σ̂² × γ), -1, 1) × position_size.

        **Validates: Requirements 9.2**
        """
        assume(math.isfinite(mu))
        assume(math.isfinite(sigma))
        assume(math.isfinite(gamma))
        assume(sigma > 0)
        assume(gamma > 0)

        # Compute expected: pi_star = mu / (sigma^2 * gamma), clip to [-1, 1], scale
        denominator = sigma**2 * gamma
        assume(math.isfinite(denominator))
        assume(denominator > 0)

        pi_star = mu / denominator
        assume(math.isfinite(pi_star))

        pi_clipped = max(-1.0, min(1.0, pi_star))
        config = ForecasterConfig(risk_aversion=gamma)
        expected_position = pi_clipped * config.position_size

        # Compute actual using the strategy
        strategy = MeanVarianceStrategy(risk_aversion=gamma)
        actual_position = strategy.compute_position(mu, sigma, config)

        assert math.isclose(actual_position, expected_position, rel_tol=1e-9, abs_tol=1e-6), (
            f"Position mismatch. "
            f"mu={mu}, sigma={sigma}, gamma={gamma}, "
            f"pi_star={pi_star}, pi_clipped={pi_clipped}, "
            f"actual={actual_position}, expected={expected_position}"
        )

    @given(
        mu=st.floats(min_value=1e-10, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_sigma_zero_positive_mu_returns_positive_position_size(self, mu: float):
        """When σ̂ = 0 and μ̂ > 0, position = +position_size.

        **Validates: Requirements 9.2**
        """
        assume(math.isfinite(mu))
        assume(mu > 0)

        config = ForecasterConfig()
        strategy = MeanVarianceStrategy()
        actual = strategy.compute_position(mu, 0.0, config)

        assert actual == config.position_size, (
            f"Expected +position_size ({config.position_size}) for mu={mu}, sigma=0, "
            f"got {actual}"
        )

    @given(
        mu=st.floats(min_value=-1e6, max_value=-1e-10, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_sigma_zero_negative_mu_returns_negative_position_size(self, mu: float):
        """When σ̂ = 0 and μ̂ < 0, position = -position_size.

        **Validates: Requirements 9.2**
        """
        assume(math.isfinite(mu))
        assume(mu < 0)

        config = ForecasterConfig()
        strategy = MeanVarianceStrategy()
        actual = strategy.compute_position(mu, 0.0, config)

        assert actual == -config.position_size, (
            f"Expected -position_size ({-config.position_size}) for mu={mu}, sigma=0, "
            f"got {actual}"
        )

    def test_sigma_zero_zero_mu_returns_zero(self):
        """When σ̂ = 0 and μ̂ = 0, position = 0.

        **Validates: Requirements 9.2**
        """
        config = ForecasterConfig()
        strategy = MeanVarianceStrategy()
        actual = strategy.compute_position(0.0, 0.0, config)

        assert actual == 0.0, (
            f"Expected 0.0 for mu=0, sigma=0, got {actual}"
        )
