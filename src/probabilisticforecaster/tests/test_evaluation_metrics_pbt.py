"""Property-based tests for Evaluation Metric Bounds.

Verifies that for any valid predictions (mu, sigma) and targets:
1. NLL (Gaussian Negative Log-Likelihood) is finite and positive
2. Directional Accuracy is in [0, 1]
3. 95% Coverage Ratio is in [0, 1]
4. RMSE is non-negative

Tests the metric computation functions directly without S3 access.

**Validates: Requirements 1.5**
"""

import math

import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from probabilisticforecaster.evaluation import (
    _compute_nll,
    _compute_directional_accuracy,
    _compute_covered_ratio_95,
    _compute_rmse,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def evaluation_inputs(draw):
    """Generate random (mu, sigma, actual) numpy arrays for metric bound testing.

    mu: any finite float (reasonable range to avoid numerical overflow in NLL)
    sigma: positive finite float (strictly > 0, required for NLL computation)
    actual: any finite float (reasonable range)

    Array size varies from 1 to 100.
    """
    n = draw(st.integers(min_value=1, max_value=100))

    mu_values = draw(
        st.lists(
            st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
        )
    )

    sigma_values = draw(
        st.lists(
            st.floats(
                min_value=1e-6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
        )
    )

    actual_values = draw(
        st.lists(
            st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
        )
    )

    mu = np.array(mu_values, dtype=np.float64)
    sigma = np.array(sigma_values, dtype=np.float64)
    actual = np.array(actual_values, dtype=np.float64)

    return mu, sigma, actual


# ---------------------------------------------------------------------------
# Property 2: Evaluation metric bounds
# ---------------------------------------------------------------------------


class TestEvaluationMetricBounds:
    """Property 2: Evaluation metric bounds.

    For any predictions and targets:
    - NLL is finite and positive
    - Directional Accuracy is in [0, 1]
    - 95% Coverage Ratio is in [0, 1]
    - RMSE is non-negative

    **Validates: Requirements 1.5**
    """

    @given(inputs=evaluation_inputs())
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_nll_is_finite(self, inputs):
        """For any valid mu, positive sigma, and actual values,
        NLL is finite (not NaN, not infinity).

        The Gaussian NLL formula is:
        NLL = mean(0.5 * (log(sigma^2) + ((actual - mu) / sigma)^2 + log(2*pi)))

        NLL is always finite when sigma > 0 and all inputs are finite.
        Note: NLL can be negative when sigma is small and predictions are
        accurate (the Gaussian density can exceed 1), so we verify finiteness
        rather than strict positivity. The lower bound is
        0.5 * (log(2*pi) + log(sigma_min^2)) which is finite for sigma > 0.

        **Validates: Requirements 1.5**
        """
        mu, sigma, actual = inputs

        nll = _compute_nll(mu, sigma, actual)

        assert math.isfinite(nll), (
            f"NLL should be finite, got {nll}. "
            f"mu range: [{mu.min()}, {mu.max()}], "
            f"sigma range: [{sigma.min()}, {sigma.max()}], "
            f"actual range: [{actual.min()}, {actual.max()}]"
        )
        # NLL has a lower bound determined by the minimum possible value
        # of the Gaussian log-likelihood. For any finite inputs with sigma > 0,
        # NLL is bounded below by 0.5 * log(2*pi*sigma_min^2) which is finite.
        # It is NOT necessarily positive (can be negative for small sigma
        # when predictions are very accurate), but it is always > -inf.
        assert nll > float("-inf"), "NLL should be bounded below"

    @given(inputs=evaluation_inputs())
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_directional_accuracy_in_unit_interval(self, inputs):
        """For any mu and actual arrays, Directional Accuracy is in [0, 1].

        DA is a proportion (count of matching signs / total count),
        so it must always be between 0 and 1 inclusive.

        **Validates: Requirements 1.5**
        """
        mu, sigma, actual = inputs

        da = _compute_directional_accuracy(mu, actual)

        assert 0.0 <= da <= 1.0, (
            f"Directional Accuracy should be in [0, 1], got {da}. "
            f"mu range: [{mu.min()}, {mu.max()}], "
            f"actual range: [{actual.min()}, {actual.max()}]"
        )

    @given(inputs=evaluation_inputs())
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_coverage_ratio_95_in_unit_interval(self, inputs):
        """For any mu, positive sigma, and actual arrays,
        95% Coverage Ratio is in [0, 1].

        CR95 is a proportion (count of actuals within 2*sigma of mu / total),
        so it must always be between 0 and 1 inclusive.

        **Validates: Requirements 1.5**
        """
        mu, sigma, actual = inputs

        cr95 = _compute_covered_ratio_95(mu, sigma, actual)

        assert 0.0 <= cr95 <= 1.0, (
            f"Coverage Ratio 95% should be in [0, 1], got {cr95}. "
            f"mu range: [{mu.min()}, {mu.max()}], "
            f"sigma range: [{sigma.min()}, {sigma.max()}], "
            f"actual range: [{actual.min()}, {actual.max()}]"
        )

    @given(inputs=evaluation_inputs())
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rmse_is_non_negative(self, inputs):
        """For any mu and actual arrays, RMSE is non-negative.

        RMSE = sqrt(mean((mu - actual)^2)) which is always >= 0
        since it's the square root of a mean of squared values.

        **Validates: Requirements 1.5**
        """
        mu, sigma, actual = inputs

        rmse = _compute_rmse(mu, actual)

        assert rmse >= 0.0, (
            f"RMSE should be non-negative, got {rmse}. "
            f"mu range: [{mu.min()}, {mu.max()}], "
            f"actual range: [{actual.min()}, {actual.max()}]"
        )
        assert math.isfinite(rmse), (
            f"RMSE should be finite, got {rmse}. "
            f"mu range: [{mu.min()}, {mu.max()}], "
            f"actual range: [{actual.min()}, {actual.max()}]"
        )
