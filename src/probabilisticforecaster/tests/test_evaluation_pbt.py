"""Property-based tests for the Evaluation module (evaluation.py).

Uses Hypothesis to verify correctness of Directional Accuracy, 95% Covered Ratio,
and RMSE formulas across randomly generated prediction triples (mu, sigma, actual).
"""

import math

import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from probabilisticforecaster.evaluation import (
    _compute_directional_accuracy,
    _compute_covered_ratio_95,
    _compute_rmse,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def prediction_triples(draw):
    """Generate random (mu, sigma, actual) numpy arrays for evaluation testing.

    mu and actual are drawn from [-10, 10] (excluding zero to avoid sign ambiguity),
    sigma from [0.01, 10] (strictly positive).
    Array size varies from 1 to 50.
    """
    n = draw(st.integers(min_value=1, max_value=50))

    mu_values = [
        draw(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ).filter(lambda x: abs(x) > 1e-9)
        )
        for _ in range(n)
    ]
    sigma_values = [
        draw(
            st.floats(
                min_value=0.01,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        for _ in range(n)
    ]
    actual_values = [
        draw(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ).filter(lambda x: abs(x) > 1e-9)
        )
        for _ in range(n)
    ]

    mu = np.array(mu_values, dtype=np.float64)
    sigma = np.array(sigma_values, dtype=np.float64)
    actual = np.array(actual_values, dtype=np.float64)

    return mu, sigma, actual


# ---------------------------------------------------------------------------
# Property 9: Evaluation Metrics Formula Correctness
# ---------------------------------------------------------------------------


class TestEvaluationMetricsFormulaCorrectness:
    """Property 9: Evaluation Metrics Formula Correctness.

    For any set of prediction triples (μ̂, σ̂, actual) where σ̂ > 0:
    - Directional Accuracy SHALL equal `count(sign(μ̂) == sign(actual)) / N`
    - 95% Covered Ratio SHALL equal `count(|actual - μ̂| ≤ 2σ̂) / N`
    - RMSE SHALL equal `sqrt(mean((μ̂ - actual)²))`

    **Validates: Requirements 8.2, 8.3, 8.4**
    """

    @given(inputs=prediction_triples())
    @settings(max_examples=100, deadline=None)
    def test_directional_accuracy_matches_formula(self, inputs):
        """Directional Accuracy equals count(sign(μ̂) == sign(actual)) / N.

        **Validates: Requirements 8.2**
        """
        mu, sigma, actual = inputs

        # Compute using the function under test
        result = _compute_directional_accuracy(mu, actual)

        # Compute expected DA manually
        n = len(mu)
        correct_count = np.sum(np.sign(mu) == np.sign(actual))
        expected_da = correct_count / n

        assert abs(result - expected_da) < 1e-10, (
            f"DA mismatch. Actual: {result}, Expected: {expected_da}, "
            f"Diff: {abs(result - expected_da)}"
        )

    @given(inputs=prediction_triples())
    @settings(max_examples=100, deadline=None)
    def test_covered_ratio_95_matches_formula(self, inputs):
        """95% Covered Ratio equals count(|actual - μ̂| ≤ 2σ̂) / N.

        **Validates: Requirements 8.3**
        """
        mu, sigma, actual = inputs

        # Compute using the function under test
        result = _compute_covered_ratio_95(mu, sigma, actual)

        # Compute expected CR95 manually
        n = len(mu)
        covered_count = np.sum(np.abs(actual - mu) <= 2 * sigma)
        expected_cr95 = covered_count / n

        assert abs(result - expected_cr95) < 1e-10, (
            f"CR95 mismatch. Actual: {result}, Expected: {expected_cr95}, "
            f"Diff: {abs(result - expected_cr95)}"
        )

    @given(inputs=prediction_triples())
    @settings(max_examples=100, deadline=None)
    def test_rmse_matches_formula(self, inputs):
        """RMSE equals sqrt(mean((μ̂ - actual)²)).

        **Validates: Requirements 8.4**
        """
        mu, sigma, actual = inputs

        # Compute using the function under test
        result = _compute_rmse(mu, actual)

        # Compute expected RMSE manually
        squared_errors = (mu - actual) ** 2
        expected_rmse = math.sqrt(np.mean(squared_errors))

        assert abs(result - expected_rmse) < 1e-10, (
            f"RMSE mismatch. Actual: {result}, Expected: {expected_rmse}, "
            f"Diff: {abs(result - expected_rmse)}"
        )
