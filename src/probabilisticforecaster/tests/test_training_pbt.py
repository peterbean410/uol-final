"""Property-based tests for the Training module (training.py).

Uses Hypothesis to verify correctness of the Gaussian NLL loss function
across randomly generated (mu, sigma, target) triples.
"""

import math

import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from probabilisticforecaster.training import gaussian_nll_loss


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def nll_inputs(draw):
    """Generate random (mu, sigma, target) tensors for NLL loss testing.

    mu and target are drawn from [-10, 10], sigma from [0.01, 10].
    Batch size varies from 1 to 16.
    """
    batch_size = draw(st.integers(min_value=1, max_value=16))

    mu_values = [
        draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        for _ in range(batch_size)
    ]
    sigma_values = [
        draw(st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False))
        for _ in range(batch_size)
    ]
    target_values = [
        draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        for _ in range(batch_size)
    ]

    mu = torch.tensor(mu_values, dtype=torch.float64)
    sigma = torch.tensor(sigma_values, dtype=torch.float64)
    target = torch.tensor(target_values, dtype=torch.float64)

    return mu, sigma, target


# ---------------------------------------------------------------------------
# Property 8: Gaussian NLL Loss Correctness
# ---------------------------------------------------------------------------


class TestGaussianNLLLossCorrectness:
    """Property 8: Gaussian NLL Loss Correctness.

    For any triple (μ, σ, target) where σ > 0, the computed Gaussian NLL loss
    SHALL equal `0.5 * (log(σ²) + ((target - μ) / σ)² + log(2π))`, averaged
    over the batch.

    **Validates: Requirements 7.1, 8.1**
    """

    @given(inputs=nll_inputs())
    @settings(max_examples=100, deadline=None)
    def test_nll_matches_manual_formula(self, inputs):
        """Gaussian NLL loss matches the expected formula for any valid (mu, sigma, target).

        **Validates: Requirements 7.1, 8.1**

        For each element in the batch, the per-element NLL is:
            0.5 * (log(sigma^2) + ((target - mu) / sigma)^2 + log(2*pi))

        The function returns the mean over the batch.
        """
        mu, sigma, target = inputs

        # Compute using the function under test
        actual_loss = gaussian_nll_loss(mu, sigma, target)

        # Compute expected loss manually
        log_variance = torch.log(sigma ** 2)
        squared_error = ((target - mu) / sigma) ** 2
        log_2pi = math.log(2 * math.pi)
        expected_loss = 0.5 * (log_variance + squared_error + log_2pi)
        expected_mean = expected_loss.mean()

        assert torch.allclose(actual_loss, expected_mean, atol=1e-6, rtol=1e-5), (
            f"NLL loss mismatch. "
            f"Actual: {actual_loss.item()}, Expected: {expected_mean.item()}, "
            f"Diff: {abs(actual_loss.item() - expected_mean.item())}"
        )
