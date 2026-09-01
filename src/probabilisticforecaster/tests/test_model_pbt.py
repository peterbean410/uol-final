"""Property-based tests for the Transformer Model (model.py).

Uses Hypothesis to verify correctness properties across randomly generated inputs.
"""

import torch
import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.model import ProbabilisticTransformer


SEQ_LEN = 36
NUM_FEATURES = 16


@st.composite
def valid_input_tensors(draw):
    """Generate valid input tensors of shape (B, 36, 16) with various value ranges.

    Includes zeros, negatives, and large values to stress-test sigma positivity.
    """
    batch_size = draw(st.integers(min_value=1, max_value=4))

    range_type = draw(st.sampled_from(["normal", "zeros", "negatives", "large", "mixed"]))

    if range_type == "normal":
        arr = draw(
            arrays(
                dtype=np.float32,
                shape=(batch_size, SEQ_LEN, NUM_FEATURES),
                elements=st.floats(
                    min_value=-4.0, max_value=4.0,
                    allow_nan=False, allow_infinity=False, width=32,
                ),
            )
        )
    elif range_type == "zeros":
        arr = np.zeros((batch_size, SEQ_LEN, NUM_FEATURES), dtype=np.float32)
    elif range_type == "negatives":
        arr = draw(
            arrays(
                dtype=np.float32,
                shape=(batch_size, SEQ_LEN, NUM_FEATURES),
                elements=st.floats(
                    min_value=-10.0, max_value=np.float32(-0.0625).item(),
                    allow_nan=False, allow_infinity=False, width=32,
                ),
            )
        )
    elif range_type == "large":
        arr = draw(
            arrays(
                dtype=np.float32,
                shape=(batch_size, SEQ_LEN, NUM_FEATURES),
                elements=st.floats(
                    min_value=-100.0, max_value=100.0,
                    allow_nan=False, allow_infinity=False, width=32,
                ),
            )
        )
    else:
        arr = draw(
            arrays(
                dtype=np.float32,
                shape=(batch_size, SEQ_LEN, NUM_FEATURES),
                elements=st.one_of(
                    st.floats(
                        min_value=-100.0, max_value=100.0,
                        allow_nan=False, allow_infinity=False, width=32,
                    ),
                    st.just(0.0),
                ),
            )
        )

    return torch.from_numpy(arr)


@st.composite
def causal_mask_input_tensors(draw):
    """Generate input tensors of shape (1, 36, 16) for causal mask testing.

    Uses numpy arrays with float32 width for Hypothesis compatibility.
    """
    arr = draw(
        arrays(
            dtype=np.float32,
            shape=(1, SEQ_LEN, NUM_FEATURES),
            elements=st.floats(
                min_value=-5.0, max_value=5.0,
                allow_nan=False, allow_infinity=False, width=32,
            ),
        )
    )
    return torch.from_numpy(arr)


class TestCausalMaskPreventsFutureLeakage:
    """Property 5: Causal Mask Prevents Future Information Leakage.

    For any input tensor of shape (B, 36, 16), the model output at position t
    SHALL be invariant to modifications of input values at positions > t.
    That is, changing future inputs does not affect past outputs.

    **Validates: Requirements 5.3**
    """

    @given(
        data=causal_mask_input_tensors(),
        t=st.integers(min_value=0, max_value=34),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.large_base_example],
    )
    def test_output_invariant_to_future_modifications(
        self, data: torch.Tensor, t: int, seed: int
    ):
        """Model output at position t is invariant to modifications at positions > t.

        **Validates: Requirements 5.3**

        Test approach:
        1. Generate a random input tensor of shape (1, 36, 16)
        2. Run the model to get outputs (mu1, sigma1)
        3. Modify input values at positions > t (for some random t in [0, 34])
        4. Run the model again to get outputs (mu2, sigma2)
        5. Assert that mu1[:, :t+1, :] == mu2[:, :t+1, :] and
           sigma1[:, :t+1, :] == sigma2[:, :t+1, :]
        """
        torch.manual_seed(seed)
        config = ForecasterConfig()
        model = ProbabilisticTransformer(config)
        model.eval()

        with torch.no_grad():
            mu1, sigma1 = model(data)

            modified_data = data.clone()
            rng = torch.Generator()
            rng.manual_seed(seed + 1)
            future_noise = torch.randn(
                1, SEQ_LEN - t - 1, NUM_FEATURES, generator=rng
            ) * 10.0
            modified_data[:, t + 1 :, :] = future_noise

            mu2, sigma2 = model(modified_data)

        assert torch.allclose(
            mu1[:, : t + 1, :], mu2[:, : t + 1, :], atol=1e-6, rtol=1e-5
        ), (
            f"mu output at positions 0..{t} changed when future positions were modified. "
            f"Max diff: {(mu1[:, :t+1, :] - mu2[:, :t+1, :]).abs().max().item()}"
        )
        assert torch.allclose(
            sigma1[:, : t + 1, :], sigma2[:, : t + 1, :], atol=1e-6, rtol=1e-5
        ), (
            f"sigma output at positions 0..{t} changed when future positions were modified. "
            f"Max diff: {(sigma1[:, :t+1, :] - sigma2[:, :t+1, :]).abs().max().item()}"
        )


class TestModelOutputShapeInvariant:
    """Property 6: Model Output Shape Invariant.

    For any valid input tensor of shape (B, 36, 16) where B > 0, the model
    SHALL produce two output tensors (μ, σ) each of shape (B, 36, 1).

    **Validates: Requirements 5.5**
    """

    @given(batch_size=st.integers(min_value=1, max_value=8))
    @settings(max_examples=50, deadline=None)
    def test_output_shape_matches_expected(self, batch_size: int):
        """Model output (μ, σ) each have shape (B, 36, 1) for any valid input.

        **Validates: Requirements 5.5**
        """
        config = ForecasterConfig()
        model = ProbabilisticTransformer(config)
        model.eval()

        x = torch.randn(batch_size, SEQ_LEN, NUM_FEATURES)

        with torch.no_grad():
            mu, sigma = model(x)

        assert mu.shape == (batch_size, SEQ_LEN, 1), (
            f"Expected mu shape ({batch_size}, 36, 1), got {mu.shape}"
        )
        assert sigma.shape == (batch_size, SEQ_LEN, 1), (
            f"Expected sigma shape ({batch_size}, 36, 1), got {sigma.shape}"
        )


class TestSigmaStrictlyPositive:
    """Property 7: Sigma Strictly Positive.

    For any valid input tensor of shape (B, 36, 16), all values in the σ output
    tensor SHALL be strictly greater than zero.

    **Validates: Requirements 5.6**
    """

    @given(x=valid_input_tensors())
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_sigma_strictly_positive(self, x: torch.Tensor):
        """All sigma output values are strictly > 0 for any valid input.

        **Validates: Requirements 5.6**
        """
        config = ForecasterConfig()
        model = ProbabilisticTransformer(config)
        model.eval()

        with torch.no_grad():
            mu, sigma = model(x)

        assert torch.all(sigma > 0), (
            f"Sigma contains non-positive values. "
            f"Min sigma: {sigma.min().item()}, "
            f"Num non-positive: {(sigma <= 0).sum().item()} / {sigma.numel()}"
        )
