"""Property-based tests for the KServe serving predictor.

Tests the inference request/response contract: for any valid feature tensor
of shape (lookback_window, 16) with finite floats, the predictor returns
mu (finite float) and sigma (positive finite float).

**Validates: Requirements 5.1, 5.2**
"""

import math
import os
import sys
import tempfile
from dataclasses import asdict
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.model import ProbabilisticTransformer


_kserve_mock = ModuleType("kserve")


class _MockKServeModel:
    """Minimal mock of kserve.Model base class."""

    def __init__(self, name: str):
        self.name = name
        self.ready = False


_kserve_mock.Model = _MockKServeModel
sys.modules.setdefault("kserve", _kserve_mock)

from probabilisticforecaster.kubeflow.serving.predictor import ForecasterPredictor  # noqa: E402


NUM_FEATURES = 16
VALID_LOOKBACK_WINDOWS = [24, 36, 48]


@st.composite
def valid_feature_tensors(draw):
    """Generate valid feature tensors of shape (lookback_window, 16) with finite floats.

    lookback_window is drawn from {24, 36, 48}.
    Feature values are finite floats in a reasonable range.
    """
    lookback_window = draw(st.sampled_from(VALID_LOOKBACK_WINDOWS))
    range_type = draw(st.sampled_from(["normal", "zeros", "small", "large", "mixed"]))

    if range_type == "normal":
        arr = draw(
            arrays(
                dtype=np.float32,
                shape=(lookback_window, NUM_FEATURES),
                elements=st.floats(
                    min_value=-4.0,
                    max_value=4.0,
                    allow_nan=False,
                    allow_infinity=False,
                    width=32,
                ),
            )
        )
    elif range_type == "zeros":
        arr = np.zeros((lookback_window, NUM_FEATURES), dtype=np.float32)
    elif range_type == "small":
        arr = draw(
            arrays(
                dtype=np.float32,
                shape=(lookback_window, NUM_FEATURES),
                elements=st.floats(
                    min_value=-0.5,
                    max_value=0.5,
                    allow_nan=False,
                    allow_infinity=False,
                    width=32,
                ),
            )
        )
    elif range_type == "large":
        arr = draw(
            arrays(
                dtype=np.float32,
                shape=(lookback_window, NUM_FEATURES),
                elements=st.floats(
                    min_value=-50.0,
                    max_value=50.0,
                    allow_nan=False,
                    allow_infinity=False,
                    width=32,
                ),
            )
        )
    else:
        arr = draw(
            arrays(
                dtype=np.float32,
                shape=(lookback_window, NUM_FEATURES),
                elements=st.one_of(
                    st.floats(
                        min_value=-50.0,
                        max_value=50.0,
                        allow_nan=False,
                        allow_infinity=False,
                        width=32,
                    ),
                    st.just(0.0),
                ),
            )
        )

    return lookback_window, arr.tolist()


def _create_predictor(lookback_window: int) -> ForecasterPredictor:
    """Create a ForecasterPredictor with a real model for the given lookback_window.

    Creates a temporary checkpoint file with a freshly initialized model,
    then loads it into the predictor.
    """
    config = ForecasterConfig(lookback_window=lookback_window)
    model = ProbabilisticTransformer(config)
    model.eval()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
    }

    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(checkpoint, f.name)
    f.close()

    predictor = ForecasterPredictor(name="test-forecaster", model_path=f.name)
    predictor.load()

    os.unlink(f.name)

    return predictor


_predictor_cache: dict[int, ForecasterPredictor] = {}


def _get_predictor(lookback_window: int) -> ForecasterPredictor:
    """Get or create a cached predictor for the given lookback_window."""
    if lookback_window not in _predictor_cache:
        _predictor_cache[lookback_window] = _create_predictor(lookback_window)
    return _predictor_cache[lookback_window]


class TestInferenceRequestResponseContract:
    """Property 6: Inference request/response contract.

    For any valid feature tensor of shape (lookback_window, 16) with finite floats,
    predictor returns mu (finite float) and sigma (positive finite float).

    **Validates: Requirements 5.1, 5.2**
    """

    @given(data=valid_feature_tensors())
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.large_base_example],
    )
    def test_valid_input_returns_finite_mu_and_positive_sigma(self, data):
        """For any valid feature tensor, predictor returns finite mu and positive sigma.

        **Validates: Requirements 5.1, 5.2**

        Test approach:
        1. Generate a valid feature tensor of shape (lookback_window, 16) with finite floats
        2. Create a ForecasterPredictor with a real ProbabilisticTransformer model
        3. Call predict() with the valid payload
        4. Assert response contains "predictions" list
        5. Assert each prediction has "mu" that is a finite float
        6. Assert each prediction has "sigma" that is a positive finite float
        """
        lookback_window, features = data

        predictor = _get_predictor(lookback_window)

        payload = {"instances": features}
        response = predictor.predict(payload)

        assert "predictions" in response, (
            f"Response missing 'predictions' key. Got keys: {list(response.keys())}"
        )
        predictions = response["predictions"]
        assert isinstance(predictions, list), (
            f"'predictions' must be a list, got {type(predictions).__name__}"
        )
        assert len(predictions) > 0, "predictions list must not be empty"

        for i, pred in enumerate(predictions):
            assert "mu" in pred, f"Prediction {i} missing 'mu' key"
            assert "sigma" in pred, f"Prediction {i} missing 'sigma' key"

            mu = pred["mu"]
            sigma = pred["sigma"]

            assert isinstance(mu, float), (
                f"Prediction {i}: mu must be a float, got {type(mu).__name__}"
            )
            assert math.isfinite(mu), (
                f"Prediction {i}: mu must be finite, got {mu}"
            )

            assert isinstance(sigma, float), (
                f"Prediction {i}: sigma must be a float, got {type(sigma).__name__}"
            )
            assert math.isfinite(sigma), (
                f"Prediction {i}: sigma must be finite, got {sigma}"
            )
            assert sigma > 0, (
                f"Prediction {i}: sigma must be positive, got {sigma}"
            )


@st.composite
def malformed_payloads(draw):
    """Generate payloads that violate the input contract.

    Categories of malformed payloads:
    - Missing 'instances' field
    - Wrong number of features (not 16)
    - Wrong sequence length (not matching lookback_window)
    - Non-numeric data in instances
    - NaN or Inf values in instances
    - Wrong dimensionality (1D, 4D+)
    - Empty instances list
    """
    lookback_window = draw(st.sampled_from(VALID_LOOKBACK_WINDOWS))
    violation_type = draw(
        st.sampled_from([
            "missing_instances",
            "wrong_features",
            "wrong_sequence_length",
            "non_numeric",
            "nan_values",
            "inf_values",
            "wrong_dimensionality_1d",
            "instances_not_list",
        ])
    )

    if violation_type == "missing_instances":
        payload = draw(
            st.fixed_dictionaries({"data": st.just([[0.0] * 16] * lookback_window)})
        )
    elif violation_type == "wrong_features":
        num_features = draw(st.integers(min_value=1, max_value=32).filter(lambda x: x != 16))
        payload = {
            "instances": [[0.1] * num_features for _ in range(lookback_window)]
        }
    elif violation_type == "wrong_sequence_length":
        wrong_length = draw(
            st.integers(min_value=1, max_value=100).filter(
                lambda x: x not in VALID_LOOKBACK_WINDOWS
            )
        )
        payload = {"instances": [[0.1] * 16 for _ in range(wrong_length)]}
    elif violation_type == "non_numeric":
        payload = {
            "instances": [["not_a_number"] * 16 for _ in range(lookback_window)]
        }
    elif violation_type == "nan_values":
        row = [float("nan")] + [0.1] * 15
        payload = {"instances": [row for _ in range(lookback_window)]}
    elif violation_type == "inf_values":
        row = [float("inf")] + [0.1] * 15
        payload = {"instances": [row for _ in range(lookback_window)]}
    elif violation_type == "wrong_dimensionality_1d":
        payload = {"instances": [0.1] * 16}
    else:
        payload = {"instances": "not_a_list"}

    return lookback_window, payload


class TestMalformedRequestRejection:
    """Property 7: Malformed request rejection.

    For any payload violating the input contract, predictor returns a ValueError
    (which KServe maps to HTTP 400) with a descriptive error message.

    **Validates: Requirements 5.6**
    """

    @given(data=malformed_payloads())
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.large_base_example],
    )
    def test_malformed_input_raises_value_error(self, data):
        """For any payload violating input contract, predictor raises ValueError.

        **Validates: Requirements 5.6**

        Test approach:
        1. Generate a malformed payload (wrong shape, non-numeric, NaN/Inf, missing fields)
        2. Create a ForecasterPredictor with a real ProbabilisticTransformer model
        3. Call predict() with the malformed payload
        4. Assert that a ValueError is raised
        5. Assert the error message is non-empty and descriptive
        """
        lookback_window, payload = data

        predictor = _get_predictor(lookback_window)

        import pytest

        with pytest.raises(ValueError) as exc_info:
            predictor.predict(payload)

        error_msg = str(exc_info.value)
        assert len(error_msg) > 0, "Error message must not be empty"
