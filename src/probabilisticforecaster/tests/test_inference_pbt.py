"""Property-based tests for the Inference module (inference.py).

Uses Hypothesis to verify that model parameters remain bitwise identical
before and after calling predict/predict_batch (weight immutability).
"""

import os
import tempfile
from dataclasses import asdict

import numpy as np
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.inference import ForecasterInference
from probabilisticforecaster.model import ProbabilisticTransformer


SEQ_LEN = 36
NUM_FEATURES = 16


@st.composite
def valid_2d_input_tensors(draw):
    """Generate valid input tensors of shape (36, 16) for single prediction.

    Covers diverse value ranges: normal, zeros, negatives, large, mixed.
    """
    range_type = draw(st.sampled_from(["normal", "zeros", "negatives", "large", "mixed"]))

    if range_type == "normal":
        arr = draw(
            arrays(
                dtype=np.float32,
                shape=(SEQ_LEN, NUM_FEATURES),
                elements=st.floats(
                    min_value=-4.0, max_value=4.0,
                    allow_nan=False, allow_infinity=False, width=32,
                ),
            )
        )
    elif range_type == "zeros":
        arr = np.zeros((SEQ_LEN, NUM_FEATURES), dtype=np.float32)
    elif range_type == "negatives":
        arr = draw(
            arrays(
                dtype=np.float32,
                shape=(SEQ_LEN, NUM_FEATURES),
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
                shape=(SEQ_LEN, NUM_FEATURES),
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
                shape=(SEQ_LEN, NUM_FEATURES),
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
def valid_3d_input_tensors(draw):
    """Generate valid input tensors of shape (batch, 36, 16) for batch prediction.

    Covers diverse value ranges and batch sizes from 1 to 4.
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


def _create_saved_model_path() -> tuple[str, ForecasterConfig]:
    """Create a temporary saved model checkpoint and return (path, config)."""
    config = ForecasterConfig()
    model = ProbabilisticTransformer(config)
    model.eval()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "training_history": {"epoch_loss": [1.0, 0.8, 0.6]},
        "metadata": {
            "symbol": "USDJPY",
            "horizon": 1,
            "trained_at": "2024-01-15T10:30:00Z",
            "train_nll": -5.23,
        },
    }

    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(checkpoint, f.name)
    f.close()
    return f.name, config


class TestInferenceWeightImmutability:
    """Property 14: Inference Weight Immutability.

    For any trained model loaded in inference mode and any valid input tensor,
    the model's parameter tensors SHALL be bitwise identical before and after
    calling predict.

    **Validates: Requirements 11.2**
    """

    @given(features=valid_2d_input_tensors())
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.large_base_example],
    )
    def test_predict_does_not_modify_weights(self, features: torch.Tensor):
        """Model parameters are bitwise identical before and after predict.

        **Validates: Requirements 11.2**

        Test approach:
        1. Load a trained model in inference mode
        2. Snapshot all model parameters (clone them)
        3. Call predict with a random valid input tensor of shape (36, 16)
        4. Compare each parameter tensor to its snapshot using torch.equal
        """
        model_path, config = _create_saved_model_path()
        try:
            inference = ForecasterInference(model_path, config)

            param_snapshots = {
                name: param.clone()
                for name, param in inference._model.named_parameters()
            }

            inference.predict(features)

            for name, param in inference._model.named_parameters():
                assert torch.equal(param, param_snapshots[name]), (
                    f"Parameter '{name}' was modified after calling predict. "
                    f"Max diff: {(param - param_snapshots[name]).abs().max().item()}"
                )
        finally:
            os.unlink(model_path)

    @given(features=valid_3d_input_tensors())
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.large_base_example],
    )
    def test_predict_batch_does_not_modify_weights(self, features: torch.Tensor):
        """Model parameters are bitwise identical before and after predict_batch.

        **Validates: Requirements 11.2**

        Test approach:
        1. Load a trained model in inference mode
        2. Snapshot all model parameters (clone them)
        3. Call predict_batch with a random valid input tensor of shape (batch, 36, 16)
        4. Compare each parameter tensor to its snapshot using torch.equal
        """
        model_path, config = _create_saved_model_path()
        try:
            inference = ForecasterInference(model_path, config)

            param_snapshots = {
                name: param.clone()
                for name, param in inference._model.named_parameters()
            }

            inference.predict_batch(features)

            for name, param in inference._model.named_parameters():
                assert torch.equal(param, param_snapshots[name]), (
                    f"Parameter '{name}' was modified after calling predict_batch. "
                    f"Max diff: {(param - param_snapshots[name]).abs().max().item()}"
                )
        finally:
            os.unlink(model_path)
