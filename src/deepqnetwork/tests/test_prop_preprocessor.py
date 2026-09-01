"""Property-based tests for deepqnetwork.preprocessor module.

# Feature: deepqnetwork, Property 1: State preprocessing preserves values
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from deepqnetwork.preprocessor import StatePreprocessor


@dataclass
class FakeStateRow:
    """Mimics environment_pb2.StateRow."""

    values: list[float] = field(default_factory=list)


@dataclass
class FakeObservation:
    """Mimics environment_pb2.Observation."""

    state_columns: list[str] = field(default_factory=list)
    state_data: list[FakeStateRow] = field(default_factory=list)
    reward: float = 0.0
    done: bool = False


finite_float32 = st.floats(
    min_value=-1e30,
    max_value=1e30,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=True,
)

float_value_lists = st.lists(
    finite_float32,
    min_size=1,
    max_size=100,
)


class TestStatePreprocessingPreservesValues:
    """Property 1: State preprocessing preserves values.

    For any valid Observation containing a StateRow with arbitrary float values,
    the State_Preprocessor SHALL produce a float32 tensor whose values are
    identical to the input (no normalisation, no transformation beyond type
    conversion).

    **Validates: Requirements 2.1, 2.2, 2.3**
    """

    @given(values=float_value_lists)
    @settings(max_examples=100)
    def test_output_values_match_input_after_float32_conversion(
        self, values: list[float]
    ):
        """For any valid float values (excluding NaN/Inf), the output tensor
        values are identical to the input values after float32 conversion.

        # Feature: deepqnetwork, Property 1: State preprocessing preserves values
        """
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        columns = [f"col_{i}" for i in range(len(values))]
        obs = FakeObservation(
            state_columns=columns,
            state_data=[FakeStateRow(values=values)],
        )

        result = preprocessor.process(obs)

        expected = np.array(values, dtype=np.float32)
        expected_tensor = torch.from_numpy(expected)

        assert result.dtype == torch.float32
        assert result.shape == (len(values),)
        assert torch.equal(result, expected_tensor), (
            f"Output tensor values differ from input.\n"
            f"Input (float32): {expected_tensor.tolist()}\n"
            f"Output: {result.tolist()}"
        )
