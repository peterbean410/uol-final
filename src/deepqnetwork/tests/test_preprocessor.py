"""Unit tests for deepqnetwork.preprocessor module."""

import logging
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from deepqnetwork.preprocessor import StatePreprocessor


# --- Fake protobuf-like objects for testing ---


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


class TestStatePreprocessorInit:
    """Tests for StatePreprocessor initialisation."""

    def test_state_dim_not_set_before_process(self):
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        with pytest.raises(RuntimeError, match="not yet initialised"):
            _ = preprocessor.state_dim

    def test_device_stored(self):
        device = torch.device("cpu")
        preprocessor = StatePreprocessor(device=device)
        assert preprocessor._device == device


class TestStatePreprocessorProcess:
    """Tests for StatePreprocessor.process method."""

    def test_basic_extraction(self):
        """Req 2.1: Extract float values from first StateRow into 1-D array."""
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        obs = FakeObservation(
            state_columns=["a", "b", "c"],
            state_data=[FakeStateRow(values=[1.0, 2.0, 3.0])],
        )
        result = preprocessor.process(obs)
        expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_no_normalisation_applied(self):
        """Req 2.2: Values are NOT normalised, passed through as-is."""
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        values = [100.5, -200.3, 0.001, 999.99]
        obs = FakeObservation(
            state_columns=["a", "b", "c", "d"],
            state_data=[FakeStateRow(values=values)],
        )
        result = preprocessor.process(obs)
        expected = torch.tensor(values, dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_output_dtype_float32(self):
        """Req 2.3: Output tensor is float32."""
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        obs = FakeObservation(
            state_columns=["x"],
            state_data=[FakeStateRow(values=[42.0])],
        )
        result = preprocessor.process(obs)
        assert result.dtype == torch.float32

    def test_output_device_cpu(self):
        """Req 2.3: Output tensor is on configured device."""
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        obs = FakeObservation(
            state_columns=["x", "y"],
            state_data=[FakeStateRow(values=[1.0, 2.0])],
        )
        result = preprocessor.process(obs)
        assert result.device == torch.device("cpu")

    def test_output_is_1d(self):
        """Req 2.1: Output is a 1-D tensor."""
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        obs = FakeObservation(
            state_columns=["a", "b", "c", "d", "e"],
            state_data=[FakeStateRow(values=[1.0, 2.0, 3.0, 4.0, 5.0])],
        )
        result = preprocessor.process(obs)
        assert result.dim() == 1
        assert result.shape == (5,)

    def test_state_dim_initialised_from_state_columns(self):
        """Req 2.4: state_dim initialised from state_columns length."""
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        obs = FakeObservation(
            state_columns=["col1", "col2", "col3"],
            state_data=[FakeStateRow(values=[1.0, 2.0, 3.0])],
        )
        preprocessor.process(obs)
        assert preprocessor.state_dim == 3

    def test_state_dim_set_only_once(self):
        """Req 2.4: state_dim is set on first observation only."""
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        obs1 = FakeObservation(
            state_columns=["a", "b", "c"],
            state_data=[FakeStateRow(values=[1.0, 2.0, 3.0])],
        )
        preprocessor.process(obs1)

        # Second observation with same dim but different column names
        obs2 = FakeObservation(
            state_columns=["x", "y", "z"],
            state_data=[FakeStateRow(values=[4.0, 5.0, 6.0])],
        )
        preprocessor.process(obs2)
        # state_dim remains 3 (from first observation)
        assert preprocessor.state_dim == 3

    def test_validates_state_data_length_mismatch(self):
        """Req 2.5: Raise ValueError if values length != state_dim."""
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        obs1 = FakeObservation(
            state_columns=["a", "b", "c"],
            state_data=[FakeStateRow(values=[1.0, 2.0, 3.0])],
        )
        preprocessor.process(obs1)

        # Now provide wrong length
        obs2 = FakeObservation(
            state_columns=["a", "b", "c"],
            state_data=[FakeStateRow(values=[1.0, 2.0])],
        )
        with pytest.raises(ValueError, match="does not match"):
            preprocessor.process(obs2)

    def test_empty_state_data_raises(self):
        """Raise ValueError when state_data is empty."""
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        obs = FakeObservation(
            state_columns=["a", "b"],
            state_data=[],
        )
        with pytest.raises(ValueError, match="empty state_data"):
            preprocessor.process(obs)

    def test_nan_replaced_with_zero(self, caplog):
        """NaN values are replaced with 0.0 and a warning is logged."""
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        obs = FakeObservation(
            state_columns=["a", "b", "c"],
            state_data=[FakeStateRow(values=[1.0, float("nan"), 3.0])],
        )
        with caplog.at_level(logging.WARNING):
            result = preprocessor.process(obs)
        assert result[1].item() == 0.0
        assert "NaN" in caplog.text

    def test_inf_replaced_with_zero(self, caplog):
        """Inf values are replaced with 0.0 and a warning is logged."""
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        obs = FakeObservation(
            state_columns=["a", "b", "c"],
            state_data=[FakeStateRow(values=[1.0, float("inf"), float("-inf")])],
        )
        with caplog.at_level(logging.WARNING):
            result = preprocessor.process(obs)
        assert result[1].item() == 0.0
        assert result[2].item() == 0.0
        assert "Inf" in caplog.text

    def test_nan_and_inf_mixed(self, caplog):
        """Both NaN and Inf are handled together."""
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        obs = FakeObservation(
            state_columns=["a", "b", "c", "d"],
            state_data=[
                FakeStateRow(values=[float("nan"), 2.0, float("inf"), float("-inf")])
            ],
        )
        with caplog.at_level(logging.WARNING):
            result = preprocessor.process(obs)
        expected = torch.tensor([0.0, 2.0, 0.0, 0.0], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_uses_first_state_row_only(self):
        """Only the first StateRow is used even if multiple are present."""
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        obs = FakeObservation(
            state_columns=["a", "b"],
            state_data=[
                FakeStateRow(values=[1.0, 2.0]),
                FakeStateRow(values=[99.0, 99.0]),
            ],
        )
        result = preprocessor.process(obs)
        expected = torch.tensor([1.0, 2.0], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_53_feature_state_vector(self):
        """Realistic test with 53 features (as per design doc)."""
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        columns = [f"feature_{i}" for i in range(53)]
        values = [float(i) * 0.1 for i in range(53)]
        obs = FakeObservation(
            state_columns=columns,
            state_data=[FakeStateRow(values=values)],
        )
        result = preprocessor.process(obs)
        assert result.shape == (53,)
        assert preprocessor.state_dim == 53
        expected = torch.tensor(values, dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_logs_state_dim_on_init(self, caplog):
        """Logs info message when state_dim is first initialised."""
        preprocessor = StatePreprocessor(device=torch.device("cpu"))
        obs = FakeObservation(
            state_columns=["a", "b"],
            state_data=[FakeStateRow(values=[1.0, 2.0])],
        )
        with caplog.at_level(logging.INFO):
            preprocessor.process(obs)
        assert "state_dim=2" in caplog.text
