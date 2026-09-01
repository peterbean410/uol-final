"""Unit tests for the inference module.

Tests ForecasterInference class: model loading, prediction, error handling.
"""

import os
import tempfile
from dataclasses import asdict

import pytest
import torch

from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.inference import ForecasterInference
from probabilisticforecaster.model import ProbabilisticTransformer


@pytest.fixture
def config():
    """Create a default ForecasterConfig for testing."""
    return ForecasterConfig()


@pytest.fixture
def saved_model_path(config):
    """Create a temporary saved model checkpoint and return its path."""
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

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(checkpoint, f.name)
        yield f.name

    os.unlink(f.name)


class TestForecasterInferenceInit:
    """Tests for ForecasterInference initialization."""

    def test_loads_model_successfully(self, saved_model_path, config):
        """Test that inference loads a valid model checkpoint."""
        inference = ForecasterInference(saved_model_path, config)
        assert inference is not None

    def test_file_not_found_raises_error(self, config):
        """Test that missing model file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Model weights not found"):
            ForecasterInference("/nonexistent/path/model.pt", config)

    def test_incompatible_config_raises_error(self, config):
        """Test that incompatible saved config raises RuntimeError."""
        incompatible_config = ForecasterConfig(num_features=32, num_heads=4, num_layers=3)
        model = ProbabilisticTransformer(incompatible_config)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": asdict(incompatible_config),
            "training_history": {},
            "metadata": {},
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(checkpoint, f.name)
            tmp_path = f.name

        try:
            with pytest.raises(RuntimeError, match="Saved model config incompatible"):
                ForecasterInference(tmp_path, config)
        finally:
            os.unlink(tmp_path)

    def test_model_in_eval_mode(self, saved_model_path, config):
        """Test that loaded model is in eval mode."""
        inference = ForecasterInference(saved_model_path, config)
        assert not inference._model.training


class TestForecasterInferencePredict:
    """Tests for single prediction."""

    def test_predict_returns_tuple_of_floats(self, saved_model_path, config):
        """Test that predict returns (mu, sigma) as floats."""
        inference = ForecasterInference(saved_model_path, config)
        features = torch.randn(36, 16)

        mu, sigma = inference.predict(features)

        assert isinstance(mu, float)
        assert isinstance(sigma, float)

    def test_predict_sigma_positive(self, saved_model_path, config):
        """Test that predicted sigma is strictly positive."""
        inference = ForecasterInference(saved_model_path, config)
        features = torch.randn(36, 16)

        _, sigma = inference.predict(features)

        assert sigma > 0

    def test_predict_wrong_feature_dim_raises_error(self, saved_model_path, config):
        """Test that wrong feature dimension raises ValueError."""
        inference = ForecasterInference(saved_model_path, config)
        features = torch.randn(36, 10)

        with pytest.raises(ValueError, match="Input feature dimension 10 != expected 16"):
            inference.predict(features)

    def test_predict_wrong_ndim_raises_error(self, saved_model_path, config):
        """Test that wrong number of dimensions raises ValueError."""
        inference = ForecasterInference(saved_model_path, config)
        features = torch.randn(2, 36, 16)

        with pytest.raises(ValueError, match="Expected 2D input"):
            inference.predict(features)


class TestForecasterInferencePredictBatch:
    """Tests for batch prediction."""

    def test_predict_batch_returns_tensors(self, saved_model_path, config):
        """Test that predict_batch returns (mu_tensor, sigma_tensor)."""
        inference = ForecasterInference(saved_model_path, config)
        features = torch.randn(4, 36, 16)

        mu, sigma = inference.predict_batch(features)

        assert isinstance(mu, torch.Tensor)
        assert isinstance(sigma, torch.Tensor)

    def test_predict_batch_output_shape(self, saved_model_path, config):
        """Test that batch output shape is (batch,)."""
        inference = ForecasterInference(saved_model_path, config)
        batch_size = 8
        features = torch.randn(batch_size, 36, 16)

        mu, sigma = inference.predict_batch(features)

        assert mu.shape == (batch_size,)
        assert sigma.shape == (batch_size,)

    def test_predict_batch_sigma_positive(self, saved_model_path, config):
        """Test that all batch sigma values are strictly positive."""
        inference = ForecasterInference(saved_model_path, config)
        features = torch.randn(4, 36, 16)

        _, sigma = inference.predict_batch(features)

        assert (sigma > 0).all()

    def test_predict_batch_wrong_feature_dim_raises_error(self, saved_model_path, config):
        """Test that wrong feature dimension raises ValueError."""
        inference = ForecasterInference(saved_model_path, config)
        features = torch.randn(4, 36, 8)

        with pytest.raises(ValueError, match="Input feature dimension 8 != expected 16"):
            inference.predict_batch(features)

    def test_predict_batch_wrong_ndim_raises_error(self, saved_model_path, config):
        """Test that wrong number of dimensions raises ValueError."""
        inference = ForecasterInference(saved_model_path, config)
        features = torch.randn(36, 16)

        with pytest.raises(ValueError, match="Expected 3D input"):
            inference.predict_batch(features)
