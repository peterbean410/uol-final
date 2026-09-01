"""Unit tests for the training module.

Tests hyperparameter usage, model persistence round-trip, and NaN loss detection.

Validates: Requirements 7.2, 7.3, 7.4
"""

import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.dataset import ForexDataset
from probabilisticforecaster.model import ProbabilisticTransformer
from probabilisticforecaster.training import train_model


def _make_synthetic_dataset(n_samples: int = 200, lookback: int = 36) -> ForexDataset:
    """Create a small synthetic ForexDataset with random tensors for testing.

    Generates contiguous 5-minute timestamps so no gaps are detected,
    and random features/close prices.
    """
    total_bars = n_samples + lookback
    timestamps = pd.date_range(
        start="2023-01-02 00:00", periods=total_bars, freq="5min"
    )
    features_df = pd.DataFrame(
        np.random.randn(total_bars, 16).astype(np.float32),
        index=timestamps,
        columns=[f"feat_{i}" for i in range(16)],
    )
    close_prices = pd.Series(
        150.0 + np.cumsum(np.random.randn(total_bars) * 0.01),
        index=timestamps,
    )
    return ForexDataset(
        features_df=features_df,
        close_prices=close_prices,
        lookback=lookback,
        horizon=1,
        stride=1,
    )


class TestHyperparameters:
    """Test that training uses the correct hyperparameters from config."""

    def test_default_config_learning_rate(self):
        """Config default learning rate is 0.001."""
        config = ForecasterConfig()
        assert config.learning_rate == 0.001

    def test_default_config_batch_size(self):
        """Config default batch size is 64."""
        config = ForecasterConfig()
        assert config.batch_size == 64

    def test_default_config_epochs(self):
        """Config default epochs is 5."""
        config = ForecasterConfig()
        assert config.epochs == 5

    @patch("probabilisticforecaster.training._upload_model_to_s3")
    def test_training_uses_config_hyperparameters(self, mock_upload):
        """Train model uses lr, batch_size, epochs from config."""
        config = ForecasterConfig(
            learning_rate=0.001,
            batch_size=64,
            epochs=5,
        )
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            config.model_path = tmp.name

            model = ProbabilisticTransformer(config)
            dataset = _make_synthetic_dataset(n_samples=200)

            with patch(
                "torch.optim.Adam", wraps=torch.optim.Adam
            ) as mock_adam:
                history = train_model(
                    model=model,
                    train_dataset=dataset,
                    config=config,
                    upload_to_s3=False,
                )

                mock_adam.assert_called_once()
                call_kwargs = mock_adam.call_args
                assert call_kwargs[1]["lr"] == 0.001 or call_kwargs.kwargs["lr"] == 0.001

            assert len(history["epoch_loss"]) == 5

    @patch("probabilisticforecaster.training._upload_model_to_s3")
    def test_training_respects_custom_epochs(self, mock_upload):
        """Train model runs the number of epochs specified in config."""
        config = ForecasterConfig(epochs=2)
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            config.model_path = tmp.name

            model = ProbabilisticTransformer(config)
            dataset = _make_synthetic_dataset(n_samples=100)

            history = train_model(
                model=model,
                train_dataset=dataset,
                config=config,
                upload_to_s3=False,
            )

            assert len(history["epoch_loss"]) == 2


class TestModelPersistenceRoundTrip:
    """Test that saving and reloading a model produces the same predictions."""

    @patch("probabilisticforecaster.training._upload_model_to_s3")
    def test_save_reload_same_predictions(self, mock_upload):
        """Model saved after training and reloaded produces identical predictions."""
        config = ForecasterConfig(epochs=2)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            config.model_path = tmp.name

        model = ProbabilisticTransformer(config)
        dataset = _make_synthetic_dataset(n_samples=100)

        train_model(
            model=model,
            train_dataset=dataset,
            config=config,
            upload_to_s3=False,
        )

        model.eval()
        test_input = torch.randn(1, 36, 16)
        with torch.no_grad():
            mu_original, sigma_original = model(test_input)

        checkpoint = torch.load(config.model_path, weights_only=False)
        new_model = ProbabilisticTransformer(config)
        new_model.load_state_dict(checkpoint["model_state_dict"])
        new_model.eval()

        with torch.no_grad():
            mu_loaded, sigma_loaded = new_model(test_input)

        assert torch.allclose(mu_original, mu_loaded, atol=1e-6)
        assert torch.allclose(sigma_original, sigma_loaded, atol=1e-6)

    @patch("probabilisticforecaster.training._upload_model_to_s3")
    def test_checkpoint_contains_required_keys(self, mock_upload):
        """Saved checkpoint contains model_state_dict, config, training_history, metadata."""
        config = ForecasterConfig(epochs=1)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            config.model_path = tmp.name

        model = ProbabilisticTransformer(config)
        dataset = _make_synthetic_dataset(n_samples=100)

        train_model(
            model=model,
            train_dataset=dataset,
            config=config,
            upload_to_s3=False,
        )

        checkpoint = torch.load(config.model_path, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "config" in checkpoint
        assert "training_history" in checkpoint
        assert "metadata" in checkpoint

    @patch("probabilisticforecaster.training._upload_model_to_s3")
    def test_checkpoint_config_matches(self, mock_upload):
        """Saved checkpoint config matches the training config."""
        config = ForecasterConfig(
            symbol="AUDJPY",
            learning_rate=0.001,
            batch_size=64,
            epochs=1,
        )
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            config.model_path = tmp.name

        model = ProbabilisticTransformer(config)
        dataset = _make_synthetic_dataset(n_samples=100)

        train_model(
            model=model,
            train_dataset=dataset,
            config=config,
            upload_to_s3=False,
        )

        checkpoint = torch.load(config.model_path, weights_only=False)
        assert checkpoint["config"]["symbol"] == "AUDJPY"
        assert checkpoint["config"]["learning_rate"] == 0.001
        assert checkpoint["config"]["batch_size"] == 64


class TestNaNLossDetection:
    """Test that NaN loss during training raises RuntimeError."""

    @patch("probabilisticforecaster.training._upload_model_to_s3")
    def test_nan_loss_raises_runtime_error(self, mock_upload):
        """Training raises RuntimeError when NaN loss is detected."""
        config = ForecasterConfig(epochs=2)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            config.model_path = tmp.name

        model = ProbabilisticTransformer(config)
        dataset = _make_synthetic_dataset(n_samples=100)

        with torch.no_grad():
            for param in model.parameters():
                param.fill_(float("nan"))

        with pytest.raises(RuntimeError, match="NaN loss"):
            train_model(
                model=model,
                train_dataset=dataset,
                config=config,
                upload_to_s3=False,
            )
