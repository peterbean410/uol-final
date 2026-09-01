"""Inference module for the Probabilistic Transformer Forecaster.

Loads trained model weights and produces predictions in eval mode
with gradient computation disabled.
"""

import os
from dataclasses import asdict

import torch

from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.model import ProbabilisticTransformer


class ForecasterInference:
    """Load a trained model and produce predictions.

    The model is set to eval mode and all predictions are made with
    gradient computation disabled to ensure weight immutability.

    Args:
        model_path: Path to the saved model checkpoint (.pt file).
        config: ForecasterConfig specifying model architecture parameters.

    Raises:
        FileNotFoundError: If model weights file does not exist at model_path.
        RuntimeError: If saved model config is incompatible with the provided config.
    """

    def __init__(self, model_path: str, config: ForecasterConfig):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        saved_config = checkpoint.get("config", {})
        self._validate_config_compatibility(saved_config, config)

        self._model = ProbabilisticTransformer(config)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

        self._config = config

    def _validate_config_compatibility(
        self, saved_config: dict, current_config: ForecasterConfig
    ) -> None:
        """Validate that saved config is compatible with the current config.

        Checks architecture-critical parameters: num_features, num_layers,
        num_heads, and dropout.

        Args:
            saved_config: Dictionary of config values from the checkpoint.
            current_config: The ForecasterConfig provided at init.

        Raises:
            RuntimeError: If any architecture parameter differs.
        """
        current_dict = asdict(current_config)
        architecture_keys = ["num_features", "num_layers", "num_heads", "dropout"]

        for key in architecture_keys:
            saved_value = saved_config.get(key)
            current_value = current_dict.get(key)
            if saved_value is not None and saved_value != current_value:
                raise RuntimeError(
                    "Saved model config incompatible with current config"
                )

    def predict(self, features: torch.Tensor) -> tuple[float, float]:
        """Predict (mu, sigma) for a single input sequence.

        Given a (36, 16) feature tensor, returns the predicted mean and
        standard deviation for the next bar's forward return.

        Args:
            features: Input tensor of shape (lookback_window, num_features),
                      typically (36, 16).

        Returns:
            Tuple of (mu, sigma) as Python floats for the next bar prediction.

        Raises:
            ValueError: If input feature dimension does not equal 16.
        """
        if features.dim() != 2:
            raise ValueError(
                f"Expected 2D input (seq_len, features), got {features.dim()}D"
            )

        feature_dim = features.shape[-1]
        if feature_dim != self._config.num_features:
            raise ValueError(
                f"Input feature dimension {feature_dim} != expected {self._config.num_features}"
            )

        x = features.unsqueeze(0)

        with torch.no_grad():
            mu, sigma = self._model(x)

        mu_val = mu[0, -1, 0].item()
        sigma_val = sigma[0, -1, 0].item()

        return mu_val, sigma_val

    def predict_batch(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict (mu, sigma) for a batch of input sequences.

        Given a (batch, 36, 16) feature tensor, returns the predicted mean
        and standard deviation tensors for the next bar's forward return.

        Args:
            features: Input tensor of shape (batch, lookback_window, num_features),
                      typically (batch, 36, 16).

        Returns:
            Tuple of (mu_tensor, sigma_tensor), each of shape (batch,),
            containing predictions for the last position of each sequence.

        Raises:
            ValueError: If input feature dimension does not equal 16.
        """
        if features.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq_len, features), got {features.dim()}D"
            )

        feature_dim = features.shape[-1]
        if feature_dim != self._config.num_features:
            raise ValueError(
                f"Input feature dimension {feature_dim} != expected {self._config.num_features}"
            )

        with torch.no_grad():
            mu, sigma = self._model(features)

        mu_out = mu[:, -1, 0]
        sigma_out = sigma[:, -1, 0]

        return mu_out, sigma_out
