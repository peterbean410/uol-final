"""KServe custom predictor for the ProbabilisticTransformer.

Serves the trained model via a REST endpoint that accepts feature tensors
and returns Gaussian distribution parameters (mu, sigma) for forex forecasting.
"""

import math
from typing import Dict

import kserve
import numpy as np
import torch

from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.model import ProbabilisticTransformer


class ForecasterPredictor(kserve.Model):
    """KServe custom predictor for the ProbabilisticTransformer.

    Accepts JSON payloads containing feature tensors of shape (lookback_window, 16)
    and returns predicted mu (mean) and sigma (standard deviation) values.
    """

    def __init__(self, name: str, model_path: str):
        """Initialize the predictor.

        Args:
            name: Model name for KServe registration.
            model_path: Path to the model checkpoint file (local or S3).
        """
        super().__init__(name)
        self.model_path = model_path
        self.model: ProbabilisticTransformer | None = None
        self.config: ForecasterConfig | None = None
        self.ready = False

    def load(self):
        """Load model weights from S3 artifact store.

        Downloads the checkpoint, instantiates ProbabilisticTransformer with
        the saved config, and loads the state dict. Sets the model to eval mode.

        Raises:
            FileNotFoundError: If model_path does not exist.
            KeyError: If checkpoint is missing required keys.
        """
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.config = ForecasterConfig(**checkpoint["config"])
        self.model = ProbabilisticTransformer(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.ready = True

    def predict(self, payload: Dict, headers: Dict = None) -> Dict:
        """Run inference on input feature tensor.

        Args:
            payload: JSON payload with structure:
                {"instances": [[...feature_values...]]}
                Feature tensor shape: (lookback_window, 16) for single instance
                or (batch, lookback_window, 16) for batch inference.
            headers: Optional HTTP headers (unused).

        Returns:
            Dictionary with structure:
                {"predictions": [{"mu": float, "sigma": float}, ...]}

        Raises:
            ValueError: If payload is malformed, has wrong shape, or contains
                non-finite values. KServe maps these to HTTP 400 responses.
        """
        instances = payload.get("instances")
        if instances is None:
            raise ValueError("Missing 'instances' field in request payload")

        if not isinstance(instances, (list, np.ndarray)):
            raise ValueError(
                "'instances' must be a list or array, "
                f"got {type(instances).__name__}"
            )

        try:
            tensor = torch.tensor(instances, dtype=torch.float32)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to convert 'instances' to numeric tensor: {e}"
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                "Input contains non-finite values (NaN or Inf). "
                "All feature values must be finite floats."
            )

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() != 3:
            raise ValueError(
                f"Expected 2D or 3D input tensor, got {tensor.dim()}D. "
                "Shape must be (lookback_window, 16) or (batch, lookback_window, 16)."
            )

        if tensor.shape[-1] != 16:
            raise ValueError(
                f"Expected 16 features in last dimension, got {tensor.shape[-1]}. "
                "Input shape must be (lookback_window, 16) or (batch, lookback_window, 16)."
            )

        if tensor.shape[1] != self.config.lookback_window:
            raise ValueError(
                f"Expected sequence length (lookback_window) of "
                f"{self.config.lookback_window}, got {tensor.shape[1]}. "
                f"Input must have exactly {self.config.lookback_window} time steps."
            )

        with torch.no_grad():
            mu, sigma = self.model(tensor)

        mu_vals = mu[:, -1, 0].numpy().tolist()
        sigma_vals = sigma[:, -1, 0].numpy().tolist()

        predictions = []
        for m, s in zip(mu_vals, sigma_vals):
            if not math.isfinite(m) or not math.isfinite(s):
                raise ValueError(
                    "Model produced non-finite output. This may indicate "
                    "corrupted model weights or extreme input values."
                )
            predictions.append({"mu": m, "sigma": s})

        return {"predictions": predictions}
