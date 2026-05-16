"""DQNAdvisor, stateless advisory library for live mode inference.

Loads a trained Q-Network from a checkpoint and provides action
recommendations without any gRPC dependency. Designed to be imported
and used programmatically by calling agents.

Usage:
    from deepqnetwork import DQNAdvisor

    advisor = DQNAdvisor.from_checkpoint("checkpoints/dqn_episode_500.pt")
    result = advisor.recommend_action(state_vector)
    print(result.action_name, result.confidence)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from .network import QNetwork
from .preprocessor import StatePreprocessor
from .utils import resolve_device

logger = logging.getLogger(__name__)

# Canonical action names matching the environment's discrete action space
ACTION_NAMES: list[str] = ["HOLD", "BUY_1", "BUY_2", "SELL_1", "SELL_2"]


@dataclass
class ActionResult:
    """Result of an action recommendation.

    Attributes:
        action: Integer action index (0-4).
        action_name: Human-readable action name (e.g. "HOLD", "BUY_1").
        q_values: Q-values for all 5 actions, shape (5,).
        confidence: Confidence measure: max(Q) - mean(Q).
    """

    action: int
    action_name: str
    q_values: np.ndarray
    confidence: float


class DQNAdvisor:
    """Stateless advisory interface for DQN action recommendations.

    Loads a trained Q-Network checkpoint and provides inference-only
    methods for action selection. No gRPC connection, no replay buffer,
    no weight updates, purely local forward passes.

    Attributes:
        device: The torch device used for inference.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        config: dict | None = None,
    ) -> None:
        """Initialise the advisor from a checkpoint.

        Args:
            checkpoint_path: Path to a saved .pt checkpoint file.
            device: Device string ("cpu", "cuda", "mps", "auto").
            config: Optional config dict to override checkpoint config.
                    Useful for specifying network architecture if the
                    checkpoint doesn't contain a config key.

        Raises:
            FileNotFoundError: If checkpoint_path does not exist.
            RuntimeError: If checkpoint is incompatible with the network.
        """
        self._device = resolve_device(device)
        self._checkpoint_path = checkpoint_path

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=self._device, weights_only=False
        )

        # Resolve network config from checkpoint or override
        ckpt_config = checkpoint.get("config", {})
        if config is not None:
            ckpt_config.update(config)

        # Extract network architecture parameters
        hidden_dims = ckpt_config.get("hidden_dims", [256, 256, 128])
        activation = ckpt_config.get("activation", "relu")
        layer_norm = ckpt_config.get("layer_norm", True)
        dropout = ckpt_config.get("dropout", 0.0)
        dueling = ckpt_config.get("dueling", False)

        # Infer state_dim from the first layer's weight shape
        state_dict = checkpoint["q_network_state_dict"]
        first_layer_key = next(
            k for k in state_dict.keys() if k.endswith(".weight")
        )
        state_dim = state_dict[first_layer_key].shape[1]

        # Build Q-Network
        self._q_network = QNetwork(
            state_dim=state_dim,
            action_dim=5,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
            dueling=dueling,
        )

        # Load weights and set to eval mode
        self._q_network.load_state_dict(state_dict)
        self._q_network.to(self._device)
        self._q_network.eval()

        # Preprocessor for converting raw arrays to tensors
        self._preprocessor = StatePreprocessor(self._device)
        # Manually set state_dim so preprocessor doesn't require a protobuf observation
        self._preprocessor._state_dim = state_dim

        self._state_dim = state_dim

        logger.info(
            "DQNAdvisor loaded from '%s' (state_dim=%d, device=%s, dueling=%s)",
            checkpoint_path,
            state_dim,
            self._device,
            dueling,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, **kwargs) -> "DQNAdvisor":
        """Convenience factory for constructing a DQNAdvisor.

        Args:
            checkpoint_path: Path to a saved .pt checkpoint file.
            **kwargs: Additional keyword arguments passed to __init__
                (e.g. device="cuda", config={...}).

        Returns:
            A ready-to-use DQNAdvisor instance.
        """
        return cls(checkpoint_path=checkpoint_path, **kwargs)

    def recommend_action(self, state: np.ndarray | list[float]) -> ActionResult:
        """Recommend the best action for the given state.

        Performs a greedy forward pass through the Q-Network and returns
        the action with the highest Q-value along with metadata.

        Args:
            state: Raw observation vector as a numpy array or list of floats.
                Must have length equal to state_dim.

        Returns:
            ActionResult containing the recommended action index, name,
            all Q-values, and a confidence score.

        Raises:
            ValueError: If state length doesn't match expected state_dim.
        """
        state_array = np.asarray(state, dtype=np.float32)

        if state_array.shape != (self._state_dim,):
            raise ValueError(
                f"State has shape {state_array.shape}, expected ({self._state_dim},)."
            )

        # Convert to tensor and add batch dimension
        state_tensor = torch.from_numpy(state_array).unsqueeze(0).to(self._device)

        # Forward pass with no gradient computation
        with torch.no_grad():
            q_values_tensor = self._q_network(state_tensor)

        # Extract Q-values as numpy array
        q_values = q_values_tensor.squeeze(0).cpu().numpy()

        # Greedy action selection
        action = int(np.argmax(q_values))
        action_name = ACTION_NAMES[action]

        # Confidence: max(Q) - mean(Q)
        confidence = float(q_values.max() - q_values.mean())

        return ActionResult(
            action=action,
            action_name=action_name,
            q_values=q_values,
            confidence=confidence,
        )

    def get_action_probabilities(
        self, state: np.ndarray | list[float], temperature: float = 1.0
    ) -> np.ndarray:
        """Compute softmax probability distribution over actions.

        Useful for ensemble strategies or probabilistic action selection.

        Args:
            state: Raw observation vector as a numpy array or list of floats.
                Must have length equal to state_dim.
            temperature: Softmax temperature. Higher values produce more
                uniform distributions; lower values sharpen towards greedy.
                Must be > 0.

        Returns:
            Numpy array of shape (5,) with non-negative values summing to 1.0.

        Raises:
            ValueError: If state length doesn't match expected state_dim
                or temperature <= 0.
        """
        if temperature <= 0:
            raise ValueError(
                f"Temperature must be > 0, got {temperature}."
            )

        state_array = np.asarray(state, dtype=np.float32)

        if state_array.shape != (self._state_dim,):
            raise ValueError(
                f"State has shape {state_array.shape}, expected ({self._state_dim},)."
            )

        # Convert to tensor and add batch dimension
        state_tensor = torch.from_numpy(state_array).unsqueeze(0).to(self._device)

        # Forward pass with no gradient computation
        with torch.no_grad():
            q_values_tensor = self._q_network(state_tensor)

        # Apply temperature-scaled softmax
        scaled = q_values_tensor.squeeze(0) / temperature
        probabilities = torch.softmax(scaled, dim=0).cpu().numpy()

        return probabilities

    @property
    def state_dim(self) -> int:
        """The expected input state dimension."""
        return self._state_dim

    @property
    def device(self) -> torch.device:
        """The torch device used for inference."""
        return self._device
