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

        checkpoint = torch.load(
            checkpoint_path, map_location=self._device, weights_only=False
        )

        ckpt_config = checkpoint.get("config", {})
        if config is not None:
            ckpt_config.update(config)

        self._config = dict(ckpt_config)

        hidden_dims = ckpt_config.get("hidden_dims", [256, 256, 128])
        activation = ckpt_config.get("activation", "relu")
        dropout = ckpt_config.get("dropout", 0.0)

        state_dict = checkpoint["q_network_state_dict"]
        first_layer_key = next(
            k for k in state_dict.keys() if k.endswith(".weight")
        )
        state_dim = state_dict[first_layer_key].shape[1]

        self._q_network = QNetwork(
            state_dim=state_dim,
            action_dim=5,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )

        self._q_network.load_state_dict(state_dict)
        self._q_network.to(self._device)
        self._q_network.eval()

        self._preprocessor = StatePreprocessor(self._device)
        self._preprocessor._state_dim = state_dim

        self._state_dim = state_dim

        logger.info(
            "DQNAdvisor loaded from '%s' (state_dim=%d, device=%s)",
            checkpoint_path,
            state_dim,
            self._device,
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

    @property
    def training_window(self) -> dict[str, object]:
        """The hour-of-day / date window this model was trained on.

        Read from the persisted ``DQNConfig`` in the checkpoint, with the same
        defaults as ``DQNConfig`` for older checkpoints that predate a field.
        Returns ``date_start``/``date_end`` (ISO strings, ``""`` = unset) and
        ``hour_of_day_start``/``hour_of_day_end`` (ints; ``hour_end >= 24`` means
        the session rolls into the next day). Used by the DQNPF backtest to align
        its evaluation episodes with the trained sessions.
        """
        return self._training_window_from_config(self._config)

    @staticmethod
    def _training_window_from_config(config: dict) -> dict[str, object]:
        """Project a persisted ``DQNConfig`` dict onto the training-window fields.

        Applies the same defaults as ``DQNConfig`` so checkpoints predating a
        field still resolve. Shared by :attr:`training_window` and
        :meth:`read_training_window`.
        """
        return {
            "date_start": config.get("date_start", ""),
            "date_end": config.get("date_end", ""),
            "hour_of_day_start": int(config.get("hour_of_day_start", 0)),
            "hour_of_day_end": int(config.get("hour_of_day_end", 23)),
        }

    @classmethod
    def read_training_window(cls, checkpoint_path: str) -> dict[str, object]:
        """Read only the training window from a checkpoint (no network build).

        Loads just the persisted ``config`` (CPU map) and returns the same dict
        as :attr:`training_window`. Lets callers that need the trained session
        window, e.g. the DQNPF backtest component, to configure modelenv's
        session liquidation before the sidecar starts, avoid the cost of
        constructing a full advisor and Q-network.
        """
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        return cls._training_window_from_config(checkpoint.get("config", {}))

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

        state_tensor = torch.from_numpy(state_array).unsqueeze(0).to(self._device)

        with torch.no_grad():
            q_values_tensor = self._q_network(state_tensor)

        q_values = q_values_tensor.squeeze(0).cpu().numpy()

        action = int(np.argmax(q_values))
        action_name = ACTION_NAMES[action]

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

        state_tensor = torch.from_numpy(state_array).unsqueeze(0).to(self._device)

        with torch.no_grad():
            q_values_tensor = self._q_network(state_tensor)

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
