"""State preprocessing for the DQN trading agent.

Converts protobuf Observation messages into PyTorch tensors suitable for
the Q-Network. No normalisation is applied; the model environment already
provides normalised state data.
"""

import logging
import warnings

import numpy as np
import torch

logger = logging.getLogger(__name__)


class StatePreprocessor:
    """Converts Observation protobuf messages to float32 tensors.

    Extracts the first StateRow's values from an Observation, validates
    dimensions, handles NaN/Inf defensively, and places the resulting
    tensor on the configured device.

    Attributes:
        device: The torch device tensors are placed on.
    """

    def __init__(self, device: torch.device) -> None:
        """Initialise the preprocessor.

        Args:
            device: The torch device to place output tensors on.
        """
        self._device = device
        self._state_dim: int | None = None

    def process(self, observation) -> torch.Tensor:
        """Extract state_data[0].values into a float32 tensor on device.

        On the first call, initialises state_dim from the observation's
        state_columns length. Subsequent calls validate that the state_data
        values length matches the expected state_dim.

        Args:
            observation: A protobuf Observation message with state_columns
                and state_data fields.

        Returns:
            A 1-D float32 tensor of shape (state_dim,) on the configured device.

        Raises:
            ValueError: If state_data is empty or its length doesn't match
                the expected state_dim.
        """
        # Validate state_data is present
        if not observation.state_data:
            raise ValueError(
                "Observation has empty state_data; cannot extract state vector."
            )

        # Extract values from the first StateRow
        values = list(observation.state_data[0].values)

        # Initialise state_dim on first observation
        if self._state_dim is None:
            self._state_dim = len(observation.state_columns)
            logger.info(
                "Initialised state_dim=%d from state_columns.", self._state_dim
            )

        # Validate length matches expected state_dim
        if len(values) != self._state_dim:
            raise ValueError(
                f"state_data values length {len(values)} does not match "
                f"expected state_dim {self._state_dim}."
            )

        # Convert to 1-D numpy array
        state_array = np.array(values, dtype=np.float32)

        # Handle NaN/Inf defensively
        nan_mask = np.isnan(state_array)
        inf_mask = np.isinf(state_array)
        if nan_mask.any() or inf_mask.any():
            num_nan = int(nan_mask.sum())
            num_inf = int(inf_mask.sum())
            logger.warning(
                "State contains %d NaN and %d Inf values; replacing with 0.0.",
                num_nan,
                num_inf,
            )
            state_array[nan_mask | inf_mask] = 0.0

        # Convert to tensor on device
        tensor = torch.from_numpy(state_array).to(self._device)
        return tensor

    @property
    def state_dim(self) -> int:
        """Feature dimension, set on first observation.

        Raises:
            RuntimeError: If accessed before processing the first observation.
        """
        if self._state_dim is None:
            raise RuntimeError(
                "state_dim is not yet initialised; call process() with an "
                "observation first."
            )
        return self._state_dim
