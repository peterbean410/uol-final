"""Q-Network architecture for the DQN trading agent.

Supports both standard and dueling architectures with configurable
hidden dimensions, activation functions, layer normalisation, and dropout.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation(name: str) -> nn.Module:
    """Return an activation module by name.

    Args:
        name: One of "relu", "leaky_relu", "gelu".

    Returns:
        The corresponding nn.Module activation.

    Raises:
        ValueError: If the activation name is not recognised.
    """
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "gelu": nn.GELU(),
    }
    if name not in activations:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Choose from: {list(activations.keys())}"
        )
    return activations[name]


class QNetwork(nn.Module):
    """Q-Network that maps state vectors to action Q-values.

    Supports standard feedforward and dueling architectures.

    Architecture (standard):
        Input(state_dim)
          → Linear → LayerNorm → Activation → Dropout  (per hidden layer)
          → Linear(hidden_dims[-1], action_dim)  [no activation]

    Architecture (dueling):
        Input(state_dim)
          → [shared hidden layers, excluding final output]
          → Value stream:  Linear(hidden_dims[-1], 1)
          → Advantage stream: Linear(hidden_dims[-1], action_dim)
          → Q = V + (A - mean(A))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 5,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        layer_norm: bool = True,
        dropout: float = 0.0,
        dueling: bool = False,
    ) -> None:
        """Initialise the Q-Network.

        Args:
            state_dim: Dimension of the input state vector.
            action_dim: Number of discrete actions (default: 5).
            hidden_dims: List of hidden layer sizes (default: [256, 256, 128]).
            activation: Activation function name ("relu", "leaky_relu", "gelu").
            layer_norm: Whether to apply layer normalisation after each hidden layer.
            dropout: Dropout rate after each activation (0.0 disables dropout).
            dueling: Whether to use dueling architecture.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.dueling = dueling

        # Build shared hidden layers
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(_get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim

        self.shared_layers = nn.Sequential(*layers)

        if dueling:
            # Value stream: outputs scalar V(s)
            self.value_stream = nn.Linear(hidden_dims[-1], 1)
            # Advantage stream: outputs A(s, a) for each action
            self.advantage_stream = nn.Linear(hidden_dims[-1], action_dim)
            self._init_output_layer(self.value_stream)
            self._init_output_layer(self.advantage_stream)
        else:
            # Standard output layer
            self.output_layer = nn.Linear(hidden_dims[-1], action_dim)
            self._init_output_layer(self.output_layer)

        # Initialise hidden layers
        self._init_hidden_layers()

    def _init_hidden_layers(self) -> None:
        """Apply Kaiming uniform init to hidden layers, biases zeroed."""
        for module in self.shared_layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _init_output_layer(layer: nn.Linear) -> None:
        """Apply uniform[-3e-3, 3e-3] init to an output layer."""
        nn.init.uniform_(layer.weight, -3e-3, 3e-3)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for the given state batch.

        Args:
            state: Input tensor of shape (batch_size, state_dim).

        Returns:
            Q-values tensor of shape (batch_size, action_dim).
        """
        features = self.shared_layers(state)

        if self.dueling:
            value = self.value_stream(features)  # (batch_size, 1)
            advantage = self.advantage_stream(features)  # (batch_size, action_dim)
            # Q = V + (A - mean(A))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.output_layer(features)

        return q_values
