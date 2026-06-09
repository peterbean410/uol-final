"""DQN Agent core logic for training.

Orchestrates Q-Network, Target Network, Replay Buffer, epsilon-greedy
exploration, Bellman target computation, and gradient updates.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from deepqnetwork.config import DQNConfig
from deepqnetwork.network import QNetwork
from deepqnetwork.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class DQNAgent:
    """Deep Q-Network agent for training.

    Orchestrates the Q-Network, Target Network, Replay Buffer, epsilon
    schedule, and gradient updates using the standard DQN algorithm.

    Args:
        config: DQN configuration dataclass.
        device: PyTorch device for computation.
        state_dim: Dimension of the state vector. If not provided, defaults to 53.
    """

    def __init__(
        self,
        config: DQNConfig,
        device: torch.device,
        state_dim: int = 53,
    ) -> None:
        self.config = config
        self.device = device
        self.state_dim = state_dim
        self.action_dim = 5  # HOLD, BUY_1, BUY_2, SELL_1, SELL_2

        # Epsilon schedule state
        self.epsilon = config.epsilon_start
        self.step_count = 0

        # Build Q-Network (online)
        self.q_network = QNetwork(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
            dueling=config.dueling,
        ).to(device)

        # Build Target Network (copy of Q-Network, frozen)
        self.target_network = QNetwork(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
            dueling=config.dueling,
        ).to(device)

        # Copy weights from Q-Network to Target Network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Freeze Target Network; no gradient computation
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.replay_buffer_size,
            device=device,
        )

        # Adam optimiser on Q-Network parameters
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

        # Loss function
        if config.loss_function == "huber":
            self.loss_fn = nn.SmoothL1Loss()
        elif config.loss_function == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(
                f"Unsupported loss function '{config.loss_function}'. "
                f"Choose from: 'huber', 'mse'."
            )

        logger.info(
            "DQNAgent initialised: state_dim=%d, action_dim=%d, device=%s, "
            "epsilon=%.4f, loss=%s, dueling=%s",
            state_dim,
            self.action_dim,
            device,
            self.epsilon,
            config.loss_function,
            config.dueling,
        )

    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Select an action using epsilon-greedy (training) or greedy (eval).

        Args:
            state: State tensor of shape (state_dim,) or (1, state_dim).
            training: If True, use epsilon-greedy; if False, use greedy.

        Returns:
            Selected action index (0-4).
        """
        if training and random.random() < self.epsilon:
            # Random exploration
            return random.randint(0, self.action_dim - 1)

        # Greedy action selection
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)  # (1, state_dim)
            q_values = self.q_network(state)
            return int(q_values.argmax(dim=1).item())

    def update(self) -> float | None:
        """Perform a single gradient update step.

        Samples a batch from the replay buffer, computes the Bellman target,
        calculates loss, and performs backpropagation with gradient clipping.

        Returns:
            The loss value as a float, or None if the buffer doesn't have
            enough transitions for a batch.
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )

        # Compute current Q-values for the taken actions
        # q_values shape: (batch_size, action_dim)
        q_values = self.q_network(states)
        # Gather Q-values for the actions that were actually taken
        # actions shape: (batch_size,) → (batch_size, 1) for gather
        q_values_for_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute Bellman target (Double DQN when use_double is True):
        #   y = r + γ * Q_target(s', argmax_a' Q_online(s', a')) * (1 - done)
        # Double DQN eliminates the maximisation bias of vanilla DQN by
        # decoupling action selection (online net) from evaluation (target net).
        with torch.no_grad():
            if self.config.use_double:
                # Select the best action using the online (q_) network
                online_next_q = self.q_network(next_states)
                best_actions = online_next_q.argmax(dim=1)
                # Evaluate that action using the frozen target network
                next_q_values = self.target_network(next_states)
                max_next_q = next_q_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_network(next_states)
                max_next_q = next_q_values.max(dim=1).values
            targets = rewards + self.config.gamma * max_next_q * (1.0 - dones)

        # Compute loss
        loss = self.loss_fn(q_values_for_actions, targets)

        # Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.grad_clip_norm
        )
        self.optimizer.step()

        return loss.item()

    def sync_target(self) -> None:
        """Synchronise Target Network with Q-Network.

        Uses soft update (Polyak averaging) with coefficient tau.
        When tau=1.0, this is a hard copy of all parameters.
        When tau<1.0, blends: target = tau * online + (1 - tau) * target.
        """
        tau = self.config.tau

        if tau == 1.0:
            # Hard copy
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            # Soft update (Polyak averaging)
            for target_param, online_param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_param.data.copy_(
                    tau * online_param.data + (1.0 - tau) * target_param.data
                )

        # Ensure target remains frozen after sync
        for param in self.target_network.parameters():
            param.requires_grad = False

    def step_epsilon(self) -> None:
        """Decay epsilon linearly from epsilon_start to epsilon_end.

        Decreases epsilon by (epsilon_start - epsilon_end) / epsilon_decay_steps
        per call, stopping at epsilon_end.
        """
        # epsilon_decay_steps <= 0 is the unresolved "auto" sentinel (train()
        # normally derives a positive horizon from the step budget before the
        # loop runs). Treat it as no decay rather than dividing by zero.
        if self.config.epsilon_decay_steps > 0 and self.epsilon > self.config.epsilon_end:
            decay = (
                (self.config.epsilon_start - self.config.epsilon_end)
                / self.config.epsilon_decay_steps
            )
            self.epsilon = max(self.config.epsilon_end, self.epsilon - decay)

        self.step_count += 1
