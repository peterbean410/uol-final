"""Experience replay buffer for DQN training.

Stores experience tuples (state, action, reward, next_state, done) in a
fixed-capacity circular buffer. States are stored as float32 numpy arrays
to minimise memory usage. On sampling, transitions are converted to PyTorch
tensors and moved to the configured device.
"""

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor


class Transition(NamedTuple):
    """A single experience tuple."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-capacity circular replay buffer for experience tuples.

    Args:
        capacity: Maximum number of transitions to store (default: 300,000).
        device: PyTorch device for sampled tensors. Defaults to CPU if None.
    """

    def __init__(
        self, capacity: int = 300_000, device: torch.device | None = None
    ) -> None:
        self._capacity = capacity
        self._device = device if device is not None else torch.device("cpu")
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer.

        When at capacity, the oldest transition is overwritten (FIFO).

        Args:
            state: Current state as a numpy array (stored as float32).
            action: Action taken (integer index 0-4).
            reward: Reward received.
            next_state: Next state as a numpy array (stored as float32).
            done: Whether the episode terminated.
        """
        state_f32 = np.asarray(state, dtype=np.float32)
        next_state_f32 = np.asarray(next_state, dtype=np.float32)
        self._buffer.append(
            Transition(state_f32, action, reward, next_state_f32, done)
        )

    def sample(
        self, batch_size: int = 64
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample a uniform random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
            on the configured device with dtypes:
              - states: float32, shape (batch_size, state_dim)
              - actions: int64, shape (batch_size,)
              - rewards: float32, shape (batch_size,)
              - next_states: float32, shape (batch_size, state_dim)
              - dones: float32, shape (batch_size,)

        Raises:
            ValueError: If batch_size exceeds current buffer size.
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer "
                f"with only {len(self._buffer)} transitions."
            )

        batch = random.sample(list(self._buffer), batch_size)

        states = torch.tensor(
            np.array([t.state for t in batch]),
            dtype=torch.float32,
            device=self._device,
        )
        actions = torch.tensor(
            [t.action for t in batch],
            dtype=torch.int64,
            device=self._device,
        )
        rewards = torch.tensor(
            [t.reward for t in batch],
            dtype=torch.float32,
            device=self._device,
        )
        next_states = torch.tensor(
            np.array([t.next_state for t in batch]),
            dtype=torch.float32,
            device=self._device,
        )
        dones = torch.tensor(
            [float(t.done) for t in batch],
            dtype=torch.float32,
            device=self._device,
        )

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return len(self._buffer)
