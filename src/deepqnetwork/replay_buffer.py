"""Experience replay buffer for DQN training.

Stores experience tuples (state, action, reward, next_state, done) in a
fixed-capacity circular buffer backed by preallocated NumPy arrays. States are
stored as float32 to minimise memory usage. On sampling, a uniform random
mini-batch (without replacement) is drawn by integer index (O(batch_size),
independent of the buffer size) converted to PyTorch tensors, and moved to the
configured device.
"""

import random

import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:
    """Fixed-capacity circular replay buffer for experience tuples.

    Backed by preallocated, contiguous NumPy arrays (a ring buffer). Pushing is
    O(1) and sampling is O(batch_size) (it draws random indices rather than
    copying the whole buffer) so per-step cost does not grow as the buffer
    fills. The state arrays are allocated lazily on the first push, once the
    state dimension is known.

    Args:
        capacity: Maximum number of transitions to store (default: 300,000).
        device: PyTorch device for sampled tensors. Defaults to CPU if None.
    """

    def __init__(
        self, capacity: int = 300_000, device: torch.device | None = None
    ) -> None:
        self._capacity = capacity
        self._device = device if device is not None else torch.device("cpu")

        # Ring-buffer bookkeeping.
        self._size = 0  # number of valid transitions, == min(pushes, capacity)
        self._pos = 0  # next write index (and, when full, the oldest entry)

        # State arrays are allocated lazily on first push (state dim unknown
        # until then). Scalar columns can be allocated up front.
        self._states: np.ndarray | None = None
        self._next_states: np.ndarray | None = None
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._dones = np.empty(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer.

        When at capacity, the oldest transition is overwritten (FIFO). Values
        are copied into the buffer's own storage, so the caller may safely
        mutate or reuse the input arrays afterwards.

        Args:
            state: Current state as a numpy array (stored as float32).
            action: Action taken (integer index 0-4).
            reward: Reward received.
            next_state: Next state as a numpy array (stored as float32).
            done: Whether the episode terminated.
        """
        state_f32 = np.asarray(state, dtype=np.float32)
        next_state_f32 = np.asarray(next_state, dtype=np.float32)

        if self._states is None:
            self._states = np.empty(
                (self._capacity, *state_f32.shape), dtype=np.float32
            )
            self._next_states = np.empty(
                (self._capacity, *next_state_f32.shape), dtype=np.float32
            )

        i = self._pos
        self._states[i] = state_f32
        self._next_states[i] = next_state_f32
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = float(done)

        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(
        self, batch_size: int = 64
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample a uniform random mini-batch of transitions.

        Draws ``batch_size`` distinct transitions uniformly at random (without
        replacement) in O(batch_size) time, independent of the buffer size.

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
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer "
                f"with only {self._size} transitions."
            )

        # O(batch_size) selection of distinct indices into the valid region.
        idx = np.fromiter(
            random.sample(range(self._size), batch_size),
            dtype=np.int64,
            count=batch_size,
        )

        # Fancy indexing produces fresh, contiguous copies the tensors can own.
        states = torch.from_numpy(self._states[idx]).to(self._device)
        actions = torch.from_numpy(self._actions[idx]).to(self._device)
        rewards = torch.from_numpy(self._rewards[idx]).to(self._device)
        next_states = torch.from_numpy(self._next_states[idx]).to(self._device)
        dones = torch.from_numpy(self._dones[idx]).to(self._device)

        return states, actions, rewards, next_states, dones

    def _ordered_states(self) -> np.ndarray:
        """Return stored states in logical FIFO order (oldest first).

        Inspection/test helper. The ring buffer keeps the most recent
        ``min(pushes, capacity)`` states; this reconstructs their oldest→newest
        ordering from the underlying storage.
        """
        if self._states is None or self._size == 0:
            return np.empty((0, 0), dtype=np.float32)
        if self._size < self._capacity:
            return self._states[: self._size].copy()
        # Full buffer: the oldest entry sits at the next write position.
        return np.concatenate(
            [self._states[self._pos :], self._states[: self._pos]]
        )

    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return self._size
