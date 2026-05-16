"""Unit tests for the ReplayBuffer."""

import numpy as np
import pytest
import torch

from deepqnetwork.replay_buffer import ReplayBuffer


class TestReplayBufferInit:
    """Tests for buffer initialization."""

    def test_default_capacity(self):
        buf = ReplayBuffer()
        assert len(buf) == 0
        assert buf._capacity == 300_000

    def test_custom_capacity(self):
        buf = ReplayBuffer(capacity=100)
        assert buf._capacity == 100

    def test_default_device_is_cpu(self):
        buf = ReplayBuffer()
        assert buf._device == torch.device("cpu")

    def test_custom_device(self):
        device = torch.device("cpu")
        buf = ReplayBuffer(device=device)
        assert buf._device == device


class TestReplayBufferPush:
    """Tests for push method."""

    def test_push_increases_length(self):
        buf = ReplayBuffer(capacity=10)
        state = np.zeros(5, dtype=np.float32)
        buf.push(state, 0, 1.0, state, False)
        assert len(buf) == 1

    def test_push_stores_float32(self):
        buf = ReplayBuffer(capacity=10)
        state = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        buf.push(state, 1, 0.5, state, True)
        stored = buf._buffer[0]
        assert stored.state.dtype == np.float32
        assert stored.next_state.dtype == np.float32

    def test_fifo_overwrite_at_capacity(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            state = np.array([float(i)], dtype=np.float32)
            buf.push(state, 0, 0.0, state, False)
        # Buffer should contain items 2, 3, 4 (oldest 0, 1 overwritten)
        assert len(buf) == 3
        assert buf._buffer[0].state[0] == 2.0
        assert buf._buffer[1].state[0] == 3.0
        assert buf._buffer[2].state[0] == 4.0

    def test_push_preserves_values(self):
        buf = ReplayBuffer(capacity=10)
        state = np.array([1.5, -2.3, 0.0], dtype=np.float32)
        next_state = np.array([3.1, 4.2, -1.0], dtype=np.float32)
        buf.push(state, 2, -0.5, next_state, True)
        t = buf._buffer[0]
        np.testing.assert_array_almost_equal(t.state, state)
        assert t.action == 2
        assert t.reward == -0.5
        np.testing.assert_array_almost_equal(t.next_state, next_state)
        assert t.done is True


class TestReplayBufferSample:
    """Tests for sample method."""

    def _fill_buffer(self, buf, n, state_dim=5):
        for i in range(n):
            state = np.random.randn(state_dim).astype(np.float32)
            next_state = np.random.randn(state_dim).astype(np.float32)
            buf.push(state, i % 5, float(i) * 0.1, next_state, i % 2 == 0)

    def test_sample_returns_correct_shapes(self):
        buf = ReplayBuffer(capacity=100)
        self._fill_buffer(buf, 50, state_dim=10)
        states, actions, rewards, next_states, dones = buf.sample(16)
        assert states.shape == (16, 10)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
        assert next_states.shape == (16, 10)
        assert dones.shape == (16,)

    def test_sample_returns_correct_dtypes(self):
        buf = ReplayBuffer(capacity=100)
        self._fill_buffer(buf, 50)
        states, actions, rewards, next_states, dones = buf.sample(8)
        assert states.dtype == torch.float32
        assert actions.dtype == torch.int64
        assert rewards.dtype == torch.float32
        assert next_states.dtype == torch.float32
        assert dones.dtype == torch.float32

    def test_sample_on_configured_device(self):
        device = torch.device("cpu")
        buf = ReplayBuffer(capacity=100, device=device)
        self._fill_buffer(buf, 50)
        states, actions, rewards, next_states, dones = buf.sample(8)
        assert states.device == device
        assert actions.device == device
        assert rewards.device == device
        assert next_states.device == device
        assert dones.device == device

    def test_sample_raises_when_buffer_too_small(self):
        buf = ReplayBuffer(capacity=100)
        self._fill_buffer(buf, 5)
        with pytest.raises(ValueError, match="Cannot sample"):
            buf.sample(10)

    def test_sample_default_batch_size(self):
        buf = ReplayBuffer(capacity=200)
        self._fill_buffer(buf, 100)
        states, actions, rewards, next_states, dones = buf.sample()
        assert states.shape[0] == 64


class TestReplayBufferLen:
    """Tests for __len__ method."""

    def test_empty_buffer(self):
        buf = ReplayBuffer(capacity=10)
        assert len(buf) == 0

    def test_partially_filled(self):
        buf = ReplayBuffer(capacity=100)
        state = np.zeros(3, dtype=np.float32)
        for _ in range(42):
            buf.push(state, 0, 0.0, state, False)
        assert len(buf) == 42

    def test_at_capacity(self):
        buf = ReplayBuffer(capacity=10)
        state = np.zeros(3, dtype=np.float32)
        for _ in range(10):
            buf.push(state, 0, 0.0, state, False)
        assert len(buf) == 10

    def test_over_capacity(self):
        buf = ReplayBuffer(capacity=10)
        state = np.zeros(3, dtype=np.float32)
        for _ in range(25):
            buf.push(state, 0, 0.0, state, False)
        assert len(buf) == 10
