"""Property-based tests for the ReplayBuffer.

Uses Hypothesis to verify universal invariants of the ReplayBuffer module.
"""

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from deepqnetwork.replay_buffer import ReplayBuffer


# Feature: deepqnetwork, Property 8: Replay buffer capacity invariant


@settings(max_examples=100)
@given(
    capacity=st.integers(min_value=1, max_value=500),
    num_pushes=st.integers(min_value=0, max_value=1000),
    state_dim=st.integers(min_value=1, max_value=20),
)
def test_replay_buffer_size_equals_min_n_c(
    capacity: int, num_pushes: int, state_dim: int
) -> None:
    """For any sequence of N push operations on a buffer with capacity C,
    the buffer size SHALL equal min(N, C).

    **Validates: Requirements 5.1, 5.2, 5.5**
    """
    buf = ReplayBuffer(capacity=capacity)

    for i in range(num_pushes):
        state = np.full(state_dim, float(i), dtype=np.float32)
        next_state = np.full(state_dim, float(i + 1), dtype=np.float32)
        buf.push(state, i % 5, float(i) * 0.1, next_state, i % 2 == 0)

    expected_size = min(num_pushes, capacity)
    assert len(buf) == expected_size, (
        f"After {num_pushes} pushes to buffer with capacity {capacity}, "
        f"expected size {expected_size}, got {len(buf)}"
    )


@settings(max_examples=100)
@given(
    capacity=st.integers(min_value=1, max_value=200),
    extra_pushes=st.integers(min_value=1, max_value=300),
    state_dim=st.integers(min_value=1, max_value=20),
)
def test_replay_buffer_fifo_order_when_over_capacity(
    capacity: int, extra_pushes: int, state_dim: int
) -> None:
    """When N > C, the buffer SHALL contain only the most recent C transitions
    in FIFO order.

    **Validates: Requirements 5.1, 5.2, 5.5**
    """
    num_pushes = capacity + extra_pushes  # Guarantee N > C
    buf = ReplayBuffer(capacity=capacity)

    for i in range(num_pushes):
        state = np.full(state_dim, float(i), dtype=np.float32)
        next_state = np.full(state_dim, float(i + 1), dtype=np.float32)
        buf.push(state, i % 5, float(i) * 0.1, next_state, i % 2 == 0)

    # Buffer should contain the most recent `capacity` transitions
    # The oldest remaining transition should be at index (num_pushes - capacity)
    start_index = num_pushes - capacity

    ordered_states = buf._ordered_states()
    for buf_idx in range(capacity):
        original_idx = start_index + buf_idx
        expected_state = np.full(state_dim, float(original_idx), dtype=np.float32)
        actual_state = ordered_states[buf_idx]

        assert np.array_equal(actual_state, expected_state), (
            f"At buffer position {buf_idx}, expected state with value "
            f"{float(original_idx)}, got {actual_state[0]}. "
            f"(N={num_pushes}, C={capacity}, start_index={start_index})"
        )


@settings(max_examples=100)
@given(
    capacity=st.integers(min_value=1, max_value=500),
    num_pushes=st.integers(min_value=0, max_value=1000),
    state_dim=st.integers(min_value=1, max_value=20),
)
def test_replay_buffer_never_exceeds_capacity(
    capacity: int, num_pushes: int, state_dim: int
) -> None:
    """The buffer SHALL never exceed capacity C regardless of how many
    transitions are pushed.

    **Validates: Requirements 5.1, 5.2, 5.5**
    """
    buf = ReplayBuffer(capacity=capacity)

    for i in range(num_pushes):
        state = np.full(state_dim, float(i), dtype=np.float32)
        next_state = np.full(state_dim, float(i + 1), dtype=np.float32)
        buf.push(state, i % 5, float(i) * 0.1, next_state, i % 2 == 0)

        # Check invariant after every push
        assert len(buf) <= capacity, (
            f"Buffer exceeded capacity after push {i + 1}: "
            f"size={len(buf)}, capacity={capacity}"
        )


# Feature: deepqnetwork, Property 9: Replay buffer sample correctness


@settings(max_examples=100)
@given(
    state_dim=st.integers(min_value=1, max_value=100),
    num_transitions=st.integers(min_value=1, max_value=200),
    batch_size_ratio=st.floats(min_value=0.1, max_value=1.0),
)
def test_replay_buffer_sample_shapes_correct(
    state_dim: int, num_transitions: int, batch_size_ratio: float
) -> None:
    """For any buffer containing at least batch_size transitions,
    sample(batch_size) SHALL return tensors of the correct shapes:
    states (batch_size, state_dim), actions (batch_size,),
    rewards (batch_size,), next_states (batch_size, state_dim),
    dones (batch_size,).

    **Validates: Requirements 5.3, 5.4**
    """
    batch_size = max(1, int(num_transitions * batch_size_ratio))
    capacity = max(num_transitions, batch_size)

    buf = ReplayBuffer(capacity=capacity)

    for i in range(num_transitions):
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        action = int(np.random.randint(0, 5))
        reward = float(np.random.uniform(-1.0, 1.0))
        done = bool(np.random.randint(0, 2))
        buf.push(state, action, reward, next_state, done)

    states, actions, rewards, next_states, dones = buf.sample(batch_size)

    assert states.shape == (batch_size, state_dim)
    assert actions.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert next_states.shape == (batch_size, state_dim)
    assert dones.shape == (batch_size,)


@settings(max_examples=100)
@given(
    state_dim=st.integers(min_value=1, max_value=100),
    num_transitions=st.integers(min_value=1, max_value=200),
    batch_size_ratio=st.floats(min_value=0.1, max_value=1.0),
)
def test_replay_buffer_sample_dtypes_correct(
    state_dim: int, num_transitions: int, batch_size_ratio: float
) -> None:
    """For any buffer containing at least batch_size transitions,
    sample(batch_size) SHALL return tensors with correct dtypes:
    states float32, actions int64, rewards float32,
    next_states float32, dones float32.

    **Validates: Requirements 5.3, 5.4**
    """
    batch_size = max(1, int(num_transitions * batch_size_ratio))
    capacity = max(num_transitions, batch_size)

    buf = ReplayBuffer(capacity=capacity)

    for i in range(num_transitions):
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        action = int(np.random.randint(0, 5))
        reward = float(np.random.uniform(-1.0, 1.0))
        done = bool(np.random.randint(0, 2))
        buf.push(state, action, reward, next_state, done)

    states, actions, rewards, next_states, dones = buf.sample(batch_size)

    assert states.dtype == torch.float32
    assert actions.dtype == torch.int64
    assert rewards.dtype == torch.float32
    assert next_states.dtype == torch.float32
    assert dones.dtype == torch.float32


@settings(max_examples=100)
@given(
    state_dim=st.integers(min_value=1, max_value=100),
    num_transitions=st.integers(min_value=1, max_value=200),
    batch_size_ratio=st.floats(min_value=0.1, max_value=1.0),
)
def test_replay_buffer_sample_device_correct(
    state_dim: int, num_transitions: int, batch_size_ratio: float
) -> None:
    """For any buffer containing at least batch_size transitions,
    all sampled tensors SHALL be on the configured device.

    **Validates: Requirements 5.3, 5.4**
    """
    batch_size = max(1, int(num_transitions * batch_size_ratio))
    capacity = max(num_transitions, batch_size)
    device = torch.device("cpu")

    buf = ReplayBuffer(capacity=capacity, device=device)

    for i in range(num_transitions):
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        action = int(np.random.randint(0, 5))
        reward = float(np.random.uniform(-1.0, 1.0))
        done = bool(np.random.randint(0, 2))
        buf.push(state, action, reward, next_state, done)

    states, actions, rewards, next_states, dones = buf.sample(batch_size)

    assert states.device == device
    assert actions.device == device
    assert rewards.device == device
    assert next_states.device == device
    assert dones.device == device
