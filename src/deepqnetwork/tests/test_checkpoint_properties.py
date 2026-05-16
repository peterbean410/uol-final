"""Property-based tests for the CheckpointManager.

Uses Hypothesis to verify universal invariants of the CheckpointManager module.
"""

from __future__ import annotations

import copy
import re
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st

from deepqnetwork.checkpoint_manager import CheckpointManager
from deepqnetwork.network import QNetwork


# Feature: deepqnetwork, Property 13: Checkpoint save/load round-trip


@settings(max_examples=100, deadline=None)
@given(
    state_dim=st.integers(min_value=10, max_value=53),
    hidden_dims=st.lists(
        st.integers(min_value=16, max_value=64), min_size=1, max_size=3
    ),
    epsilon=st.floats(min_value=0.0, max_value=1.0),
    step_count=st.integers(min_value=0, max_value=100_000),
    episode_count=st.integers(min_value=0, max_value=5000),
    learning_rate=st.floats(min_value=1e-5, max_value=1e-2),
)
def test_checkpoint_save_load_round_trip(
    state_dim: int,
    hidden_dims: list[int],
    epsilon: float,
    step_count: int,
    episode_count: int,
    learning_rate: float,
) -> None:
    """For any agent state (Q_Network weights, Target_Network weights,
    optimizer state, epsilon, step count, episode count), saving a checkpoint
    and loading it back SHALL restore all values identically.

    **Validates: Requirements 8.1, 8.6**
    """
    # Create Q-Network and Target Network with random weights
    q_network = QNetwork(state_dim=state_dim, action_dim=5, hidden_dims=hidden_dims)
    target_network = QNetwork(state_dim=state_dim, action_dim=5, hidden_dims=hidden_dims)

    # Randomise weights by running a forward pass with random data and backprop
    # to ensure optimizer state is non-trivial
    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
    dummy_input = torch.randn(4, state_dim)
    dummy_target = torch.randn(4, 5)
    loss = nn.functional.mse_loss(q_network(dummy_input), dummy_target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Capture original state dicts
    original_q_state = {
        k: v.clone() for k, v in q_network.state_dict().items()
    }
    original_target_state = {
        k: v.clone() for k, v in target_network.state_dict().items()
    }
    original_optimizer_state = _deep_copy_optimizer_state(optimizer.state_dict())

    # Save checkpoint using a temporary directory (no S3)
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = CheckpointManager(checkpoint_dir=tmp_dir, s3_prefix=None)

        saved_path = manager.save(
            episode=episode_count,
            q_network=q_network,
            target_network=target_network,
            optimizer=optimizer,
            epsilon=epsilon,
            step_count=step_count,
        )

        # Load checkpoint back
        loaded = manager.load(saved_path)

    # Verify all values are restored identically
    assert loaded != {}, "Checkpoint load returned empty dict"

    # Verify epsilon
    assert loaded["epsilon"] == epsilon, (
        f"Epsilon mismatch: expected {epsilon}, got {loaded['epsilon']}"
    )

    # Verify step_count
    assert loaded["step_count"] == step_count, (
        f"step_count mismatch: expected {step_count}, got {loaded['step_count']}"
    )

    # Verify episode_count
    assert loaded["episode_count"] == episode_count, (
        f"episode_count mismatch: expected {episode_count}, "
        f"got {loaded['episode_count']}"
    )

    # Verify Q-Network state dict
    loaded_q_state = loaded["q_network_state_dict"]
    for key in original_q_state:
        assert key in loaded_q_state, f"Missing Q-Network key: {key}"
        assert torch.equal(original_q_state[key], loaded_q_state[key]), (
            f"Q-Network state mismatch for key '{key}'"
        )

    # Verify Target Network state dict
    loaded_target_state = loaded["target_network_state_dict"]
    for key in original_target_state:
        assert key in loaded_target_state, f"Missing Target Network key: {key}"
        assert torch.equal(original_target_state[key], loaded_target_state[key]), (
            f"Target Network state mismatch for key '{key}'"
        )

    # Verify optimizer state dict
    loaded_optimizer_state = loaded["optimizer_state_dict"]
    _assert_optimizer_states_equal(
        original_optimizer_state, loaded_optimizer_state
    )


def _deep_copy_optimizer_state(state_dict: dict) -> dict:
    """Create a deep copy of an optimizer state dict, cloning tensors."""
    copied = {"state": {}, "param_groups": copy.deepcopy(state_dict["param_groups"])}
    for param_id, param_state in state_dict["state"].items():
        copied["state"][param_id] = {}
        for k, v in param_state.items():
            if isinstance(v, torch.Tensor):
                copied["state"][param_id][k] = v.clone()
            else:
                copied["state"][param_id][k] = v
    return copied


def _assert_optimizer_states_equal(original: dict, loaded: dict) -> None:
    """Assert that two optimizer state dicts are equal."""
    # Compare param_groups (excluding 'params' which are just indices)
    assert len(original["param_groups"]) == len(loaded["param_groups"]), (
        "Optimizer param_groups count mismatch"
    )
    for i, (orig_group, load_group) in enumerate(
        zip(original["param_groups"], loaded["param_groups"])
    ):
        for key in orig_group:
            if key == "params":
                continue
            assert orig_group[key] == load_group[key], (
                f"Optimizer param_group[{i}] key '{key}' mismatch: "
                f"{orig_group[key]} vs {load_group[key]}"
            )

    # Compare per-parameter state
    assert len(original["state"]) == len(loaded["state"]), (
        "Optimizer state count mismatch"
    )
    for param_id in original["state"]:
        assert param_id in loaded["state"], (
            f"Missing optimizer state for param {param_id}"
        )
        orig_param = original["state"][param_id]
        load_param = loaded["state"][param_id]
        for key in orig_param:
            orig_val = orig_param[key]
            load_val = load_param[key]
            if isinstance(orig_val, torch.Tensor):
                assert torch.equal(orig_val, load_val), (
                    f"Optimizer state tensor mismatch for param {param_id}, "
                    f"key '{key}'"
                )
            else:
                assert orig_val == load_val, (
                    f"Optimizer state scalar mismatch for param {param_id}, "
                    f"key '{key}': {orig_val} vs {load_val}"
                )


# Feature: deepqnetwork, Property 14: S3 checkpoint timestamp formatting


@settings(max_examples=100)
@given(
    dt=st.datetimes(timezones=st.just(timezone.utc)),
)
def test_s3_filename_matches_timestamp_pattern(dt: datetime) -> None:
    """For any UTC datetime, the S3 checkpoint filename SHALL match the pattern
    `{YYYYMMDD}T{HH}0000Z.pt` with minutes and seconds zeroed to the hour.

    **Validates: Requirements 8.5, 8.9**
    """
    mgr = CheckpointManager(checkpoint_dir="/tmp/test_checkpoints")

    with patch("deepqnetwork.checkpoint_manager.datetime") as mock_datetime:
        mock_datetime.now.return_value = dt
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        filename = mgr._generate_s3_filename()

    # Verify the filename matches the expected regex pattern
    pattern = r"^\d{8}T\d{2}0000Z\.pt$"
    assert re.match(pattern, filename), (
        f"Filename '{filename}' does not match pattern '{pattern}' "
        f"for datetime {dt}"
    )


@settings(max_examples=100)
@given(
    dt=st.datetimes(timezones=st.just(timezone.utc)),
)
def test_s3_filename_date_matches_generated_datetime(dt: datetime) -> None:
    """For any UTC datetime, the YYYYMMDD part of the S3 filename SHALL match
    the generated date.

    **Validates: Requirements 8.5, 8.9**
    """
    mgr = CheckpointManager(checkpoint_dir="/tmp/test_checkpoints")

    with patch("deepqnetwork.checkpoint_manager.datetime") as mock_datetime:
        mock_datetime.now.return_value = dt
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        filename = mgr._generate_s3_filename()

    # Extract YYYYMMDD from filename
    date_part = filename[:8]
    expected_date = dt.strftime("%Y%m%d")

    assert date_part == expected_date, (
        f"Date part '{date_part}' does not match expected '{expected_date}' "
        f"for datetime {dt}"
    )


@settings(max_examples=100)
@given(
    dt=st.datetimes(timezones=st.just(timezone.utc)),
)
def test_s3_filename_hour_matches_generated_datetime(dt: datetime) -> None:
    """For any UTC datetime, the HH part of the S3 filename SHALL match
    the generated hour.

    **Validates: Requirements 8.5, 8.9**
    """
    mgr = CheckpointManager(checkpoint_dir="/tmp/test_checkpoints")

    with patch("deepqnetwork.checkpoint_manager.datetime") as mock_datetime:
        mock_datetime.now.return_value = dt
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        filename = mgr._generate_s3_filename()

    # Extract HH from filename (position 9-10, after "YYYYMMDDT")
    hour_part = filename[9:11]
    expected_hour = f"{dt.hour:02d}"

    assert hour_part == expected_hour, (
        f"Hour part '{hour_part}' does not match expected '{expected_hour}' "
        f"for datetime {dt}"
    )


@settings(max_examples=100)
@given(
    dt=st.datetimes(timezones=st.just(timezone.utc)),
)
def test_s3_filename_minutes_seconds_always_zeroed(dt: datetime) -> None:
    """For any UTC datetime, the minutes and seconds in the S3 filename
    SHALL always be "0000".

    **Validates: Requirements 8.5, 8.9**
    """
    mgr = CheckpointManager(checkpoint_dir="/tmp/test_checkpoints")

    with patch("deepqnetwork.checkpoint_manager.datetime") as mock_datetime:
        mock_datetime.now.return_value = dt
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        filename = mgr._generate_s3_filename()

    # Extract minutes/seconds part (position 11-14, after "YYYYMMDDTHH")
    min_sec_part = filename[11:15]

    assert min_sec_part == "0000", (
        f"Minutes/seconds part '{min_sec_part}' is not '0000' "
        f"for datetime {dt} (minutes={dt.minute}, seconds={dt.second})"
    )


@settings(max_examples=100)
@given(
    dt=st.datetimes(timezones=st.just(timezone.utc)),
)
def test_s3_filename_ends_with_z_pt(dt: datetime) -> None:
    """For any UTC datetime, the S3 filename SHALL end with "Z.pt".

    **Validates: Requirements 8.5, 8.9**
    """
    mgr = CheckpointManager(checkpoint_dir="/tmp/test_checkpoints")

    with patch("deepqnetwork.checkpoint_manager.datetime") as mock_datetime:
        mock_datetime.now.return_value = dt
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        filename = mgr._generate_s3_filename()

    assert filename.endswith("Z.pt"), (
        f"Filename '{filename}' does not end with 'Z.pt' "
        f"for datetime {dt}"
    )
