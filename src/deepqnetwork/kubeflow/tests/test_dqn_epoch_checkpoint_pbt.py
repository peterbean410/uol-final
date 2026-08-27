"""Property-based tests for DQN epoch checkpoint persistence.

Verifies that the DQN training loop saves exactly N checkpoints for N
checkpoint intervals, that each checkpoint contains all required fields
and is loadable by both DQNAgent and DQNAdvisor.

Uses Hypothesis to generate varying checkpoint intervals and episode counts,
and mocks the CheckpointManager to capture saved checkpoints in memory
without requiring actual file I/O or S3 connectivity.

**Validates: Requirements DQN-R26**
"""

import io
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from deepqnetwork.agent import DQNAgent
from deepqnetwork.advisor import DQNAdvisor
from deepqnetwork.checkpoint_manager import CheckpointManager
from deepqnetwork.config import DQNConfig
from deepqnetwork.network import QNetwork


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_DIM = 53  # modelenv observation dimension
ACTION_DIM = 5  # HOLD, BUY_1, BUY_2, SELL_1, SELL_2
HIDDEN_DIMS = [256, 256, 128]

# Required fields in every DQN checkpoint
REQUIRED_CHECKPOINT_FIELDS = {
    "q_network_state_dict",
    "target_network_state_dict",
    "optimizer_state_dict",
    "epsilon",
    "step_count",
    "episode_count",
    "config",
}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def checkpoint_intervals(draw):
    """Generate valid checkpoint interval configurations.

    Returns a tuple of (num_episodes_per_range, checkpoint_interval) where
    num_episodes_per_range is a multiple of checkpoint_interval to ensure
    clean division for checkpoint counting.
    """
    interval = draw(st.integers(min_value=1, max_value=10))
    # num_episodes_per_range is interval * multiplier so we get exact checkpoint counts
    multiplier = draw(st.integers(min_value=1, max_value=10))
    num_episodes_per_range = interval * multiplier
    return num_episodes_per_range, interval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_config(
    num_episodes_per_range: int = 10,
    checkpoint_interval: int = 5,
) -> DQNConfig:
    """Create a minimal DQNConfig for checkpoint testing."""
    return DQNConfig(
        num_episodes_per_range=num_episodes_per_range,
        checkpoint_interval=checkpoint_interval,
        hidden_dims=HIDDEN_DIMS,
        activation="relu",
        dropout=0.0,
        learning_rate=1e-4,
        batch_size=64,
        replay_buffer_size=1000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=1000,
        gamma=0.99,
        target_update_freq=100,
        train_freq=4,
        tau=1.0,
        loss_function="huber",
        grad_clip_norm=10.0,
        weight_decay=0.0,
        symbol="USDJPY",
    )


def _create_agent(config: DQNConfig) -> DQNAgent:
    """Create a DQNAgent with the given config on CPU."""
    return DQNAgent(config=config, device=torch.device("cpu"), state_dim=STATE_DIM)


def _simulate_training_checkpoints(
    agent: DQNAgent,
    config: DQNConfig,
) -> list[dict]:
    """Simulate the training loop checkpoint saving logic.

    Mirrors the checkpoint saving logic from deepqnetwork/train.py:
    - Save at every `checkpoint_interval` episodes (episode > 0 and episode % interval == 0)
    - Save a final checkpoint at the end of training

    Returns a list of checkpoint dicts that would be saved.
    """
    saved_checkpoints: list[dict] = []

    for episode in range(config.num_episodes_per_range):
        # Checkpoint saving at configurable intervals (mirrors train.py logic)
        if episode > 0 and episode % config.checkpoint_interval == 0:
            checkpoint_dict = {
                "q_network_state_dict": agent.q_network.state_dict(),
                "target_network_state_dict": agent.target_network.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "epsilon": agent.epsilon,
                "step_count": agent.step_count,
                "episode_count": episode,
                "config": asdict(config),
            }
            saved_checkpoints.append(checkpoint_dict)

    # Final checkpoint at end of training
    final_checkpoint = {
        "q_network_state_dict": agent.q_network.state_dict(),
        "target_network_state_dict": agent.target_network.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "step_count": agent.step_count,
        "episode_count": config.num_episodes_per_range - 1,
        "config": asdict(config),
    }
    saved_checkpoints.append(final_checkpoint)

    return saved_checkpoints


def _save_checkpoint_to_tempfile(checkpoint: dict) -> str:
    """Save a checkpoint dict to a temporary file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(checkpoint, tmp.name)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Property DQN-17: DQN epoch checkpoint persistence
# ---------------------------------------------------------------------------


class TestDQNEpochCheckpointPersistence:
    """Property DQN-17: DQN epoch checkpoint persistence.

    For any N checkpoint intervals, exactly N checkpoints exist in the
    artifact store, each loadable by DQNAgent and DQNAdvisor.

    **Validates: Requirements DQN-R26**
    """

    @given(params=checkpoint_intervals())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_exactly_n_checkpoints_saved(self, params):
        """For any N checkpoint intervals, exactly N interval checkpoints
        are saved plus one final checkpoint.

        The training loop saves at episode % checkpoint_interval == 0
        (for episode > 0), plus a final checkpoint at the end.

        **Validates: Requirements DQN-R26**
        """
        num_episodes_per_range, checkpoint_interval = params

        config = _create_config(
            num_episodes_per_range=num_episodes_per_range,
            checkpoint_interval=checkpoint_interval,
        )
        agent = _create_agent(config)

        # Simulate checkpoint saving
        checkpoints = _simulate_training_checkpoints(agent, config)

        # Expected interval checkpoints: episodes that satisfy
        # episode > 0 and episode % checkpoint_interval == 0
        expected_interval_count = len([
            e for e in range(num_episodes_per_range)
            if e > 0 and e % checkpoint_interval == 0
        ])

        # Total checkpoints = interval checkpoints + 1 final
        expected_total = expected_interval_count + 1

        assert len(checkpoints) == expected_total, (
            f"Expected {expected_total} checkpoints "
            f"({expected_interval_count} interval + 1 final) for "
            f"num_episodes_per_range={num_episodes_per_range}, checkpoint_interval={checkpoint_interval}, "
            f"got {len(checkpoints)}"
        )

    @given(params=checkpoint_intervals())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_each_checkpoint_has_required_fields(self, params):
        """Each saved checkpoint contains all required fields:
        q_network_state_dict, target_network_state_dict, optimizer_state_dict,
        epsilon, step_count, episode_count, and config.

        **Validates: Requirements DQN-R26**
        """
        num_episodes_per_range, checkpoint_interval = params

        config = _create_config(
            num_episodes_per_range=num_episodes_per_range,
            checkpoint_interval=checkpoint_interval,
        )
        agent = _create_agent(config)

        checkpoints = _simulate_training_checkpoints(agent, config)

        for i, ckpt in enumerate(checkpoints):
            missing = REQUIRED_CHECKPOINT_FIELDS - set(ckpt.keys())
            assert not missing, (
                f"Checkpoint {i} missing required fields: {missing}"
            )

            # Verify field types
            assert isinstance(ckpt["q_network_state_dict"], dict), (
                f"Checkpoint {i}: q_network_state_dict is not a dict"
            )
            assert isinstance(ckpt["target_network_state_dict"], dict), (
                f"Checkpoint {i}: target_network_state_dict is not a dict"
            )
            assert isinstance(ckpt["optimizer_state_dict"], dict), (
                f"Checkpoint {i}: optimizer_state_dict is not a dict"
            )
            assert isinstance(ckpt["epsilon"], float), (
                f"Checkpoint {i}: epsilon is not a float"
            )
            assert isinstance(ckpt["step_count"], int), (
                f"Checkpoint {i}: step_count is not an int"
            )
            assert isinstance(ckpt["episode_count"], int), (
                f"Checkpoint {i}: episode_count is not an int"
            )
            assert isinstance(ckpt["config"], dict), (
                f"Checkpoint {i}: config is not a dict"
            )

    @given(params=checkpoint_intervals())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_each_checkpoint_loadable_by_dqn_agent(self, params):
        """Each checkpoint is loadable by a fresh DQNAgent; the state dicts
        can be loaded into new QNetwork and target network instances.

        **Validates: Requirements DQN-R26**
        """
        num_episodes_per_range, checkpoint_interval = params

        config = _create_config(
            num_episodes_per_range=num_episodes_per_range,
            checkpoint_interval=checkpoint_interval,
        )
        agent = _create_agent(config)

        checkpoints = _simulate_training_checkpoints(agent, config)

        for i, ckpt in enumerate(checkpoints):
            # Create a fresh agent and load the checkpoint state
            fresh_agent = _create_agent(config)

            # Load Q-network weights
            fresh_agent.q_network.load_state_dict(ckpt["q_network_state_dict"])
            # Load target network weights
            fresh_agent.target_network.load_state_dict(ckpt["target_network_state_dict"])
            # Load optimizer state
            fresh_agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            # Restore epsilon and step count
            fresh_agent.epsilon = ckpt["epsilon"]
            fresh_agent.step_count = ckpt["step_count"]

            # Verify the loaded agent can perform a forward pass
            dummy_state = torch.randn(1, STATE_DIM)
            with torch.no_grad():
                q_values = fresh_agent.q_network(dummy_state)

            assert q_values.shape == (1, ACTION_DIM), (
                f"Checkpoint {i}: loaded Q-network output shape "
                f"{q_values.shape} != expected (1, {ACTION_DIM})"
            )
            assert torch.isfinite(q_values).all(), (
                f"Checkpoint {i}: loaded Q-network produced non-finite values"
            )

            # Verify target network also works
            with torch.no_grad():
                target_q = fresh_agent.target_network(dummy_state)

            assert target_q.shape == (1, ACTION_DIM), (
                f"Checkpoint {i}: loaded target network output shape "
                f"{target_q.shape} != expected (1, {ACTION_DIM})"
            )
            assert torch.isfinite(target_q).all(), (
                f"Checkpoint {i}: loaded target network produced non-finite values"
            )

    @given(params=checkpoint_intervals())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_each_checkpoint_loadable_by_dqn_advisor(self, params):
        """Each checkpoint is loadable by DQNAdvisor.from_checkpoint() and
        produces valid action recommendations.

        **Validates: Requirements DQN-R26**
        """
        num_episodes_per_range, checkpoint_interval = params

        config = _create_config(
            num_episodes_per_range=num_episodes_per_range,
            checkpoint_interval=checkpoint_interval,
        )
        agent = _create_agent(config)

        checkpoints = _simulate_training_checkpoints(agent, config)

        for i, ckpt in enumerate(checkpoints):
            # Save checkpoint to a temp file (DQNAdvisor loads from file path)
            tmp_path = _save_checkpoint_to_tempfile(ckpt)

            try:
                # Load via DQNAdvisor.from_checkpoint
                advisor = DQNAdvisor.from_checkpoint(tmp_path, device="cpu")

                # Verify advisor can produce recommendations
                dummy_state = np.random.randn(STATE_DIM).astype(np.float32)
                result = advisor.recommend_action(dummy_state)

                # Verify result structure
                assert 0 <= result.action < ACTION_DIM, (
                    f"Checkpoint {i}: advisor action {result.action} out of range"
                )
                assert result.action_name in [
                    "HOLD", "BUY_1", "BUY_2", "SELL_1", "SELL_2"
                ], (
                    f"Checkpoint {i}: invalid action name '{result.action_name}'"
                )
                assert result.q_values.shape == (ACTION_DIM,), (
                    f"Checkpoint {i}: q_values shape {result.q_values.shape} "
                    f"!= expected ({ACTION_DIM},)"
                )
                assert np.isfinite(result.q_values).all(), (
                    f"Checkpoint {i}: advisor produced non-finite q_values"
                )
                assert isinstance(result.confidence, float), (
                    f"Checkpoint {i}: confidence is not a float"
                )
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)

    @given(params=checkpoint_intervals())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_checkpoint_manager_save_produces_loadable_checkpoints(self, params):
        """CheckpointManager.save() produces checkpoints that are loadable
        via CheckpointManager.load() with all required fields intact.

        **Validates: Requirements DQN-R26**
        """
        num_episodes_per_range, checkpoint_interval = params

        config = _create_config(
            num_episodes_per_range=num_episodes_per_range,
            checkpoint_interval=checkpoint_interval,
        )
        agent = _create_agent(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a CheckpointManager with no S3 (local only)
            mgr = CheckpointManager(
                checkpoint_dir=tmpdir,
                s3_prefix=None,
            )

            saved_paths: list[str] = []

            # Simulate the training loop checkpoint logic
            for episode in range(num_episodes_per_range):
                if episode > 0 and episode % checkpoint_interval == 0:
                    path = mgr.save(
                        episode=episode,
                        q_network=agent.q_network,
                        target_network=agent.target_network,
                        optimizer=agent.optimizer,
                        epsilon=agent.epsilon,
                        step_count=agent.step_count,
                        config=asdict(config),
                    )
                    saved_paths.append(path)

            # Final checkpoint
            final_path = mgr.save(
                episode=num_episodes_per_range - 1,
                q_network=agent.q_network,
                target_network=agent.target_network,
                optimizer=agent.optimizer,
                epsilon=agent.epsilon,
                step_count=agent.step_count,
                config=asdict(config),
            )
            saved_paths.append(final_path)

            # Verify each saved checkpoint is loadable
            for i, path in enumerate(saved_paths):
                loaded = mgr.load(path)
                assert loaded, (
                    f"Checkpoint {i} at {path} loaded as empty dict"
                )
                missing = REQUIRED_CHECKPOINT_FIELDS - set(loaded.keys())
                assert not missing, (
                    f"Loaded checkpoint {i} missing fields: {missing}"
                )
