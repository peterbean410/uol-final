"""Property-based tests for DQN Training component checkpoint output.

Tests that the build_dqn_config function produces configs that can instantiate
DQNAgent and DQNAdvisor, and that finetune mode correctly applies the reduced
learning rate.

**Validates: Requirements DQN-R7, DQN-R8**
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from deepqnetwork.agent import DQNAgent
from deepqnetwork.advisor import DQNAdvisor
from deepqnetwork.config import DQNConfig
from deepqnetwork.kubeflow.components.dqn_training.component import build_dqn_config
from deepqnetwork.kubeflow.pipeline.config_schema import DQNPipelineConfig


VALID_SYMBOLS = ("USDJPY", "AUDJPY")
VALID_ACTIVATIONS = ("relu", "leaky_relu", "gelu")
VALID_LOSS_FUNCTIONS = ("huber", "mse")
VALID_TRAINING_MODES = ("scratch", "finetune")


@st.composite
def valid_dqn_pipeline_configs_for_training(draw):
    """Generate valid DQNPipelineConfig instances with reduced episodes for test speed."""
    epsilon_start = draw(
        st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    epsilon_end = draw(
        st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)
    )
    assume(epsilon_end <= epsilon_start)

    training_mode = draw(st.sampled_from(VALID_TRAINING_MODES))

    hidden_dims = draw(
        st.lists(st.integers(min_value=8, max_value=64), min_size=1, max_size=3)
    )

    return DQNPipelineConfig(
        grpc_address="localhost:50051",
        symbol=draw(st.sampled_from(VALID_SYMBOLS)),
        episode_start_ts=draw(st.integers(min_value=0, max_value=2_000_000_000)),
        episode_end_ts=draw(st.integers(min_value=0, max_value=2_000_000_000)),
        step_size_seconds=draw(st.integers(min_value=1, max_value=60)),
        gamma=draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=draw(st.integers(min_value=1, max_value=200_000)),
        batch_size=draw(st.integers(min_value=1, max_value=128)),
        replay_buffer_size=draw(st.integers(min_value=100, max_value=10_000)),
        target_update_freq=draw(st.integers(min_value=1, max_value=10_000)),
        train_freq=draw(st.integers(min_value=1, max_value=16)),
        tau=draw(
            st.floats(
                min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        ),
        hidden_dims=hidden_dims,
        activation=draw(st.sampled_from(VALID_ACTIVATIONS)),
        dropout=draw(
            st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)
        ),
        learning_rate=draw(
            st.floats(min_value=1e-6, max_value=0.01, allow_nan=False, allow_infinity=False)
        ),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=draw(
            st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False)
        ),
        grad_clip_norm=draw(
            st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        loss_function=draw(st.sampled_from(VALID_LOSS_FUNCTIONS)),
        num_episodes_per_range=draw(st.integers(min_value=1, max_value=5)),
        max_steps_per_episode=draw(st.integers(min_value=1, max_value=100)),
        checkpoint_interval=draw(st.integers(min_value=1, max_value=10)),
        num_workers=1,
        max_wall_time_hours=1,
        training_mode=training_mode,
        finetune_learning_rate=draw(
            st.floats(min_value=1e-6, max_value=0.01, allow_nan=False, allow_infinity=False)
        ),
        finetune_num_episodes_per_range=draw(st.integers(min_value=1, max_value=5)),
    )


def _make_args(training_mode: str = "scratch", production_checkpoint_path: str = "") -> argparse.Namespace:
    """Create a minimal argparse.Namespace mimicking component CLI args."""
    return argparse.Namespace(
        training_mode=training_mode,
        production_checkpoint_path=production_checkpoint_path,
    )


class TestTrainingProducesValidCheckpoint:
    """Property DQN-5: Training produces valid checkpoint.

    For any valid DQNPipelineConfig (within reduced episode bounds for test speed),
    the build_dqn_config function produces a config that can be used to instantiate
    a DQNAgent, save a checkpoint, and load it with DQNAdvisor.

    **Validates: Requirements DQN-R7**
    """

    @given(pipeline_config=valid_dqn_pipeline_configs_for_training())
    @settings(max_examples=50, deadline=None)
    def test_config_produces_loadable_checkpoint(self, pipeline_config: DQNPipelineConfig):
        """For any valid DQNPipelineConfig, build_dqn_config produces a config
        that instantiates a DQNAgent whose checkpoint is loadable by DQNAdvisor.

        **Validates: Requirements DQN-R7**
        """
        args = _make_args(training_mode=pipeline_config.training_mode)
        dqn_config = build_dqn_config(pipeline_config, args)

        assert isinstance(dqn_config, DQNConfig)

        device = torch.device("cpu")
        agent = DQNAgent(dqn_config, device)

        assert agent.q_network is not None
        assert agent.target_network is not None
        assert agent.optimizer is not None

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            from dataclasses import asdict

            checkpoint_data = {
                "q_network_state_dict": agent.q_network.state_dict(),
                "target_network_state_dict": agent.target_network.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "epsilon": agent.epsilon,
                "step_count": agent.step_count,
                "episode_count": 0,
                "config": asdict(dqn_config),
            }
            torch.save(checkpoint_data, checkpoint_path)

            new_agent = DQNAgent(dqn_config, device)
            loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            new_agent.q_network.load_state_dict(loaded_checkpoint["q_network_state_dict"])
            new_agent.target_network.load_state_dict(loaded_checkpoint["target_network_state_dict"])
            new_agent.optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])

            advisor = DQNAdvisor(checkpoint_path=checkpoint_path, device="cpu")
            assert advisor.state_dim == 53

            state = np.random.randn(53).astype(np.float32)
            result = advisor.recommend_action(state)
            assert 0 <= result.action <= 4
            assert result.q_values.shape == (5,)

        finally:
            os.unlink(checkpoint_path)


class TestFinetuneUsesLowerLearningRate:
    """Property DQN-6: Finetune uses lower learning rate.

    When training_mode is 'finetune', the effective learning rate in the built
    DQNConfig equals finetune_learning_rate, not the base learning_rate.

    **Validates: Requirements DQN-R8**
    """

    @given(pipeline_config=valid_dqn_pipeline_configs_for_training())
    @settings(max_examples=50, deadline=None)
    def test_finetune_mode_uses_finetune_learning_rate(self, pipeline_config: DQNPipelineConfig):
        """When training_mode is 'finetune', the effective learning rate equals
        finetune_learning_rate, not the base learning_rate.

        **Validates: Requirements DQN-R8**
        """
        pipeline_config = pipeline_config.override(training_mode="finetune")

        assume(pipeline_config.finetune_learning_rate != pipeline_config.learning_rate)

        args = _make_args(training_mode="finetune")
        dqn_config = build_dqn_config(pipeline_config, args)

        assert dqn_config.learning_rate == pipeline_config.finetune_learning_rate, (
            f"Expected finetune_learning_rate={pipeline_config.finetune_learning_rate}, "
            f"got learning_rate={dqn_config.learning_rate}"
        )

        assert dqn_config.learning_rate != pipeline_config.learning_rate or (
            pipeline_config.finetune_learning_rate == pipeline_config.learning_rate
        ), (
            f"Finetune mode should use finetune_learning_rate, not base learning_rate. "
            f"Got: {dqn_config.learning_rate}, base: {pipeline_config.learning_rate}"
        )

        assert dqn_config.num_episodes_per_range == pipeline_config.finetune_num_episodes_per_range, (
            f"Expected finetune_num_episodes_per_range={pipeline_config.finetune_num_episodes_per_range}, "
            f"got num_episodes_per_range={dqn_config.num_episodes_per_range}"
        )

    @given(pipeline_config=valid_dqn_pipeline_configs_for_training())
    @settings(max_examples=50, deadline=None)
    def test_scratch_mode_uses_base_learning_rate(self, pipeline_config: DQNPipelineConfig):
        """When training_mode is 'scratch', the effective learning rate equals
        the base learning_rate, not finetune_learning_rate.

        **Validates: Requirements DQN-R8**
        """
        pipeline_config = pipeline_config.override(training_mode="scratch")

        assume(pipeline_config.finetune_learning_rate != pipeline_config.learning_rate)

        args = _make_args(training_mode="scratch")
        dqn_config = build_dqn_config(pipeline_config, args)

        assert dqn_config.learning_rate == pipeline_config.learning_rate, (
            f"Expected base learning_rate={pipeline_config.learning_rate}, "
            f"got learning_rate={dqn_config.learning_rate}"
        )

        assert dqn_config.num_episodes_per_range == pipeline_config.num_episodes_per_range, (
            f"Expected num_episodes_per_range={pipeline_config.num_episodes_per_range}, "
            f"got num_episodes_per_range={dqn_config.num_episodes_per_range}"
        )
