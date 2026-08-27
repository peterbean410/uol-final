"""Property-based tests for DQN Pipeline Configuration (config_schema.py).

Uses Hypothesis to verify correctness properties across randomly generated configurations.

**Validates: Requirements DQN-R1, DQN-R2, DQN-R3**
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict

import yaml
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from deepqnetwork.kubeflow.pipeline.config_schema import DQNPipelineConfig


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

VALID_SYMBOLS = ("USDJPY", "AUDJPY")
VALID_ACTIVATIONS = ("relu", "leaky_relu", "gelu")
VALID_LOSS_FUNCTIONS = ("huber", "mse")
VALID_TRAINING_MODES = ("scratch", "finetune")


@st.composite
def valid_dqn_pipeline_configs(draw):
    """Generate valid DQNPipelineConfig instances that pass validation."""
    epsilon_start = draw(
        st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    epsilon_end = draw(
        st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)
    )
    assume(epsilon_end <= epsilon_start)

    training_mode = draw(st.sampled_from(VALID_TRAINING_MODES))

    return DQNPipelineConfig(
        grpc_address=draw(
            st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz0123456789:.")
        ),
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
        batch_size=draw(st.integers(min_value=1, max_value=512)),
        replay_buffer_size=draw(st.integers(min_value=1, max_value=1_000_000)),
        target_update_freq=draw(st.integers(min_value=1, max_value=10_000)),
        train_freq=draw(st.integers(min_value=1, max_value=16)),
        tau=draw(
            st.floats(
                min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        ),
        hidden_dims=draw(
            st.lists(st.integers(min_value=1, max_value=512), min_size=1, max_size=5)
        ),
        activation=draw(st.sampled_from(VALID_ACTIVATIONS)),
        dropout=draw(
            st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False)
        ),
        learning_rate=draw(
            st.floats(min_value=1e-6, max_value=0.01, allow_nan=False, allow_infinity=False)
        ),
        betas=(
            draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
            draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        ),
        eps=draw(
            st.floats(min_value=1e-10, max_value=1e-6, allow_nan=False, allow_infinity=False)
        ),
        weight_decay=draw(
            st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False)
        ),
        grad_clip_norm=draw(
            st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        loss_function=draw(st.sampled_from(VALID_LOSS_FUNCTIONS)),
        num_episodes_per_range=draw(st.integers(min_value=1, max_value=10_000)),
        max_steps_per_episode=draw(st.integers(min_value=1, max_value=100_000)),
        checkpoint_interval=draw(st.integers(min_value=1, max_value=500)),
        num_workers=draw(st.integers(min_value=1, max_value=4)),
        max_wall_time_hours=draw(st.integers(min_value=1, max_value=48)),  # Keep simple to avoid conditional validation
        katib_max_trials=draw(st.integers(min_value=1, max_value=100)),
        katib_parallel_trials=draw(st.integers(min_value=1, max_value=10)),
        katib_trial_timeout_hours=draw(st.integers(min_value=1, max_value=24)),
        serving_min_replicas=draw(st.integers(min_value=0, max_value=4)),
        serving_max_replicas=draw(st.integers(min_value=1, max_value=10)),
        serving_target_concurrency=draw(st.integers(min_value=1, max_value=100)),
        alert_webhook_url=draw(
            st.text(min_size=0, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz0123456789:/.@-_")
        ),
        sharpe_degradation_threshold=draw(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
        sharpe_absolute_threshold=draw(
            st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)
        ),
        pnl_absolute_threshold=draw(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        ),
        training_mode=training_mode,
        finetune_learning_rate=draw(
            st.floats(min_value=1e-6, max_value=0.01, allow_nan=False, allow_infinity=False)
        ),
        finetune_num_episodes_per_range=draw(st.integers(min_value=1, max_value=5000)),
    )


@st.composite
def invalid_dqn_pipeline_configs(draw):
    """Generate DQNPipelineConfig instances with at least one invalid field.

    Deliberately sets one or more fields to out-of-range values that should
    be caught by validate().
    """
    invalid_field = draw(
        st.sampled_from([
            "learning_rate_too_high",
            "learning_rate_too_low",
            "activation_invalid",
            "gamma_negative",
            "gamma_above_one",
            "loss_function_invalid",
            "hidden_dims_empty",
            "hidden_dims_negative",
            "num_workers_out_of_range",
        ])
    )

    # Start with valid defaults
    config_kwargs: dict = dict(
        symbol="USDJPY",
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        learning_rate=0.0001,
        activation="relu",
        loss_function="huber",
        hidden_dims=[256, 256, 128],
        num_workers=1,
    )

    # Invalidate the chosen field
    if invalid_field == "learning_rate_too_high":
        config_kwargs["learning_rate"] = draw(
            st.floats(min_value=0.011, max_value=1.0, allow_nan=False, allow_infinity=False)
        )
    elif invalid_field == "learning_rate_too_low":
        config_kwargs["learning_rate"] = draw(
            st.floats(min_value=1e-10, max_value=9e-7, allow_nan=False, allow_infinity=False)
        )
    elif invalid_field == "activation_invalid":
        config_kwargs["activation"] = draw(
            st.text(min_size=1, max_size=15, alphabet="abcdefghijklmnopqrstuvwxyz_")
            .filter(lambda s: s not in VALID_ACTIVATIONS)
        )
    elif invalid_field == "gamma_negative":
        config_kwargs["gamma"] = draw(
            st.floats(min_value=-10.0, max_value=-0.001, allow_nan=False, allow_infinity=False)
        )
    elif invalid_field == "gamma_above_one":
        config_kwargs["gamma"] = draw(
            st.floats(min_value=1.001, max_value=10.0, allow_nan=False, allow_infinity=False)
        )
    elif invalid_field == "loss_function_invalid":
        config_kwargs["loss_function"] = draw(
            st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz")
            .filter(lambda s: s not in VALID_LOSS_FUNCTIONS)
        )
    elif invalid_field == "hidden_dims_empty":
        config_kwargs["hidden_dims"] = []
    elif invalid_field == "hidden_dims_negative":
        config_kwargs["hidden_dims"] = [256, draw(st.integers(min_value=-100, max_value=0)), 128]
    elif invalid_field == "num_workers_out_of_range":
        config_kwargs["num_workers"] = draw(
            st.one_of(
                st.integers(min_value=-10, max_value=0),
                st.integers(min_value=5, max_value=100),
            )
        )

    return DQNPipelineConfig(**config_kwargs)


# ---------------------------------------------------------------------------
# Property DQN-1: YAML Configuration Round-Trip
# ---------------------------------------------------------------------------


class TestYAMLConfigurationRoundTrip:
    """Property DQN-1: YAML configuration round-trip.

    For any valid DQNPipelineConfig, serialize to YAML and deserialize back
    produces equivalent config.

    **Validates: Requirements DQN-R1**
    """

    @given(config=valid_dqn_pipeline_configs())
    @settings(max_examples=100, deadline=None)
    def test_yaml_round_trip_preserves_config(self, config: DQNPipelineConfig):
        """Serializing a valid DQNPipelineConfig to YAML and loading it back
        produces an equivalent configuration.

        **Validates: Requirements DQN-R1**
        """
        config_dict = asdict(config)

        # Convert tuples to lists for YAML-safe serialization (yaml.safe_load
        # cannot handle Python-specific tuple tags)
        yaml_dict = {}
        for key, value in config_dict.items():
            if isinstance(value, tuple):
                yaml_dict[key] = list(value)
            else:
                yaml_dict[key] = value

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_dict, f, default_flow_style=False)
            tmp_path = f.name

        try:
            loaded_config = DQNPipelineConfig.from_yaml(tmp_path)
            loaded_dict = asdict(loaded_config)

            for field_name, original_value in config_dict.items():
                loaded_value = loaded_dict[field_name]
                if isinstance(original_value, float):
                    assert abs(original_value - loaded_value) < 1e-10, (
                        f"Field '{field_name}' differs after round-trip: "
                        f"original={original_value}, loaded={loaded_value}"
                    )
                elif isinstance(original_value, tuple):
                    # from_yaml converts lists back to tuples for betas
                    assert tuple(loaded_value) == original_value, (
                        f"Field '{field_name}' differs after round-trip: "
                        f"original={original_value}, loaded={loaded_value}"
                    )
                else:
                    assert original_value == loaded_value, (
                        f"Field '{field_name}' differs after round-trip: "
                        f"original={original_value}, loaded={loaded_value}"
                    )
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Property DQN-2: CLI Argument Generation
# ---------------------------------------------------------------------------


class TestCLIArgumentGeneration:
    """Property DQN-2: CLI argument generation.

    For any valid DQNPipelineConfig, `to_cli_args()` produces a list of strings
    that `load_config()` can parse back to equivalent values.

    **Validates: Requirements DQN-R2**
    """

    @given(config=valid_dqn_pipeline_configs())
    @settings(max_examples=100, deadline=None)
    def test_cli_args_parseable_by_load_config(self, config: DQNPipelineConfig):
        """For any valid DQNPipelineConfig, to_cli_args() produces arguments
        that load_config() can parse back to equivalent key field values.

        **Validates: Requirements DQN-R2**
        """
        from deepqnetwork.config import load_config

        cli_args = config.to_cli_args()

        # All elements must be strings
        assert isinstance(cli_args, list)
        assert all(isinstance(a, str) for a in cli_args)

        # Parse with load_config (needs --config pointing to a valid YAML)
        full_args = ["--config", "deepqnetwork/config.yaml"] + cli_args
        parsed = load_config(full_args)

        # Verify key fields match (fields that to_cli_args() emits)
        assert parsed.symbol == config.symbol
        assert parsed.gamma == config.gamma
        assert parsed.batch_size == config.batch_size
        assert parsed.max_steps_per_episode == config.max_steps_per_episode
        assert parsed.checkpoint_interval == config.checkpoint_interval
        assert parsed.epsilon_start == config.epsilon_start
        assert parsed.epsilon_end == config.epsilon_end
        assert parsed.epsilon_decay_steps == config.epsilon_decay_steps
        assert parsed.replay_buffer_size == config.replay_buffer_size
        assert parsed.target_update_freq == config.target_update_freq
        assert parsed.train_freq == config.train_freq
        assert parsed.tau == config.tau
        assert parsed.grpc_address == config.grpc_address
        assert parsed.episode_start_ts == config.episode_start_ts
        assert parsed.episode_end_ts == config.episode_end_ts
        assert parsed.step_size_seconds == config.step_size_seconds
        assert parsed.dropout == config.dropout
        assert parsed.grad_clip_norm == config.grad_clip_norm
        assert parsed.weight_decay == config.weight_decay
        assert parsed.loss_function == config.loss_function
        assert parsed.mode == "train"

        # Learning rate depends on training mode
        if config.training_mode == "finetune":
            assert parsed.learning_rate == config.finetune_learning_rate
        else:
            assert parsed.learning_rate == config.learning_rate

        # Num episodes depends on training mode
        if config.training_mode == "finetune":
            assert parsed.num_episodes_per_range == config.finetune_num_episodes_per_range
        else:
            assert parsed.num_episodes_per_range == config.num_episodes_per_range

        # Dueling flag


# ---------------------------------------------------------------------------
# Property DQN-3: Parameter Validation Rejects Invalid Configs
# ---------------------------------------------------------------------------


class TestParameterValidationRejectsInvalid:
    """Property DQN-3: Parameter validation rejects invalid configs.

    Configs with out-of-range learning_rate, invalid activation, or negative
    gamma produce non-empty error lists.

    **Validates: Requirements DQN-R3**
    """

    @given(config=invalid_dqn_pipeline_configs())
    @settings(max_examples=100, deadline=None)
    def test_invalid_configs_produce_errors(self, config: DQNPipelineConfig):
        """Configurations with out-of-range values produce non-empty error lists.

        **Validates: Requirements DQN-R3**
        """
        errors = config.validate()
        assert len(errors) > 0, (
            f"Expected validation errors for invalid config but got none. "
            f"Config: learning_rate={config.learning_rate}, "
            f"activation={config.activation}, gamma={config.gamma}, "
            f"loss_function={config.loss_function}, "
            f"hidden_dims={config.hidden_dims}, "
            f"num_workers={config.num_workers}"
        )

    @given(config=valid_dqn_pipeline_configs())
    @settings(max_examples=100, deadline=None)
    def test_valid_configs_produce_no_errors(self, config: DQNPipelineConfig):
        """Valid configurations produce empty error lists (complementary check).

        **Validates: Requirements DQN-R3**
        """
        errors = config.validate()
        assert len(errors) == 0, (
            f"Expected no validation errors for valid config but got: {errors}. "
            f"Config: symbol={config.symbol}, gamma={config.gamma}, "
            f"epsilon_start={config.epsilon_start}, epsilon_end={config.epsilon_end}, "
            f"learning_rate={config.learning_rate}, activation={config.activation}, "
            f"hidden_dims={config.hidden_dims}, num_workers={config.num_workers}, "
            f"dropout={config.dropout}, tau={config.tau}"
        )
