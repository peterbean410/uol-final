"""Unit tests for deepqnetwork.config module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from deepqnetwork.config import DQNConfig, load_config


class TestDQNConfig:
    """Tests for DQNConfig dataclass defaults."""

    def test_default_environment_fields(self):
        config = DQNConfig()
        assert config.grpc_address == "localhost:50051"
        assert config.symbol == "USDJPY"
        assert config.episode_start_ts == 0
        assert config.episode_end_ts == 0
        assert config.step_size_seconds == 5

    def test_default_agent_fields(self):
        config = DQNConfig()
        assert config.gamma == 0.99
        assert config.epsilon_start == 1.0
        assert config.epsilon_end == 0.01
        assert config.epsilon_decay_steps == 50_000
        assert config.batch_size == 64
        assert config.replay_buffer_size == 300_000
        assert config.target_update_freq == 1000
        assert config.train_freq == 4
        assert config.tau == 1.0

    def test_default_network_fields(self):
        config = DQNConfig()
        assert config.hidden_dims == [256, 256, 128]
        assert config.activation == "relu"
        assert config.layer_norm is True
        assert config.dropout == 0.0
        assert config.dueling is False
        assert config.learning_rate == 1e-4
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8
        assert config.weight_decay == 0.0
        assert config.grad_clip_norm == 10.0
        assert config.loss_function == "huber"

    def test_default_training_fields(self):
        config = DQNConfig()
        assert config.num_episodes == 3000
        assert config.max_steps_per_episode == 30_000
        assert config.checkpoint_interval == 50
        assert config.checkpoint_dir == "deepqnetwork/checkpoints/"
        assert config.log_interval == 10

    def test_default_mode_fields(self):
        config = DQNConfig()
        assert config.mode == "train"
        assert config.checkpoint is None
        assert config.device == "auto"

    def test_default_s3_fields(self):
        config = DQNConfig()
        assert config.s3_checkpoint_prefix is None
        assert config.model_version is None

    def test_hidden_dims_not_shared_between_instances(self):
        c1 = DQNConfig()
        c2 = DQNConfig()
        c1.hidden_dims.append(64)
        assert c2.hidden_dims == [256, 256, 128]


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_yaml(self):
        yaml_content = {
            "symbol": "EURUSD",
            "gamma": 0.95,
            "batch_size": 32,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            config = load_config(["--config", f.name])

        assert config.symbol == "EURUSD"
        assert config.gamma == 0.95
        assert config.batch_size == 32
        # Defaults still apply for unset fields
        assert config.mode == "train"
        assert config.learning_rate == 1e-4

    def test_cli_overrides_yaml(self):
        yaml_content = {
            "symbol": "EURUSD",
            "device": "cuda",
            "batch_size": 32,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            config = load_config([
                "--config", f.name,
                "--device", "cpu",
                "--batch-size", "128",
            ])

        # CLI overrides
        assert config.device == "cpu"
        assert config.batch_size == 128
        # YAML value preserved when no CLI override
        assert config.symbol == "EURUSD"

    def test_missing_yaml_uses_defaults(self):
        config = load_config(["--config", "/nonexistent/path.yaml"])
        assert config.symbol == "USDJPY"
        assert config.gamma == 0.99

    def test_live_mode_requires_checkpoint(self):
        with pytest.raises(ValueError, match="Live mode requires"):
            load_config(["--mode", "live"])

    def test_live_mode_with_checkpoint_succeeds(self):
        config = load_config(["--mode", "live", "--checkpoint", "model.pt"])
        assert config.mode == "live"
        assert config.checkpoint == "model.pt"

    def test_s3_arguments(self):
        config = load_config([
            "--s3-checkpoint-prefix", "s3://bucket/path",
            "--model-version", "v3",
        ])
        assert config.s3_checkpoint_prefix == "s3://bucket/path"
        assert config.model_version == "v3"

    def test_yaml_betas_converted_to_tuple(self):
        yaml_content = {"betas": [0.8, 0.99]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            config = load_config(["--config", f.name])

        assert config.betas == (0.8, 0.99)

    def test_yaml_hidden_dims_preserved_as_list(self):
        yaml_content = {"hidden_dims": [512, 256]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            config = load_config(["--config", f.name])

        assert config.hidden_dims == [512, 256]

    def test_default_config_yaml_path(self):
        """Loading with no --config uses deepqnetwork/config.yaml."""
        config = load_config([])
        assert config.symbol == "USDJPY"
        assert config.gamma == 0.99

    def test_train_mode_without_checkpoint_succeeds(self):
        config = load_config(["--mode", "train"])
        assert config.mode == "train"
        assert config.checkpoint is None


# ---------------------------------------------------------------------------
# Feature: deepqnetwork, Property 17: CLI overrides YAML configuration
# ---------------------------------------------------------------------------
from hypothesis import given, settings
from hypothesis import strategies as st


# Strategies for CLI-overridable configuration values
# Exclude leading '-' to avoid argparse interpreting values as flags
_str_values = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_./"),
    min_size=1,
    max_size=20,
).filter(lambda s: not s.startswith("-"))
_positive_int = st.integers(min_value=1, max_value=1_000_000)
_positive_float = st.floats(min_value=1e-8, max_value=10.0, allow_nan=False, allow_infinity=False)
_unit_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Each entry: (cli_flag, config_attr, yaml_strategy, cli_strategy)
# We use two independent strategies so YAML and CLI values differ with high probability.
_OVERRIDABLE_PARAMS = [
    ("--device", "device", _str_values, _str_values),
    ("--symbol", "symbol", _str_values, _str_values),
    ("--grpc-address", "grpc_address", _str_values, _str_values),
    ("--checkpoint-dir", "checkpoint_dir", _str_values, _str_values),
    ("--s3-checkpoint-prefix", "s3_checkpoint_prefix", _str_values, _str_values),
    ("--model-version", "model_version", _str_values, _str_values),
    ("--gamma", "gamma", _positive_float, _positive_float),
    ("--epsilon-start", "epsilon_start", _unit_float, _unit_float),
    ("--epsilon-end", "epsilon_end", _unit_float, _unit_float),
    ("--epsilon-decay-steps", "epsilon_decay_steps", _positive_int, _positive_int),
    ("--batch-size", "batch_size", _positive_int, _positive_int),
    ("--replay-buffer-size", "replay_buffer_size", _positive_int, _positive_int),
    ("--target-update-freq", "target_update_freq", _positive_int, _positive_int),
    ("--train-freq", "train_freq", _positive_int, _positive_int),
    ("--tau", "tau", _positive_float, _positive_float),
    ("--learning-rate", "learning_rate", _positive_float, _positive_float),
    ("--grad-clip-norm", "grad_clip_norm", _positive_float, _positive_float),
    ("--weight-decay", "weight_decay", _positive_float, _positive_float),
    ("--dropout", "dropout", _unit_float, _unit_float),
    ("--num-episodes", "num_episodes", _positive_int, _positive_int),
    ("--max-steps-per-episode", "max_steps_per_episode", _positive_int, _positive_int),
    ("--checkpoint-interval", "checkpoint_interval", _positive_int, _positive_int),
    ("--log-interval", "log_interval", _positive_int, _positive_int),
    ("--loss-function", "loss_function", st.just("huber"), st.just("mse")),
]


@given(param_index=st.integers(min_value=0, max_value=len(_OVERRIDABLE_PARAMS) - 1), data=st.data())
@settings(max_examples=100)
def test_cli_overrides_yaml_property(param_index, data):
    """**Validates: Requirements 10.2**

    For any configuration key that is set in both a YAML file and as a CLI
    argument with different values, the resolved configuration SHALL use the
    CLI argument value.
    """
    cli_flag, config_attr, yaml_strat, cli_strat = _OVERRIDABLE_PARAMS[param_index]

    yaml_value = data.draw(yaml_strat, label="yaml_value")
    cli_value = data.draw(cli_strat.filter(lambda v, y=yaml_value: v != y), label="cli_value")

    # Build YAML content using the config attribute name (underscore form)
    yaml_content = {config_attr: yaml_value}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_content, f)
        f.flush()
        config = load_config([
            "--config", f.name,
            cli_flag, str(cli_value),
        ])

    resolved = getattr(config, config_attr)

    # For numeric types, compare with appropriate tolerance
    if isinstance(cli_value, float):
        assert abs(resolved - cli_value) < 1e-9, (
            f"{config_attr}: expected CLI value {cli_value}, got {resolved} "
            f"(YAML had {yaml_value})"
        )
    elif isinstance(cli_value, int):
        assert resolved == cli_value, (
            f"{config_attr}: expected CLI value {cli_value}, got {resolved} "
            f"(YAML had {yaml_value})"
        )
    else:
        assert resolved == str(cli_value), (
            f"{config_attr}: expected CLI value '{cli_value}', got '{resolved}' "
            f"(YAML had '{yaml_value}')"
        )
