"""Unit tests for DQNPipelineConfig."""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from deepqnetwork.kubeflow.pipeline.config_schema import DQNPipelineConfig


class TestFromYaml:
    """Tests for DQNPipelineConfig.from_yaml()."""

    def test_loads_default_yaml(self) -> None:
        config = DQNPipelineConfig.from_yaml(
            "deepqnetwork/kubeflow/config/dqn_pipeline_config.yaml"
        )
        assert config.symbol == "USDJPY"
        assert config.gamma == 0.99
        assert config.learning_rate == 0.0001
        assert config.hidden_dims == [256, 256, 128]
        assert config.betas == (0.9, 0.999)
        assert config.sharpe_absolute_threshold == 1.0
        assert config.pnl_absolute_threshold == 0.0

    def test_ignores_unknown_keys(self, tmp_path: Path) -> None:
        yaml_content = {"symbol": "AUDJPY", "unknown_key": "should_be_ignored"}
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml.dump(yaml_content))

        config = DQNPipelineConfig.from_yaml(str(config_path))
        assert config.symbol == "AUDJPY"

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        config = DQNPipelineConfig.from_yaml(str(config_path))
        assert config.symbol == "USDJPY"
        assert config.gamma == 0.99

    def test_betas_loaded_as_tuple(self, tmp_path: Path) -> None:
        yaml_content = {"betas": [0.8, 0.99]}
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml.dump(yaml_content))

        config = DQNPipelineConfig.from_yaml(str(config_path))
        assert config.betas == (0.8, 0.99)
        assert isinstance(config.betas, tuple)


class TestOverride:
    """Tests for DQNPipelineConfig.override()."""

    def test_returns_new_instance(self) -> None:
        config = DQNPipelineConfig()
        overridden = config.override(symbol="AUDJPY")
        assert overridden is not config
        assert overridden.symbol == "AUDJPY"
        assert config.symbol == "USDJPY"

    def test_multiple_overrides(self) -> None:
        config = DQNPipelineConfig()
        overridden = config.override(
            symbol="AUDJPY", learning_rate=0.001, num_episodes_per_range=500
        )
        assert overridden.symbol == "AUDJPY"
        assert overridden.learning_rate == 0.001
        assert overridden.num_episodes_per_range == 500

    def test_preserves_betas_as_tuple(self) -> None:
        config = DQNPipelineConfig()
        overridden = config.override(symbol="AUDJPY")
        assert isinstance(overridden.betas, tuple)


class TestValidate:
    """Tests for DQNPipelineConfig.validate()."""

    def test_default_config_is_valid(self) -> None:
        config = DQNPipelineConfig()
        assert config.validate() == []

    def test_yaml_config_is_valid(self) -> None:
        config = DQNPipelineConfig.from_yaml(
            "deepqnetwork/kubeflow/config/dqn_pipeline_config.yaml"
        )
        assert config.validate() == []

    def test_invalid_symbol(self) -> None:
        config = DQNPipelineConfig(symbol="GBPUSD")
        errors = config.validate()
        assert any("symbol" in e.lower() for e in errors)

    def test_negative_gamma(self) -> None:
        config = DQNPipelineConfig(gamma=-0.1)
        errors = config.validate()
        assert any("gamma" in e for e in errors)

    def test_gamma_above_one(self) -> None:
        config = DQNPipelineConfig(gamma=1.5)
        errors = config.validate()
        assert any("gamma" in e for e in errors)

    def test_invalid_activation(self) -> None:
        config = DQNPipelineConfig(activation="sigmoid")
        errors = config.validate()
        assert any("activation" in e.lower() for e in errors)

    def test_learning_rate_too_high(self) -> None:
        config = DQNPipelineConfig(learning_rate=0.1)
        errors = config.validate()
        assert any("learning_rate" in e for e in errors)

    def test_learning_rate_too_low(self) -> None:
        config = DQNPipelineConfig(learning_rate=1e-7)
        errors = config.validate()
        assert any("learning_rate" in e for e in errors)

    def test_invalid_loss_function(self) -> None:
        config = DQNPipelineConfig(loss_function="l1")
        errors = config.validate()
        assert any("loss_function" in e for e in errors)

    def test_empty_hidden_dims(self) -> None:
        config = DQNPipelineConfig(hidden_dims=[])
        errors = config.validate()
        assert any("hidden_dims" in e for e in errors)

    def test_negative_hidden_dim(self) -> None:
        config = DQNPipelineConfig(hidden_dims=[256, -1, 128])
        errors = config.validate()
        assert any("hidden_dims" in e for e in errors)

    def test_num_workers_out_of_range(self) -> None:
        config = DQNPipelineConfig(num_workers=10)
        errors = config.validate()
        assert any("num_workers" in e for e in errors)

    def test_epsilon_end_greater_than_start(self) -> None:
        config = DQNPipelineConfig(epsilon_start=0.5, epsilon_end=0.9)
        errors = config.validate()
        assert any("epsilon_end" in e for e in errors)

    def test_finetune_validates_lr(self) -> None:
        config = DQNPipelineConfig(
            training_mode="finetune", finetune_learning_rate=0.1
        )
        errors = config.validate()
        assert any("finetune_learning_rate" in e for e in errors)


class TestToCliArgs:
    """Tests for DQNPipelineConfig.to_cli_args()."""

    def test_returns_list_of_strings(self) -> None:
        config = DQNPipelineConfig()
        args = config.to_cli_args()
        assert isinstance(args, list)
        assert all(isinstance(a, str) for a in args)

    def test_contains_required_flags(self) -> None:
        config = DQNPipelineConfig()
        args = config.to_cli_args()
        assert "--symbol" in args
        assert "--learning-rate" in args
        assert "--num-episodes-per-range" in args
        assert "--mode" in args

    def test_scratch_mode_uses_base_lr(self) -> None:
        config = DQNPipelineConfig(learning_rate=0.001, training_mode="scratch")
        args = config.to_cli_args()
        lr_idx = args.index("--learning-rate")
        assert args[lr_idx + 1] == "0.001"

    def test_finetune_mode_uses_finetune_lr(self) -> None:
        config = DQNPipelineConfig(
            learning_rate=0.001,
            finetune_learning_rate=0.0001,
            training_mode="finetune",
        )
        args = config.to_cli_args()
        lr_idx = args.index("--learning-rate")
        assert args[lr_idx + 1] == "0.0001"

    def test_finetune_mode_uses_finetune_episodes(self) -> None:
        config = DQNPipelineConfig(
            num_episodes_per_range=3000,
            finetune_num_episodes_per_range=500,
            training_mode="finetune",
        )
        args = config.to_cli_args()
        ep_idx = args.index("--num-episodes-per-range")
        assert args[ep_idx + 1] == "500"


    def test_parseable_by_load_config(self) -> None:
        """CLI args generated by to_cli_args() can be parsed by load_config()."""
        from deepqnetwork.config import load_config

        config = DQNPipelineConfig(
            symbol="USDJPY",
            gamma=0.95,
            learning_rate=0.001,
            batch_size=128,
            num_episodes_per_range=1000,
        )
        cli_args = ["--config", "deepqnetwork/config.yaml"] + config.to_cli_args()
        parsed = load_config(cli_args)

        assert parsed.symbol == "USDJPY"
        assert parsed.gamma == 0.95
        assert parsed.learning_rate == 0.001
        assert parsed.batch_size == 128
        assert parsed.num_episodes_per_range == 1000
        assert parsed.mode == "train"
