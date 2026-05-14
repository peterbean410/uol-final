"""Property-based tests for Pipeline Configuration (config_schema.py).

Uses Hypothesis to verify correctness properties across randomly generated configurations.

**Validates: Requirements 8.1, 8.2, 8.3**
"""

import tempfile
import os
from dataclasses import asdict

import yaml
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from probabilisticforecaster.kubeflow.pipeline.config_schema import PipelineConfig


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

VALID_SYMBOLS = ("USDJPY", "AUDJPY")
VALID_FORECAST_HORIZONS = (1, 3, 6, 12)
VALID_NUM_LAYERS = (2, 3, 4)
VALID_NUM_HEADS = (2, 4, 8)
VALID_D_FF = (32, 64, 128)
VALID_BATCH_SIZES = (32, 64, 128)
VALID_LOOKBACK_WINDOWS = (24, 36, 48)
VALID_SCHEDULE_MODES = ("daily", "weekly", "on-demand", "drift-triggered")
VALID_TRAINING_MODES = ("scratch", "finetune")


@st.composite
def valid_pipeline_configs(draw):
    """Generate valid PipelineConfig instances that pass validation."""
    train_pct = draw(st.floats(min_value=0.1, max_value=0.8))
    test_pct = draw(st.floats(min_value=0.1, max_value=0.8))
    assume(train_pct + test_pct <= 1.0)

    return PipelineConfig(
        symbol=draw(st.sampled_from(VALID_SYMBOLS)),
        forecast_horizon=draw(st.sampled_from(VALID_FORECAST_HORIZONS)),
        lookback_window=draw(st.sampled_from(VALID_LOOKBACK_WINDOWS)),
        historical_window=draw(st.integers(min_value=100, max_value=5000)),
        num_features=16,
        num_layers=draw(st.sampled_from(VALID_NUM_LAYERS)),
        num_heads=draw(st.sampled_from(VALID_NUM_HEADS)),
        d_ff=draw(st.sampled_from(VALID_D_FF)),
        dropout=draw(
            st.floats(min_value=0.05, max_value=0.3, allow_nan=False, allow_infinity=False)
        ),
        learning_rate=draw(
            st.floats(min_value=0.0001, max_value=0.01, allow_nan=False, allow_infinity=False)
        ),
        batch_size=draw(st.sampled_from(VALID_BATCH_SIZES)),
        epochs=draw(st.integers(min_value=1, max_value=100)),
        random_seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
        train_start="2012-01-01",
        train_end="2022-12-31",
        test_start="2023-01-01",
        test_end="2026-04-30",
        data_start="2012-01-01",
        train_pct=train_pct,
        test_pct=test_pct,
        gpu_enabled=draw(st.booleans()),
        num_workers=draw(st.integers(min_value=1, max_value=4)),
        max_wall_time_hours=draw(st.integers(min_value=1, max_value=48)),
        katib_enabled=draw(st.booleans()),
        katib_max_trials=draw(st.integers(min_value=1, max_value=100)),
        katib_parallel_trials=draw(st.integers(min_value=1, max_value=10)),
        katib_trial_timeout_hours=draw(st.integers(min_value=1, max_value=24)),
        serving_min_replicas=draw(st.integers(min_value=0, max_value=4)),
        serving_max_replicas=draw(st.integers(min_value=1, max_value=10)),
        serving_target_concurrency=draw(st.integers(min_value=1, max_value=100)),
        canary_traffic_percent=draw(st.integers(min_value=0, max_value=100)),
        alert_webhook_url=draw(st.text(min_size=0, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz0123456789:/.@-_")),
        nll_degradation_threshold=draw(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
        da_degradation_threshold=draw(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
        schedule_mode=draw(st.sampled_from(VALID_SCHEDULE_MODES)),
        training_mode=draw(st.sampled_from(VALID_TRAINING_MODES)),
        finetune_epochs=draw(st.integers(min_value=1, max_value=10)),
        finetune_learning_rate=draw(
            st.floats(min_value=0.00001, max_value=0.01, allow_nan=False, allow_infinity=False)
        ),
    )


@st.composite
def invalid_pipeline_configs(draw):
    """Generate PipelineConfig instances with at least one invalid field.

    Deliberately sets one or more fields to out-of-range values.
    """
    # Pick which field(s) to invalidate
    invalid_field = draw(
        st.sampled_from([
            "symbol",
            "forecast_horizon",
            "learning_rate",
            "num_layers",
            "num_heads",
            "d_ff",
            "dropout",
            "batch_size",
            "lookback_window",
            "num_workers",
            "max_wall_time_hours",
            "train_pct",
            "test_pct",
            "pct_sum",
        ])
    )

    # Start with valid defaults
    config_kwargs = dict(
        symbol="USDJPY",
        forecast_horizon=1,
        lookback_window=36,
        num_layers=3,
        num_heads=4,
        d_ff=64,
        dropout=0.1,
        learning_rate=0.001,
        batch_size=64,
        num_workers=1,
        max_wall_time_hours=8,
        train_pct=0.75,
        test_pct=0.25,
    )

    # Invalidate the chosen field
    if invalid_field == "symbol":
        config_kwargs["symbol"] = draw(
            st.text(min_size=1, max_size=10, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            .filter(lambda s: s not in VALID_SYMBOLS)
        )
    elif invalid_field == "forecast_horizon":
        config_kwargs["forecast_horizon"] = draw(
            st.integers(min_value=-100, max_value=100)
            .filter(lambda x: x not in VALID_FORECAST_HORIZONS)
        )
    elif invalid_field == "learning_rate":
        config_kwargs["learning_rate"] = draw(
            st.one_of(
                st.floats(min_value=-1.0, max_value=0.00009, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.011, max_value=1.0, allow_nan=False, allow_infinity=False),
            )
        )
    elif invalid_field == "num_layers":
        config_kwargs["num_layers"] = draw(
            st.integers(min_value=-10, max_value=100)
            .filter(lambda x: x not in VALID_NUM_LAYERS)
        )
    elif invalid_field == "num_heads":
        config_kwargs["num_heads"] = draw(
            st.integers(min_value=-10, max_value=100)
            .filter(lambda x: x not in VALID_NUM_HEADS)
        )
    elif invalid_field == "d_ff":
        config_kwargs["d_ff"] = draw(
            st.integers(min_value=-10, max_value=1000)
            .filter(lambda x: x not in VALID_D_FF)
        )
    elif invalid_field == "dropout":
        config_kwargs["dropout"] = draw(
            st.one_of(
                st.floats(min_value=-1.0, max_value=0.049, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.301, max_value=1.0, allow_nan=False, allow_infinity=False),
            )
        )
    elif invalid_field == "batch_size":
        config_kwargs["batch_size"] = draw(
            st.integers(min_value=-10, max_value=1000)
            .filter(lambda x: x not in VALID_BATCH_SIZES)
        )
    elif invalid_field == "lookback_window":
        config_kwargs["lookback_window"] = draw(
            st.integers(min_value=-10, max_value=1000)
            .filter(lambda x: x not in VALID_LOOKBACK_WINDOWS)
        )
    elif invalid_field == "num_workers":
        config_kwargs["num_workers"] = draw(
            st.one_of(
                st.integers(min_value=-10, max_value=0),
                st.integers(min_value=5, max_value=100),
            )
        )
    elif invalid_field == "max_wall_time_hours":
        config_kwargs["max_wall_time_hours"] = draw(
            st.integers(min_value=-100, max_value=0)
        )
    elif invalid_field == "train_pct":
        config_kwargs["train_pct"] = draw(
            st.one_of(
                st.floats(min_value=-1.0, max_value=0.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            )
        )
    elif invalid_field == "test_pct":
        config_kwargs["test_pct"] = draw(
            st.one_of(
                st.floats(min_value=-1.0, max_value=0.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            )
        )
    elif invalid_field == "pct_sum":
        # Both valid individually but sum > 1.0
        config_kwargs["train_pct"] = draw(
            st.floats(min_value=0.6, max_value=0.9, allow_nan=False, allow_infinity=False)
        )
        config_kwargs["test_pct"] = draw(
            st.floats(min_value=0.6, max_value=0.9, allow_nan=False, allow_infinity=False)
        )
        assume(config_kwargs["train_pct"] + config_kwargs["test_pct"] > 1.0)

    return PipelineConfig(**config_kwargs)


# Fields that can be overridden in the override test
OVERRIDABLE_FIELDS = {
    "symbol": st.sampled_from(VALID_SYMBOLS),
    "forecast_horizon": st.sampled_from(VALID_FORECAST_HORIZONS),
    "lookback_window": st.sampled_from(VALID_LOOKBACK_WINDOWS),
    "num_layers": st.sampled_from(VALID_NUM_LAYERS),
    "num_heads": st.sampled_from(VALID_NUM_HEADS),
    "d_ff": st.sampled_from(VALID_D_FF),
    "dropout": st.floats(min_value=0.05, max_value=0.3, allow_nan=False, allow_infinity=False),
    "learning_rate": st.floats(min_value=0.0001, max_value=0.01, allow_nan=False, allow_infinity=False),
    "batch_size": st.sampled_from(VALID_BATCH_SIZES),
    "epochs": st.integers(min_value=1, max_value=100),
    "num_workers": st.integers(min_value=1, max_value=4),
    "max_wall_time_hours": st.integers(min_value=1, max_value=48),
    "gpu_enabled": st.booleans(),
    "katib_enabled": st.booleans(),
    "training_mode": st.sampled_from(VALID_TRAINING_MODES),
    "schedule_mode": st.sampled_from(VALID_SCHEDULE_MODES),
}


# ---------------------------------------------------------------------------
# Property 16: YAML Configuration Round-Trip
# ---------------------------------------------------------------------------


class TestYAMLConfigurationRoundTrip:
    """Property 16: YAML configuration round-trip.

    For any valid PipelineConfig, serialize to YAML and deserialize back
    produces equivalent config.

    **Validates: Requirements 8.1, 8.2, 8.3**
    """

    @given(config=valid_pipeline_configs())
    @settings(max_examples=100, deadline=None)
    def test_yaml_round_trip_preserves_config(self, config: PipelineConfig):
        """Serializing a valid PipelineConfig to YAML and loading it back
        produces an equivalent configuration.

        **Validates: Requirements 8.1**
        """
        # Serialize config to YAML via a temp file
        config_dict = asdict(config)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            tmp_path = f.name

        try:
            # Deserialize back using from_yaml
            loaded_config = PipelineConfig.from_yaml(tmp_path)

            # All fields should be equivalent
            loaded_dict = asdict(loaded_config)
            for field_name, original_value in config_dict.items():
                loaded_value = loaded_dict[field_name]
                if isinstance(original_value, float):
                    # Float comparison with tolerance for YAML serialization
                    assert abs(original_value - loaded_value) < 1e-10, (
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
# Property 17: Parameter Override Precedence
# ---------------------------------------------------------------------------


class TestParameterOverridePrecedence:
    """Property 17: Parameter override precedence.

    Overriding a single parameter preserves all other values.

    **Validates: Requirements 8.1, 8.2**
    """

    @given(
        config=valid_pipeline_configs(),
        field_name=st.sampled_from(list(OVERRIDABLE_FIELDS.keys())),
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_override_preserves_other_fields(
        self, config: PipelineConfig, field_name: str, data
    ):
        """Overriding a single parameter preserves all other parameter values.

        **Validates: Requirements 8.2**
        """
        # Draw a new value for the chosen field
        new_value = data.draw(OVERRIDABLE_FIELDS[field_name])

        # Apply the override
        overridden = config.override(**{field_name: new_value})

        # The overridden field should have the new value
        original_dict = asdict(config)
        overridden_dict = asdict(overridden)

        assert overridden_dict[field_name] == new_value, (
            f"Override did not apply: expected {field_name}={new_value}, "
            f"got {overridden_dict[field_name]}"
        )

        # All other fields should remain unchanged
        for other_field, original_value in original_dict.items():
            if other_field == field_name:
                continue
            overridden_value = overridden_dict[other_field]
            if isinstance(original_value, float):
                assert abs(original_value - overridden_value) < 1e-10, (
                    f"Field '{other_field}' changed after overriding '{field_name}': "
                    f"original={original_value}, after_override={overridden_value}"
                )
            else:
                assert original_value == overridden_value, (
                    f"Field '{other_field}' changed after overriding '{field_name}': "
                    f"original={original_value}, after_override={overridden_value}"
                )


# ---------------------------------------------------------------------------
# Property 18: Parameter Validation Rejects Invalid Configurations
# ---------------------------------------------------------------------------


class TestParameterValidationRejectsInvalid:
    """Property 18: Parameter validation rejects invalid configurations.

    Configs with out-of-range values produce non-empty error lists.

    **Validates: Requirements 8.3**
    """

    @given(config=invalid_pipeline_configs())
    @settings(max_examples=100, deadline=None)
    def test_invalid_configs_produce_errors(self, config: PipelineConfig):
        """Configurations with out-of-range values produce non-empty error lists.

        **Validates: Requirements 8.3**
        """
        errors = config.validate()
        assert len(errors) > 0, (
            f"Expected validation errors for invalid config but got none. "
            f"Config: symbol={config.symbol}, forecast_horizon={config.forecast_horizon}, "
            f"learning_rate={config.learning_rate}, num_layers={config.num_layers}, "
            f"num_heads={config.num_heads}, d_ff={config.d_ff}, dropout={config.dropout}, "
            f"batch_size={config.batch_size}, lookback_window={config.lookback_window}, "
            f"num_workers={config.num_workers}, max_wall_time_hours={config.max_wall_time_hours}, "
            f"train_pct={config.train_pct}, test_pct={config.test_pct}"
        )

    @given(config=valid_pipeline_configs())
    @settings(max_examples=100, deadline=None)
    def test_valid_configs_produce_no_errors(self, config: PipelineConfig):
        """Valid configurations produce empty error lists (complementary check).

        **Validates: Requirements 8.3**
        """
        errors = config.validate()
        assert len(errors) == 0, (
            f"Expected no validation errors for valid config but got: {errors}"
        )
