"""Property-based tests for dqnpf-intraday Airflow DAGs.

Tests the precondition logic (parent model availability check) and parameter
forwarding for the dqnpf_revalidate DAG.

Uses Hypothesis for property-based testing with smart generators that
cover the full space of valid inputs.

**Validates: Requirements 23.6**
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Boolean strategies for parent model existence
parent_exists = st.booleans()

# Unix timestamps: positive integers representing valid episode ranges
valid_timestamps = st.integers(min_value=1_000_000_000, max_value=2_000_000_000)


# ---------------------------------------------------------------------------
# Helper: create a mock resolve_production_checkpoint
# ---------------------------------------------------------------------------


def _make_mock_resolve(dqn_exists: bool, forecaster_exists: bool):
    """Create a mock resolve_production_checkpoint function.

    Args:
        dqn_exists: Whether the DQN model has a production version.
        forecaster_exists: Whether the Forecaster model has a production version.

    Returns:
        A callable that mimics resolve_production_checkpoint behaviour.
    """

    def mock_resolve(model_name, lifecycle_stage="production"):
        if "deepqnetwork" in model_name:
            if not dqn_exists:
                raise ValueError(
                    f"No production-stage version found for model: {model_name}"
                )
            return "s3://bucket/dqn/checkpoint.pt"
        elif "probabilisticforecaster" in model_name:
            if not forecaster_exists:
                raise ValueError(
                    f"No production-stage version found for model: {model_name}"
                )
            return "s3://bucket/forecaster/checkpoint.pt"
        raise ValueError(f"Unknown model: {model_name}")

    return mock_resolve


# ---------------------------------------------------------------------------
# Property DQNPF-AF-1: Precondition fails fast on missing parent
# ---------------------------------------------------------------------------


class TestPreconditionFailsFastOnMissingParent:
    """Property DQNPF-AF-1: Precondition fails fast on missing parent.

    When either parent has no production version, the DAG raises before
    submitting the KFP run. When both parents have production versions,
    the precondition check passes without raising.

    **Validates: Requirements 23.6**
    """

    @given(
        dqn_exists=parent_exists,
        forecaster_exists=parent_exists,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_precondition_raises_when_parent_missing(
        self, dqn_exists, forecaster_exists
    ):
        """check_parent_models_available raises RuntimeError when either parent is missing.

        **Validates: Requirements 23.6**
        """
        mock_resolve = _make_mock_resolve(dqn_exists, forecaster_exists)

        # Mock the registry module that check_parent_models_available imports from
        mock_registry_module = MagicMock()
        mock_registry_module.resolve_production_checkpoint = mock_resolve

        with patch.dict(
            sys.modules,
            {
                "tradingmodel.intraday.dqnpf.kubeflow.registry.registry_client": mock_registry_module,
            },
        ):
            from tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils import (
                check_parent_models_available,
            )

            if dqn_exists and forecaster_exists:
                # Both parents available (should NOT raise
                check_parent_models_available()
            else:
                # At least one parent missing) MUST raise RuntimeError
                with pytest.raises(RuntimeError, match="Precondition failed"):
                    check_parent_models_available()

    @given(
        forecaster_exists=parent_exists,
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_precondition_mentions_dqn_when_dqn_missing(
        self, forecaster_exists
    ):
        """Error message mentions DQN parent when DQN model is missing.

        **Validates: Requirements 23.6**
        """
        mock_resolve = _make_mock_resolve(
            dqn_exists=False, forecaster_exists=forecaster_exists
        )

        mock_registry_module = MagicMock()
        mock_registry_module.resolve_production_checkpoint = mock_resolve

        with patch.dict(
            sys.modules,
            {
                "tradingmodel.intraday.dqnpf.kubeflow.registry.registry_client": mock_registry_module,
            },
        ):
            from tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils import (
                check_parent_models_available,
            )

            with pytest.raises(RuntimeError) as exc_info:
                check_parent_models_available()

            # Error message should mention the DQN parent
            assert "DQN parent model" in str(exc_info.value)

    @given(
        dqn_exists=parent_exists,
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_precondition_mentions_forecaster_when_forecaster_missing(
        self, dqn_exists
    ):
        """Error message mentions Forecaster parent when Forecaster model is missing.

        **Validates: Requirements 23.6**
        """
        mock_resolve = _make_mock_resolve(
            dqn_exists=dqn_exists, forecaster_exists=False
        )

        mock_registry_module = MagicMock()
        mock_registry_module.resolve_production_checkpoint = mock_resolve

        with patch.dict(
            sys.modules,
            {
                "tradingmodel.intraday.dqnpf.kubeflow.registry.registry_client": mock_registry_module,
            },
        ):
            from tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils import (
                check_parent_models_available,
            )

            with pytest.raises(RuntimeError) as exc_info:
                check_parent_models_available()

            # Error message should mention the Forecaster parent
            assert "Forecaster parent model" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Property DQNPF-AF-2: Parameter forwarding
# ---------------------------------------------------------------------------


class TestParameterForwarding:
    """Property DQNPF-AF-2: Parameter forwarding.

    `dqnpf_revalidate` passes operator-specified episode range to the KFP run.
    For any valid episode_start_ts and episode_end_ts, the submit_run() call
    receives the exact values specified by the operator.

    **Validates: Requirements 23.6**
    """

    @given(
        episode_start_ts=valid_timestamps,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_revalidate_forwards_episode_range_to_submit_run(
        self, episode_start_ts
    ):
        """dqnpf_revalidate passes operator-specified episode range to submit_run.

        **Validates: Requirements 23.6**
        """
        # Ensure end > start
        episode_end_ts = episode_start_ts + 86400  # 1 day later

        with patch(
            "tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils.Client"
        ) as mock_client_cls, patch(
            "tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils.tenacity.nap.time"
        ), patch(
            "tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils.logger"
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_run = MagicMock()
            mock_run.run_id = "dqnpf-revalidate-run-001"
            mock_client.create_run_from_pipeline_func.return_value = mock_run

            from tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils import (
                DqnpfKFPTrigger,
            )

            trigger = DqnpfKFPTrigger.__new__(DqnpfKFPTrigger)
            trigger.client = mock_client

            run_id = trigger.submit_run(
                integration_config_yaml="/etc/dqnpf/config/dqnpf_pipeline_config.yaml",
                dqn_model_registry_name="deepqnetwork-usdjpy",
                forecaster_model_registry_name="probabilisticforecaster-usdjpy",
                episode_start_ts=episode_start_ts,
                episode_end_ts=episode_end_ts,
            )

            assert run_id == "dqnpf-revalidate-run-001"

            # Verify the arguments dict passed to create_run_from_pipeline_func
            call_args = mock_client.create_run_from_pipeline_func.call_args
            arguments = call_args.kwargs.get("arguments") or call_args[1].get(
                "arguments"
            )

            # Episode range MUST be forwarded exactly
            assert arguments["episode_start_ts"] == episode_start_ts
            assert arguments["episode_end_ts"] == episode_end_ts

    @given(
        episode_start_ts=valid_timestamps,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_revalidate_includes_registry_names_alongside_episode_range(
        self, episode_start_ts
    ):
        """submit_run includes both registry names and episode range in arguments.

        **Validates: Requirements 23.6**
        """
        episode_end_ts = episode_start_ts + 3600

        with patch(
            "tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils.Client"
        ) as mock_client_cls, patch(
            "tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils.tenacity.nap.time"
        ), patch(
            "tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils.logger"
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_run = MagicMock()
            mock_run.run_id = "dqnpf-revalidate-run-002"
            mock_client.create_run_from_pipeline_func.return_value = mock_run

            from tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils import (
                DqnpfKFPTrigger,
            )

            trigger = DqnpfKFPTrigger.__new__(DqnpfKFPTrigger)
            trigger.client = mock_client

            dqn_name = "deepqnetwork-usdjpy"
            forecaster_name = "probabilisticforecaster-usdjpy"
            config_yaml = "/etc/dqnpf/config/custom_config.yaml"

            trigger.submit_run(
                integration_config_yaml=config_yaml,
                dqn_model_registry_name=dqn_name,
                forecaster_model_registry_name=forecaster_name,
                episode_start_ts=episode_start_ts,
                episode_end_ts=episode_end_ts,
            )

            call_args = mock_client.create_run_from_pipeline_func.call_args
            arguments = call_args.kwargs.get("arguments") or call_args[1].get(
                "arguments"
            )

            # All parameters must be present
            assert arguments["integration_config_yaml"] == config_yaml
            assert arguments["dqn_model_registry_name"] == dqn_name
            assert arguments["forecaster_model_registry_name"] == forecaster_name
            assert arguments["episode_start_ts"] == episode_start_ts
            assert arguments["episode_end_ts"] == episode_end_ts

    @given(
        episode_start_ts=valid_timestamps,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_weekly_backtest_does_not_include_episode_range(
        self, episode_start_ts
    ):
        """Weekly backtest submit_run without episode range omits those keys.

        **Validates: Requirements 23.6**
        """
        with patch(
            "tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils.Client"
        ) as mock_client_cls, patch(
            "tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils.tenacity.nap.time"
        ), patch(
            "tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils.logger"
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_run = MagicMock()
            mock_run.run_id = "dqnpf-weekly-run-001"
            mock_client.create_run_from_pipeline_func.return_value = mock_run

            from tradingmodel.intraday.dqnpf.kubeflow.airflow.dqnpf_kfp_utils import (
                DqnpfKFPTrigger,
            )

            trigger = DqnpfKFPTrigger.__new__(DqnpfKFPTrigger)
            trigger.client = mock_client

            # Weekly backtest does NOT pass episode range
            trigger.submit_run(
                integration_config_yaml="/etc/dqnpf/config/dqnpf_pipeline_config.yaml",
                dqn_model_registry_name="deepqnetwork-usdjpy",
                forecaster_model_registry_name="probabilisticforecaster-usdjpy",
            )

            call_args = mock_client.create_run_from_pipeline_func.call_args
            arguments = call_args.kwargs.get("arguments") or call_args[1].get(
                "arguments"
            )

            # Episode range keys should NOT be present
            assert "episode_start_ts" not in arguments
            assert "episode_end_ts" not in arguments
