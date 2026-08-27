"""Property-based tests for DQNPF combined Model Registry operations.

Tests the logic of the DqnpfRegistryClient including threshold gating,
production resolution, and retention policy enforcement.

Uses an in-memory fake Model Registry to validate logic without requiring a real
Kubeflow Model Registry server.

**Validates: Requirements 23.1, 23.2, 23.3**
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

# Mock the model_registry package before importing the registry client
_mock_model_registry = MagicMock()
sys.modules["model_registry"] = _mock_model_registry
sys.modules["model_registry.types"] = MagicMock()

from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from tradingmodel.intraday.dqnpf.backtest import BacktestComparison, ThresholdReport
from tradingmodel.intraday.dqnpf.config import IntegrationConfig


# ---------------------------------------------------------------------------
# In-memory fake Model Registry
# ---------------------------------------------------------------------------


class FakeModelVersion:
    """Simulates a model version stored in the registry."""

    def __init__(
        self,
        name: str,
        version: str,
        model_name: str,
        uri: str,
        description: str,
        custom_properties: dict,
    ):
        self.name = name
        self.version = version
        self.model_name = model_name
        self.uri = uri
        self.description = description
        self.custom_properties = custom_properties


class FakeRegisteredModel:
    """Simulates a registered model in the registry."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


class FakeModelRegistry:
    """In-memory fake of the Kubeflow Model Registry.

    Stores registered models and their versions in dictionaries,
    allowing the DqnpfRegistryClient logic to be tested without
    a real server.
    """

    def __init__(self, **kwargs):
        self.registered_models: dict[str, FakeRegisteredModel] = {}
        self.model_versions: dict[str, list[FakeModelVersion]] = {}

    def get_registered_model(self, name: str) -> FakeRegisteredModel:
        if name in self.registered_models:
            return self.registered_models[name]
        raise Exception(f"Model {name} not found")

    def register_model(
        self, name: str, uri: str = "", description: str = ""
    ) -> FakeRegisteredModel:
        rm = FakeRegisteredModel(name=name, description=description)
        self.registered_models[name] = rm
        self.model_versions[name] = []
        return rm

    def register_model_version(
        self,
        name: str,
        version: str,
        model_name: str,
        uri: str,
        description: str,
        custom_properties: dict,
    ) -> FakeModelVersion:
        mv = FakeModelVersion(
            name=name,
            version=version,
            model_name=model_name,
            uri=uri,
            description=description,
            custom_properties=custom_properties,
        )
        if model_name not in self.model_versions:
            self.model_versions[model_name] = []
        self.model_versions[model_name].append(mv)
        return mv

    def get_registered_models(self) -> list[FakeRegisteredModel]:
        return list(self.registered_models.values())

    def get_model_versions(self, model_name: str) -> list[FakeModelVersion]:
        return self.model_versions.get(model_name, [])

    def update_model_version(
        self, version: FakeModelVersion, custom_properties: dict
    ) -> None:
        version.custom_properties.update(custom_properties)


# ---------------------------------------------------------------------------
# Import the registry client (after mocking model_registry)
# ---------------------------------------------------------------------------

from tradingmodel.intraday.dqnpf.kubeflow.registry.registry_client import (
    DqnpfRegistryClient,
    resolve_production_checkpoint,
)


# ---------------------------------------------------------------------------
# Helper: create a DqnpfRegistryClient with the fake registry
# ---------------------------------------------------------------------------


def create_test_client() -> DqnpfRegistryClient:
    """Create a DqnpfRegistryClient backed by the in-memory fake."""
    with patch(
        "tradingmodel.intraday.dqnpf.kubeflow.registry.registry_client.ModelRegistry",
        FakeModelRegistry,
    ):
        client = DqnpfRegistryClient(registry_url="http://fake:8080")
    return client


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

symbols = st.sampled_from(["USDJPY", "AUDJPY", "EURUSD", "GBPUSD"])

sharpe_strategy = st.floats(
    min_value=-2.0, max_value=5.0, allow_nan=False, allow_infinity=False
)
pnl_strategy = st.floats(
    min_value=-10000.0, max_value=50000.0, allow_nan=False, allow_infinity=False
)
drawdown_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
suppression_rate_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
trade_count_strategy = st.integers(min_value=0, max_value=10000)
time_fraction_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


@st.composite
def quarterly_pnl(draw):
    """Generate a quarterly PnL dict."""
    quarters = draw(
        st.lists(
            st.sampled_from(["2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4", "2024-Q1"]),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    return {q: draw(pnl_strategy) for q in quarters}


@st.composite
def suppression_by_reason(draw):
    """Generate a suppression_by_reason dict."""
    reasons = ["budget_exhausted", "directional_conflict"]
    return {r: draw(st.integers(min_value=0, max_value=500)) for r in reasons}


@st.composite
def backtest_comparison(draw):
    """Generate a valid BacktestComparison instance."""
    return BacktestComparison(
        combined_return=draw(pnl_strategy),
        baseline_return=draw(pnl_strategy),
        combined_sharpe=draw(sharpe_strategy),
        baseline_sharpe=draw(sharpe_strategy),
        suppression_rate=draw(suppression_rate_strategy),
        suppression_by_reason=draw(suppression_by_reason()),
        high_sigma_pnl_combined=draw(pnl_strategy),
        high_sigma_pnl_baseline=draw(pnl_strategy),
        low_sigma_pnl_combined=draw(pnl_strategy),
        low_sigma_pnl_baseline=draw(pnl_strategy),
        trades_combined=draw(trade_count_strategy),
        trades_baseline=draw(trade_count_strategy),
        high_sigma_trades_combined=draw(trade_count_strategy),
        high_sigma_trades_baseline=draw(trade_count_strategy),
        high_sigma_negative_pnl_proportion_combined=draw(suppression_rate_strategy),
        high_sigma_negative_pnl_proportion_baseline=draw(suppression_rate_strategy),
        quarterly_pnl_combined=draw(quarterly_pnl()),
        quarterly_pnl_baseline=draw(quarterly_pnl()),
        high_sigma_time_fraction=draw(time_fraction_strategy),
    )


@st.composite
def threshold_report_failed(draw):
    """Generate a ThresholdReport with passed=False."""
    failures = draw(
        st.lists(
            st.sampled_from([
                "combined_sharpe < baseline_sharpe",
                "suppression_rate > 0.6",
                "high_sigma_pnl_combined < 0",
                "combined_return < baseline_return",
            ]),
            min_size=1,
            max_size=3,
        )
    )
    return ThresholdReport(passed=False, failures=failures)


def threshold_report_passed():
    """Create a ThresholdReport with passed=True."""
    return ThresholdReport(passed=True, failures=[])


@st.composite
def integration_config(draw):
    """Generate a valid IntegrationConfig instance."""
    return IntegrationConfig(
        symbol=draw(symbols),
        variance_threshold=draw(
            st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        ),
        max_risk_long_units=draw(st.integers(min_value=1, max_value=5)),
        max_risk_short_units=draw(st.integers(min_value=1, max_value=5)),
        forecast_horizon=draw(st.sampled_from([1, 3, 6, 12])),
        step_size_seconds=draw(st.sampled_from([60, 300])),
    )


s3_paths = st.builds(
    lambda prefix, uid: f"s3://prod-fintech-forex-sg/dqnpf/{prefix}/{uid}/checkpoint.pt",
    prefix=st.sampled_from(["combined", "staging", "prod"]),
    uid=st.uuids().map(str),
)

pipeline_run_ids = st.builds(lambda uid: f"run-{uid}", uid=st.uuids().map(str))

version_strings = st.builds(lambda uid: f"v-{uid}", uid=st.uuids().map(str))


# ---------------------------------------------------------------------------
# Property DQNPF-REG-1: Threshold-failed runs never enter staging
# ---------------------------------------------------------------------------


class TestThresholdFailedRunsNeverEnterStaging:
    """Property DQNPF-REG-1: Threshold-failed runs never enter staging.

    `register_combined_version()` with `threshold_report.passed == False`
    raises and does not write an entry.

    **Validates: Requirements 23.2**
    """

    @given(
        comparison=backtest_comparison(),
        report=threshold_report_failed(),
        config=integration_config(),
        dqn_version=version_strings,
        forecaster_version=version_strings,
        run_id=pipeline_run_ids,
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_failed_threshold_raises_and_no_entry_written(
        self,
        comparison: BacktestComparison,
        report: ThresholdReport,
        config: IntegrationConfig,
        dqn_version: str,
        forecaster_version: str,
        run_id: str,
    ):
        """Registering with a failed threshold report raises ValueError and writes nothing."""
        client = create_test_client()

        # Count entries before the attempt
        versions_before = sum(
            len(vs) for vs in client.registry.model_versions.values()
        )

        # Attempt to register (should raise
        try:
            client.register_combined_version(
                comparison=comparison,
                threshold_report=report,
                integration_config=config,
                parent_dqn_version=dqn_version,
                parent_forecaster_version=forecaster_version,
                pipeline_run_id=run_id,
            )
            # If we get here, the function did NOT raise) that's a failure
            assert False, (
                "register_combined_version() should raise when "
                "threshold_report.passed == False"
            )
        except (ValueError, RuntimeError):
            pass  # Expected: registration rejected

        # Verify no entry was written
        versions_after = sum(
            len(vs) for vs in client.registry.model_versions.values()
        )
        assert versions_after == versions_before, (
            f"Expected no new entries after failed threshold, but "
            f"count went from {versions_before} to {versions_after}"
        )


# ---------------------------------------------------------------------------
# Property DQNPF-REG-2: Production lookup returns latest
# ---------------------------------------------------------------------------


class TestProductionLookupReturnsLatest:
    """Property DQNPF-REG-2: Production lookup returns latest.

    Among multiple production-stage entries for a model,
    `resolve_production_checkpoint()` returns the most recent `registered_at`.

    **Validates: Requirements 23.1**
    """

    @given(
        comparisons=st.lists(backtest_comparison(), min_size=2, max_size=5),
        configs=st.lists(integration_config(), min_size=2, max_size=5),
        s3_path_list=st.lists(s3_paths, min_size=2, max_size=5),
    )
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_resolve_returns_most_recent_production_entry(
        self,
        comparisons: list[BacktestComparison],
        configs: list[IntegrationConfig],
        s3_path_list: list[str],
    ):
        """Resolving production checkpoint returns the entry with the latest registered_at."""
        # Ensure lists are same length
        n = min(len(comparisons), len(configs), len(s3_path_list))
        assume(n >= 2)
        comparisons = comparisons[:n]
        configs = configs[:n]
        s3_path_list = s3_path_list[:n]

        client = create_test_client()
        model_name = "dqnpf-intraday-usdjpy"

        # Register the model in the fake registry
        client.registry.register_model(name=model_name)

        # Register multiple versions with increasing timestamps and promote to production
        registered_at_times: list[datetime] = []
        for i in range(n):
            ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
            registered_at_times.append(ts)

            version_id = str(uuid.uuid4())
            custom_props = {
                "version_id": version_id,
                "lifecycle_stage": "production",
                "registered_at": ts.isoformat(),
                "s3_path": s3_path_list[i],
                "pipeline_run_id": f"run-{uuid.uuid4()}",
                "parent_dqn_version": f"dqn-v{i}",
                "parent_forecaster_version": f"fc-v{i}",
            }
            client.registry.register_model_version(
                name=f"{model_name}-v{i}",
                version=str(i),
                model_name=model_name,
                uri=s3_path_list[i],
                description=f"Combined version {i}",
                custom_properties=custom_props,
            )

        # The latest entry is the one with the highest registered_at
        expected_path = s3_path_list[n - 1]

        # Resolve production checkpoint
        with patch(
            "tradingmodel.intraday.dqnpf.kubeflow.registry.registry_client.ModelRegistry",
            FakeModelRegistry,
        ):
            # Use the client's internal registry for resolution
            result = client.resolve_production_checkpoint(model_name)

        assert result == expected_path, (
            f"Expected latest production path '{expected_path}', got '{result}'"
        )


# ---------------------------------------------------------------------------
# Property DQNPF-REG-2b: Parent models resolve via artifact uri (no s3_path)
# ---------------------------------------------------------------------------


class TestParentModelResolvesViaUri:
    """Parent models (DQN/forecaster) carry no ``s3_path`` custom property.

    They are registered via ModelRegistry.register_model(uri=...), so the
    checkpoint path lives on the version/artifact uri. resolve_production_
    checkpoint() must return that uri rather than raising KeyError on a
    missing ``s3_path`` key (regression for the dqnpf-backtest failure).

    **Validates: Requirements 23.1**
    """

    @given(uri=s3_paths, created_offset=st.integers(min_value=0, max_value=10))
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_resolve_returns_uri_when_no_s3_path(
        self, uri: str, created_offset: int
    ):
        """A production version without s3_path resolves to its uri."""
        client = create_test_client()
        model_name = "deepqnetwork-usdjpy"
        client.registry.register_model(name=model_name)

        created_at = (
            datetime(2024, 1, 1, tzinfo=timezone.utc)
            + timedelta(days=created_offset)
        ).isoformat()
        # Parent-model custom_properties: created_at, lifecycle_stage; NO s3_path.
        client.registry.register_model_version(
            name=str(uuid.uuid4()),
            version="1",
            model_name=model_name,
            uri=uri,
            description="Parent DQN version",
            custom_properties={
                "version_id": str(uuid.uuid4()),
                "lifecycle_stage": "production",
                "created_at": created_at,
            },
        )

        assert client.resolve_production_checkpoint(model_name) == uri


# ---------------------------------------------------------------------------
# Property DQNPF-REG-3: Archive respects retention
# ---------------------------------------------------------------------------


class TestArchiveRespectsRetention:
    """Property DQNPF-REG-3: Archive respects retention.

    Archiving a version younger than 90 days raises.

    **Validates: Requirements 23.3**
    """

    @given(
        comparison=backtest_comparison(),
        config=integration_config(),
        age_days=st.integers(min_value=0, max_value=89),
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_archive_version_younger_than_90_days_raises(
        self,
        comparison: BacktestComparison,
        config: IntegrationConfig,
        age_days: int,
    ):
        """Archiving a version younger than 90 days raises an error."""
        client = create_test_client()
        model_name = "dqnpf-intraday-usdjpy"

        # Register the model
        client.registry.register_model(name=model_name)

        # Create a version with registered_at = age_days ago
        registered_at = datetime.now(timezone.utc) - timedelta(days=age_days)
        version_id = str(uuid.uuid4())
        custom_props = {
            "version_id": version_id,
            "lifecycle_stage": "staging",
            "registered_at": registered_at.isoformat(),
            "s3_path": f"s3://bucket/dqnpf/{version_id}/checkpoint.pt",
            "pipeline_run_id": f"run-{uuid.uuid4()}",
            "parent_dqn_version": "dqn-v1",
            "parent_forecaster_version": "fc-v1",
        }
        client.registry.register_model_version(
            name=f"{model_name}-{version_id}",
            version="1",
            model_name=model_name,
            uri=custom_props["s3_path"],
            description="Test version",
            custom_properties=custom_props,
        )

        # Attempt to archive, should raise for versions < 90 days old
        try:
            client.archive_version(version_id)
            assert False, (
                f"archive_version() should raise for version {age_days} days old "
                f"(< 90 days retention)"
            )
        except (ValueError, RuntimeError):
            pass  # Expected: retention policy blocks archival

    @given(
        comparison=backtest_comparison(),
        config=integration_config(),
        age_days=st.integers(min_value=90, max_value=365),
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_archive_version_90_days_or_older_succeeds(
        self,
        comparison: BacktestComparison,
        config: IntegrationConfig,
        age_days: int,
    ):
        """Archiving a version 90 days or older succeeds without error."""
        client = create_test_client()
        model_name = "dqnpf-intraday-usdjpy"

        # Register the model
        client.registry.register_model(name=model_name)

        # Create a version with registered_at = age_days ago
        registered_at = datetime.now(timezone.utc) - timedelta(days=age_days)
        version_id = str(uuid.uuid4())
        custom_props = {
            "version_id": version_id,
            "lifecycle_stage": "staging",
            "registered_at": registered_at.isoformat(),
            "s3_path": f"s3://bucket/dqnpf/{version_id}/checkpoint.pt",
            "pipeline_run_id": f"run-{uuid.uuid4()}",
            "parent_dqn_version": "dqn-v1",
            "parent_forecaster_version": "fc-v1",
        }
        client.registry.register_model_version(
            name=f"{model_name}-{version_id}",
            version="1",
            model_name=model_name,
            uri=custom_props["s3_path"],
            description="Test version",
            custom_properties=custom_props,
        )

        # Archiving should succeed for versions >= 90 days old
        client.archive_version(version_id)

        # Verify the version is now archived
        versions = client.registry.get_model_versions(model_name)
        archived_version = next(
            (v for v in versions if v.custom_properties.get("version_id") == version_id),
            None,
        )
        assert archived_version is not None
        assert archived_version.custom_properties["lifecycle_stage"] == "archived"
