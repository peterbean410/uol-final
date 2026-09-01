"""Property-based tests for DQN Model Registry operations.

Tests the logic of the DQNRegistryClient including metadata completeness,
lifecycle state transitions, and retention policy enforcement.

Uses an in-memory fake registry to validate logic without requiring a real
Kubeflow Model Registry server.

**Validates: Requirements DQN-R19, DQN-R20**
"""

import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import patch, MagicMock

_mock_model_registry = MagicMock()
sys.modules["model_registry"] = _mock_model_registry
sys.modules["model_registry.types"] = MagicMock()

from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from deepqnetwork.kubeflow.registry.dqn_registry_client import (
    DQNRegistryClient,
    DQNModelMetadata,
    _DQN_VALID_TRANSITIONS,
)
from deepqnetwork.kubeflow.registry.dqn_lifecycle import DQNLifecycleManager


class FakeModelVersion:
    """Simulates a model version stored in the registry."""

    def __init__(self, name: str, version: str, model_name: str, uri: str,
                 description: str, custom_properties: dict):
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
    allowing the DQNRegistryClient logic to be tested without
    a real server.
    """

    def __init__(self, **kwargs):
        self.registered_models: dict[str, FakeRegisteredModel] = {}
        self.model_versions: dict[str, list[FakeModelVersion]] = {}

    def get_registered_model(self, name: str) -> FakeRegisteredModel:
        if name in self.registered_models:
            return self.registered_models[name]
        raise Exception(f"Model {name} not found")

    def register_model(self, name: str, uri: str = "", description: str = "") -> FakeRegisteredModel:
        rm = FakeRegisteredModel(name=name, description=description)
        self.registered_models[name] = rm
        self.model_versions[name] = []
        return rm

    def register_model_version(self, name: str, version: str, model_name: str,
                               uri: str, description: str,
                               custom_properties: dict) -> FakeModelVersion:
        mv = FakeModelVersion(
            name=name, version=version, model_name=model_name,
            uri=uri, description=description, custom_properties=custom_properties,
        )
        if model_name not in self.model_versions:
            self.model_versions[model_name] = []
        self.model_versions[model_name].append(mv)
        return mv

    def get_registered_models(self) -> list[FakeRegisteredModel]:
        return list(self.registered_models.values())

    def get_model_versions(self, model_name: str) -> list[FakeModelVersion]:
        return self.model_versions.get(model_name, [])

    def update_model_version(self, version: FakeModelVersion,
                             custom_properties: dict) -> None:
        version.custom_properties = custom_properties


def create_test_client() -> DQNRegistryClient:
    """Create a DQNRegistryClient backed by the in-memory fake."""
    with patch(
        "deepqnetwork.kubeflow.registry.dqn_registry_client.ModelRegistry",
        FakeModelRegistry,
    ):
        client = DQNRegistryClient(registry_url="http://fake:8080")
    return client


def create_test_lifecycle_manager() -> tuple[DQNLifecycleManager, DQNRegistryClient]:
    """Create a DQNLifecycleManager with a fake-backed registry client."""
    client = create_test_client()
    manager = DQNLifecycleManager(registry_client=client)
    return manager, client


symbols = st.sampled_from(["USDJPY", "AUDJPY", "EURUSD", "GBPUSD"])
step_sizes = st.sampled_from([60, 300, 900, 3600])
sharpe_strategy = st.floats(min_value=-2.0, max_value=5.0, allow_nan=False, allow_infinity=False)
pnl_strategy = st.floats(min_value=-10000.0, max_value=50000.0, allow_nan=False, allow_infinity=False)
drawdown_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
win_rate_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
stages = st.sampled_from(["staging", "production", "archived"])


@st.composite
def dqn_model_metadata(draw):
    """Generate a valid DQNModelMetadata instance."""
    symbol = draw(symbols)
    step_size = draw(step_sizes)
    cumulative_pnl = draw(pnl_strategy)
    sharpe_ratio = draw(sharpe_strategy)
    max_drawdown = draw(drawdown_strategy)
    win_rate = draw(win_rate_strategy)
    hyperparams = {
        "learning_rate": draw(st.floats(min_value=1e-5, max_value=1e-2,
                                        allow_nan=False, allow_infinity=False)),
        "hidden_dims": draw(st.sampled_from([[128, 128], [256, 256, 128], [512, 256, 128]])),
        "gamma": draw(st.floats(min_value=0.9, max_value=0.999,
                                allow_nan=False, allow_infinity=False)),
        "batch_size": draw(st.sampled_from([32, 64, 128])),
    }
    pipeline_run_id = f"run-{draw(st.uuids())}"
    episode_start = draw(st.dates(
        min_value=datetime(2020, 1, 1).date(),
        max_value=datetime(2023, 6, 1).date(),
    ))
    episode_end = draw(st.dates(
        min_value=episode_start,
        max_value=datetime(2024, 1, 1).date(),
    ))
    training_timestamp = datetime.now(timezone.utc).isoformat()

    return DQNModelMetadata(
        symbol=symbol,
        episode_start_ts=episode_start.isoformat(),
        episode_end_ts=episode_end.isoformat(),
        step_size_seconds=step_size,
        training_timestamp=training_timestamp,
        cumulative_pnl=cumulative_pnl,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        hyperparameters=hyperparams,
        pipeline_run_id=pipeline_run_id,
        lifecycle_stage="staging",
    )


class TestDQNMetadataCompleteness:
    """Property DQN-11: Registered DQN models return complete metadata and lineage.

    After registering a model, querying it back must return all metadata
    fields (symbol, episode range, step_size, metrics, hyperparams) and
    lineage information (pipeline_run_id).

    **Validates: Requirements DQN-R19**
    """

    @given(metadata=dqn_model_metadata())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_registered_model_returns_complete_metadata(self, metadata: DQNModelMetadata):
        """Registering a DQN model and querying it back returns all metadata fields."""
        client = create_test_client()

        version_id = client.register_model(
            model_path="s3://bucket/dqn-models/checkpoint.pt",
            metadata=metadata,
        )

        results = client.query_models(symbol=metadata.symbol)

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        result = results[0]

        assert result.symbol == metadata.symbol
        assert result.episode_start_ts == metadata.episode_start_ts
        assert result.episode_end_ts == metadata.episode_end_ts
        assert result.step_size_seconds == metadata.step_size_seconds
        assert result.training_timestamp == metadata.training_timestamp
        assert result.cumulative_pnl == metadata.cumulative_pnl
        assert result.sharpe_ratio == metadata.sharpe_ratio
        assert result.max_drawdown == metadata.max_drawdown
        assert result.win_rate == metadata.win_rate
        assert result.hyperparameters == metadata.hyperparameters

        assert result.pipeline_run_id == metadata.pipeline_run_id

        assert result.lifecycle_stage == "staging"

    @given(metadata=dqn_model_metadata())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_version_id_is_valid_uuid(self, metadata: DQNModelMetadata):
        """Each registered model version has a valid UUID identifier."""
        client = create_test_client()

        version_id = client.register_model(
            model_path="s3://bucket/dqn-models/checkpoint.pt",
            metadata=metadata,
        )

        uuid.UUID(version_id)


class TestDQNLifecycleStateTransitions:
    """Property DQN-12: Only valid transitions succeed.

    Valid transitions: staging→production, staging→archived, production→archived.
    Archived is a terminal state. Invalid transitions must be rejected.

    **Validates: Requirements DQN-R20**
    """

    @given(
        current_stage=stages,
        target_stage=stages,
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_valid_transitions_succeed_invalid_fail(self, current_stage: str, target_stage: str):
        """Only valid lifecycle transitions succeed; invalid ones are rejected.

        Tests the _DQN_VALID_TRANSITIONS mapping directly to verify the state machine.
        """
        manager, _ = create_test_lifecycle_manager()

        valid_targets = _DQN_VALID_TRANSITIONS.get(current_stage, set())
        is_valid = target_stage in valid_targets

        result = manager.validate_transition(current_stage, target_stage)
        assert result == is_valid, (
            f"Transition {current_stage}→{target_stage}: "
            f"expected valid={is_valid}, got {result}"
        )

    @given(metadata=dqn_model_metadata())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_promote_from_staging_succeeds(self, metadata: DQNModelMetadata):
        """Promoting a DQN model from staging to production succeeds."""
        client = create_test_client()

        version_id = client.register_model(
            model_path="s3://bucket/dqn-models/checkpoint.pt",
            metadata=metadata,
        )

        result = client.promote_to_production(version_id)

        assert result is True, "Promotion from staging to production should succeed"

    @given(metadata=dqn_model_metadata())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_promote_from_production_fails(self, metadata: DQNModelMetadata):
        """Promoting a DQN model already in production fails (production→production is invalid)."""
        client = create_test_client()

        version_id = client.register_model(
            model_path="s3://bucket/dqn-models/checkpoint.pt",
            metadata=metadata,
        )

        client.promote_to_production(version_id)

        result = client.promote_to_production(version_id)

        assert result is False, "Promotion from production to production should fail"

    @given(metadata=dqn_model_metadata())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_lifecycle_manager_transition_staging_to_production(self, metadata: DQNModelMetadata):
        """DQNLifecycleManager.transition_stage from staging to production succeeds."""
        manager, client = create_test_lifecycle_manager()

        version_id = client.register_model(
            model_path="s3://bucket/dqn-models/checkpoint.pt",
            metadata=metadata,
        )

        result = manager.transition_stage(version_id, "production")

        assert result is True, "Transition staging→production via lifecycle manager should succeed"

    @given(metadata=dqn_model_metadata())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_lifecycle_manager_transition_staging_to_archived(self, metadata: DQNModelMetadata):
        """DQNLifecycleManager.transition_stage from staging to archived succeeds."""
        manager, client = create_test_lifecycle_manager()

        version_id = client.register_model(
            model_path="s3://bucket/dqn-models/checkpoint.pt",
            metadata=metadata,
        )

        result = manager.transition_stage(version_id, "archived")
        assert result is True, "Transition staging→archived via lifecycle manager should succeed"

    @given(metadata=dqn_model_metadata())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_lifecycle_manager_transition_archived_to_production_fails(self, metadata: DQNModelMetadata):
        """DQNLifecycleManager.transition_stage from archived to production fails."""
        manager, client = create_test_lifecycle_manager()

        version_id = client.register_model(
            model_path="s3://bucket/dqn-models/checkpoint.pt",
            metadata=metadata,
        )

        manager.transition_stage(version_id, "archived")

        result = manager.transition_stage(version_id, "production")
        assert result is False, "Transition archived→production should fail (archived is terminal)"

    def test_archived_is_terminal(self):
        """Archived state has no valid transitions (terminal state)."""
        assert _DQN_VALID_TRANSITIONS["archived"] == set(), (
            "Archived should be a terminal state with no valid transitions"
        )

    def test_staging_can_transition_to_production_and_archived(self):
        """Staging can transition to production or archived."""
        assert _DQN_VALID_TRANSITIONS["staging"] == {"production", "archived"}

    def test_production_can_only_transition_to_archived(self):
        """Production can only transition to archived."""
        assert _DQN_VALID_TRANSITIONS["production"] == {"archived"}


class TestDQNRetentionPolicyEnforcement:
    """Property DQN-13: Models <90 days cannot be archived, models ≥90 days can.

    The can_archive() method enforces the 90-day retention policy based on
    the model's created_at timestamp.

    **Validates: Requirements DQN-R20**
    """

    @given(
        metadata=dqn_model_metadata(),
        age_days=st.integers(min_value=0, max_value=89),
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_models_under_90_days_cannot_be_archived(self, metadata: DQNModelMetadata, age_days: int):
        """DQN models less than 90 days old cannot be archived."""
        client = create_test_client()

        version_id = client.register_model(
            model_path="s3://bucket/dqn-models/checkpoint.pt",
            metadata=metadata,
        )

        created_at = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
        for rm in client.registry.get_registered_models():
            for mv in client.registry.get_model_versions(rm.name):
                if mv.custom_properties.get("version_id") == version_id:
                    mv.custom_properties["created_at"] = created_at

        result = client.can_archive(version_id)
        assert result is False, (
            f"DQN model {age_days} days old should NOT be archivable (< 90 days)"
        )

    @given(
        metadata=dqn_model_metadata(),
        age_days=st.integers(min_value=90, max_value=365),
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_models_90_days_or_older_can_be_archived(self, metadata: DQNModelMetadata, age_days: int):
        """DQN models 90 days or older can be archived."""
        client = create_test_client()

        version_id = client.register_model(
            model_path="s3://bucket/dqn-models/checkpoint.pt",
            metadata=metadata,
        )

        created_at = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
        for rm in client.registry.get_registered_models():
            for mv in client.registry.get_model_versions(rm.name):
                if mv.custom_properties.get("version_id") == version_id:
                    mv.custom_properties["created_at"] = created_at

        result = client.can_archive(version_id)
        assert result is True, (
            f"DQN model {age_days} days old should be archivable (>= 90 days)"
        )

    def test_retention_days_constant_is_90(self):
        """The RETENTION_DAYS constant is set to 90."""
        assert DQNRegistryClient.RETENTION_DAYS == 90
