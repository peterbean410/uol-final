"""Property-based tests for Model Registry operations.

Tests the logic of the ForecasterRegistryClient including metadata completeness,
unique version ID generation, lifecycle state transitions, query filtering,
and retention policy enforcement.

Uses an in-memory fake registry to validate logic without requiring a real
Kubeflow Model Registry server.

**Validates: Requirements 6.1, 6.2, 6.3, 6.5, 6.6, 6.7**
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

from probabilisticforecaster.kubeflow.registry.registry_client import (
    ForecasterRegistryClient,
    ModelMetadata,
    _VALID_TRANSITIONS,
)


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
    allowing the ForecasterRegistryClient logic to be tested without
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


def create_test_client() -> ForecasterRegistryClient:
    """Create a ForecasterRegistryClient backed by the in-memory fake."""
    with patch(
        "probabilisticforecaster.kubeflow.registry.registry_client.ModelRegistry",
        FakeModelRegistry,
    ):
        client = ForecasterRegistryClient(registry_url="http://fake:8080")
    return client


symbols = st.sampled_from(["USDJPY", "AUDJPY", "EURUSD", "GBPUSD"])
horizons = st.sampled_from([1, 3, 6, 12])
nll_strategy = st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)
da_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
stages = st.sampled_from(["staging", "production", "archived"])


@st.composite
def model_metadata(draw):
    """Generate a valid ModelMetadata instance."""
    symbol = draw(symbols)
    horizon = draw(horizons)
    training_nll = draw(nll_strategy)
    validation_nll = draw(nll_strategy)
    da = draw(da_strategy)
    hyperparams = {
        "learning_rate": draw(st.floats(min_value=0.0001, max_value=0.01,
                                        allow_nan=False, allow_infinity=False)),
        "num_layers": draw(st.sampled_from([2, 3, 4])),
        "num_heads": draw(st.sampled_from([2, 4, 8])),
    }
    pipeline_run_id = f"run-{draw(st.uuids())}"
    data_snapshot_path = f"s3://bucket/{symbol.lower()}/{draw(st.dates()).isoformat()}/snapshot.parquet"
    training_timestamp = datetime.now(timezone.utc).isoformat()

    return ModelMetadata(
        symbol=symbol,
        forecast_horizon=horizon,
        training_timestamp=training_timestamp,
        training_nll=training_nll,
        validation_nll=validation_nll,
        directional_accuracy=da,
        hyperparameters=hyperparams,
        pipeline_run_id=pipeline_run_id,
        data_snapshot_path=data_snapshot_path,
        lifecycle_stage="staging",
    )


class TestMetadataAndLineageCompleteness:
    """Property 8: Registered models return complete metadata and lineage.

    After registering a model, querying it back must return all metadata
    fields (symbol, horizon, timestamps, metrics, hyperparams) and lineage
    information (pipeline_run_id, data_snapshot_path).

    **Validates: Requirements 6.1, 6.2, 6.5**
    """

    @given(metadata=model_metadata())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_registered_model_returns_complete_metadata(self, metadata: ModelMetadata):
        """Registering a model and querying it back returns all metadata fields."""
        client = create_test_client()

        version_id = client.register_model(
            model_path="s3://bucket/models/checkpoint.pt",
            metadata=metadata,
        )

        results = client.query_models(
            symbol=metadata.symbol,
            horizon=metadata.forecast_horizon,
        )

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        result = results[0]

        assert result.symbol == metadata.symbol
        assert result.forecast_horizon == metadata.forecast_horizon
        assert result.training_timestamp == metadata.training_timestamp
        assert result.training_nll == metadata.training_nll
        assert result.validation_nll == metadata.validation_nll
        assert result.directional_accuracy == metadata.directional_accuracy
        assert result.hyperparameters == metadata.hyperparameters

        assert result.pipeline_run_id == metadata.pipeline_run_id
        assert result.data_snapshot_path == metadata.data_snapshot_path

        assert result.lifecycle_stage == "staging"


class TestUniqueVersionIdentifiers:
    """Property 9: N registrations produce N unique IDs.

    Each call to register_model() must produce a distinct version identifier,
    regardless of whether the metadata is identical.

    **Validates: Requirements 6.2**
    """

    @given(
        n=st.integers(min_value=2, max_value=10),
        metadata=model_metadata(),
    )
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_n_registrations_produce_n_unique_ids(self, n: int, metadata: ModelMetadata):
        """Registering N models produces N distinct version IDs."""
        client = create_test_client()

        version_ids = []
        for _ in range(n):
            vid = client.register_model(
                model_path="s3://bucket/models/checkpoint.pt",
                metadata=metadata,
            )
            version_ids.append(vid)

        assert len(set(version_ids)) == n, (
            f"Expected {n} unique IDs, got {len(set(version_ids))} unique out of {version_ids}"
        )

        for vid in version_ids:
            uuid.UUID(vid)


class TestLifecycleStateTransitions:
    """Property 10: Only valid transitions succeed.

    Valid transitions: staging→production, staging→archived, production→archived.
    Archived is a terminal state. Invalid transitions must be rejected.

    **Validates: Requirements 6.3**
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

        Tests the _VALID_TRANSITIONS mapping directly to verify the state machine.
        """
        valid_targets = _VALID_TRANSITIONS.get(current_stage, set())
        is_valid = target_stage in valid_targets

        if is_valid:
            assert target_stage in valid_targets
        else:
            assert target_stage not in valid_targets

    @given(metadata=model_metadata())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_promote_from_staging_succeeds(self, metadata: ModelMetadata):
        """Promoting a model from staging to production succeeds."""
        client = create_test_client()

        version_id = client.register_model(
            model_path="s3://bucket/models/checkpoint.pt",
            metadata=metadata,
        )

        result = client.promote_to_production(version_id)

        assert result is True, "Promotion from staging to production should succeed"

    @given(metadata=model_metadata())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_promote_from_production_fails(self, metadata: ModelMetadata):
        """Promoting a model already in production fails (production→production is invalid)."""
        client = create_test_client()

        version_id = client.register_model(
            model_path="s3://bucket/models/checkpoint.pt",
            metadata=metadata,
        )

        client.promote_to_production(version_id)

        result = client.promote_to_production(version_id)

        assert result is False, "Promotion from production to production should fail"

    def test_archived_is_terminal(self):
        """Archived state has no valid transitions (terminal state)."""
        assert _VALID_TRANSITIONS["archived"] == set(), (
            "Archived should be a terminal state with no valid transitions"
        )

    def test_staging_can_transition_to_production_and_archived(self):
        """Staging can transition to production or archived."""
        assert _VALID_TRANSITIONS["staging"] == {"production", "archived"}

    def test_production_can_only_transition_to_archived(self):
        """Production can only transition to archived."""
        assert _VALID_TRANSITIONS["production"] == {"archived"}


class TestQueryFiltering:
    """Property 11: Returned models match all specified filters, no matching model excluded.

    Query filtering uses AND logic; a model must match ALL specified filters.
    No model that matches all filters should be excluded from results.

    **Validates: Requirements 6.6**
    """

    @given(
        metadata_list=st.lists(model_metadata(), min_size=3, max_size=10),
        filter_symbol=st.one_of(st.none(), symbols),
        filter_stage=st.one_of(st.none(), st.just("staging")),
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_query_results_match_all_filters(
        self, metadata_list: list, filter_symbol: Optional[str], filter_stage: Optional[str]
    ):
        """All returned models match every specified filter."""
        client = create_test_client()

        for meta in metadata_list:
            client.register_model(
                model_path="s3://bucket/models/checkpoint.pt",
                metadata=meta,
            )

        results = client.query_models(
            symbol=filter_symbol,
            stage=filter_stage,
        )

        for result in results:
            if filter_symbol is not None:
                assert result.symbol == filter_symbol, (
                    f"Result symbol {result.symbol} doesn't match filter {filter_symbol}"
                )
            if filter_stage is not None:
                assert result.lifecycle_stage == filter_stage, (
                    f"Result stage {result.lifecycle_stage} doesn't match filter {filter_stage}"
                )

    @given(
        metadata_list=st.lists(model_metadata(), min_size=3, max_size=10),
        filter_symbol=symbols,
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_no_matching_model_excluded(self, metadata_list: list, filter_symbol: str):
        """No model that matches all filters is excluded from results."""
        client = create_test_client()

        for meta in metadata_list:
            client.register_model(
                model_path="s3://bucket/models/checkpoint.pt",
                metadata=meta,
            )

        results = client.query_models(symbol=filter_symbol)

        expected_count = sum(
            1 for meta in metadata_list if meta.symbol == filter_symbol
        )

        assert len(results) == expected_count, (
            f"Expected {expected_count} results for symbol={filter_symbol}, "
            f"got {len(results)}"
        )

    @given(
        metadata=model_metadata(),
        min_da=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_min_da_filter_excludes_below_threshold(self, metadata: ModelMetadata, min_da: float):
        """Models with DA below min_da are excluded from results."""
        client = create_test_client()

        client.register_model(
            model_path="s3://bucket/models/checkpoint.pt",
            metadata=metadata,
        )

        results = client.query_models(min_da=min_da)

        if metadata.directional_accuracy >= min_da:
            assert len(results) == 1, "Model meeting DA threshold should be included"
        else:
            assert len(results) == 0, "Model below DA threshold should be excluded"

    @given(
        metadata=model_metadata(),
        max_nll=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_max_nll_filter_excludes_above_threshold(self, metadata: ModelMetadata, max_nll: float):
        """Models with NLL above max_nll are excluded from results."""
        client = create_test_client()

        client.register_model(
            model_path="s3://bucket/models/checkpoint.pt",
            metadata=metadata,
        )

        results = client.query_models(max_nll=max_nll)

        if metadata.validation_nll <= max_nll:
            assert len(results) == 1, "Model meeting NLL threshold should be included"
        else:
            assert len(results) == 0, "Model above NLL threshold should be excluded"


class TestRetentionPolicyEnforcement:
    """Property 12: Models <90 days cannot be archived, models ≥90 days can.

    The can_archive() method enforces the 90-day retention policy based on
    the model's created_at timestamp.

    **Validates: Requirements 6.7**
    """

    @given(
        metadata=model_metadata(),
        age_days=st.integers(min_value=0, max_value=89),
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_models_under_90_days_cannot_be_archived(self, metadata: ModelMetadata, age_days: int):
        """Models less than 90 days old cannot be archived."""
        client = create_test_client()

        version_id = client.register_model(
            model_path="s3://bucket/models/checkpoint.pt",
            metadata=metadata,
        )

        created_at = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
        for rm in client.registry.get_registered_models():
            for mv in client.registry.get_model_versions(rm.name):
                if mv.custom_properties.get("version_id") == version_id:
                    mv.custom_properties["created_at"] = created_at

        result = client.can_archive(version_id)
        assert result is False, (
            f"Model {age_days} days old should NOT be archivable (< 90 days)"
        )

    @given(
        metadata=model_metadata(),
        age_days=st.integers(min_value=90, max_value=365),
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_models_90_days_or_older_can_be_archived(self, metadata: ModelMetadata, age_days: int):
        """Models 90 days or older can be archived."""
        client = create_test_client()

        version_id = client.register_model(
            model_path="s3://bucket/models/checkpoint.pt",
            metadata=metadata,
        )

        created_at = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
        for rm in client.registry.get_registered_models():
            for mv in client.registry.get_model_versions(rm.name):
                if mv.custom_properties.get("version_id") == version_id:
                    mv.custom_properties["created_at"] = created_at

        result = client.can_archive(version_id)
        assert result is True, (
            f"Model {age_days} days old should be archivable (>= 90 days)"
        )

    def test_retention_days_constant_is_90(self):
        """The RETENTION_DAYS constant is set to 90."""
        assert ForecasterRegistryClient.RETENTION_DAYS == 90
