"""Model Registry client wrapping Kubeflow Model Registry for the forecaster.

Provides model registration with full metadata and lineage, lifecycle stage
management (staging → production → archived), query filtering, and retention
policy enforcement.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from model_registry import ModelRegistry
from model_registry.types import ModelArtifact, ModelVersion, RegisteredModel

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata stored with each model version.

    Captures all information required for experiment tracking, lineage,
    and model comparison.
    """

    symbol: str
    forecast_horizon: int
    training_timestamp: str
    training_nll: float
    validation_nll: float
    directional_accuracy: float
    hyperparameters: dict
    pipeline_run_id: str
    data_snapshot_path: str
    lifecycle_stage: str = "staging"  # staging | production | archived


# Valid lifecycle stage transitions
_VALID_TRANSITIONS = {
    "staging": {"production", "archived"},
    "production": {"archived"},
    "archived": set(),  # terminal state
}


class ForecasterRegistryClient:
    """Wrapper around Kubeflow Model Registry for the forecaster.

    Handles model registration with full metadata, lifecycle transitions,
    query filtering, and retention policy enforcement.
    """

    RETENTION_DAYS = 90
    REGISTERED_MODEL_PREFIX = "probabilistic-transformer"

    def __init__(self, registry_url: str):
        """Initialize the registry client.

        Args:
            registry_url: URL of the Kubeflow Model Registry server.
        """
        self.registry = ModelRegistry(
            server_address=registry_url, author="forecaster-pipeline"
        )

    def _registered_model_name(self, symbol: str, horizon: int) -> str:
        """Construct the registered model name for a symbol/horizon pair."""
        return f"{self.REGISTERED_MODEL_PREFIX}-{symbol.lower()}-h{horizon}"

    def _ensure_registered_model(self, symbol: str, horizon: int) -> RegisteredModel:
        """Get or create the RegisteredModel for a symbol/horizon pair.

        Args:
            symbol: Trading pair symbol (e.g., "USDJPY").
            horizon: Forecast horizon in bars.

        Returns:
            The RegisteredModel instance.
        """
        name = self._registered_model_name(symbol, horizon)
        try:
            return self.registry.get_registered_model(name)
        except Exception:
            # Model doesn't exist yet, create it
            return self.registry.register_model(
                name,
                uri="",  # URI set per version
                description=(
                    f"ProbabilisticTransformer for {symbol} "
                    f"with forecast horizon {horizon}"
                ),
            )

    def register_model(self, model_path: str, metadata: ModelMetadata) -> str:
        """Register a new model version with full metadata and lineage.

        Stores the model artifact with all metadata fields (symbol, horizon,
        timestamps, metrics, hyperparams) and lineage information (pipeline_run_id,
        data_snapshot_path).

        Args:
            model_path: S3 URI or local path to the model checkpoint artifact.
            metadata: Complete metadata for this model version.

        Returns:
            Unique version identifier (UUID string).

        Requirements: 6.1, 6.2, 6.5
        """
        version_id = str(uuid.uuid4())

        # Ensure the parent registered model exists
        registered_model = self._ensure_registered_model(
            metadata.symbol, metadata.forecast_horizon
        )

        # Build custom properties dict with all metadata and lineage
        custom_properties = {
            "version_id": version_id,
            "symbol": metadata.symbol,
            "forecast_horizon": str(metadata.forecast_horizon),
            "training_timestamp": metadata.training_timestamp,
            "training_nll": str(metadata.training_nll),
            "validation_nll": str(metadata.validation_nll),
            "directional_accuracy": str(metadata.directional_accuracy),
            "hyperparameters": json.dumps(metadata.hyperparameters),
            "pipeline_run_id": metadata.pipeline_run_id,
            "data_snapshot_path": metadata.data_snapshot_path,
            "lifecycle_stage": metadata.lifecycle_stage,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Register the model version with the registry
        model_version = self.registry.register_model_version(
            name=version_id,
            version=version_id,
            model_name=registered_model.name,
            uri=model_path,
            description=f"Trained on {metadata.symbol} h{metadata.forecast_horizon}",
            custom_properties=custom_properties,
        )

        logger.info(
            "Registered model version %s for %s h%d (NLL=%.4f, DA=%.4f)",
            version_id,
            metadata.symbol,
            metadata.forecast_horizon,
            metadata.validation_nll,
            metadata.directional_accuracy,
        )

        return version_id

    def promote_to_production(self, version_id: str) -> bool:
        """Promote a model version to production lifecycle stage.

        Validates the state transition (must be in staging), updates the
        lifecycle stage, and triggers a KServe InferenceService update
        to deploy the new model version.

        Args:
            version_id: The unique version identifier to promote.

        Returns:
            True if promotion succeeded, False if the transition is invalid
            or blocked by policy.

        Requirements: 6.3, 6.4
        """
        version_data = self._get_version_data(version_id)
        if version_data is None:
            logger.error("Version %s not found in registry", version_id)
            return False

        current_stage = version_data.get("lifecycle_stage", "staging")

        # Validate state transition
        if "production" not in _VALID_TRANSITIONS.get(current_stage, set()):
            logger.warning(
                "Invalid transition from %s to production for version %s",
                current_stage,
                version_id,
            )
            return False

        # Demote current production model to staging (only one production at a time)
        self._demote_current_production(
            version_data["symbol"], int(version_data["forecast_horizon"])
        )

        # Update lifecycle stage to production
        self._update_lifecycle_stage(version_id, "production")

        # Trigger KServe InferenceService update
        self._trigger_kserve_update(version_data)

        logger.info("Promoted version %s to production", version_id)
        return True

    def query_models(
        self,
        symbol: Optional[str] = None,
        horizon: Optional[int] = None,
        min_da: Optional[float] = None,
        max_nll: Optional[float] = None,
        stage: Optional[str] = None,
    ) -> list[ModelMetadata]:
        """Query models with filters.

        Returns all model versions matching the specified filter criteria.
        Filters are combined with AND logic; a model must match all specified
        filters to be included.

        Args:
            symbol: Filter by trading pair symbol.
            horizon: Filter by forecast horizon.
            min_da: Minimum directional accuracy threshold.
            max_nll: Maximum validation NLL threshold.
            stage: Filter by lifecycle stage (staging/production/archived).

        Returns:
            List of ModelMetadata instances matching all filters.

        Requirements: 6.6
        """
        results = []

        # Determine which registered models to query
        registered_models = self._list_registered_models()

        for rm in registered_models:
            versions = self._list_versions_for_model(rm)
            for version_data in versions:
                # Apply filters
                if symbol and version_data.get("symbol") != symbol:
                    continue
                if horizon and int(version_data.get("forecast_horizon", 0)) != horizon:
                    continue
                if min_da is not None:
                    da = float(version_data.get("directional_accuracy", 0))
                    if da < min_da:
                        continue
                if max_nll is not None:
                    nll = float(version_data.get("validation_nll", float("inf")))
                    if nll > max_nll:
                        continue
                if stage and version_data.get("lifecycle_stage") != stage:
                    continue

                # Build ModelMetadata from stored properties
                metadata = ModelMetadata(
                    symbol=version_data["symbol"],
                    forecast_horizon=int(version_data["forecast_horizon"]),
                    training_timestamp=version_data["training_timestamp"],
                    training_nll=float(version_data["training_nll"]),
                    validation_nll=float(version_data["validation_nll"]),
                    directional_accuracy=float(version_data["directional_accuracy"]),
                    hyperparameters=json.loads(
                        version_data.get("hyperparameters", "{}")
                    ),
                    pipeline_run_id=version_data["pipeline_run_id"],
                    data_snapshot_path=version_data["data_snapshot_path"],
                    lifecycle_stage=version_data.get("lifecycle_stage", "staging"),
                )
                results.append(metadata)

        return results

    def can_archive(self, version_id: str) -> bool:
        """Check if a model version is old enough to archive.

        Enforces the 90-day retention policy: models must be at least 90 days
        old before they can be archived.

        Args:
            version_id: The unique version identifier to check.

        Returns:
            True if the model is >= 90 days old and can be archived,
            False if it's too recent or not found.

        Requirements: 6.7
        """
        version_data = self._get_version_data(version_id)
        if version_data is None:
            return False

        created_at_str = version_data.get("created_at")
        if not created_at_str:
            return False

        created_at = datetime.fromisoformat(created_at_str)
        now = datetime.now(timezone.utc)
        age_days = (now - created_at).days

        return age_days >= self.RETENTION_DAYS

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_version_data(self, version_id: str) -> Optional[dict]:
        """Retrieve custom properties for a model version by its version_id.

        Searches across all registered models to find the version with the
        matching version_id in its custom properties.

        Args:
            version_id: The unique version identifier.

        Returns:
            Dictionary of custom properties, or None if not found.
        """
        registered_models = self._list_registered_models()
        for rm in registered_models:
            versions = self._list_versions_for_model(rm)
            for version_data in versions:
                if version_data.get("version_id") == version_id:
                    return version_data
        return None

    def _list_registered_models(self) -> list:
        """List all registered models managed by this client."""
        try:
            return self.registry.get_registered_models()
        except Exception:
            return []

    def _list_versions_for_model(self, registered_model) -> list[dict]:
        """List all version custom properties for a registered model.

        Args:
            registered_model: The RegisteredModel instance.

        Returns:
            List of custom property dictionaries for each version.
        """
        try:
            versions = self.registry.get_model_versions(registered_model.name)
            return [
                v.custom_properties if hasattr(v, "custom_properties") else {}
                for v in versions
            ]
        except Exception:
            return []

    def _update_lifecycle_stage(self, version_id: str, new_stage: str) -> None:
        """Update the lifecycle stage of a model version.

        Args:
            version_id: The unique version identifier.
            new_stage: The new lifecycle stage to set.
        """
        registered_models = self._list_registered_models()
        for rm in registered_models:
            try:
                versions = self.registry.get_model_versions(rm.name)
                for v in versions:
                    props = (
                        v.custom_properties
                        if hasattr(v, "custom_properties")
                        else {}
                    )
                    if props.get("version_id") == version_id:
                        props["lifecycle_stage"] = new_stage
                        self.registry.update_model_version(
                            v, custom_properties=props
                        )
                        return
            except Exception:
                continue

    def _demote_current_production(self, symbol: str, horizon: int) -> None:
        """Demote the current production model back to staging.

        Ensures only one model version is in production at a time for a
        given symbol/horizon pair.

        Args:
            symbol: Trading pair symbol.
            horizon: Forecast horizon.
        """
        model_name = self._registered_model_name(symbol, horizon)
        try:
            versions = self.registry.get_model_versions(model_name)
            for v in versions:
                props = (
                    v.custom_properties if hasattr(v, "custom_properties") else {}
                )
                if props.get("lifecycle_stage") == "production":
                    props["lifecycle_stage"] = "staging"
                    self.registry.update_model_version(v, custom_properties=props)
                    logger.info(
                        "Demoted version %s from production to staging",
                        props.get("version_id"),
                    )
        except Exception:
            pass  # No existing production model

    def _trigger_kserve_update(self, version_data: dict) -> None:
        """Trigger KServe InferenceService update for the promoted model.

        Updates the InferenceService to point to the new model artifact URI.

        Args:
            version_data: Custom properties dict of the promoted version.

        Requirements: 6.4
        """
        try:
            from kubernetes import client, config as k8s_config

            k8s_config.load_incluster_config()
            api = client.CustomObjectsApi()

            symbol = version_data["symbol"]
            horizon = int(version_data["forecast_horizon"])
            service_name = f"forecaster-{symbol.lower()}-h{horizon}"

            # Patch the InferenceService to update the model URI
            patch_body = {
                "spec": {
                    "predictor": {
                        "containers": [
                            {
                                "name": "forecaster-predictor",
                                "env": [
                                    {
                                        "name": "MODEL_PATH",
                                        "value": version_data.get(
                                            "model_uri", ""
                                        ),
                                    }
                                ],
                            }
                        ]
                    }
                }
            }

            api.patch_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace="kubeflow",
                plural="inferenceservices",
                name=service_name,
                body=patch_body,
            )

            logger.info(
                "Triggered KServe update for %s with new model version",
                service_name,
            )
        except ImportError:
            logger.warning(
                "kubernetes client not available; skipping KServe update"
            )
        except Exception as e:
            logger.error("Failed to trigger KServe update: %s", e)
