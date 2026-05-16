"""DQN Model Registry client wrapping Kubeflow Model Registry.

Provides model registration with full metadata and lineage, lifecycle stage
management (staging → production → archived), query filtering, and retention
policy enforcement for the Deep Q-Network trading agent.

Requirements: DQN-R19, DQN-R20
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
class DQNModelMetadata:
    """Metadata stored with each DQN model version.

    Captures all information required for experiment tracking, lineage,
    and model comparison specific to the DQN trading agent.
    """

    symbol: str
    episode_start_ts: str
    episode_end_ts: str
    step_size_seconds: int
    cumulative_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    hyperparameters: dict
    pipeline_run_id: str
    training_timestamp: str
    lifecycle_stage: str = "staging"  # staging | production | archived


# Valid lifecycle stage transitions
_DQN_VALID_TRANSITIONS = {
    "staging": {"production", "archived"},
    "production": {"archived"},
    "archived": set(),  # terminal state
}


class DQNRegistryClient:
    """Wrapper around Kubeflow Model Registry for the DQN agent.

    Handles model registration with full metadata, lifecycle transitions,
    query filtering, and retention policy enforcement.
    """

    RETENTION_DAYS = 90
    REGISTERED_MODEL_PREFIX = "dqn-agent"

    def __init__(self, registry_url: str):
        """Initialize the registry client.

        Args:
            registry_url: URL of the Kubeflow Model Registry server.
        """
        self.registry = ModelRegistry(
            server_address=registry_url, author="dqn-pipeline"
        )

    def _registered_model_name(self, symbol: str) -> str:
        """Construct the registered model name for a symbol."""
        return f"{self.REGISTERED_MODEL_PREFIX}-{symbol.lower()}"

    def _ensure_registered_model(self, symbol: str) -> RegisteredModel:
        """Get or create the RegisteredModel for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "USDJPY").

        Returns:
            The RegisteredModel instance.
        """
        name = self._registered_model_name(symbol)
        try:
            return self.registry.get_registered_model(name)
        except Exception:
            # Model doesn't exist yet, create it
            return self.registry.register_model(
                name,
                uri="",  # URI set per version
                description=f"DQN trading agent for {symbol}",
            )

    def register_model(self, model_path: str, metadata: DQNModelMetadata) -> str:
        """Register a new DQN model version with full metadata and lineage.

        Stores the model artifact with all metadata fields (symbol, episode range,
        step_size, metrics, hyperparams) and lineage information (pipeline_run_id).

        Args:
            model_path: S3 URI or local path to the model checkpoint artifact.
            metadata: Complete metadata for this model version.

        Returns:
            Unique version identifier (UUID string).

        Requirements: DQN-R19
        """
        version_id = str(uuid.uuid4())

        # Ensure the parent registered model exists
        registered_model = self._ensure_registered_model(metadata.symbol)

        # Build custom properties dict with all metadata and lineage
        custom_properties = {
            "version_id": version_id,
            "symbol": metadata.symbol,
            "episode_start_ts": metadata.episode_start_ts,
            "episode_end_ts": metadata.episode_end_ts,
            "step_size_seconds": str(metadata.step_size_seconds),
            "cumulative_pnl": str(metadata.cumulative_pnl),
            "sharpe_ratio": str(metadata.sharpe_ratio),
            "max_drawdown": str(metadata.max_drawdown),
            "win_rate": str(metadata.win_rate),
            "hyperparameters": json.dumps(metadata.hyperparameters),
            "pipeline_run_id": metadata.pipeline_run_id,
            "training_timestamp": metadata.training_timestamp,
            "lifecycle_stage": metadata.lifecycle_stage,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Register the model version with the registry
        model_version = self.registry.register_model_version(
            name=version_id,
            version=version_id,
            model_name=registered_model.name,
            uri=model_path,
            description=(
                f"DQN agent for {metadata.symbol} "
                f"({metadata.episode_start_ts} to {metadata.episode_end_ts})"
            ),
            custom_properties=custom_properties,
        )

        logger.info(
            "Registered DQN model version %s for %s "
            "(Sharpe=%.4f, P&L=%.4f, WinRate=%.4f)",
            version_id,
            metadata.symbol,
            metadata.sharpe_ratio,
            metadata.cumulative_pnl,
            metadata.win_rate,
        )

        return version_id

    def promote_to_production(self, version_id: str) -> bool:
        """Promote a DQN model version to production lifecycle stage.

        Validates the state transition (must be in staging) and updates the
        lifecycle stage in the Model Registry. The dqnpf-intraday
        InferenceService picks up the new production-stage entry via its
        registry-polling hot-reload watcher (see
        ``tradingmodel/intraday/dqnpf/kubeflow/serving/dqnpf_predictor.py``).
        No KServe patching is performed here; the standalone DQN
        InferenceService was deprecated when serving moved to dqnpf-intraday
        (see kubeflow-ml-pipeline spec, Requirement 15).

        Args:
            version_id: The unique version identifier to promote.

        Returns:
            True if promotion succeeded, False if the transition is invalid
            or blocked by policy.

        Requirements: DQN-R20
        """
        version_data = self._get_version_data(version_id)
        if version_data is None:
            logger.error("Version %s not found in registry", version_id)
            return False

        current_stage = version_data.get("lifecycle_stage", "staging")

        # Validate state transition
        if "production" not in _DQN_VALID_TRANSITIONS.get(current_stage, set()):
            logger.warning(
                "Invalid transition from %s to production for version %s",
                current_stage,
                version_id,
            )
            return False

        # Demote current production model to staging (only one production at a time)
        self._demote_current_production(version_data["symbol"])

        # Update lifecycle stage to production. The dqnpf-intraday predictor's
        # hot-reload watcher (Task 30.3) resolves the new production-stage
        # checkpoint on its next poll and atomically swaps in the model.
        self._update_lifecycle_stage(version_id, "production")

        logger.info(
            "Promoted DQN version %s to production; dqnpf-intraday "
            "hot-reload will pick up the new checkpoint on its next poll",
            version_id,
        )
        return True

    def query_models(
        self,
        symbol: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        min_pnl: Optional[float] = None,
        stage: Optional[str] = None,
    ) -> list[DQNModelMetadata]:
        """Query DQN models with filters.

        Returns all model versions matching the specified filter criteria.
        Filters are combined with AND logic; a model must match all specified
        filters to be included.

        Args:
            symbol: Filter by trading pair symbol.
            min_sharpe: Minimum Sharpe ratio threshold.
            min_pnl: Minimum cumulative P&L threshold.
            stage: Filter by lifecycle stage (staging/production/archived).

        Returns:
            List of DQNModelMetadata instances matching all filters.

        Requirements: DQN-R19
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
                if min_sharpe is not None:
                    sharpe = float(version_data.get("sharpe_ratio", float("-inf")))
                    if sharpe < min_sharpe:
                        continue
                if min_pnl is not None:
                    pnl = float(version_data.get("cumulative_pnl", float("-inf")))
                    if pnl < min_pnl:
                        continue
                if stage and version_data.get("lifecycle_stage") != stage:
                    continue

                # Build DQNModelMetadata from stored properties
                metadata = DQNModelMetadata(
                    symbol=version_data["symbol"],
                    episode_start_ts=version_data["episode_start_ts"],
                    episode_end_ts=version_data["episode_end_ts"],
                    step_size_seconds=int(version_data["step_size_seconds"]),
                    cumulative_pnl=float(version_data["cumulative_pnl"]),
                    sharpe_ratio=float(version_data["sharpe_ratio"]),
                    max_drawdown=float(version_data["max_drawdown"]),
                    win_rate=float(version_data["win_rate"]),
                    hyperparameters=json.loads(
                        version_data.get("hyperparameters", "{}")
                    ),
                    pipeline_run_id=version_data["pipeline_run_id"],
                    training_timestamp=version_data["training_timestamp"],
                    lifecycle_stage=version_data.get("lifecycle_stage", "staging"),
                )
                results.append(metadata)

        return results

    def can_archive(self, version_id: str) -> bool:
        """Check if a DQN model version is old enough to archive.

        Enforces the 90-day retention policy: models must be at least 90 days
        old before they can be archived.

        Args:
            version_id: The unique version identifier to check.

        Returns:
            True if the model is >= 90 days old and can be archived,
            False if it's too recent or not found.

        Requirements: DQN-R19
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

    def _demote_current_production(self, symbol: str) -> None:
        """Demote the current production model back to staging.

        Ensures only one model version is in production at a time for a
        given symbol.

        Args:
            symbol: Trading pair symbol.
        """
        model_name = self._registered_model_name(symbol)
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
                        "Demoted DQN version %s from production to staging",
                        props.get("version_id"),
                    )
        except Exception:
            pass  # No existing production model

