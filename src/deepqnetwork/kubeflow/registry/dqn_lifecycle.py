"""Lifecycle management for the DQN model registry.

Provides a higher-level lifecycle management layer on top of the
DQNRegistryClient, handling state transition validation and initial deployment
bootstrap (auto-promoting the first model when no production model exists).

The standalone DQN KServe InferenceService is deprecated (see
kubeflow-ml-pipeline spec, Requirement 15). Promotion to ``production`` now
only updates the registry metadata; the dqnpf-intraday combined predictor's
hot-reload watcher (Task 30.3) picks up the new production checkpoint on its
next poll. No KServe patching occurs here.

Requirements: DQN-R20
"""

import logging
from typing import Optional

from deepqnetwork.kubeflow.registry.dqn_registry_client import (
    DQNRegistryClient,
    DQNModelMetadata,
    _DQN_VALID_TRANSITIONS,
)

logger = logging.getLogger(__name__)


class DQNLifecycleManager:
    """Manages DQN model lifecycle transitions for the registry.

    Wraps the DQNRegistryClient to provide:
    - State transition validation (staging→production, staging→archived,
      production→archived; archived is terminal)
    - Initial deployment bootstrap: if no production model exists for a
      symbol, auto-promotes the first registered model

    Promotion to ``production`` is registry-metadata-only. The dqnpf-intraday
    InferenceService's hot-reload watcher (see
    ``tradingmodel/intraday/dqnpf/kubeflow/serving/dqnpf_predictor.py``) is the
    consumer that swaps in the new checkpoint at serving time.
    """

    def __init__(self, registry_client: DQNRegistryClient):
        """Initialize the DQN lifecycle manager.

        Args:
            registry_client: An initialized DQNRegistryClient instance.
        """
        self.registry_client = registry_client

    def validate_transition(self, current_stage: str, target_stage: str) -> bool:
        """Validate whether a lifecycle state transition is allowed.

        Valid transitions:
            - staging → production
            - staging → archived
            - production → archived

        Archived is a terminal state with no outgoing transitions.

        Args:
            current_stage: The current lifecycle stage of the model.
            target_stage: The desired target lifecycle stage.

        Returns:
            True if the transition is valid, False otherwise.
        """
        valid_targets = _DQN_VALID_TRANSITIONS.get(current_stage, set())
        is_valid = target_stage in valid_targets

        if not is_valid:
            logger.warning(
                "Invalid DQN lifecycle transition: %s → %s (valid targets: %s)",
                current_stage,
                target_stage,
                valid_targets or "none (terminal state)",
            )

        return is_valid

    def transition_stage(self, version_id: str, target_stage: str) -> bool:
        """Transition a DQN model version to a new lifecycle stage.

        Validates the transition, performs the stage change, and triggers
        any side effects (e.g., KServe update on promotion to production).

        Args:
            version_id: The unique version identifier of the model.
            target_stage: The target lifecycle stage.

        Returns:
            True if the transition succeeded, False if invalid or failed.
        """
        version_data = self.registry_client._get_version_data(version_id)
        if version_data is None:
            logger.error(
                "Cannot transition DQN version %s: not found in registry",
                version_id,
            )
            return False

        current_stage = version_data.get("lifecycle_stage", "staging")

        if not self.validate_transition(current_stage, target_stage):
            return False

        if target_stage == "production":
            return self.promote_to_production(version_id)
        elif target_stage == "archived":
            return self._archive_model(version_id)
        else:
            logger.error("Unsupported target stage: %s", target_stage)
            return False

    def promote_to_production(self, version_id: str) -> bool:
        """Promote a DQN model version to production lifecycle stage.

        Delegates to the registry client's promote_to_production method,
        which handles:
        1. Validating the state transition (must be in staging)
        2. Demoting the current production model
        3. Updating the lifecycle stage

        The dqnpf-intraday combined predictor's hot-reload watcher picks up
        the new production-stage checkpoint on its next poll (Task 30.3 in
        the kubeflow-ml-pipeline spec); no KServe patching is performed here.

        Args:
            version_id: The unique version identifier to promote.

        Returns:
            True if promotion succeeded, False otherwise.

        Requirements: DQN-R20
        """
        version_data = self.registry_client._get_version_data(version_id)
        if version_data is None:
            logger.error(
                "Cannot promote DQN version %s: not found in registry",
                version_id,
            )
            return False

        current_stage = version_data.get("lifecycle_stage", "staging")
        if not self.validate_transition(current_stage, "production"):
            return False

        result = self.registry_client.promote_to_production(version_id)

        if result:
            logger.info(
                "DQN model %s promoted to production; dqnpf-intraday "
                "hot-reload watcher will swap the checkpoint on its next poll",
                version_id,
            )

        return result

    def bootstrap_initial_deployment(self, symbol: str) -> Optional[str]:
        """Bootstrap initial deployment by auto-promoting the first DQN model.

        If no production model exists for the given symbol, automatically
        promotes the first registered model (in staging) to production.
        This handles the cold-start case where the dqnpf-intraday predictor's
        registry resolver would otherwise raise on a missing production-stage
        entry.

        Args:
            symbol: Trading pair symbol (e.g., "USDJPY").

        Returns:
            The version_id of the auto-promoted model, or None if no model
            was available to promote or a production model already exists.

        Requirements: DQN-R20
        """
        # Check if a production model already exists
        production_models = self.registry_client.query_models(
            symbol=symbol, stage="production"
        )

        if production_models:
            logger.info(
                "Production DQN model already exists for %s; "
                "skipping bootstrap (dqnpf-intraday hot-reload owns rollout)",
                symbol,
            )
            return None

        # Find staging models for this symbol
        staging_models = self.registry_client.query_models(
            symbol=symbol, stage="staging"
        )

        if not staging_models:
            logger.warning(
                "No staging DQN models available for %s; "
                "cannot bootstrap initial deployment",
                symbol,
            )
            return None

        # Find the version_id of the first staging model
        first_model = staging_models[0]
        version_id = self._find_version_id(first_model, symbol)

        if version_id is None:
            logger.error(
                "Could not find version_id for first staging DQN model of %s",
                symbol,
            )
            return None

        logger.info(
            "No production DQN model for %s; auto-promoting first model %s",
            symbol,
            version_id,
        )

        success = self.promote_to_production(version_id)
        if success:
            return version_id

        logger.error(
            "Failed to auto-promote DQN model %s for initial deployment",
            version_id,
        )
        return None

    def _archive_model(self, version_id: str) -> bool:
        """Archive a DQN model version.

        Updates the lifecycle stage to 'archived'. This is a terminal state.

        Args:
            version_id: The unique version identifier to archive.

        Returns:
            True if archival succeeded, False otherwise.
        """
        try:
            self.registry_client._update_lifecycle_stage(version_id, "archived")
            logger.info("Archived DQN model version %s", version_id)
            return True
        except Exception as e:
            logger.error("Failed to archive DQN version %s: %s", version_id, e)
            return False

    def _find_version_id(
        self, metadata: DQNModelMetadata, symbol: str
    ) -> Optional[str]:
        """Find the version_id for a DQN model matching the given metadata.

        Searches the registry for a staging model matching the symbol
        and returns its version_id from custom properties.

        Args:
            metadata: The DQNModelMetadata to match against.
            symbol: Trading pair symbol.

        Returns:
            The version_id string, or None if not found.
        """
        model_name = self.registry_client._registered_model_name(symbol)
        try:
            versions = self.registry_client.registry.get_model_versions(model_name)
            for v in versions:
                props = (
                    v.custom_properties if hasattr(v, "custom_properties") else {}
                )
                if (
                    props.get("lifecycle_stage") == "staging"
                    and props.get("symbol") == symbol
                    and props.get("pipeline_run_id") == metadata.pipeline_run_id
                ):
                    return props.get("version_id")
        except Exception:
            pass
        return None

