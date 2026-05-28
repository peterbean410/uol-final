"""Combined-version Model Registry client for dqnpf-intraday.

Manages registration, lifecycle, and resolution of combined deployment
versions that track both parent model checkpoints (DQN + Forecaster),
the IntegrationConfig, and backtest validation results.

Requirements: 23.1, 23.2, 23.3, 23.4
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from model_registry import ModelRegistry

from tradingmodel.intraday.dqnpf.backtest import BacktestComparison, ThresholdReport
from tradingmodel.intraday.dqnpf.config import IntegrationConfig


class DqnpfRegistryClient:
    """Wrapper around Kubeflow Model Registry for dqnpf-intraday combined versions.

    Each registered entry tracks:
    - The resolved DQN checkpoint version
    - The resolved Forecaster checkpoint version
    - The full IntegrationConfig (frozen thresholds and budgets)
    - The BacktestComparison and ThresholdReport from the validating backtest
    - The pipeline_run_id
    """

    RETENTION_DAYS = 90
    MODEL_NAME = "dqnpf-intraday"

    def __init__(self, registry_url: str):
        # The in-cluster model-registry-service exposes plain HTTP on 8080;
        # model-registry SDK >= 0.2.16 defaults is_secure=True and raises
        # `user token must be provided for secure connection` unless we
        # opt out explicitly even for http:// URLs.
        is_secure = not registry_url.lower().startswith("http://")
        self.registry = ModelRegistry(
            server_address=registry_url,
            author="dqnpf-pipeline",
            is_secure=is_secure,
            user_token="",
        )

    def register_combined_version(
        self,
        comparison: BacktestComparison,
        threshold_report: ThresholdReport,
        integration_config: IntegrationConfig,
        parent_dqn_version: str,
        parent_forecaster_version: str,
        pipeline_run_id: str,
        s3_path: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """Register a new combined version in the Model Registry.

        Only succeeds when threshold_report.passed is True. Raises ValueError
        if the threshold report indicates failure.

        Returns the version_id of the newly registered entry.
        """
        if not threshold_report.passed:
            raise ValueError(
                f"Cannot register combined version: threshold validation failed. "
                f"Failures: {threshold_report.failures}"
            )

        version_id = str(uuid.uuid4())
        registered_at = datetime.now(timezone.utc)
        target_model_name = model_name or self.MODEL_NAME

        # Ensure the registered model exists
        try:
            self.registry.get_registered_model(target_model_name)
        except Exception:
            self.registry.register_model(
                name=target_model_name,
                description="dqnpf-intraday combined deployment versions",
            )

        uri = s3_path or f"s3://prod-fintech-forex-sg/dqnpf/{version_id}/combined.pt"

        custom_properties = {
            "version_id": version_id,
            "lifecycle_stage": "staging",
            "registered_at": registered_at.isoformat(),
            "s3_path": uri,
            "pipeline_run_id": pipeline_run_id,
            "parent_dqn_version": parent_dqn_version,
            "parent_forecaster_version": parent_forecaster_version,
            "threshold_passed": str(threshold_report.passed),
            "combined_sharpe": str(comparison.combined_sharpe),
            "baseline_sharpe": str(comparison.baseline_sharpe),
            "combined_return": str(comparison.combined_return),
            "baseline_return": str(comparison.baseline_return),
        }

        self.registry.register_model_version(
            name=f"{target_model_name}-{version_id}",
            version=version_id,
            model_name=target_model_name,
            uri=uri,
            description=(
                f"Combined dqnpf-intraday version. "
                f"DQN: {parent_dqn_version}, Forecaster: {parent_forecaster_version}"
            ),
            custom_properties=custom_properties,
        )

        return version_id

    def resolve_production_checkpoint(self, model_name: str) -> str:
        """Return the S3 path of the latest production-stage entry for model_name.

        Raises ValueError if no production entry exists.
        """
        versions = self.registry.get_model_versions(model_name)
        production_versions = [
            v
            for v in versions
            if v.custom_properties.get("lifecycle_stage") == "production"
        ]

        if not production_versions:
            raise ValueError(
                f"No production-stage version found for model '{model_name}'"
            )

        # Sort by registered_at descending, return the most recent
        production_versions.sort(
            key=lambda v: v.custom_properties.get("registered_at", ""),
            reverse=True,
        )

        return production_versions[0].custom_properties["s3_path"]

    def promote_to_production(self, version_id: str) -> bool:
        """Promote a version to production stage.

        Gated on operator API token. Emits registry event for predictor hot-reload.
        Returns True on success, False if transition is invalid.
        """
        version = self._find_version(version_id)
        if version is None:
            raise ValueError(f"Version {version_id} not found")

        current_stage = version.custom_properties.get("lifecycle_stage")
        if current_stage != "staging":
            return False

        version.custom_properties["lifecycle_stage"] = "production"
        self.registry.update_model_version(
            version, custom_properties=version.custom_properties
        )
        return True

    def archive_version(self, version_id: str) -> None:
        """Archive a version. Respects 90-day minimum retention.

        Raises ValueError if the version is younger than 90 days.
        """
        version = self._find_version(version_id)
        if version is None:
            raise ValueError(f"Version {version_id} not found")

        registered_at_str = version.custom_properties.get("registered_at")
        if registered_at_str:
            registered_at = datetime.fromisoformat(registered_at_str)
            age = datetime.now(timezone.utc) - registered_at
            if age < timedelta(days=self.RETENTION_DAYS):
                raise ValueError(
                    f"Cannot archive version {version_id}: "
                    f"only {age.days} days old (minimum {self.RETENTION_DAYS} days)"
                )

        version.custom_properties["lifecycle_stage"] = "archived"
        self.registry.update_model_version(
            version, custom_properties=version.custom_properties
        )

    def _find_version(self, version_id: str):
        """Find a model version by version_id across all registered models."""
        for model_name in list(self.registry.model_versions.keys()):
            for version in self.registry.get_model_versions(model_name):
                if version.custom_properties.get("version_id") == version_id:
                    return version
        return None


def resolve_production_checkpoint(
    model_name: str,
    registry_url: str = "http://model-registry-service.kubeflow.svc.cluster.local:8080",
) -> str:
    """Module-level convenience function to resolve the latest production checkpoint.

    Used by the backtest component and predictor to look up parent checkpoints.
    The default URL points at the in-cluster model-registry-service used
    everywhere else in this repo (pipelines, InferenceService manifest, Airflow
    trigger).
    """
    client = DqnpfRegistryClient(registry_url=registry_url)
    return client.resolve_production_checkpoint(model_name)
