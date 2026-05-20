"""KFP Pipeline Definition for the Probabilistic Forex Forecaster.

Defines the forecaster_pipeline using KFP v2 SDK with @dsl.container_component
decorators. Wires data_preparation -> model_training -> model_evaluation -> backtesting
as a DAG with retry(3) on each component and caching enabled on data_preparation.

Includes Model Registry registration after model_evaluation with degradation gate
integration, initial deployment bootstrap (auto-promote when no production model
exists), and Katib best-parameter injection.

Pipeline accepts parameters: symbol, forecast_horizon, epochs, batch_size,
learning_rate, snapshot_date, training_mode, katib_best_params_json.

Registered via Gitea Action on push to pipeline-related files.

Requirements: 1.1, 1.2, 1.7, 1.8, 1.9, 6.2, 6.4, 9.4
"""

import json
from dataclasses import asdict
from typing import NamedTuple

from kfp import dsl, kubernetes
from kfp.dsl import Dataset, Input, Metrics, Model, Output

from probabilisticforecaster.kubeflow.pipeline.config_schema import PipelineConfig


def _mount_minio_creds(task) -> None:
    """Inject MinIO accesskey/secretkey from the mlpipeline-minio-artifact secret.

    The container components use `probabilisticforecaster.artifact_io` to route
    reads/writes by URI scheme; when the URI is `minio://...` it needs
    MINIO_ACCESS_KEY / MINIO_SECRET_KEY env vars to reach the in-cluster MinIO.
    The Secret is created by the KFP install in the same namespace as each
    pipeline run, so this works without any extra RBAC.
    """
    kubernetes.use_secret_as_env(
        task,
        secret_name="mlpipeline-minio-artifact",
        secret_key_to_env={
            "accesskey": "MINIO_ACCESS_KEY",
            "secretkey": "MINIO_SECRET_KEY",
        },
    )

# ECR registry base for all component images
ECR_BASE = "731833471586.dkr.ecr.ap-southeast-1.amazonaws.com"


# ---------------------------------------------------------------------------
# Container Components
# ---------------------------------------------------------------------------


@dsl.container_component
def data_preparation(
    symbol: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    lookback_window: int,
    forecast_horizon: int,
    historical_window: int,
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
):
    """Load S3 parquet data, compute 16 features, build train/test splits.

    Outputs train and test datasets as KFP artifacts to S3.
    """
    return dsl.ContainerSpec(
        image=f"{ECR_BASE}/forecaster/data-preparation:latest",
        command=["python", "component.py"],
        args=[
            "--symbol", symbol,
            "--train-start", train_start,
            "--train-end", train_end,
            "--test-start", test_start,
            "--test-end", test_end,
            "--lookback-window", str(lookback_window),
            "--forecast-horizon", str(forecast_horizon),
            "--historical-window", str(historical_window),
            "--train-dataset-path", train_dataset.uri,
            "--test-dataset-path", test_dataset.uri,
        ],
    )


@dsl.container_component
def model_training(
    train_dataset: Input[Dataset],
    config_json: str,
    training_mode: str,
    production_model_uri: str,
    model_checkpoint: Output[Model],
    training_metrics: Output[Metrics],
):
    """Train ProbabilisticTransformer with Gaussian NLL loss.

    Supports both scratch (random init, full epochs) and finetune
    (load production weights, reduced LR, fewer epochs) modes.

    ``production_model_uri`` is the URI of the current production checkpoint
    to fine-tune from. It is required when ``training_mode="finetune"`` and
    ignored when ``training_mode="scratch"``. Pass an empty string for
    initial scratch runs (no production model exists yet).
    """
    return dsl.ContainerSpec(
        image=f"{ECR_BASE}/forecaster/model-training:latest",
        command=["python", "component.py"],
        args=[
            "--train-dataset-path", train_dataset.uri,
            "--checkpoint-path", model_checkpoint.uri,
            "--training-mode", training_mode,
            "--production-model-path", production_model_uri,
            "--config-json", config_json,
        ],
    )


@dsl.container_component
def model_evaluation(
    model_checkpoint: Input[Model],
    test_dataset: Input[Dataset],
    config_json: str,
    evaluation_metrics: Output[Metrics],
):
    """Compute NLL, directional accuracy, 95% coverage ratio, RMSE on test set.

    Includes forgetting check and degradation gate logic.
    """
    return dsl.ContainerSpec(
        image=f"{ECR_BASE}/forecaster/model-evaluation:latest",
        command=["python", "component.py"],
        args=[
            "--model-checkpoint-path", model_checkpoint.uri,
            "--test-dataset-path", test_dataset.uri,
            "--output-path", evaluation_metrics.uri,
            "--config-json", config_json,
        ],
    )


@dsl.container_component
def backtesting(
    model_checkpoint: Input[Model],
    test_dataset: Input[Dataset],
    config_json: str,
    backtest_results: Output[Metrics],
):
    """Run all trading strategies and output performance metrics.

    Outputs annualised return, Sharpe ratio, and max drawdown per strategy.
    """
    return dsl.ContainerSpec(
        image=f"{ECR_BASE}/forecaster/backtesting:latest",
        command=["python", "component.py"],
        args=[
            "--model-checkpoint-path", model_checkpoint.uri,
            "--test-dataset-path", test_dataset.uri,
            "--output-path", backtest_results.uri,
            "--config-json", config_json,
        ],
    )


# ---------------------------------------------------------------------------
# Lightweight Python component for config resolution at runtime
# ---------------------------------------------------------------------------


@dsl.component(base_image="python:3.11-slim", packages_to_install=["pyyaml"])
def resolve_config(
    symbol: str,
    forecast_horizon: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    snapshot_date: str,
    training_mode: str,
    katib_best_params_json: str = "",
) -> NamedTuple(
    "ConfigOutputs",
    [
        ("config_json", str),
        ("train_start", str),
        ("train_end", str),
        ("test_start", str),
        ("test_end", str),
        ("lookback_window", int),
        ("historical_window", int),
    ],
):
    """Resolve pipeline config: apply overrides, Katib params, date ranges, and validate.

    This lightweight component runs at the start of the pipeline to:
    1. Build PipelineConfig with parameter overrides
    2. Apply Katib best hyperparameters when katib_best_params_json is provided
    3. Apply training mode adjustments (finetune → reduced LR/epochs)
    4. Resolve date ranges from snapshot_date
    5. Validate the configuration
    6. Output resolved config JSON and individual date parameters

    Args:
        katib_best_params_json: Optional JSON string of Katib best hyperparameters.
            When provided, overrides the corresponding config fields.
            Expected keys: learning_rate, num_layers, num_heads, d_ff, dropout,
            batch_size, lookback_window.

    Raises:
        ValueError: If configuration validation fails.
    """
    import json
    from collections import namedtuple
    from dataclasses import asdict, dataclass
    from datetime import date, timedelta

    @dataclass
    class _Config:
        symbol: str = "USDJPY"
        forecast_horizon: int = 1
        lookback_window: int = 36
        historical_window: int = 1440
        num_features: int = 16
        num_layers: int = 3
        num_heads: int = 4
        d_ff: int = 64
        dropout: float = 0.1
        learning_rate: float = 0.001
        batch_size: int = 64
        epochs: int = 5
        random_seed: int = 42
        train_start: str = "2012-01-01"
        train_end: str = "2022-12-31"
        test_start: str = "2023-01-01"
        test_end: str = "2026-04-30"
        data_start: str = "2012-01-01"
        train_pct: float = 0.75
        test_pct: float = 0.25
        gpu_enabled: bool = True
        num_workers: int = 1
        max_wall_time_hours: int = 8
        katib_enabled: bool = False
        katib_max_trials: int = 30
        katib_parallel_trials: int = 5
        katib_trial_timeout_hours: int = 2
        serving_min_replicas: int = 0
        serving_max_replicas: int = 4
        serving_target_concurrency: int = 10
        canary_traffic_percent: int = 0
        alert_webhook_url: str = ""
        nll_degradation_threshold: float = 0.1
        da_degradation_threshold: float = 0.05
        nll_absolute_threshold: float = 3.5
        da_absolute_threshold: float = 0.50
        schedule_mode: str = "drift-triggered"
        training_mode: str = "scratch"
        finetune_epochs: int = 2
        finetune_learning_rate: float = 0.0001

    cfg = _Config(
        symbol=symbol,
        forecast_horizon=forecast_horizon,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_mode=training_mode,
    )

    # Apply Katib best hyperparameters when provided
    if katib_best_params_json:
        katib_params = json.loads(katib_best_params_json)
        katib_overrides = {
            k: v
            for k, v in katib_params.items()
            if k in _Config.__dataclass_fields__
        }
        for k, v in katib_overrides.items():
            setattr(cfg, k, v)
        cfg.katib_enabled = True

    # Apply training mode adjustments
    if training_mode == "finetune":
        cfg.epochs = cfg.finetune_epochs
        cfg.learning_rate = cfg.finetune_learning_rate

    # Resolve date ranges
    if snapshot_date:
        end_d = date.fromisoformat(snapshot_date)
        start_d = date.fromisoformat(cfg.data_start)
        total_days = (end_d - start_d).days
        test_days = int(total_days * cfg.test_pct)
        test_end_d = end_d
        test_start_d = end_d - timedelta(days=test_days)
        train_end_d = test_start_d - timedelta(days=1)
        train_start_d = start_d
        cfg.train_start = train_start_d.isoformat()
        cfg.train_end = train_end_d.isoformat()
        cfg.test_start = test_start_d.isoformat()
        cfg.test_end = test_end_d.isoformat()

    # Validate
    errors: list = []
    if cfg.symbol not in ("USDJPY", "AUDJPY"):
        errors.append(f"Invalid symbol: {cfg.symbol}")
    if cfg.forecast_horizon not in (1, 3, 6, 12):
        errors.append(f"Invalid forecast_horizon: {cfg.forecast_horizon}")
    if not (0.0001 <= cfg.learning_rate <= 0.01):
        errors.append(f"learning_rate out of range: {cfg.learning_rate}")
    if cfg.batch_size not in (32, 64, 128):
        errors.append(f"Invalid batch_size: {cfg.batch_size}")
    if errors:
        raise ValueError(
            f"Pipeline configuration validation failed: {'; '.join(errors)}"
        )

    config_json = json.dumps(asdict(cfg))

    ConfigOutputs = namedtuple(
        "ConfigOutputs",
        [
            "config_json",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "lookback_window",
            "historical_window",
        ],
    )
    return ConfigOutputs(
        config_json=config_json,
        train_start=cfg.train_start,
        train_end=cfg.train_end,
        test_start=cfg.test_start,
        test_end=cfg.test_end,
        lookback_window=cfg.lookback_window,
        historical_window=cfg.historical_window,
    )


# ---------------------------------------------------------------------------
# Config Helpers (client-side, for use at submission time)
# ---------------------------------------------------------------------------


def build_pipeline_config(
    symbol: str = "USDJPY",
    forecast_horizon: int = 1,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    snapshot_date: str = "",
    training_mode: str = "scratch",
) -> PipelineConfig:
    """Build and validate a PipelineConfig from pipeline parameters.

    This function is called at pipeline submission time (client-side) to
    resolve date ranges and validate configuration before the pipeline runs.
    Provides fail-fast validation before submitting to the KFP engine.

    Args:
        symbol: Currency pair to forecast (USDJPY or AUDJPY).
        forecast_horizon: Number of bars ahead to predict (1, 3, 6, or 12).
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        snapshot_date: ISO date for rolling-window date resolution.
            Empty string uses static date ranges from config.
        training_mode: "scratch" for full training or "finetune" for
            incremental training on production model weights.

    Returns:
        Validated PipelineConfig with resolved date ranges.

    Raises:
        ValueError: If configuration validation fails.
    """
    config = PipelineConfig()

    # Apply pipeline parameter overrides
    config = config.override(
        symbol=symbol,
        forecast_horizon=forecast_horizon,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_mode=training_mode,
    )

    # Apply training mode adjustments
    if training_mode == "finetune":
        config = config.override(
            epochs=config.finetune_epochs,
            learning_rate=config.finetune_learning_rate,
        )

    # Resolve date ranges based on snapshot_date
    resolved_snapshot = snapshot_date if snapshot_date else None
    config = config.resolve_date_ranges(resolved_snapshot)

    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(
            f"Pipeline configuration validation failed: {'; '.join(errors)}"
        )

    return config


# ---------------------------------------------------------------------------
# Lightweight Python component for resolving the current production checkpoint
# ---------------------------------------------------------------------------


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["model-registry"],
)
def resolve_production_checkpoint(
    registry_url: str,
    symbol: str,
    forecast_horizon: int,
) -> str:
    """Return the URI of the current production checkpoint, or ""  if none.

    Looks up the registered model ``probabilistic-transformer-<symbol>-h<H>``
    in the Model Registry and returns the ``uri`` of the version whose
    ``custom_properties.lifecycle_stage`` is ``"production"``. When no
    production version exists (initial deployment) or the registry is
    unreachable, returns an empty string; the caller is expected to pass
    that through to ``model_training`` and use scratch mode.

    This is the input that lets ``model_training --training-mode=finetune``
    locate the prior weights it needs to warm-start from.
    """
    import json
    import sys
    from datetime import datetime, timezone

    def _log(msg: str, extra: dict | None = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "logger": __name__,
            "message": msg,
            "component": "resolve_production_checkpoint",
        }
        if extra:
            entry.update(extra)
        print(json.dumps(entry), file=sys.stdout)

    model_name = (
        f"probabilistic-transformer-{symbol.lower()}-h{forecast_horizon}"
    )

    try:
        from model_registry import ModelRegistry
    except ImportError as e:
        _log(
            "Model Registry SDK not available; returning empty URI",
            extra={"error": str(e)},
        )
        return ""

    try:
        registry = ModelRegistry(
            server_address=registry_url, author="forecaster-pipeline"
        )
        versions = list(registry.get_model_versions(model_name))
    except Exception as e:
        _log(
            "Registry lookup failed; returning empty URI (treat as no production)",
            extra={"error": str(e), "model_name": model_name},
        )
        return ""

    production_uri = ""
    for v in versions:
        props = v.custom_properties if hasattr(v, "custom_properties") else {}
        # custom_properties values may be wrapped (MetadataStringValue) or
        # plain, read both shapes.
        stage_raw = props.get("lifecycle_stage")
        if hasattr(stage_raw, "string_value"):
            stage = stage_raw.string_value
        else:
            stage = stage_raw
        if stage == "production":
            production_uri = getattr(v, "uri", "") or ""
            break

    _log(
        "Resolved production checkpoint",
        extra={
            "model_name": model_name,
            "uri": production_uri,
            "found": bool(production_uri),
        },
    )
    return production_uri


# ---------------------------------------------------------------------------
# Lightweight Python component for Model Registry registration
# ---------------------------------------------------------------------------


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3", "model-registry", "kubernetes"],
)
def model_registration(
    model_checkpoint: Input[Model],
    evaluation_metrics: Input[Metrics],
    pipeline_run_id: str,
    registry_url: str,
    symbol: str,
    forecast_horizon: int,
    config_json: str,
):
    """Register the trained model in the Model Registry after evaluation.

    Reads evaluation metrics from S3, checks the degradation gate decision,
    and registers/promotes the model when the gate passes. Handles initial
    deployment bootstrap (auto-promotes the first model when no production
    model exists).

    Gracefully degrades: if the Model Registry is unreachable, logs a warning
    and returns without failing the pipeline (model checkpoint is still in S3).

    Args:
        model_checkpoint: KFP Model artifact from model_training.
        evaluation_metrics: KFP Metrics artifact from model_evaluation.
        pipeline_run_id: Unique run ID of this pipeline execution.
        registry_url: URL of the Kubeflow Model Registry server.
        symbol: Currency pair symbol (USDJPY or AUDJPY).
        forecast_horizon: Forecast horizon in bars.
        config_json: JSON string of the resolved pipeline configuration.

    Requirements: 6.2, 6.3, 6.4, 9.4
    """
    import json
    import logging
    import os
    import sys
    import uuid
    from datetime import datetime, timezone

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model_checkpoint_uri = model_checkpoint.uri
    evaluation_metrics_uri = evaluation_metrics.uri

    def _log(msg: str, extra: dict = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "logger": __name__,
            "message": msg,
            "component": "model_registration",
        }
        if extra:
            entry.update(extra)
        print(json.dumps(entry), file=sys.stdout)

    # Step 1: Read evaluation metrics ----------------------------------------
    _log("Reading evaluation metrics", extra={"uri": evaluation_metrics_uri})

    import boto3
    from urllib.parse import urlparse

    def _get_object_bytes(uri: str) -> bytes:
        """URI-aware read. minio://b/k → MinIO; s3://b/k → AWS S3;
        bare key → AWS S3 against the default forex bucket.
        """
        parsed = urlparse(uri)
        scheme = (parsed.scheme or "").lower()
        if scheme == "minio":
            client = boto3.client(
                "s3",
                endpoint_url="http://minio-service.kubeflow:9000",
                aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
                aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
                region_name="us-east-1",
            )
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
        elif scheme == "s3":
            client = boto3.client("s3")
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
        else:
            client = boto3.client("s3")
            bucket = "prod-fintech-forex-sg-731833471586"
            key = uri.lstrip("/")
        return client.get_object(Bucket=bucket, Key=key)["Body"].read()

    try:
        eval_data = json.loads(_get_object_bytes(evaluation_metrics_uri).decode("utf-8"))
    except Exception as e:
        _log(
            "Failed to read evaluation metrics; failing pipeline step",
            extra={"error": str(e)},
        )
        raise

    # Step 2: Check degradation gate -----------------------------------------
    gate = eval_data.get("degradation_gate", {})
    gate_passed = gate.get("gate_passed", False)
    gate_skipped = gate.get("gate_skipped", False)
    gate_reason = gate.get("reason", "")

    _log(
        "Degradation gate result",
        extra={
            "gate_passed": gate_passed,
            "gate_skipped": gate_skipped,
            "reason": gate_reason,
        },
    )

    if not gate_passed:
        _log(
            "Model did not pass degradation gate; skipping registration. "
            "Model checkpoint retained in S3 for manual review.",
            extra={"gate_reason": gate_reason},
        )
        return

    # Step 3: Register model in Model Registry -------------------------------
    _log(
        "Gate passed; registering model in Model Registry",
        extra={"registry_url": registry_url},
    )

    try:
        from model_registry import ModelRegistry

        registry = ModelRegistry(
            server_address=registry_url, author="forecaster-pipeline"
        )

        # Build model name from symbol and horizon
        model_name = (
            f"probabilistic-transformer-{symbol.lower()}-h{forecast_horizon}"
        )

        # Build metadata
        version_id = str(uuid.uuid4())
        config = json.loads(config_json)
        test_metrics = eval_data.get("test_metrics", {})

        # Look up any existing production version *before* registering this one,
        # so we can demote it after the new version lands.
        existing_production: list = []
        try:
            for v in registry.get_model_versions(model_name):
                if (v.custom_properties or {}).get(
                    "lifecycle_stage"
                ) == "production":
                    existing_production.append(v)
        except Exception:
            # Registered model does not exist yet, first deployment.
            pass

        has_production = bool(existing_production)
        if not has_production:
            _log("No production model exists; bootstrapping initial deployment")

        custom_properties = {
            "version_id": version_id,
            "symbol": symbol,
            "forecast_horizon": str(forecast_horizon),
            "training_timestamp": eval_data.get("timestamp", ""),
            "training_nll": str(config.get("training_nll", "")),
            "validation_nll": str(test_metrics.get("nll", "")),
            "directional_accuracy": str(test_metrics.get("directional_accuracy", "")),
            "coverage_ratio_95": str(test_metrics.get("coverage_ratio_95", "")),
            "rmse": str(test_metrics.get("rmse", "")),
            "hyperparameters": json.dumps(
                {
                    "num_layers": config.get("num_layers"),
                    "num_heads": config.get("num_heads"),
                    "d_ff": config.get("d_ff"),
                    "dropout": config.get("dropout"),
                    "learning_rate": config.get("learning_rate"),
                    "batch_size": config.get("batch_size"),
                    "lookback_window": config.get("lookback_window"),
                }
            ),
            "pipeline_run_id": pipeline_run_id,
            "data_snapshot_path": "",
            # New version starts in production on bootstrap; otherwise staging
            # until we demote the incumbent below.
            "lifecycle_stage": "production" if not has_production else "staging",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Single SDK call registers (or upserts) the RegisteredModel and creates
        # the ModelVersion with the supplied custom properties.
        registry.register_model(
            name=model_name,
            uri=model_checkpoint_uri,
            model_format_name="pytorch",
            model_format_version="2",
            version=version_id,
            description=(
                f"Trained on {symbol} h{forecast_horizon} "
                f"(run={pipeline_run_id})"
            ),
            version_description=f"version {version_id}",
            metadata=custom_properties,
        )

        _log(
            "Model registered successfully",
            extra={
                "version_id": version_id,
                "model_name": model_name,
                "nll": test_metrics.get("nll"),
                "directional_accuracy": test_metrics.get("directional_accuracy"),
            },
        )

        # Promote: demote the prior production version (if any) and ensure the
        # new version is marked production.
        if has_production:
            new_version = registry.get_model_version(model_name, version_id)
            if new_version is not None:
                new_version.custom_properties["lifecycle_stage"] = "production"
                registry.update(new_version)

            for prev in existing_production:
                prev.custom_properties["lifecycle_stage"] = "staging"
                registry.update(prev)
                _log(
                    "Demoted previous production model",
                    extra={
                        "version_id": (prev.custom_properties or {}).get(
                            "version_id"
                        )
                    },
                )

        _log(
            "Model promoted to production",
            extra={"version_id": version_id, "model_name": model_name},
        )

        # Note: standalone Forecaster KServe InferenceService is deprecated
        # (kubeflow-ml-pipeline spec, Requirement 5). The dqnpf-intraday
        # combined predictor's hot-reload watcher (Task 30.3) resolves the
        # new production checkpoint on its next poll; no KServe patching
        # is performed from this pipeline step.

        _log(
            "Model registration component completed successfully",
            extra={
                "version_id": version_id,
                "promoted": True,
                "bootstrap": not has_production,
            },
        )

    except ImportError as e:
        _log(
            "Model Registry SDK not available; skipping registration",
            extra={"error": str(e)},
        )
    except Exception as e:
        _log(
            "Model registration failed; failing the pipeline step. "
            "Model checkpoint remains in S3.",
            extra={"error": str(e)},
        )
        raise


# ---------------------------------------------------------------------------
# Pipeline Definition
# ---------------------------------------------------------------------------


@dsl.pipeline(
    name="forecaster-pipeline",
    description=(
        "Probabilistic Forex Forecaster Pipeline: "
        "data preparation → model training → model evaluation "
        "→ model registration → backtesting"
    ),
)
def forecaster_pipeline(
    symbol: str = "USDJPY",
    forecast_horizon: int = 1,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    snapshot_date: str = "",
    training_mode: str = "scratch",
    katib_best_params_json: str = "",
    model_registry_url: str = "http://model-registry-service.kubeflow.svc.cluster.local:8080",
):
    """Forecaster Pipeline: data → train → evaluate → register → backtest.

    Config loading and validation is performed both:
    - At submission time via build_pipeline_config() (client-side, fail-fast)
    - At runtime via the resolve_config component (in-cluster validation)

    The pipeline DAG wires six steps:

    1. resolve_config: validates params, applies Katib best params, resolves dates
    2. data_preparation: loads S3 data, computes features, outputs datasets
    3. model_training: trains the model, outputs checkpoint
    4. model_evaluation: evaluates metrics, runs degradation gate
    5. model_registration: registers in Model Registry if gate passes,
       auto-promotes on initial deployment (bootstrap)
    6. backtesting: runs trading strategies on test set

    DAG structure:
        resolve_config → data_preparation → model_training
            → model_evaluation → model_registration
            → backtesting
        (model_evaluation/model_registration and backtesting run in parallel
         after training)

    Args:
        symbol: Currency pair to forecast (USDJPY or AUDJPY).
        forecast_horizon: Number of bars ahead to predict (1, 3, 6, or 12).
        epochs: Number of training epochs (overridden in finetune mode).
        batch_size: Training batch size.
        learning_rate: Learning rate (overridden in finetune mode).
        snapshot_date: ISO date for rolling-window date resolution.
            Empty string uses static date ranges from config.
        training_mode: "scratch" for full training or "finetune" for
            incremental training on production model weights.
        katib_best_params_json: JSON string of Katib best hyperparameters.
            When provided, overrides the corresponding config fields
            (learning_rate, num_layers, num_heads, d_ff, dropout, batch_size,
            lookback_window).
        model_registry_url: URL of the Kubeflow Model Registry server.
    """
    # -----------------------------------------------------------------------
    # Step 0: Config resolution and validation (lightweight Python component)
    # -----------------------------------------------------------------------
    config_task = resolve_config(
        symbol=symbol,
        forecast_horizon=forecast_horizon,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        snapshot_date=snapshot_date,
        training_mode=training_mode,
        katib_best_params_json=katib_best_params_json,
    )

    # -----------------------------------------------------------------------
    # Step 1: Data Preparation
    # -----------------------------------------------------------------------
    dp_task = data_preparation(
        symbol=symbol,
        train_start=config_task.outputs["train_start"],
        train_end=config_task.outputs["train_end"],
        test_start=config_task.outputs["test_start"],
        test_end=config_task.outputs["test_end"],
        lookback_window=config_task.outputs["lookback_window"],
        forecast_horizon=forecast_horizon,
        historical_window=config_task.outputs["historical_window"],
    )
    dp_task.set_retry(num_retries=3)
    dp_task.set_caching_options(enable_caching=True)
    _mount_minio_creds(dp_task)

    # -----------------------------------------------------------------------
    # Step 1b: Resolve the current production checkpoint (for finetune mode)
    # -----------------------------------------------------------------------
    # Always runs, when no production version exists yet, returns "" and
    # model_training (in scratch mode) ignores it. When training_mode is
    # "finetune", the trainer requires a non-empty URI and will fail with a
    # clear ValueError if this is empty (i.e. you tried to fine-tune before
    # any production model existed).
    prod_ckpt_task = resolve_production_checkpoint(
        registry_url=model_registry_url,
        symbol=symbol,
        forecast_horizon=forecast_horizon,
    )

    # -----------------------------------------------------------------------
    # Step 2: Model Training
    # -----------------------------------------------------------------------
    mt_task = model_training(
        train_dataset=dp_task.outputs["train_dataset"],
        config_json=config_task.outputs["config_json"],
        training_mode=training_mode,
        production_model_uri=prod_ckpt_task.output,
    )
    mt_task.set_retry(num_retries=3)
    _mount_minio_creds(mt_task)

    # -----------------------------------------------------------------------
    # Step 3: Model Evaluation
    # -----------------------------------------------------------------------
    me_task = model_evaluation(
        model_checkpoint=mt_task.outputs["model_checkpoint"],
        test_dataset=dp_task.outputs["test_dataset"],
        config_json=config_task.outputs["config_json"],
    )
    me_task.set_retry(num_retries=3)
    _mount_minio_creds(me_task)

    # -----------------------------------------------------------------------
    # Step 4: Model Registration (after evaluation)
    # -----------------------------------------------------------------------
    # pipeline_run_id identifies this run for lineage tracking in the Model
    # Registry. Uses the pipeline job name placeholder which KFP resolves
    # to the actual job name at runtime.
    reg_task = model_registration(
        model_checkpoint=mt_task.outputs["model_checkpoint"],
        evaluation_metrics=me_task.outputs["evaluation_metrics"],
        pipeline_run_id=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        registry_url=model_registry_url,
        symbol=symbol,
        forecast_horizon=forecast_horizon,
        config_json=config_task.outputs["config_json"],
    )
    _mount_minio_creds(reg_task)

    # -----------------------------------------------------------------------
    # Step 5: Backtesting (runs in parallel with evaluation + registration)
    # -----------------------------------------------------------------------
    bt_task = backtesting(
        model_checkpoint=mt_task.outputs["model_checkpoint"],
        test_dataset=dp_task.outputs["test_dataset"],
        config_json=config_task.outputs["config_json"],
    )
    bt_task.set_retry(num_retries=3)
    _mount_minio_creds(bt_task)
