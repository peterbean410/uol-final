"""KFP component: run the dqnpf-intraday backtest against frozen parent checkpoints.

Responsibilities (Requirement 21):

1. Read the integration config YAML and resolve checkpoint paths from the
   Model Registry for the DQN and Forecaster parent models.
2. Launch ``modelenv-server`` as a subprocess sidecar (mounting the warm
   modelenv-cache PVC) and wait for the gRPC server to come up on
   ``localhost:50051``.
3. Invoke :func:`tradingmodel.intraday.dqnpf.backtest.run_backtest` and then
   :func:`validate_thresholds`.
4. Write a single JSON artifact at the KFP output path containing the
   comparison, threshold report, and the integration config used.
5. Stream structured-JSON logs for suppression counts, budget utilisation,
   and conditional PnL so the run is queryable in CloudWatch / Loki.

The orchestration is deliberately kept thin and testable: the
:func:`_resolve_checkpoint` and ``run_backtest`` calls are dependency-injected
so unit tests can substitute fakes.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import yaml

from tradingmodel.intraday.dqnpf import backtest
from tradingmodel.intraday.dqnpf.backtest import (
    BacktestComparison,
    ThresholdReport,
    validate_thresholds,
)
from tradingmodel.intraday.dqnpf.config import IntegrationConfig
from tradingmodel.intraday.dqnpf.kubeflow.pipeline.config_schema import (
    DqnpfPipelineConfig,
)

logger = logging.getLogger(__name__)

# dqn/base installs the binary as /usr/local/bin/modelenv-server (see
# Dockerfile.dqn-base); the DQN training/backtest components reference the same
# path. Match it here.
_MODELENV_BINARY = os.environ.get(
    "MODELENV_BINARY", "/usr/local/bin/modelenv-server"
)
# modelenv runs in Training mode (its default) and preloads market data before
# binding the gRPC port. Even reading from the warm cache PVC this takes longer
# than 30s, so allow the same headroom the DQN backtest component uses.
_MODELENV_HEALTHCHECK_TIMEOUT_S = 60.0


# ---------------------------------------------------------------------------
# modelenv sidecar lifecycle
# ---------------------------------------------------------------------------


def _wait_for_port(
    host: str, port: int, timeout_s: float, proc: subprocess.Popen
) -> None:
    """Block until ``host:port`` accepts a TCP connection, or raise.

    Fails fast (rather than waiting out the full timeout) if the modelenv
    process exits before the port opens. modelenv's own logs stream to this
    pod's stdout/stderr, so the cause is visible above this error.
    """
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        returncode = proc.poll()
        if returncode is not None:
            raise RuntimeError(
                f"modelenv exited before binding {host}:{port} "
                f"(returncode={returncode}); see modelenv logs above"
            )
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_err = exc
            time.sleep(0.5)
    raise TimeoutError(
        f"modelenv healthcheck timed out after {timeout_s:.1f}s at "
        f"{host}:{port}: {last_err}"
    )


def _start_modelenv_sidecar(symbol: str, port: int = 50051) -> subprocess.Popen:
    """Launch modelenv as a subprocess and wait for the port to open."""
    logger.info(
        json.dumps(
            {
                "event": "modelenv.start",
                "symbol": symbol,
                "port": port,
                "binary": _MODELENV_BINARY,
            }
        )
    )
    # modelenv-server takes --addr host:port (default 0.0.0.0:50051) and
    # rejects unknown flags, so --port is invalid. Bind on all interfaces so
    # the localhost healthcheck below connects. Inherit this process's
    # stdout/stderr so modelenv's startup logs land in the pod logs for
    # debugging (rather than being swallowed by an undrained pipe).
    proc = subprocess.Popen(
        [_MODELENV_BINARY, "--addr", f"0.0.0.0:{port}", "--symbol", symbol],
    )
    _wait_for_port("localhost", port, _MODELENV_HEALTHCHECK_TIMEOUT_S, proc)
    logger.info(json.dumps({"event": "modelenv.ready", "port": port}))
    return proc


def _stop_modelenv_sidecar(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    logger.info(json.dumps({"event": "modelenv.stopped"}))


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def _load_pipeline_config(yaml_path: str) -> DqnpfPipelineConfig:
    cfg = DqnpfPipelineConfig.from_yaml(yaml_path)
    errors = cfg.validate()
    if errors:
        raise ValueError(
            "dqnpf pipeline config validation failed: " + "; ".join(errors)
        )
    return cfg


def _build_integration_config(
    pipeline_cfg: DqnpfPipelineConfig,
    *,
    dqn_checkpoint_path: str,
    forecaster_checkpoint_path: str,
) -> IntegrationConfig:
    """Materialise an IntegrationConfig with resolved registry paths."""
    cfg = pipeline_cfg.to_integration_config()
    cfg.dqn_checkpoint_path = dqn_checkpoint_path
    cfg.forecaster_checkpoint_path = forecaster_checkpoint_path
    return cfg


def _serialise_result(
    comparison: BacktestComparison,
    report: ThresholdReport,
    integration_config: IntegrationConfig,
) -> dict:
    """Build the JSON payload written to the output Artifact path."""
    return {
        "comparison": asdict(comparison),
        "threshold_report": asdict(report),
        "integration_config": asdict(integration_config),
    }


def _emit_summary_logs(
    comparison: BacktestComparison, report: ThresholdReport
) -> None:
    """Emit structured-JSON summary logs (Req 21.6, 24.1)."""
    logger.info(
        json.dumps(
            {
                "event": "backtest.summary",
                "combined_return": comparison.combined_return,
                "baseline_return": comparison.baseline_return,
                "combined_sharpe": comparison.combined_sharpe,
                "baseline_sharpe": comparison.baseline_sharpe,
                "suppression_rate": comparison.suppression_rate,
                "suppression_by_reason": comparison.suppression_by_reason,
                "trades_combined": comparison.trades_combined,
                "trades_baseline": comparison.trades_baseline,
                "high_sigma_time_fraction": comparison.high_sigma_time_fraction,
                "quarterly_pnl_combined": comparison.quarterly_pnl_combined,
            }
        )
    )
    logger.info(
        json.dumps(
            {
                "event": "backtest.threshold_validation",
                "passed": report.passed,
                "failures": report.failures,
            }
        )
    )


# ---------------------------------------------------------------------------
# Registry resolver, injected at call sites so tests can stub it
# ---------------------------------------------------------------------------


CheckpointResolver = Callable[[str, str], str]
"""Signature: ``resolve(model_name, lifecycle_stage) -> checkpoint_s3_path``."""


def _default_checkpoint_resolver(model_name: str, lifecycle_stage: str) -> str:
    """Default resolver delegates to the combined Model Registry client.

    ``lifecycle_stage`` is part of the resolver protocol but
    ``resolve_production_checkpoint`` only looks up the ``production``
    stage (its only contract); we accept the arg for API stability but
    do not forward it. The registry URL is read from the
    ``MODEL_REGISTRY_URL`` env var (matches the predictor's
    InferenceService and Airflow trigger conventions) and falls back to
    the in-cluster service DNS.
    """
    import os

    from tradingmodel.intraday.dqnpf.kubeflow.registry.registry_client import (
        resolve_production_checkpoint,
    )

    registry_url = os.environ.get(
        "MODEL_REGISTRY_URL",
        "http://model-registry-service.kubeflow.svc.cluster.local:8080",
    )
    return resolve_production_checkpoint(model_name, registry_url=registry_url)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_dqnpf_backtest(
    integration_config_yaml: str,
    dqn_model_registry_name: str,
    forecaster_model_registry_name: str,
    output_artifact_path: str,
    *,
    checkpoint_resolver: CheckpointResolver | None = None,
    run_backtest_fn: Callable[[IntegrationConfig], BacktestComparison] | None = None,
    start_sidecar: bool = True,
) -> dict:
    """Run the dqnpf-intraday backtest and write the artifact to ``output_artifact_path``.

    Args:
        integration_config_yaml: Path to a YAML file with ``DqnpfPipelineConfig``
            fields. The YAML is parsed, validated, and converted to an
            ``IntegrationConfig`` for the backtest.
        dqn_model_registry_name: Model Registry entry name for the DQN parent.
        forecaster_model_registry_name: Model Registry entry name for the
            Forecaster parent.
        output_artifact_path: Filesystem path KFP provides for the output
            Artifact. JSON payload is written here.
        checkpoint_resolver: Optional dependency-injected resolver
            ``(model_name, lifecycle_stage) -> s3_path`` for unit tests.
        run_backtest_fn: Optional dependency-injected backtest runner for unit
            tests. Defaults to :func:`backtest.run_backtest`.
        start_sidecar: If True (default), launches modelenv as a subprocess.
            Set to False for unit tests that supply their own gRPC env or use
            an injected ``run_backtest_fn``.

    Returns:
        The payload dict that was written to ``output_artifact_path``.
    """
    resolve = checkpoint_resolver or _default_checkpoint_resolver
    runner = run_backtest_fn or backtest.run_backtest

    pipeline_cfg = _load_pipeline_config(integration_config_yaml)
    dqn_path = resolve(dqn_model_registry_name, pipeline_cfg.dqn_lifecycle_stage)
    fc_path = resolve(
        forecaster_model_registry_name, pipeline_cfg.forecaster_lifecycle_stage
    )

    integration_cfg = _build_integration_config(
        pipeline_cfg,
        dqn_checkpoint_path=dqn_path,
        forecaster_checkpoint_path=fc_path,
    )

    logger.info(
        json.dumps(
            {
                "event": "backtest.start",
                "symbol": integration_cfg.symbol,
                "num_episodes": integration_cfg.num_episodes,
                "dqn_checkpoint_path": dqn_path,
                "forecaster_checkpoint_path": fc_path,
            }
        )
    )

    sidecar = (
        _start_modelenv_sidecar(integration_cfg.symbol) if start_sidecar else None
    )
    try:
        comparison = runner(integration_cfg)
        report = validate_thresholds(comparison)
    finally:
        if sidecar is not None:
            _stop_modelenv_sidecar(sidecar)

    payload = _serialise_result(comparison, report, integration_cfg)
    Path(output_artifact_path).write_text(json.dumps(payload, default=str, indent=2))
    _emit_summary_logs(comparison, report)

    return payload


# Re-export for ergonomic imports in tests
__all__ = [
    "CheckpointResolver",
    "run_dqnpf_backtest",
    "_build_integration_config",
    "_load_pipeline_config",
    "_serialise_result",
]
