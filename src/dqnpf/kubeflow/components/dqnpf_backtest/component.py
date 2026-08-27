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
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

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
# Seconds to wait for the modelenv sidecar's port. Its startup preload can take
# minutes on a cold cache; 60s killed it after the eoh/eod/eow/eom consolidation
# made the "latest" snapshots full-history. Env-overridable; mirrors the DQN
# train/backtest components' fix.
_MODELENV_HEALTHCHECK_TIMEOUT_S = float(
    os.environ.get("MODELENV_HEALTH_CHECK_TIMEOUT", "3600")
)


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


def _start_modelenv_sidecar(
    symbol: str,
    port: int = 50051,
    trade_log_path: str | None = None,
    swap_rate_long: float = 0.0,
    swap_rate_short: float = 0.0,
    trading_session_hour_start: int | None = None,
    trading_session_hour_end: int | None = None,
    training_date_start: str | None = None,
    training_date_end: str | None = None,
) -> subprocess.Popen:
    """Launch modelenv as a subprocess and wait for the port to open.

    When ``trade_log_path`` is set, modelenv is started with ``--trade-log``,
    so every order fill and position close is appended as JSONL to that path
    for offline review. The trade log is a pure side-channel; it does not
    affect observations, rewards, or the backtest result.
    swap-rate values; the clean way to run a swap-free comparison backtest. By
    default it is left unset so modelenv applies its built-in default table and
    the backtest's PnL reflects the same swap regime the DQN trained under.

    ``swap_rate_long`` / ``swap_rate_short`` are the daily overnight-financing
    rates (per unit volume) charged to BUY / SELL positions held across a day
    boundary; only a genuinely
    non-zero rate is forwarded as ``--swap-rate-long`` / ``--swap-rate-short`` so
    the backtest's PnL and trade log reflect overnight swap costs.

    ``trading_session_hour_start`` / ``trading_session_hour_end`` align
    modelenv's end-of-session liquidation boundary: at each session end
    modelenv closes all shorts + losing longs (net of swap) and keeps
    winning/break-even longs. They are forwarded as
    ``--trading-session-hour-start`` / ``--trading-session-hour-end`` only when
    set (date-range backtests), so the liquidation boundary matches the
    hour-bound episode windows the backtest runs. ``None`` (legacy fixed-window
    mode) forwards nothing, in which case modelenv's own default applies,
    liquidation is ON by default at 23h UTC (modelenv T-6.3-08); a fully
    liquidation-free run must start modelenv with ``--no-session-liquidation``.
    """
    # Coerce to float before the truthiness guards below. A string like "0.0"
    # (e.g. a KFP str pipeline param) is truthy, which would forward
    # `--swap-rate-long 0.0` and wrongly OVERRIDE modelenv's built-in default
    # table with zero swap. After coercion only a genuinely non-zero rate is
    # forwarded; 0.0 (float) is falsy and leaves the built-in table in place.
    def _as_rate(value: object) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    swap_rate_long = _as_rate(swap_rate_long)
    swap_rate_short = _as_rate(swap_rate_short)

    logger.info(
        json.dumps(
            {
                "event": "modelenv.start",
                "symbol": symbol,
                "port": port,
                "binary": _MODELENV_BINARY,
                "trade_log_path": trade_log_path,
                "swap_rate_long": swap_rate_long,
                "swap_rate_short": swap_rate_short,
                "trading_session_hour_start": trading_session_hour_start,
                "trading_session_hour_end": trading_session_hour_end,
            }
        )
    )
    # modelenv-server takes --addr host:port (default 0.0.0.0:50051) and
    # rejects unknown flags, so --port is invalid. Bind on all interfaces so
    # the localhost healthcheck below connects. Inherit this process's
    # stdout/stderr so modelenv's startup logs land in the pod logs for
    # debugging (rather than being swallowed by an undrained pipe).
    cmd = [_MODELENV_BINARY, "--addr", f"0.0.0.0:{port}", "--symbol", symbol]
    if trade_log_path:
        cmd += ["--trade-log", trade_log_path]
    else:
        if swap_rate_long:
            cmd += ["--swap-rate-long", str(swap_rate_long)]
        if swap_rate_short:
            cmd += ["--swap-rate-short", str(swap_rate_short)]
    # End-of-session liquidation: forward the session-hour window so modelenv
    # liquidates at the same boundary the date-range episodes are sliced on.
    if trading_session_hour_start is not None:
        cmd += ["--trading-session-hour-start", str(trading_session_hour_start)]
    if trading_session_hour_end is not None:
        cmd += ["--trading-session-hour-end", str(trading_session_hour_end)]
    # Scope modelenv's startup preload to the backtest window. Without this,
    # modelenv preloads the full M1 reference span = the LATEST snapshot, which
    # since the snapshot consolidation is the full 2012->now history (5.3M M1
    # rows); that OOMs the 8Gi pod and blows the readiness timeout. Passing the
    # window makes the preload resolve/load only the eval range.
    if training_date_start:
        cmd += ["--training-date-start", training_date_start]
    if training_date_end:
        cmd += ["--training-date-end", training_date_end]
    proc = subprocess.Popen(cmd)
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
    model_provenance: dict,
) -> dict:
    """Build the JSON payload written to the output Artifact path.

    ``model_provenance`` records *which* checkpoints the run actually used:
    the Model Registry entry names and the resolved registry URIs
    (``minio://``/``s3://`` ...``model_checkpoint``). ``integration_config``
    only carries the ephemeral local temp paths the checkpoints were
    downloaded to, which don't identify the source artifact.
    """
    return {
        "comparison": asdict(comparison),
        "threshold_report": asdict(report),
        "integration_config": asdict(integration_config),
        "model_provenance": model_provenance,
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


def _materialize_checkpoint(uri: str, dest_dir: str, name: str) -> str:
    """Resolve a checkpoint URI to a readable local file path.

    The Model Registry hands back the KFP artifact URI of the parent
    checkpoint, e.g. ``minio://mlpipeline/v2/artifacts/.../model_checkpoint``
    (or ``s3://...``). ``torch.load`` can't open those schemes, so download
    the object to a local file via the same URI-routing the DQN/forecaster
    components use. Bare local paths (unit tests, pre-downloaded files) pass
    through unchanged.

    ``name`` disambiguates the local filename: both the DQN and forecaster
    artifact URIs end in ``model_checkpoint``, so deriving the filename from
    the URI basename alone would collide and the second download would clobber
    the first.
    """
    scheme = urlparse(uri).scheme.lower()
    if scheme not in ("minio", "s3"):
        return uri

    # deepqnetwork.artifact_io routes minio:// to the in-cluster MinIO (using
    # MINIO_ACCESS_KEY/MINIO_SECRET_KEY) and s3:// to AWS S3.
    from deepqnetwork import artifact_io

    basename = os.path.basename(urlparse(uri).path) or "checkpoint"
    local_path = os.path.join(dest_dir, f"{name}-{basename}")
    artifact_io.download_file(uri, local_path)
    return local_path


def _write_output_bytes(path: str, data: bytes, content_type: str) -> None:
    """Write ``data`` to ``path``, routing minio:///s3:// through artifact_io.

    KFP hands us the artifact's minio:// (or s3://) URI; torch/Path can't write
    to those schemes, so route uploads through artifact_io like the DQN side.
    Bare local paths (KFP outputPath, unit tests) are written directly, creating
    parent directories as needed.
    """
    if urlparse(path).scheme.lower() in ("minio", "s3"):
        from deepqnetwork import artifact_io

        artifact_io.put_object_bytes(path, data, content_type=content_type)
    else:
        out = Path(path)
        if out.parent != Path(""):
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)


def _export_trade_log(local_trade_log: str, trade_log_output_path: str) -> None:
    """Copy the modelenv trade log to the downloadable output artifact.

    modelenv writes the JSONL trade log to ``local_trade_log`` (a pod-local
    file). To make it downloadable after the run we upload its contents to
    ``trade_log_output_path`` (a KFP outputPath / minio:// / s3:// URI). The
    declared output must always exist, so if no trade log was produced (e.g. the
    run placed no trades) an empty file is written instead.
    """
    if os.path.exists(local_trade_log):
        data = Path(local_trade_log).read_bytes()
        lines = data.count(b"\n")
    else:
        data = b""
        lines = 0
    _write_output_bytes(trade_log_output_path, data, "application/x-ndjson")
    logger.info(
        json.dumps(
            {
                "event": "backtest.trade_log_exported",
                "records": lines,
                "output": trade_log_output_path,
            }
        )
    )


def run_dqnpf_backtest(
    integration_config_yaml: str,
    dqn_model_registry_name: str,
    forecaster_model_registry_name: str,
    output_artifact_path: str,
    *,
    checkpoint_resolver: CheckpointResolver | None = None,
    run_backtest_fn: Callable[[IntegrationConfig], BacktestComparison] | None = None,
    trade_log_output_path: str | None = None,
    swap_rate_long: float = 0.0,
    swap_rate_short: float = 0.0,
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
        trade_log_output_path: Optional destination for modelenv's JSONL trade
            log (every fill + position close), exported so it can be downloaded
            after the run. modelenv writes it to a pod-local temp file via
            ``--trade-log``; once the backtest finishes the file is uploaded
            here (a KFP outputPath, or a ``minio://`` / ``s3://`` URI). Only
            takes effect when the sidecar is started by this component. The
            trade log is a side-channel and does not affect the backtest result.
        swap_rate_long: Daily overnight-financing rate (per unit volume) charged
            to BUY positions held across a day boundary. Default 0.0 = leave
            modelenv's built-in default table in place; a non-zero value is
            forwarded as ``--swap-rate-long``.
        swap_rate_short: Same as ``swap_rate_long`` for SELL positions, passed as
            ``--swap-rate-short``. Both only take effect when this component
            starts the sidecar.

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

    # The resolved paths are KFP artifact URIs (minio://, s3://); torch.load
    # needs local files, so download them before the backtest opens them.
    ckpt_dir = tempfile.mkdtemp(prefix="dqnpf-ckpt-")
    dqn_local = _materialize_checkpoint(dqn_path, ckpt_dir, "dqn")
    fc_local = _materialize_checkpoint(fc_path, ckpt_dir, "forecaster")

    integration_cfg = _build_integration_config(
        pipeline_cfg,
        dqn_checkpoint_path=dqn_local,
        forecaster_checkpoint_path=fc_local,
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

    # modelenv only writes the trade log to a local file, so point it at a
    # pod-local temp path and export that to the downloadable output afterwards.
    local_trade_log = (
        os.path.join(tempfile.mkdtemp(prefix="dqnpf-tradelog-"), "trades.jsonl")
        if trade_log_output_path
        else None
    )

    # In date-range mode, align modelenv's end-of-session liquidation with the
    # hour-bound episode windows the backtest runs. The session hours default to
    # the window the DQN was trained on (read straight from the checkpoint,
    # without building a full advisor) unless the config overrides them; legacy
    # fixed-window mode resolves to None and leaves liquidation off.
    session_hour_start: int | None = None
    session_hour_end: int | None = None
    if integration_cfg.date_start:
        from deepqnetwork.advisor import DQNAdvisor

        session_hours = backtest._resolve_session_hours(
            integration_cfg, DQNAdvisor.read_training_window(dqn_local)
        )
        if session_hours is not None:
            session_hour_start, session_hour_end = session_hours

    sidecar = _start_modelenv_sidecar(
            integration_cfg.symbol,
            trade_log_path=local_trade_log,
            swap_rate_long=swap_rate_long,
            swap_rate_short=swap_rate_short,
            trading_session_hour_start=session_hour_start,
            trading_session_hour_end=session_hour_end,
            training_date_start=integration_cfg.date_start or None,
        training_date_end=integration_cfg.date_end or None,
    )
    try:
        comparison = runner(integration_cfg)
        report = validate_thresholds(comparison)
    finally:
        # Stop the sidecar first so the trade log is flushed/closed before export.
        if sidecar is not None:
            _stop_modelenv_sidecar(sidecar)
        if trade_log_output_path and local_trade_log is not None:
            _export_trade_log(local_trade_log, trade_log_output_path)

    payload = _serialise_result(
        comparison,
        report,
        integration_cfg,
        model_provenance={
            "dqn_model_registry_name": dqn_model_registry_name,
            "forecaster_model_registry_name": forecaster_model_registry_name,
            # Resolved registry URIs (pre-download), the canonical checkpoint
            # identity. integration_config only has the local temp paths.
            "dqn_checkpoint_uri": dqn_path,
            "forecaster_checkpoint_uri": fc_path,
        },
    )
    payload_json = json.dumps(payload, default=str, indent=2)
    _write_output_bytes(
        output_artifact_path, payload_json.encode("utf-8"), "application/json"
    )
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
