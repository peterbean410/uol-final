"""Unit tests for the dqnpf_backtest KFP component entry point.

Validates the orchestration layer with the registry resolver and
``run_backtest`` injected as fakes; the heavy gRPC + checkpoint paths are
out of scope for this layer (those are covered by the integration tests
under ``tradingmodel/intraday/dqnpf/tests/test_integration_e2e.py``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from tradingmodel.intraday.dqnpf.backtest import BacktestComparison
from tradingmodel.intraday.dqnpf.config import IntegrationConfig
from tradingmodel.intraday.dqnpf.kubeflow.components.dqnpf_backtest import component
from tradingmodel.intraday.dqnpf.kubeflow.components.dqnpf_backtest.component import (
    _build_integration_config,
    _load_pipeline_config,
    _serialise_result,
    _start_modelenv_sidecar,
    run_dqnpf_backtest,
)
from tradingmodel.intraday.dqnpf.kubeflow.components.dqnpf_backtest.__main__ import (
    _build_parser,
)
from tradingmodel.intraday.dqnpf.kubeflow.pipeline.config_schema import (
    DqnpfPipelineConfig,
)


def _passing_comparison() -> BacktestComparison:
    return BacktestComparison(
        combined_return=1.0,
        baseline_return=0.5,
        combined_sharpe=0.8,
        baseline_sharpe=0.4,
        suppression_rate=0.2,
        suppression_by_reason={"budget_exhausted": 1},
        high_sigma_pnl_combined=0.4,
        high_sigma_pnl_baseline=0.2,
        low_sigma_pnl_combined=0.6,
        low_sigma_pnl_baseline=0.6,
        trades_combined=5,
        trades_baseline=10,
        high_sigma_trades_combined=1,
        high_sigma_trades_baseline=5,
        high_sigma_negative_pnl_proportion_combined=0.2,
        high_sigma_negative_pnl_proportion_baseline=0.5,
        quarterly_pnl_combined={"2024-Q1": 0.3, "2024-Q2": 0.4, "2024-Q3": 0.3},
        quarterly_pnl_baseline={"2024-Q1": 0.2, "2024-Q2": 0.2, "2024-Q3": 0.1},
        high_sigma_time_fraction=0.3,
        # Money-PnL counterparts: validate_thresholds evaluates Req 14.1/14.3 on
        # these (raw_pnl_delta), not the reward-based *_sharpe / *_proportion.
        combined_sharpe_pnl=0.8,
        baseline_sharpe_pnl=0.4,
        high_sigma_negative_raw_pnl_proportion_combined=0.2,
        high_sigma_negative_raw_pnl_proportion_baseline=0.5,
    )


def _failing_comparison() -> BacktestComparison:
    cmp = _passing_comparison()
    cmp.combined_sharpe_pnl = 0.0  # fails Req 14.1 (money-PnL Sharpe)
    cmp.trades_combined = 100  # fails Req 14.2
    return cmp


def _stub_materialize_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the real S3/MinIO checkpoint download in unit tests.

    ``run_dqnpf_backtest`` calls ``_materialize_checkpoint`` to download the
    resolved URI to a local file for ``torch.load``. With an injected
    ``run_backtest_fn`` the checkpoint is never opened, so stub it to a no-op
    pass-through; the assertions expect the resolved URI to flow through to
    ``IntegrationConfig`` unchanged.
    """
    monkeypatch.setattr(
        component, "_materialize_checkpoint", lambda uri, dest_dir, name: uri
    )


def _write_pipeline_yaml(tmp_path: Path, **overrides) -> Path:
    cfg = DqnpfPipelineConfig(**overrides)
    yaml_path = tmp_path / "pipeline.yaml"
    cfg.to_yaml(yaml_path)
    return yaml_path


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_load_pipeline_config_accepts_valid_yaml(tmp_path: Path) -> None:
    yaml_path = _write_pipeline_yaml(tmp_path)
    cfg = _load_pipeline_config(str(yaml_path))
    assert isinstance(cfg, DqnpfPipelineConfig)
    assert cfg.symbol == "USDJPY"


def test_load_pipeline_config_rejects_invalid(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(yaml.safe_dump({"variance_threshold": -1.0}))
    with pytest.raises(ValueError, match="variance_threshold"):
        _load_pipeline_config(str(yaml_path))


def test_build_integration_config_applies_resolved_paths() -> None:
    pcfg = DqnpfPipelineConfig()
    cfg = _build_integration_config(
        pcfg,
        dqn_checkpoint_path="s3://bucket/dqn.pt",
        forecaster_checkpoint_path="s3://bucket/fc.pt",
    )
    assert isinstance(cfg, IntegrationConfig)
    assert cfg.dqn_checkpoint_path == "s3://bucket/dqn.pt"
    assert cfg.forecaster_checkpoint_path == "s3://bucket/fc.pt"
    assert cfg.symbol == pcfg.symbol


def test_serialise_result_contains_required_sections() -> None:
    comparison = _passing_comparison()
    from tradingmodel.intraday.dqnpf.backtest import validate_thresholds

    report = validate_thresholds(comparison)
    cfg = IntegrationConfig()
    provenance = {
        "dqn_model_registry_name": "deepqnetwork-usdjpy",
        "forecaster_model_registry_name": "probabilistic-transformer-usdjpy-h1",
        "dqn_checkpoint_uri": "minio://bucket/dqn/model_checkpoint",
        "forecaster_checkpoint_uri": "minio://bucket/fc/model_checkpoint",
    }
    payload = _serialise_result(comparison, report, cfg, provenance)
    assert set(payload.keys()) == {
        "comparison",
        "threshold_report",
        "integration_config",
        "model_provenance",
    }
    assert payload["comparison"]["combined_return"] == comparison.combined_return
    assert payload["threshold_report"]["passed"] is True
    assert payload["integration_config"]["symbol"] == cfg.symbol
    assert payload["model_provenance"] == provenance


# ---------------------------------------------------------------------------
# Entry point, fully injected
# ---------------------------------------------------------------------------


def test_run_dqnpf_backtest_writes_artifact_on_passing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_materialize_checkpoint(monkeypatch)
    yaml_path = _write_pipeline_yaml(tmp_path)
    output_path = tmp_path / "report.json"

    calls = {"resolver": [], "runner": 0}

    def resolver(name: str, stage: str) -> str:
        calls["resolver"].append((name, stage))
        return f"s3://bucket/{name}/{stage}.pt"

    def runner(cfg: IntegrationConfig) -> BacktestComparison:
        calls["runner"] += 1
        # Confirm the injected resolver values flowed through.
        assert cfg.dqn_checkpoint_path == "s3://bucket/deepqnetwork-usdjpy/production.pt"
        assert (
            cfg.forecaster_checkpoint_path
            == "s3://bucket/probabilistic-transformer-usdjpy-h1/production.pt"
        )
        return _passing_comparison()

    payload = run_dqnpf_backtest(
        integration_config_yaml=str(yaml_path),
        dqn_model_registry_name="deepqnetwork-usdjpy",
        forecaster_model_registry_name="probabilistic-transformer-usdjpy-h1",
        output_artifact_path=str(output_path),
        checkpoint_resolver=resolver,
        run_backtest_fn=runner,
        start_sidecar=False,
    )

    assert calls["runner"] == 1
    assert calls["resolver"] == [
        ("deepqnetwork-usdjpy", "production"),
        ("probabilistic-transformer-usdjpy-h1", "production"),
    ]
    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written == payload
    assert written["threshold_report"]["passed"] is True
    # The report records which checkpoints were used: registry names + the
    # resolved URIs (pre-download), not the ephemeral local temp paths.
    assert written["model_provenance"] == {
        "dqn_model_registry_name": "deepqnetwork-usdjpy",
        "forecaster_model_registry_name": "probabilistic-transformer-usdjpy-h1",
        "dqn_checkpoint_uri": "s3://bucket/deepqnetwork-usdjpy/production.pt",
        "forecaster_checkpoint_uri": (
            "s3://bucket/probabilistic-transformer-usdjpy-h1/production.pt"
        ),
    }


def test_run_dqnpf_backtest_records_failing_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_materialize_checkpoint(monkeypatch)
    yaml_path = _write_pipeline_yaml(tmp_path)
    output_path = tmp_path / "report.json"

    payload = run_dqnpf_backtest(
        integration_config_yaml=str(yaml_path),
        dqn_model_registry_name="deepqnetwork-usdjpy",
        forecaster_model_registry_name="probabilistic-transformer-usdjpy-h1",
        output_artifact_path=str(output_path),
        checkpoint_resolver=lambda n, s: f"s3://bucket/{n}.pt",
        run_backtest_fn=lambda cfg: _failing_comparison(),
        start_sidecar=False,
    )

    report = payload["threshold_report"]
    assert report["passed"] is False
    assert report["failures"]
    assert any("14.1" in f for f in report["failures"])


def test_run_dqnpf_backtest_uses_pipeline_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_materialize_checkpoint(monkeypatch)
    yaml_path = _write_pipeline_yaml(
        tmp_path,
        dqn_lifecycle_stage="staging",
        forecaster_lifecycle_stage="staging",
    )
    output_path = tmp_path / "report.json"
    stages: list[tuple[str, str]] = []

    run_dqnpf_backtest(
        integration_config_yaml=str(yaml_path),
        dqn_model_registry_name="deepqnetwork-usdjpy",
        forecaster_model_registry_name="probabilistic-transformer-usdjpy-h1",
        output_artifact_path=str(output_path),
        checkpoint_resolver=lambda n, s: stages.append((n, s)) or f"s3://{n}/{s}.pt",
        run_backtest_fn=lambda cfg: _passing_comparison(),
        start_sidecar=False,
    )

    assert stages == [
        ("deepqnetwork-usdjpy", "staging"),
        ("probabilistic-transformer-usdjpy-h1", "staging"),
    ]


def _patch_sidecar_capture(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Patch Popen + the port healthcheck so the sidecar cmd can be inspected."""
    captured: list[list[str]] = []

    class _FakeProc:
        def poll(self) -> None:
            return None

    def fake_popen(cmd, *args, **kwargs):
        captured.append(list(cmd))
        return _FakeProc()

    monkeypatch.setattr(component.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(component, "_wait_for_port", lambda *a, **k: None)
    return captured


def test_sidecar_omits_trade_log_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_sidecar_capture(monkeypatch)
    _start_modelenv_sidecar("USDJPY")
    assert "--trade-log" not in captured[0]


def test_sidecar_passes_trade_log_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_sidecar_capture(monkeypatch)
    _start_modelenv_sidecar("USDJPY", trade_log_path="/mnt/cache/trades.jsonl")
    cmd = captured[0]
    assert cmd[cmd.index("--trade-log") + 1] == "/mnt/cache/trades.jsonl"


def test_sidecar_omits_session_hours_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_sidecar_capture(monkeypatch)
    _start_modelenv_sidecar("USDJPY")
    # Legacy fixed-window mode: no session liquidation configured.
    assert "--trading-session-hour-start" not in captured[0]
    assert "--trading-session-hour-end" not in captured[0]


def test_sidecar_passes_session_hours_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_sidecar_capture(monkeypatch)
    _start_modelenv_sidecar(
        "USDJPY",
        trading_session_hour_start=15,
        trading_session_hour_end=39,
    )
    cmd = captured[0]
    assert cmd[cmd.index("--trading-session-hour-start") + 1] == "15"
    assert cmd[cmd.index("--trading-session-hour-end") + 1] == "39"


def test_cli_trade_log_output_path_defaults_to_none() -> None:
    args = _build_parser().parse_args(
        [
            "--integration-config-yaml=cfg.yaml",
            "--dqn-model-registry-name=dqn",
            "--forecaster-model-registry-name=fc",
            "--output-artifact-path=out.json",
        ]
    )
    assert args.trade_log_output_path is None


def test_cli_trade_log_output_path_parsed() -> None:
    args = _build_parser().parse_args(
        [
            "--integration-config-yaml=cfg.yaml",
            "--dqn-model-registry-name=dqn",
            "--forecaster-model-registry-name=fc",
            "--output-artifact-path=out.json",
            "--trade-log-output-path=/mnt/out/trades.jsonl",
        ]
    )
    assert args.trade_log_output_path == "/mnt/out/trades.jsonl"


def test_sidecar_omits_swap_rates_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_sidecar_capture(monkeypatch)
    _start_modelenv_sidecar("USDJPY")
    # By default the sidecar applies modelenv's built-in swap table: no
    # --no-swap and no per-side rate overrides.
    assert "--no-swap" not in captured[0]
    assert "--swap-rate-long" not in captured[0]
    assert "--swap-rate-short" not in captured[0]


def test_sidecar_passes_swap_rates_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_sidecar_capture(monkeypatch)
    _start_modelenv_sidecar("USDJPY", swap_rate_long=-0.5, swap_rate_short=0.25)
    cmd = captured[0]
    assert cmd[cmd.index("--swap-rate-long") + 1] == "-0.5"
    assert cmd[cmd.index("--swap-rate-short") + 1] == "0.25"


def test_sidecar_passes_no_swap_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_sidecar_capture(monkeypatch)
    _start_modelenv_sidecar("USDJPY", no_swap=True)
    assert "--no-swap" in captured[0]


def test_sidecar_no_swap_overrides_swap_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_sidecar_capture(monkeypatch)
    _start_modelenv_sidecar(
        "USDJPY", no_swap=True, swap_rate_long=-0.5, swap_rate_short=0.25
    )
    cmd = captured[0]
    assert "--no-swap" in cmd
    assert "--swap-rate-long" not in cmd
    assert "--swap-rate-short" not in cmd


def test_cli_swap_rates_default_to_zero() -> None:
    args = _build_parser().parse_args(
        [
            "--integration-config-yaml=cfg.yaml",
            "--dqn-model-registry-name=dqn",
            "--forecaster-model-registry-name=fc",
            "--output-artifact-path=out.json",
        ]
    )
    assert args.swap_rate_long == 0.0
    assert args.swap_rate_short == 0.0


def test_cli_swap_rates_parsed_as_float() -> None:
    args = _build_parser().parse_args(
        [
            "--integration-config-yaml=cfg.yaml",
            "--dqn-model-registry-name=dqn",
            "--forecaster-model-registry-name=fc",
            "--output-artifact-path=out.json",
            "--swap-rate-long=-0.5",
            "--swap-rate-short=0.25",
        ]
    )
    assert args.swap_rate_long == -0.5
    assert args.swap_rate_short == 0.25


def test_cli_no_swap_defaults_false() -> None:
    args = _build_parser().parse_args(
        [
            "--integration-config-yaml=cfg.yaml",
            "--dqn-model-registry-name=dqn",
            "--forecaster-model-registry-name=fc",
            "--output-artifact-path=out.json",
        ]
    )
    # Default: swap ON (built-in table), matching DQN training/backtest.
    assert args.no_swap is False


def test_cli_no_swap_parsed_true() -> None:
    args = _build_parser().parse_args(
        [
            "--integration-config-yaml=cfg.yaml",
            "--dqn-model-registry-name=dqn",
            "--forecaster-model-registry-name=fc",
            "--output-artifact-path=out.json",
            "--no-swap=true",
        ]
    )
    assert args.no_swap is True


def test_export_trade_log_copies_contents(tmp_path: Path) -> None:
    src = tmp_path / "trades.jsonl"
    src.write_text('{"type":"fill"}\n{"type":"close"}\n')
    dest = tmp_path / "out" / "trade_log.jsonl"
    component._export_trade_log(str(src), str(dest))
    assert dest.read_text() == '{"type":"fill"}\n{"type":"close"}\n'


def test_export_trade_log_writes_empty_when_no_trades(tmp_path: Path) -> None:
    # modelenv never created the file (no trades placed); the declared output
    # artifact must still exist, so an empty file is written.
    missing = tmp_path / "never-written.jsonl"
    dest = tmp_path / "trade_log.jsonl"
    component._export_trade_log(str(missing), str(dest))
    assert dest.exists()
    assert dest.read_text() == ""


def test_run_dqnpf_backtest_emits_summary_log(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_materialize_checkpoint(monkeypatch)
    yaml_path = _write_pipeline_yaml(tmp_path)
    output_path = tmp_path / "report.json"

    with caplog.at_level(
        logging.INFO,
        logger="tradingmodel.intraday.dqnpf.kubeflow.components.dqnpf_backtest.component",
    ):
        run_dqnpf_backtest(
            integration_config_yaml=str(yaml_path),
            dqn_model_registry_name="deepqnetwork-usdjpy",
            forecaster_model_registry_name="probabilistic-transformer-usdjpy-h1",
            output_artifact_path=str(output_path),
            checkpoint_resolver=lambda n, s: "s3://stub",
            run_backtest_fn=lambda cfg: _passing_comparison(),
            start_sidecar=False,
        )

    events: list[str] = []
    for rec in caplog.records:
        try:
            events.append(json.loads(rec.message)["event"])
        except (KeyError, ValueError):
            continue
    assert "backtest.start" in events
    assert "backtest.summary" in events
    assert "backtest.threshold_validation" in events
