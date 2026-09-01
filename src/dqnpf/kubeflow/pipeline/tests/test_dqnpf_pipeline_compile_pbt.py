"""Property-based tests for dqnpf-intraday KFP pipeline compilation.

Feature: kubeflow-ml-pipeline (dqnpf-intraday section)

Property DQNPF-PIPE-1: Pipeline compiles deterministically, repeated
compilation of the same ``dqnpf_intraday_pipeline`` produces byte-identical
IR YAML. The pipeline currently takes no caller-side variation in its
component definition, so determinism collapses to "same code → same bytes".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from hypothesis import given, settings, strategies as st

from dqnpf.kubeflow.pipeline.dqnpf_pipeline import (
    DEFAULT_IR_PATH,
    PIPELINE_NAME,
    compile_pipeline,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@settings(max_examples=5, deadline=None)
@given(suffix=st.integers(min_value=0, max_value=10_000))
def test_compile_pipeline_is_deterministic(
    tmp_path_factory, suffix: int
) -> None:
    tmp = tmp_path_factory.mktemp("dqnpf_pipeline_compile")
    first = tmp / f"first_{suffix}.yaml"
    second = tmp / f"second_{suffix}.yaml"

    compile_pipeline(first)
    compile_pipeline(second)

    assert first.exists() and second.exists()
    assert _digest(first) == _digest(second), (
        "compile_pipeline produced different bytes across two invocations"
    )


def test_shipped_ir_yaml_matches_fresh_compile(tmp_path) -> None:
    """The committed IR YAML next to the pipeline module is the latest output."""
    fresh = tmp_path / "fresh.yaml"
    compile_pipeline(fresh)
    assert DEFAULT_IR_PATH.exists(), (
        f"shipped IR YAML missing at {DEFAULT_IR_PATH}; run "
        "`python -m dqnpf.kubeflow.pipeline.dqnpf_pipeline`"
    )
    assert _digest(fresh) == _digest(DEFAULT_IR_PATH), (
        "shipped dqnpf_pipeline.yaml is stale, re-run compile_pipeline()"
    )


def test_compiled_ir_contains_pipeline_name(tmp_path) -> None:
    out = tmp_path / "pipeline.yaml"
    compile_pipeline(out)
    body = out.read_text()
    assert f"name: {PIPELINE_NAME}" in body
    assert "exec-dqnpf-backtest" in body
    assert "dqnpf-intraday-backtest:latest" in body


def test_compiled_ir_exposes_three_input_parameters(tmp_path) -> None:
    """The pipeline accepts exactly the three documented parameters."""
    out = tmp_path / "pipeline.yaml"
    compile_pipeline(out)
    body = out.read_text()
    for param in (
        "integration_config_yaml",
        "dqn_model_registry_name",
        "forecaster_model_registry_name",
    ):
        assert param in body, f"compiled IR missing parameter {param!r}"


def test_compiled_ir_disables_caching(tmp_path) -> None:
    """Backtest task must run fresh every time (no KFP cache).

    KFP emits ``cachingOptions: {}`` when caching is explicitly disabled (the
    map only contains ``enableCache`` when set to ``true``). The presence of
    the empty map plus the absence of ``enableCache: true`` is the disabled
    signal.
    """
    out = tmp_path / "pipeline.yaml"
    compile_pipeline(out)
    body = out.read_text()
    assert "cachingOptions: {}" in body
    assert "enableCache: true" not in body
