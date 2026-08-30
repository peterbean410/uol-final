"""Unit tests for the threshold-tuning grid search (Task 9.4).

These tests inject fake ``run_fn`` / ``validate_fn`` callables so the grid
logic is exercised without ``torch`` or a live modelenv. A ``BacktestComparison``
is built via a lightweight stand-in to avoid importing ``backtest`` (which
transitively imports torch).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from dqnpf.config import IntegrationConfig
from dqnpf.tune import (
    DEFAULT_GRID,
    GridPoint,
    GridResult,
    frozen_config,
    grid_result_to_row,
    grid_search,
    select_best,
    write_results_json,
    write_winning_config_yaml,
)


# --- Lightweight stand-ins (avoid importing backtest -> torch) -------------


@dataclass
class FakeComparison:
    """Minimal duck-typed stand-in for BacktestComparison."""

    combined_sharpe_pnl: float = 0.0
    baseline_sharpe_pnl: float = 0.0


@dataclass
class FakeReport:
    """Minimal duck-typed stand-in for ThresholdReport."""

    passed: bool = True
    failures: list[str] = field(default_factory=list)


def _base_config(**overrides) -> IntegrationConfig:
    defaults = dict(
        symbol="USDJPY",
        dqn_checkpoint_path="/tmp/dqn.pt",
        forecaster_checkpoint_path="/tmp/fc.pt",
        episode_start_ts=1640995200,
        episode_end_ts=1672531200,
        seed=7,
    )
    defaults.update(overrides)
    return IntegrationConfig(**defaults)


# --- DEFAULT_GRID ----------------------------------------------------------


def test_default_grid_is_full_cartesian_product() -> None:
    assert len(DEFAULT_GRID) == 5 * 4 * 2
    # Every combination is unique.
    assert len({(p.variance_threshold, p.max_risk_long_units, p.max_risk_short_units)
                for p in DEFAULT_GRID}) == 40
    assert {p.variance_threshold for p in DEFAULT_GRID} == {2.0, 3.0, 4.5, 6.0, 8.0}
    assert {p.max_risk_long_units for p in DEFAULT_GRID} == {1, 2, 3, 4}
    assert {p.max_risk_short_units for p in DEFAULT_GRID} == {1, 2}


# --- grid_search -----------------------------------------------------------


def test_grid_search_overrides_only_tunable_fields_and_holds_rest() -> None:
    seen: list[IntegrationConfig] = []

    def fake_run(cfg: IntegrationConfig) -> FakeComparison:
        seen.append(cfg)
        return FakeComparison(combined_sharpe_pnl=1.0, baseline_sharpe_pnl=0.5)

    base = _base_config()
    grid = [GridPoint(3.0, 2, 1), GridPoint(8.0, 4, 2)]
    grid_search(base, grid, run_fn=fake_run, validate_fn=lambda c: FakeReport())

    assert [c.variance_threshold for c in seen] == [3.0, 8.0]
    assert [c.max_risk_long_units for c in seen] == [2, 4]
    assert [c.max_risk_short_units for c in seen] == [1, 2]
    # Non-tunable fields are held fixed across the grid.
    for c in seen:
        assert c.symbol == "USDJPY"
        assert c.seed == 7
        assert c.episode_start_ts == base.episode_start_ts
        assert c.dqn_checkpoint_path == "/tmp/dqn.pt"


def test_grid_search_captures_metrics_and_pass_state() -> None:
    def fake_run(cfg: IntegrationConfig) -> FakeComparison:
        # Sharpe scales with the variance threshold for a deterministic order.
        return FakeComparison(
            combined_sharpe_pnl=cfg.variance_threshold,
            baseline_sharpe_pnl=1.0,
        )

    def fake_validate(cmp: FakeComparison) -> FakeReport:
        ok = cmp.combined_sharpe_pnl > cmp.baseline_sharpe_pnl
        return FakeReport(passed=ok, failures=[] if ok else ["14.1"])

    grid = [GridPoint(0.5, 1, 1), GridPoint(4.5, 2, 1)]
    results = grid_search(
        _base_config(), grid, run_fn=fake_run, validate_fn=fake_validate
    )

    assert len(results) == 2
    assert results[0].point == GridPoint(0.5, 1, 1)
    assert results[0].combined_sharpe_pnl == 0.5
    assert results[0].passed is False
    assert results[0].failures == ["14.1"]
    assert results[1].passed is True


def test_grid_search_rejects_empty_grid() -> None:
    with pytest.raises(ValueError, match="at least one GridPoint"):
        grid_search(_base_config(), [], run_fn=lambda c: FakeComparison())


def test_grid_search_failures_are_copied_not_aliased() -> None:
    shared = ["14.2"]

    def fake_validate(cmp: FakeComparison) -> FakeReport:
        return FakeReport(passed=False, failures=shared)

    results = grid_search(
        _base_config(),
        [GridPoint(4.5, 2, 1)],
        run_fn=lambda c: FakeComparison(),
        validate_fn=fake_validate,
    )
    shared.append("mutated")
    assert results[0].failures == ["14.2"]


# --- select_best -----------------------------------------------------------


def _result(vt: float, sharpe: float, passed: bool) -> GridResult:
    return GridResult(
        point=GridPoint(vt, 2, 1),
        combined_sharpe_pnl=sharpe,
        baseline_sharpe_pnl=0.0,
        passed=passed,
        failures=[] if passed else ["14.1"],
        comparison=FakeComparison(combined_sharpe_pnl=sharpe),
    )


def test_select_best_prefers_passing_even_with_lower_sharpe() -> None:
    results = [
        _result(2.0, sharpe=9.0, passed=False),  # higher sharpe but fails gates
        _result(4.5, sharpe=3.0, passed=True),
    ]
    best = select_best(results)
    assert best.point.variance_threshold == 4.5
    assert best.passed is True


def test_select_best_maximises_sharpe_among_passing() -> None:
    results = [
        _result(2.0, sharpe=3.0, passed=True),
        _result(4.5, sharpe=5.0, passed=True),
        _result(8.0, sharpe=4.0, passed=True),
    ]
    assert select_best(results).point.variance_threshold == 4.5


def test_select_best_falls_back_to_best_failing_when_none_pass() -> None:
    results = [
        _result(2.0, sharpe=1.0, passed=False),
        _result(4.5, sharpe=2.5, passed=False),
    ]
    best = select_best(results)
    assert best.point.variance_threshold == 4.5
    assert best.passed is False


def test_select_best_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        select_best([])


# --- serialization ---------------------------------------------------------


def test_grid_result_to_row_is_json_shaped() -> None:
    row = grid_result_to_row(_result(4.5, sharpe=2.0, passed=True))
    assert row == {
        "variance_threshold": 4.5,
        "max_risk_long_units": 2,
        "max_risk_short_units": 1,
        "combined_sharpe_pnl": 2.0,
        "baseline_sharpe_pnl": 0.0,
        "passed": True,
        "failures": [],
    }


def test_write_results_json_round_trips(tmp_path) -> None:
    import json

    results = [_result(2.0, 1.0, False), _result(4.5, 3.0, True)]
    path = tmp_path / "grid.json"
    write_results_json(results, str(path))
    rows = json.loads(path.read_text())
    assert len(rows) == 2
    assert rows[1]["variance_threshold"] == 4.5
    assert rows[1]["passed"] is True


def test_frozen_config_applies_winning_thresholds_and_keeps_rest() -> None:
    base = _base_config(variance_threshold=4.5, max_risk_long_units=2)
    best = _result(8.0, sharpe=5.0, passed=True)
    best = GridResult(
        point=GridPoint(8.0, 4, 2),
        combined_sharpe_pnl=5.0,
        baseline_sharpe_pnl=0.0,
        passed=True,
        failures=[],
        comparison=FakeComparison(),
    )
    data = frozen_config(base, best)
    assert data["variance_threshold"] == 8.0
    assert data["max_risk_long_units"] == 4
    assert data["max_risk_short_units"] == 2
    # Untouched fields survive.
    assert data["symbol"] == "USDJPY"
    assert data["seed"] == 7
    assert data["episode_start_ts"] == base.episode_start_ts


def test_write_winning_config_yaml_loads_into_valid_config(tmp_path) -> None:
    import yaml

    base = _base_config()
    best = GridResult(
        point=GridPoint(6.0, 3, 2),
        combined_sharpe_pnl=4.0,
        baseline_sharpe_pnl=1.0,
        passed=True,
        failures=[],
        comparison=FakeComparison(),
    )
    path = tmp_path / "tuned.yaml"
    write_winning_config_yaml(base, best, str(path))
    data = yaml.safe_load(path.read_text())
    # The emitted YAML is a complete, valid IntegrationConfig.
    cfg = IntegrationConfig(**data)
    assert cfg.variance_threshold == 6.0
    assert cfg.max_risk_long_units == 3
    assert cfg.max_risk_short_units == 2
