"""Threshold tuning entry point: grid-search the integration thresholds.

Implements Task 9.4, tune ``variance_threshold`` / ``max_risk_long_units`` /
``max_risk_short_units`` on a validation window (the last 12 months of the
training range, 2022) and select the config with the best money-PnL Sharpe,
preferring configs that clear every Req 14 gate.

The module mirrors ``backtest.py``'s split:

* **Pure / injectable core** (``grid_search``, ``select_best``, and the
  serialization helpers) is deterministic and unit-tested. ``grid_search``
  takes the backtest runner and the threshold validator as injectable
  callables, defaulting to the real ones, so the grid logic can be exercised
  without ``torch`` / modelenv.
* **Orchestration** (``main``) loads a base config, drives the live runs via
  ``run_backtest``, and writes the result artifacts.

To keep the module import torch-free (``backtest`` transitively imports
``torch`` via ``forecaster_bridge``), the real ``run_backtest`` /
``validate_thresholds`` are imported lazily inside ``grid_search`` only when
no override is supplied.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Sequence

import yaml

from tradingmodel.intraday.dqnpf.config import IntegrationConfig, load_config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tradingmodel.intraday.dqnpf.backtest import (
        BacktestComparison,
        ThresholdReport,
    )

logger = logging.getLogger(__name__)

# Validation window for threshold tuning: the last 12 months of the training
# range (Req 12.3). Unix seconds, UTC. 2022-01-01 .. 2023-01-01.
VALIDATION_START_TS = 1640995200  # 2022-01-01 00:00:00 UTC
VALIDATION_END_TS = 1672531200  # 2023-01-01 00:00:00 UTC


@dataclass(frozen=True)
class GridPoint:
    """One combination of tunable thresholds."""

    variance_threshold: float
    max_risk_long_units: int
    max_risk_short_units: int


@dataclass
class GridResult:
    """Backtest outcome for a single ``GridPoint``."""

    point: GridPoint
    combined_sharpe_pnl: float
    baseline_sharpe_pnl: float
    passed: bool
    failures: list[str]
    comparison: "BacktestComparison"


# Grid from Task 9.4: 5 x 4 x 2 = 40 configs.
DEFAULT_GRID: list[GridPoint] = [
    GridPoint(v, long_cap, short_cap)
    for v in (2.0, 3.0, 4.5, 6.0, 8.0)
    for long_cap in (1, 2, 3, 4)
    for short_cap in (1, 2)
]

# Injectable callable signatures.
RunBacktest = Callable[[IntegrationConfig], "BacktestComparison"]
ValidateThresholds = Callable[["BacktestComparison"], "ThresholdReport"]


def grid_search(
    base_config: IntegrationConfig,
    grid: Sequence[GridPoint] = DEFAULT_GRID,
    *,
    run_fn: RunBacktest | None = None,
    validate_fn: ValidateThresholds | None = None,
) -> list[GridResult]:
    """Run ``run_fn`` over ``grid``, validating each against the Req 14 gates.

    Each grid point overrides only the three tunable thresholds on
    ``base_config``; every other field (symbol, checkpoints, episode window,
    seed) is held fixed so the runs are comparable. ``run_fn`` and
    ``validate_fn`` default to the real ``run_backtest`` /
    ``validate_thresholds`` but can be injected for testing.
    """
    if not grid:
        raise ValueError("grid must contain at least one GridPoint")

    if run_fn is None or validate_fn is None:
        # Lazy import: pulls torch via forecaster_bridge, so only at runtime.
        from tradingmodel.intraday.dqnpf.backtest import (
            run_backtest,
            validate_thresholds,
        )

        run_fn = run_fn or run_backtest
        validate_fn = validate_fn or validate_thresholds

    results: list[GridResult] = []
    for i, point in enumerate(grid, start=1):
        logger.info(
            "grid point %d/%d: variance_threshold=%.2f max_risk_long=%d "
            "max_risk_short=%d",
            i,
            len(grid),
            point.variance_threshold,
            point.max_risk_long_units,
            point.max_risk_short_units,
        )
        config = replace(
            base_config,
            variance_threshold=point.variance_threshold,
            max_risk_long_units=point.max_risk_long_units,
            max_risk_short_units=point.max_risk_short_units,
        )
        comparison = run_fn(config)
        report = validate_fn(comparison)
        results.append(
            GridResult(
                point=point,
                combined_sharpe_pnl=comparison.combined_sharpe_pnl,
                baseline_sharpe_pnl=comparison.baseline_sharpe_pnl,
                passed=report.passed,
                failures=list(report.failures),
                comparison=comparison,
            )
        )
        logger.info(
            "  -> combined_sharpe_pnl=%.4f baseline_sharpe_pnl=%.4f passed=%s",
            comparison.combined_sharpe_pnl,
            comparison.baseline_sharpe_pnl,
            report.passed,
        )
    return results


def select_best(results: Sequence[GridResult]) -> GridResult:
    """Pick the best grid result.

    Prefers configs that clear every Req 14 gate; among those (or among all
    results if none pass), maximises money-PnL Sharpe. Ties break toward the
    earliest grid point for determinism.
    """
    if not results:
        raise ValueError("results must be non-empty")
    passing = [r for r in results if r.passed]
    pool = passing or list(results)
    if not passing:
        logger.warning(
            "no grid config cleared all Req 14 gates; selecting best by "
            "money-PnL Sharpe among %d failing configs",
            len(results),
        )
    # max() keeps the first element on ties, preserving grid order.
    return max(pool, key=lambda r: r.combined_sharpe_pnl)


def grid_result_to_row(result: GridResult) -> dict[str, Any]:
    """Flatten a ``GridResult`` to a JSON-serialisable summary row."""
    return {
        "variance_threshold": result.point.variance_threshold,
        "max_risk_long_units": result.point.max_risk_long_units,
        "max_risk_short_units": result.point.max_risk_short_units,
        "combined_sharpe_pnl": result.combined_sharpe_pnl,
        "baseline_sharpe_pnl": result.baseline_sharpe_pnl,
        "passed": result.passed,
        "failures": result.failures,
    }


def write_results_json(results: Sequence[GridResult], path: str) -> None:
    """Write the full grid (one row per config) to ``path`` as JSON."""
    rows = [grid_result_to_row(r) for r in results]
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    logger.info("wrote %d grid rows to %s", len(rows), path)


def frozen_config(base_config: IntegrationConfig, best: GridResult) -> dict[str, Any]:
    """Materialise the winning config as a plain dict for YAML emission.

    Starts from ``base_config`` and applies the winning thresholds, so the
    output is a complete, frozen ``IntegrationConfig`` ready to feed the 9.5
    out-of-sample evaluation.
    """
    data = asdict(base_config)
    data["variance_threshold"] = best.point.variance_threshold
    data["max_risk_long_units"] = best.point.max_risk_long_units
    data["max_risk_short_units"] = best.point.max_risk_short_units
    return data


def write_winning_config_yaml(
    base_config: IntegrationConfig, best: GridResult, path: str
) -> None:
    """Write the winning thresholds as a frozen config YAML to ``path``."""
    data = frozen_config(base_config, best)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=True)
    logger.info(
        "wrote winning config (variance_threshold=%.2f max_risk_long=%d "
        "max_risk_short=%d, passed=%s) to %s",
        best.point.variance_threshold,
        best.point.max_risk_long_units,
        best.point.max_risk_short_units,
        best.passed,
        path,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: tune thresholds on the validation window and emit artifacts.

    Loads a base ``IntegrationConfig`` (checkpoints, symbol, grpc address) via
    the shared ``load_config``, pins the episode window to the 2022 validation
    range unless the caller already set one, runs the grid, and writes both the
    full results JSON and the winning frozen config YAML.
    """
    logging.basicConfig(level=logging.INFO)
    base_config = load_config(argv)

    # Default the episode window to the 2022 validation range (Req 12.3) unless
    # the caller pinned an explicit window via config/CLI.
    if base_config.episode_start_ts == 0 and base_config.episode_end_ts == 0:
        base_config = replace(
            base_config,
            episode_start_ts=VALIDATION_START_TS,
            episode_end_ts=VALIDATION_END_TS,
        )
        logger.info(
            "no episode window set; defaulting to 2022 validation window "
            "[%d, %d)",
            VALIDATION_START_TS,
            VALIDATION_END_TS,
        )

    results = grid_search(base_config)
    best = select_best(results)

    write_results_json(results, "tune_results.json")
    write_winning_config_yaml(base_config, best, "tuned_config.yaml")

    logger.info(
        "best config: variance_threshold=%.2f max_risk_long=%d "
        "max_risk_short=%d combined_sharpe_pnl=%.4f passed=%s",
        best.point.variance_threshold,
        best.point.max_risk_long_units,
        best.point.max_risk_short_units,
        best.combined_sharpe_pnl,
        best.passed,
    )
    if not best.passed:
        logger.warning(
            "best config did NOT clear all Req 14 gates; failures: %s",
            best.failures,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
