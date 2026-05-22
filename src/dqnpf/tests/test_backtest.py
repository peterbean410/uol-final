"""Unit tests for backtest pure helpers and threshold validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tradingmodel.intraday.dqnpf.backtest import (
    BacktestComparison,
    StepRecord,
    ThresholdReport,
    compare_results,
    compute_sharpe,
    conditional_pnl,
    count_trades,
    count_trades_in_regime,
    negative_pnl_proportion,
    quarterly_pnl,
    suppression_stats,
    validate_thresholds,
)


def _ts(year: int, month: int = 1, day: int = 1) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1_000_000_000)


def _rec(
    *,
    reward: float = 0.0,
    final_action: int = 1,
    dqn_action: int = 1,
    reason: str = "pass",
    sigma: float = 5.0,
    high_sigma: bool = True,
    mu: float = 0.0,
    timestamp_ns: int | None = None,
) -> StepRecord:
    return StepRecord(
        timestamp_ns=timestamp_ns if timestamp_ns is not None else _ts(2024),
        dqn_action=dqn_action,
        final_action=final_action,
        reason=reason,
        mu=mu,
        sigma=sigma,
        reward=reward,
        high_sigma=high_sigma,
    )


# ---------------------------------------------------------------------------
# compute_sharpe
# ---------------------------------------------------------------------------


def test_compute_sharpe_empty_returns_zero() -> None:
    assert compute_sharpe([]) == 0.0


def test_compute_sharpe_single_value_returns_zero() -> None:
    assert compute_sharpe([0.42]) == 0.0


def test_compute_sharpe_zero_std_returns_zero() -> None:
    assert compute_sharpe([1.0, 1.0, 1.0]) == 0.0


def test_compute_sharpe_basic() -> None:
    # mean = 0.2, sample std with n=5 -> ~0.158
    rewards = [0.1, 0.2, 0.3, 0.2, 0.2]
    out = compute_sharpe(rewards)
    assert out > 0
    # Quick sanity check that result equals mean/std
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
    expected = mean / var**0.5
    assert out == pytest.approx(expected)


def test_compute_sharpe_negative_mean() -> None:
    assert compute_sharpe([-0.1, -0.2, -0.3]) < 0


# ---------------------------------------------------------------------------
# quarterly_pnl
# ---------------------------------------------------------------------------


def test_quarterly_pnl_groups_by_year_quarter() -> None:
    records = [
        _rec(reward=1.0, timestamp_ns=_ts(2024, 1, 15)),  # Q1
        _rec(reward=2.0, timestamp_ns=_ts(2024, 3, 31)),  # Q1
        _rec(reward=3.0, timestamp_ns=_ts(2024, 4, 1)),  # Q2
        _rec(reward=4.0, timestamp_ns=_ts(2024, 12, 31)),  # Q4
        _rec(reward=5.0, timestamp_ns=_ts(2025, 1, 1)),  # 2025 Q1
    ]
    out = quarterly_pnl(records)
    assert out == {"2024-Q1": 3.0, "2024-Q2": 3.0, "2024-Q4": 4.0, "2025-Q1": 5.0}


def test_quarterly_pnl_empty() -> None:
    assert quarterly_pnl([]) == {}


# ---------------------------------------------------------------------------
# suppression_stats
# ---------------------------------------------------------------------------


def test_suppression_stats_empty() -> None:
    rate, by_reason = suppression_stats([])
    assert rate == 0.0
    assert by_reason == {}


def test_suppression_stats_mix_of_reasons() -> None:
    records = [
        _rec(reason="pass"),
        _rec(reason="pass"),
        _rec(reason="budget_exhausted", final_action=0),
        _rec(reason="directional_conflict", final_action=0),
        _rec(reason="budget_exhausted", final_action=0),
    ]
    rate, by_reason = suppression_stats(records)
    assert rate == 3 / 5
    assert by_reason == {"budget_exhausted": 2, "directional_conflict": 1}


def test_suppression_stats_ignores_baseline_reason() -> None:
    records = [
        _rec(reason="baseline"),
        _rec(reason="baseline"),
    ]
    rate, by_reason = suppression_stats(records)
    assert rate == 0.0
    assert by_reason == {}


# ---------------------------------------------------------------------------
# conditional_pnl / counts / negative proportion
# ---------------------------------------------------------------------------


def test_conditional_pnl_separates_regimes() -> None:
    records = [
        _rec(reward=1.0, high_sigma=True),
        _rec(reward=2.0, high_sigma=False),
        _rec(reward=-0.5, high_sigma=True),
    ]
    assert conditional_pnl(records, high_sigma=True) == 0.5
    assert conditional_pnl(records, high_sigma=False) == 2.0


def test_count_trades_skips_hold() -> None:
    records = [
        _rec(final_action=0),
        _rec(final_action=1),
        _rec(final_action=2),
        _rec(final_action=0),
    ]
    assert count_trades(records) == 2


def test_count_trades_in_regime() -> None:
    records = [
        _rec(final_action=1, high_sigma=True),
        _rec(final_action=0, high_sigma=True),  # HOLD: skipped
        _rec(final_action=2, high_sigma=False),
        _rec(final_action=3, high_sigma=True),
    ]
    assert count_trades_in_regime(records, high_sigma=True) == 2
    assert count_trades_in_regime(records, high_sigma=False) == 1


def test_negative_pnl_proportion_empty_regime_returns_zero() -> None:
    records = [_rec(high_sigma=False, reward=1.0)]
    assert negative_pnl_proportion(records, high_sigma=True) == 0.0


def test_negative_pnl_proportion_basic() -> None:
    records = [
        _rec(reward=1.0, high_sigma=True),
        _rec(reward=-0.5, high_sigma=True),
        _rec(reward=-1.0, high_sigma=True),
        _rec(reward=0.5, high_sigma=False),
    ]
    assert negative_pnl_proportion(records, high_sigma=True) == 2 / 3
    assert negative_pnl_proportion(records, high_sigma=False) == 0.0


# ---------------------------------------------------------------------------
# compare_results
# ---------------------------------------------------------------------------


def test_compare_results_basic_aggregation() -> None:
    combined = [
        _rec(reward=0.5, reason="pass", final_action=1, high_sigma=True),
        _rec(reward=-0.1, reason="budget_exhausted", final_action=0, high_sigma=True),
        _rec(reward=0.2, reason="pass", final_action=2, high_sigma=False),
    ]
    baseline = [
        _rec(reward=0.4, reason="baseline", final_action=1, high_sigma=True),
        _rec(reward=-0.3, reason="baseline", final_action=1, high_sigma=True),
        _rec(reward=0.2, reason="baseline", final_action=2, high_sigma=False),
    ]
    cmp = compare_results(combined, baseline)
    assert isinstance(cmp, BacktestComparison)
    assert cmp.combined_return == pytest.approx(0.6)
    assert cmp.baseline_return == pytest.approx(0.3)
    assert cmp.trades_combined == 2
    assert cmp.trades_baseline == 3
    assert cmp.high_sigma_trades_combined == 1
    assert cmp.high_sigma_trades_baseline == 2
    assert cmp.suppression_rate == pytest.approx(1 / 3)
    assert cmp.suppression_by_reason == {"budget_exhausted": 1}
    assert cmp.high_sigma_pnl_combined == pytest.approx(0.4)
    assert cmp.low_sigma_pnl_combined == pytest.approx(0.2)
    assert cmp.high_sigma_negative_pnl_proportion_combined == pytest.approx(0.5)
    assert cmp.high_sigma_negative_pnl_proportion_baseline == pytest.approx(0.5)
    assert cmp.high_sigma_time_fraction == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# validate_thresholds
# ---------------------------------------------------------------------------


def _passing_comparison() -> BacktestComparison:
    """Build a BacktestComparison that passes every Req 14 threshold."""
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
        quarterly_pnl_combined={"2024-Q1": 0.3, "2024-Q2": 0.3, "2024-Q3": 0.4},
        quarterly_pnl_baseline={"2024-Q1": 0.2, "2024-Q2": 0.2, "2024-Q3": 0.1},
        high_sigma_time_fraction=0.3,
    )


def test_threshold_report_passes_on_good_comparison() -> None:
    report = validate_thresholds(_passing_comparison())
    assert isinstance(report, ThresholdReport)
    assert report.passed is True
    assert report.failures == []


def test_threshold_141_fails_when_sharpe_not_better() -> None:
    cmp = _passing_comparison()
    cmp.combined_sharpe = 0.3
    report = validate_thresholds(cmp)
    assert report.passed is False
    assert any("14.1" in f for f in report.failures)


def test_threshold_142_fails_when_trades_not_reduced() -> None:
    cmp = _passing_comparison()
    cmp.trades_combined = 12  # > baseline
    report = validate_thresholds(cmp)
    assert any("14.2 trades" in f for f in report.failures)


def test_threshold_142_fails_when_reduction_not_in_high_sigma() -> None:
    # Reduce trades but in low-sigma only.
    cmp = _passing_comparison()
    cmp.trades_combined = 5
    cmp.trades_baseline = 10
    cmp.high_sigma_trades_combined = 5
    cmp.high_sigma_trades_baseline = 5  # no high-sigma reduction
    report = validate_thresholds(cmp)
    assert any("14.2 concentration" in f for f in report.failures)


def test_threshold_143_fails_when_negative_pnl_proportion_worse() -> None:
    cmp = _passing_comparison()
    cmp.high_sigma_negative_pnl_proportion_combined = 0.6  # worse than baseline 0.5
    report = validate_thresholds(cmp)
    assert any("14.3" in f for f in report.failures)


def test_threshold_144_fails_on_low_sigma_degradation() -> None:
    cmp = _passing_comparison()
    cmp.low_sigma_pnl_baseline = 1.0
    cmp.low_sigma_pnl_combined = 0.5  # 50% drop, far over 5% tolerance
    report = validate_thresholds(cmp)
    assert any("14.4" in f for f in report.failures)


def test_threshold_144_passes_within_tolerance() -> None:
    cmp = _passing_comparison()
    cmp.low_sigma_pnl_baseline = 1.0
    cmp.low_sigma_pnl_combined = 0.96  # 4% drop, within 5% tolerance
    report = validate_thresholds(cmp)
    assert all("14.4" not in f for f in report.failures)


def test_threshold_145_fails_on_quarterly_concentration() -> None:
    cmp = _passing_comparison()
    cmp.combined_return = 1.0
    cmp.quarterly_pnl_combined = {"2024-Q1": 0.9, "2024-Q2": 0.1}
    report = validate_thresholds(cmp)
    assert any("14.5" in f for f in report.failures)


def test_threshold_145_passes_when_distributed() -> None:
    cmp = _passing_comparison()
    cmp.combined_return = 1.0
    cmp.quarterly_pnl_combined = {"2024-Q1": 0.3, "2024-Q2": 0.4, "2024-Q3": 0.3}
    report = validate_thresholds(cmp)
    assert all("14.5" not in f for f in report.failures)


def test_threshold_146_failures_listed_when_multiple_break() -> None:
    cmp = _passing_comparison()
    cmp.combined_sharpe = 0.0
    cmp.trades_combined = 100
    report = validate_thresholds(cmp)
    assert report.passed is False
    assert len(report.failures) >= 2


# ---------------------------------------------------------------------------
# StepRecord.screened
# ---------------------------------------------------------------------------


def test_step_record_screened_property() -> None:
    assert _rec(reason="pass").screened is False
    assert _rec(reason="baseline").screened is False
    assert _rec(reason="budget_exhausted").screened is True
    assert _rec(reason="directional_conflict").screened is True
