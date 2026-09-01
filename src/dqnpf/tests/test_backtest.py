"""Unit tests for backtest pure helpers and threshold validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dqnpf.config import IntegrationConfig
from dqnpf.backtest import (
    BacktestComparison,
    StepRecord,
    ThresholdReport,
    _distribution,
    _resolve_episode_windows,
    _resolve_session_hours,
    compare_results,
    compute_sharpe,
    conditional_pnl,
    count_trades,
    count_trades_in_regime,
    forecaster_position,
    negative_pnl_proportion,
    negative_raw_pnl_proportion,
    quarterly_pnl,
    suppression_stats,
    validate_thresholds,
)

_HOLD, _BUY_1, _BUY_2, _SELL_1, _SELL_2 = 0, 1, 2, 3, 4


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
    max_total_margin: float = 0.0,
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
        max_total_margin=max_total_margin,
    )


def test_compute_sharpe_empty_returns_zero() -> None:
    assert compute_sharpe([]) == 0.0


def test_compute_sharpe_single_value_returns_zero() -> None:
    assert compute_sharpe([0.42]) == 0.0


def test_compute_sharpe_zero_std_returns_zero() -> None:
    assert compute_sharpe([1.0, 1.0, 1.0]) == 0.0


def test_compute_sharpe_basic() -> None:
    rewards = [0.1, 0.2, 0.3, 0.2, 0.2]
    out = compute_sharpe(rewards)
    assert out > 0
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
    expected = mean / var**0.5
    assert out == pytest.approx(expected)


def test_compute_sharpe_negative_mean() -> None:
    assert compute_sharpe([-0.1, -0.2, -0.3]) < 0


def test_quarterly_pnl_groups_by_year_quarter() -> None:
    records = [
        _rec(reward=1.0, timestamp_ns=_ts(2024, 1, 15)),
        _rec(reward=2.0, timestamp_ns=_ts(2024, 3, 31)),
        _rec(reward=3.0, timestamp_ns=_ts(2024, 4, 1)),
        _rec(reward=4.0, timestamp_ns=_ts(2024, 12, 31)),
        _rec(reward=5.0, timestamp_ns=_ts(2025, 1, 1)),
    ]
    out = quarterly_pnl(records)
    assert out == {"2024-Q1": 3.0, "2024-Q2": 3.0, "2024-Q4": 4.0, "2025-Q1": 5.0}


def test_quarterly_pnl_empty() -> None:
    assert quarterly_pnl([]) == {}


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
        _rec(final_action=0, high_sigma=True),
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
        combined_sharpe_pnl=0.8,
        baseline_sharpe_pnl=0.4,
        high_sigma_negative_raw_pnl_proportion_combined=0.2,
        high_sigma_negative_raw_pnl_proportion_baseline=0.5,
        combined_raw_pnl=1.0,
        quarterly_raw_pnl_combined={"2024-Q1": 0.3, "2024-Q2": 0.3, "2024-Q3": 0.4},
    )


def test_threshold_report_passes_on_good_comparison() -> None:
    report = validate_thresholds(_passing_comparison())
    assert isinstance(report, ThresholdReport)
    assert report.passed is True
    assert report.failures == []


def test_threshold_141_fails_when_sharpe_not_better() -> None:
    cmp = _passing_comparison()
    cmp.combined_sharpe_pnl = 0.3
    report = validate_thresholds(cmp)
    assert report.passed is False
    assert any("14.1" in f for f in report.failures)


def test_threshold_142_fails_when_trades_not_reduced() -> None:
    cmp = _passing_comparison()
    cmp.trades_combined = 12
    report = validate_thresholds(cmp)
    assert any("14.2 trades" in f for f in report.failures)


def test_threshold_142_fails_when_reduction_not_in_high_sigma() -> None:
    cmp = _passing_comparison()
    cmp.trades_combined = 5
    cmp.trades_baseline = 10
    cmp.high_sigma_trades_combined = 5
    cmp.high_sigma_trades_baseline = 5
    report = validate_thresholds(cmp)
    assert any("14.2 concentration" in f for f in report.failures)


def test_threshold_143_fails_when_negative_pnl_proportion_worse() -> None:
    cmp = _passing_comparison()
    cmp.high_sigma_negative_raw_pnl_proportion_combined = 0.6
    report = validate_thresholds(cmp)
    assert any("14.3" in f for f in report.failures)


def test_threshold_144_fails_on_low_sigma_degradation() -> None:
    cmp = _passing_comparison()
    cmp.low_sigma_pnl_baseline = 1.0
    cmp.low_sigma_pnl_combined = 0.5
    report = validate_thresholds(cmp)
    assert any("14.4" in f for f in report.failures)


def test_threshold_144_passes_within_tolerance() -> None:
    cmp = _passing_comparison()
    cmp.low_sigma_pnl_baseline = 1.0
    cmp.low_sigma_pnl_combined = 0.96
    report = validate_thresholds(cmp)
    assert all("14.4" not in f for f in report.failures)


def test_threshold_146_failures_listed_when_multiple_break() -> None:
    cmp = _passing_comparison()
    cmp.combined_sharpe_pnl = 0.0
    cmp.trades_combined = 100
    report = validate_thresholds(cmp)
    assert report.passed is False
    assert len(report.failures) >= 2


def test_step_record_screened_property() -> None:
    assert _rec(reason="pass").screened is False
    assert _rec(reason="baseline").screened is False
    assert _rec(reason="budget_exhausted").screened is True
    assert _rec(reason="directional_conflict").screened is True


def test_distribution_empty_returns_zeros() -> None:
    assert _distribution([]) == {"min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}


def test_distribution_single_value_all_equal() -> None:
    assert _distribution([3.5]) == {"min": 3.5, "median": 3.5, "p95": 3.5, "max": 3.5}


def test_distribution_is_order_independent() -> None:
    ascending = _distribution([1.0, 2.0, 3.0, 4.0, 5.0])
    shuffled = _distribution([3.0, 1.0, 5.0, 2.0, 4.0])
    assert ascending == shuffled


def test_distribution_min_max_median_p95() -> None:
    d = _distribution([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert d["min"] == 0.0
    assert d["max"] == 9.0
    assert d["median"] == 4.0
    assert d["p95"] == 9.0


def test_distribution_handles_negative_and_unsorted() -> None:
    d = _distribution([-2.0, 5.0, -10.0, 0.0])
    assert d["min"] == -10.0
    assert d["max"] == 5.0


def test_compare_results_reports_signal_distributions() -> None:
    combined = [
        _rec(mu=0.5, sigma=1.0, high_sigma=False),
        _rec(mu=2.0, sigma=3.0, high_sigma=False),
        _rec(mu=-3.5, sigma=4.0, high_sigma=False),
        _rec(mu=0.2, sigma=2.0, high_sigma=False),
        _rec(mu=1.5, sigma=6.0, high_sigma=True),
    ]
    baseline = [_rec(reason="baseline", mu=0.0, sigma=1.0, high_sigma=False)]

    cmp = compare_results(combined, baseline, directional_tolerance=1.0)

    assert cmp.sigma_distribution_combined["min"] == 1.0
    assert cmp.sigma_distribution_combined["max"] == 6.0
    assert cmp.abs_mu_distribution_combined["max"] == 3.5
    assert cmp.abs_mu_above_tolerance_fraction == pytest.approx(0.6)


def test_compare_results_empty_signal_distributions_are_zeroed() -> None:
    cmp = compare_results([], [], directional_tolerance=1.0)
    assert cmp.sigma_distribution_combined == {
        "min": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "max": 0.0,
    }
    assert cmp.abs_mu_above_tolerance_fraction == 0.0


def test_negative_raw_pnl_proportion_uses_money_not_reward() -> None:
    recs = [
        _rec(reward=1.0, high_sigma=True),
        _rec(reward=1.0, high_sigma=True),
        _rec(reward=1.0, high_sigma=True),
    ]
    recs[0].raw_pnl_delta = -2.0
    recs[1].raw_pnl_delta = -1.0
    recs[2].raw_pnl_delta = 3.0
    assert negative_raw_pnl_proportion(recs, high_sigma=True) == pytest.approx(2 / 3)
    assert negative_pnl_proportion(recs, high_sigma=True) == 0.0


def test_compare_results_money_fields_from_raw_pnl() -> None:
    combined = [
        _rec(reward=-1.0, high_sigma=True),
        _rec(reward=-1.0, high_sigma=True),
    ]
    combined[0].raw_pnl_delta = 2.0
    combined[1].raw_pnl_delta = -1.0
    baseline = [_rec(reward=5.0, reason="baseline", high_sigma=True)]
    baseline[0].raw_pnl_delta = -3.0

    cmp = compare_results(combined, baseline)

    assert cmp.high_sigma_negative_raw_pnl_proportion_combined == pytest.approx(0.5)
    assert cmp.high_sigma_negative_raw_pnl_proportion_baseline == pytest.approx(1.0)
    assert cmp.combined_sharpe_pnl == pytest.approx(
        compute_sharpe([2.0, -1.0])
    )


def test_compare_results_max_total_margin() -> None:
    """Peak total open volume is the max of per-step max_total_margin values."""
    combined = [
        _rec(max_total_margin=0.0),
        _rec(max_total_margin=1.0),
        _rec(max_total_margin=3.0),
        _rec(max_total_margin=2.0),
    ]
    baseline = [
        _rec(reason="baseline", max_total_margin=0.0),
        _rec(reason="baseline", max_total_margin=5.0),
    ]
    cmp = compare_results(combined, baseline)
    assert cmp.max_total_margin_combined == 3.0
    assert cmp.max_total_margin_baseline == 5.0


def test_forecaster_position_direction_is_sign_of_mu() -> None:
    assert forecaster_position(2.0, 2.0, risk_aversion=0.2) > 0
    assert forecaster_position(-2.0, 2.0, risk_aversion=0.2) < 0


def test_forecaster_position_scales_with_sigma() -> None:
    near = forecaster_position(1.0, 6.0, risk_aversion=0.2)
    far = forecaster_position(1.0, 20.0, risk_aversion=0.2)
    assert 0 < far < near <= 1.0


def test_forecaster_position_truncates_to_unit_interval() -> None:
    assert forecaster_position(4.0, 1.0, risk_aversion=0.1) == 1.0
    assert forecaster_position(-4.0, 1.0, risk_aversion=0.1) == -1.0


def test_forecaster_position_matches_mean_variance_formula() -> None:
    assert forecaster_position(1.0, 5.0, risk_aversion=0.2) == pytest.approx(
        1.0 / (25.0 * 0.2)
    )


def test_forecaster_position_sigma_nonpositive_full_position() -> None:
    assert forecaster_position(1.0, 0.0, risk_aversion=0.1) == 1.0
    assert forecaster_position(-1.0, 0.0, risk_aversion=0.1) == -1.0


_TRAINING_WINDOW = {
    "date_start": "",
    "date_end": "",
    "hour_of_day_start": 15,
    "hour_of_day_end": 39,
}


def test_resolve_episode_windows_legacy_fixed_window() -> None:
    """Dates unset -> num_episodes repeats of the explicit fixed window."""
    config = IntegrationConfig(
        episode_start_ts=1000, episode_end_ts=2000, num_episodes=3
    )
    assert _resolve_episode_windows(config, _TRAINING_WINDOW) == [
        (1000, 2000),
        (1000, 2000),
        (1000, 2000),
    ]


def test_resolve_episode_windows_date_range_inherits_trained_hours() -> None:
    """Date-range mode with no hour override inherits the DQN's trained session."""
    config = IntegrationConfig(date_start="2012-01-01", date_end="2012-01-03")
    windows = _resolve_episode_windows(config, _TRAINING_WINDOW)
    assert len(windows) == 3
    start = datetime(2012, 1, 1, 15, tzinfo=timezone.utc)
    end = datetime(2012, 1, 2, 15, tzinfo=timezone.utc)
    assert windows[0] == (int(start.timestamp()), int(end.timestamp()))


def test_resolve_episode_windows_explicit_hours_override_training_window() -> None:
    """Explicit hour_of_day_* on the config override the trained session."""
    config = IntegrationConfig(
        date_start="2012-01-01",
        date_end="2012-01-01",
        hour_of_day_start=0,
        hour_of_day_end=23,
    )
    windows = _resolve_episode_windows(config, _TRAINING_WINDOW)
    start = datetime(2012, 1, 1, 0, tzinfo=timezone.utc)
    end = datetime(2012, 1, 1, 23, tzinfo=timezone.utc)
    assert windows == [(int(start.timestamp()), int(end.timestamp()))]


def test_resolve_session_hours_legacy_returns_none() -> None:
    """Legacy fixed-window mode has no session window (liquidation stays off)."""
    config = IntegrationConfig(episode_start_ts=1000, episode_end_ts=2000)
    assert _resolve_session_hours(config, _TRAINING_WINDOW) is None


def test_resolve_session_hours_inherits_training_window() -> None:
    """Date-range mode with no override inherits the DQN's trained session."""
    config = IntegrationConfig(date_start="2012-01-01", date_end="2012-01-03")
    assert _resolve_session_hours(config, _TRAINING_WINDOW) == (15, 39)


def test_resolve_session_hours_config_overrides_training_window() -> None:
    """Explicit config hours win over the trained session."""
    config = IntegrationConfig(
        date_start="2012-01-01",
        date_end="2012-01-01",
        hour_of_day_start=0,
        hour_of_day_end=23,
    )
    assert _resolve_session_hours(config, _TRAINING_WINDOW) == (0, 23)
