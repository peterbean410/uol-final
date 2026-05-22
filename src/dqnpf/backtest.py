"""Backtest entry point: combined system vs. DQN-only baseline.

The module is split into:

* **Pure helpers** that turn per-step records into a ``BacktestComparison`` and
  a ``ThresholdReport`` (Sharpe, quarterly grouping, suppression stats,
  conditional PnL, trade counts, threshold validation). These are
  deterministic, side-effect-free, and unit-tested.
* **Orchestration helpers** (``_run_episode_combined`` / ``_run_episode_baseline``
  / ``run_backtest``) that drive the live gRPC loop. They depend on the
  modelenv service and trained checkpoints; they are exercised end-to-end in
  the integration test layer rather than unit-tested here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from tradingmodel.intraday.dqnpf.config import IntegrationConfig
from tradingmodel.intraday.dqnpf.forecaster_bridge import ForecasterBridge
from tradingmodel.intraday.dqnpf.integration import IntegrationLayer
from tradingmodel.intraday.dqnpf.signal_cache import SignalCache

logger = logging.getLogger(__name__)

_HOLD = 0


@dataclass
class StepRecord:
    """Per-step trace from a backtest run."""

    timestamp_ns: int
    dqn_action: int
    final_action: int
    reason: str  # "pass"/"budget_exhausted"/"directional_conflict" or "baseline"
    mu: float
    sigma: float
    reward: float
    high_sigma: bool

    @property
    def screened(self) -> bool:
        return self.reason not in {"pass", "baseline"}


@dataclass
class BacktestComparison:
    """Combined-system vs. DQN-only baseline summary."""

    combined_return: float
    baseline_return: float
    combined_sharpe: float
    baseline_sharpe: float
    suppression_rate: float
    suppression_by_reason: dict[str, int]
    high_sigma_pnl_combined: float
    high_sigma_pnl_baseline: float
    low_sigma_pnl_combined: float
    low_sigma_pnl_baseline: float
    trades_combined: int
    trades_baseline: int
    high_sigma_trades_combined: int
    high_sigma_trades_baseline: int
    high_sigma_negative_pnl_proportion_combined: float
    high_sigma_negative_pnl_proportion_baseline: float
    quarterly_pnl_combined: dict[str, float]
    quarterly_pnl_baseline: dict[str, float]
    high_sigma_time_fraction: float


@dataclass
class ThresholdReport:
    """Result of validating a BacktestComparison against Req 14.1–14.5."""

    passed: bool
    failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def compute_sharpe(rewards: Sequence[float]) -> float:
    """Mean-over-std Sharpe ratio. Returns 0.0 for empty or zero-std series."""
    n = len(rewards)
    if n == 0:
        return 0.0
    mean = sum(rewards) / n
    if n == 1:
        return 0.0
    var = sum((r - mean) ** 2 for r in rewards) / (n - 1)
    if var <= 0:
        return 0.0
    std = var ** 0.5
    return mean / std


def quarterly_pnl(records: Sequence[StepRecord]) -> dict[str, float]:
    """Group rewards by calendar quarter (``YYYY-Qn`` keys)."""
    out: dict[str, float] = {}
    for r in records:
        dt = datetime.fromtimestamp(r.timestamp_ns / 1_000_000_000, tz=timezone.utc)
        quarter = (dt.month - 1) // 3 + 1
        key = f"{dt.year}-Q{quarter}"
        out[key] = out.get(key, 0.0) + r.reward
    return out


def suppression_stats(
    records: Sequence[StepRecord],
) -> tuple[float, dict[str, int]]:
    """Compute screened-action rate and counts by reason (combined run only)."""
    if not records:
        return 0.0, {}
    by_reason: dict[str, int] = {}
    screened = 0
    for r in records:
        if r.reason == "baseline":
            continue
        if r.reason != "pass":
            screened += 1
            by_reason[r.reason] = by_reason.get(r.reason, 0) + 1
    return screened / len(records), by_reason


def conditional_pnl(
    records: Sequence[StepRecord], *, high_sigma: bool
) -> float:
    """Sum rewards on steps matching the requested sigma regime."""
    return sum(r.reward for r in records if r.high_sigma is high_sigma)


def count_trades(records: Sequence[StepRecord]) -> int:
    """Count steps that placed a non-HOLD final action."""
    return sum(1 for r in records if r.final_action != _HOLD)


def count_trades_in_regime(
    records: Sequence[StepRecord], *, high_sigma: bool
) -> int:
    return sum(
        1
        for r in records
        if r.final_action != _HOLD and r.high_sigma is high_sigma
    )


def negative_pnl_proportion(
    records: Sequence[StepRecord], *, high_sigma: bool
) -> float:
    """Fraction of steps in the regime with reward < 0. Returns 0.0 if none."""
    in_regime = [r for r in records if r.high_sigma is high_sigma]
    if not in_regime:
        return 0.0
    return sum(1 for r in in_regime if r.reward < 0) / len(in_regime)


def compare_results(
    combined: Sequence[StepRecord], baseline: Sequence[StepRecord]
) -> BacktestComparison:
    """Aggregate two record streams into a BacktestComparison."""
    combined_rewards = [r.reward for r in combined]
    baseline_rewards = [r.reward for r in baseline]

    supp_rate, supp_by_reason = suppression_stats(combined)
    high_time_fraction = (
        sum(1 for r in combined if r.high_sigma) / len(combined)
        if combined
        else 0.0
    )

    return BacktestComparison(
        combined_return=sum(combined_rewards),
        baseline_return=sum(baseline_rewards),
        combined_sharpe=compute_sharpe(combined_rewards),
        baseline_sharpe=compute_sharpe(baseline_rewards),
        suppression_rate=supp_rate,
        suppression_by_reason=supp_by_reason,
        high_sigma_pnl_combined=conditional_pnl(combined, high_sigma=True),
        high_sigma_pnl_baseline=conditional_pnl(baseline, high_sigma=True),
        low_sigma_pnl_combined=conditional_pnl(combined, high_sigma=False),
        low_sigma_pnl_baseline=conditional_pnl(baseline, high_sigma=False),
        trades_combined=count_trades(combined),
        trades_baseline=count_trades(baseline),
        high_sigma_trades_combined=count_trades_in_regime(combined, high_sigma=True),
        high_sigma_trades_baseline=count_trades_in_regime(baseline, high_sigma=True),
        high_sigma_negative_pnl_proportion_combined=negative_pnl_proportion(
            combined, high_sigma=True
        ),
        high_sigma_negative_pnl_proportion_baseline=negative_pnl_proportion(
            baseline, high_sigma=True
        ),
        quarterly_pnl_combined=quarterly_pnl(combined),
        quarterly_pnl_baseline=quarterly_pnl(baseline),
        high_sigma_time_fraction=high_time_fraction,
    )


def validate_thresholds(
    comparison: BacktestComparison, low_sigma_tolerance: float = 0.05
) -> ThresholdReport:
    """Verify Req 14.1–14.5 against a comparison; return failure list."""
    failures: list[str] = []

    # Req 14.1: combined Sharpe > baseline Sharpe
    if not (comparison.combined_sharpe > comparison.baseline_sharpe):
        failures.append(
            f"14.1 sharpe: combined={comparison.combined_sharpe:.4f} "
            f"<= baseline={comparison.baseline_sharpe:.4f}"
        )

    # Req 14.2: fewer trades, reduction concentrated in high-sigma
    if not (comparison.trades_combined < comparison.trades_baseline):
        failures.append(
            f"14.2 trades: combined={comparison.trades_combined} "
            f">= baseline={comparison.trades_baseline}"
        )
    else:
        total_reduction = comparison.trades_baseline - comparison.trades_combined
        high_reduction = (
            comparison.high_sigma_trades_baseline
            - comparison.high_sigma_trades_combined
        )
        if total_reduction > 0 and high_reduction / total_reduction < 0.5:
            failures.append(
                f"14.2 concentration: high-sigma reduction "
                f"{high_reduction}/{total_reduction} below 50%"
            )

    # Req 14.3: combined high-sigma negative-PnL proportion < baseline
    if not (
        comparison.high_sigma_negative_pnl_proportion_combined
        < comparison.high_sigma_negative_pnl_proportion_baseline
    ):
        failures.append(
            "14.3 high-sigma neg-PnL proportion: "
            f"combined={comparison.high_sigma_negative_pnl_proportion_combined:.4f} "
            f">= baseline={comparison.high_sigma_negative_pnl_proportion_baseline:.4f}"
        )

    # Req 14.4: low-sigma PnL degradation ≤ tolerance
    base_low = comparison.low_sigma_pnl_baseline
    comb_low = comparison.low_sigma_pnl_combined
    if base_low > 0:
        floor = base_low * (1 - low_sigma_tolerance)
        if comb_low < floor:
            failures.append(
                f"14.4 low-sigma degradation: combined={comb_low:.4f} "
                f"< floor={floor:.4f} (baseline={base_low:.4f}, "
                f"tol={low_sigma_tolerance:.2f})"
            )
    else:
        # Baseline non-positive: require combined no worse by tolerance * |baseline|.
        slack = low_sigma_tolerance * abs(base_low)
        if comb_low < base_low - slack:
            failures.append(
                f"14.4 low-sigma degradation: combined={comb_low:.4f} "
                f"< baseline - slack={base_low - slack:.4f}"
            )

    # Req 14.5: no single quarter > 50% of total PnL
    total = comparison.combined_return
    quarters = comparison.quarterly_pnl_combined
    if total != 0 and quarters:
        max_share = max(abs(v) for v in quarters.values()) / abs(total)
        if max_share > 0.5:
            failures.append(
                f"14.5 quarterly concentration: max share {max_share:.2%} > 50%"
            )

    return ThresholdReport(passed=not failures, failures=failures)


# ---------------------------------------------------------------------------
# Orchestration (live infra)
# ---------------------------------------------------------------------------


def _latest_bar_ts(env_client: Any, symbol: str) -> int:
    """Return the latest completed M5 bar timestamp, or 0 if unavailable."""
    response = env_client.recent_bars(symbol)
    bars = getattr(response, "bars", None)
    if not bars or "M5" not in bars:
        return 0
    series = bars["M5"].bars
    return int(series[-1].timestamp_ns) if series else 0


def _step_timestamp_ns(config: IntegrationConfig, step_idx: int) -> int:
    return (config.episode_start_ts + step_idx * config.step_size_seconds) * 1_000_000_000


def _run_episode(
    *,
    env_client: Any,
    dqn: Any,
    preprocessor: Any,
    bridge: ForecasterBridge,
    cache: SignalCache,
    integration: IntegrationLayer | None,
    config: IntegrationConfig,
) -> list[StepRecord]:
    """Single-episode loop. Pass ``integration=None`` for the DQN-only baseline."""
    cache.invalidate()
    obs = env_client.reset(
        symbol=config.symbol,
        episode_start_ts=config.episode_start_ts,
        episode_end_ts=config.episode_end_ts,
        step_size_seconds=config.step_size_seconds,
    )
    records: list[StepRecord] = []
    step_idx = 0
    while not getattr(obs, "done", False):
        state = preprocessor.process(obs)
        dqn_result = dqn.recommend_action(state)

        latest_ts = _latest_bar_ts(env_client, config.symbol)
        mu, sigma = cache.get_or_compute(latest_ts, bridge.compute_signal)
        high_sigma = sigma > config.variance_threshold

        if integration is not None:
            screened = integration.screen(dqn_result, mu, sigma)
            final_action = screened.action
            reason = screened.reason
        else:
            final_action = dqn_result.action
            reason = "baseline"

        order_id = f"{config.symbol}-{step_idx}"
        step_response = env_client.step(final_action, order_id)
        obs = getattr(step_response, "data", step_response)

        records.append(
            StepRecord(
                timestamp_ns=_step_timestamp_ns(config, step_idx),
                dqn_action=int(dqn_result.action),
                final_action=int(final_action),
                reason=reason,
                mu=float(mu),
                sigma=float(sigma),
                reward=float(getattr(obs, "reward", 0.0)),
                high_sigma=high_sigma,
            )
        )
        step_idx += 1

    return records


def run_backtest(config: IntegrationConfig) -> BacktestComparison:
    """Orchestrate the full backtest. Requires live infra and trained checkpoints.

    Wires together ``DQNAdvisor``, ``ForecasterInference``, ``EnvironmentClient``,
    and ``IntegrationLayer``, runs ``config.num_episodes`` paired episodes
    (combined system vs. DQN-only baseline) on identical episodes, and returns
    the resulting ``BacktestComparison``.
    """
    # Imports kept local to avoid pulling heavy deps when only the pure helpers
    # are used in unit tests.
    from deepqnetwork.advisor import DQNAdvisor
    from deepqnetwork.environment_client import EnvironmentClient
    from deepqnetwork.preprocessor import StatePreprocessor
    from deepqnetwork.utils import resolve_device
    from probabilisticforecaster.config import ForecasterConfig
    from probabilisticforecaster.inference import ForecasterInference

    if config.dqn_checkpoint_path is None:
        raise ValueError("dqn_checkpoint_path is required for run_backtest")
    if config.forecaster_checkpoint_path is None:
        raise ValueError("forecaster_checkpoint_path is required for run_backtest")

    device = resolve_device(config.device)
    env_client = EnvironmentClient(config.grpc_address)
    dqn = DQNAdvisor(config.dqn_checkpoint_path, device=str(device))
    forecaster = ForecasterInference(
        config.forecaster_checkpoint_path,
        ForecasterConfig(forecast_horizon=config.forecast_horizon),
    )
    preprocessor = StatePreprocessor(device)
    bridge = ForecasterBridge(forecaster, config.symbol, env_client)
    cache = SignalCache()
    integration = IntegrationLayer(dqn, bridge, cache, config)

    combined_records: list[StepRecord] = []
    baseline_records: list[StepRecord] = []
    for episode in range(config.num_episodes):
        logger.info("backtest episode %d/%d (combined)", episode + 1, config.num_episodes)
        combined_records.extend(
            _run_episode(
                env_client=env_client,
                dqn=dqn,
                preprocessor=preprocessor,
                bridge=bridge,
                cache=cache,
                integration=integration,
                config=config,
            )
        )
        logger.info("backtest episode %d/%d (baseline)", episode + 1, config.num_episodes)
        baseline_records.extend(
            _run_episode(
                env_client=env_client,
                dqn=dqn,
                preprocessor=preprocessor,
                bridge=bridge,
                cache=cache,
                integration=None,
                config=config,
            )
        )

    comparison = compare_results(combined_records, baseline_records)
    report = validate_thresholds(comparison)

    logger.info(
        "backtest comparison: combined_return=%.4f baseline_return=%.4f "
        "combined_sharpe=%.4f baseline_sharpe=%.4f suppression_rate=%.4f "
        "trades_combined=%d trades_baseline=%d high_sigma_time=%.2f%%",
        comparison.combined_return,
        comparison.baseline_return,
        comparison.combined_sharpe,
        comparison.baseline_sharpe,
        comparison.suppression_rate,
        comparison.trades_combined,
        comparison.trades_baseline,
        100 * comparison.high_sigma_time_fraction,
    )
    logger.info(
        "suppression_by_reason=%s quarterly_combined=%s",
        comparison.suppression_by_reason,
        comparison.quarterly_pnl_combined,
    )
    if report.passed:
        logger.info("threshold validation: PASSED")
    else:
        logger.warning(
            "threshold validation: FAILED, config invalid (Req 14.6). "
            "Failures: %s",
            report.failures,
        )

    return comparison
