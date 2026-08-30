"""Drive the three backtest arms through the REAL integration layer.

This is the heart of the prototype: it produces ``StepRecord`` streams for the
combined (DQN -> screen), DQN-only baseline, and forecaster-only arms by
replaying a price slice, and it does so using the production code unchanged,

* ``dqnpf.integration.IntegrationLayer`` (screen + gate),
* ``dqnpf.backtest`` pure helpers (compare_results,
  validate_thresholds, forecaster_position, StepRecord),
* ``dqnpf.action_mapper`` / ``config``.

Only the *replay loop* is new; it stands in for ``backtest._run_episode``'s gRPC
loop, mirroring its arm logic (one session per UTC day, per-day budget reset,
``begin_session()`` at each boundary, next-bar mark-to-market PnL).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from dqnpf.action_mapper import ACTION_NAMES
from dqnpf.backtest import (
    BacktestComparison,
    StepRecord,
    ThresholdReport,
    compare_results,
    forecaster_position,
    validate_thresholds,
)
from dqnpf.config import IntegrationConfig
from dqnpf.integration import IntegrationLayer

logger = logging.getLogger(__name__)

_HOLD, _BUY_1, _SELL_1 = 0, 1, 3
_NANOS_PER_DAY = 86_400_000_000_000


@dataclass
class _ActionResult:
    """Minimal stand-in for deepqnetwork.advisor.ActionResult (screen reads both)."""

    action: int
    action_name: str = ""


@dataclass
class ArmResults:
    regime: str
    gate_enabled: bool
    combined: list[StepRecord]
    baseline: list[StepRecord]
    gate_active_series: list[tuple[int, bool]]  # (timestamp_ns, gate_active) per combined step
    comparison: BacktestComparison
    report: ThresholdReport
    meta: dict


def _run_policy_arm(env, policy, mu_bps, sigma_bps, ts_ns, close, config, integration):
    """One pass over the slice for a policy-driven arm (combined or baseline)."""
    env.reset()
    records: list[StepRecord] = []
    gate_series: list[tuple[int, bool]] = []
    last_day: int | None = None
    if integration is not None:
        integration.begin_session()
    done = False
    while not done:
        t = env.t
        obs = env.observe()
        day = int(ts_ns[t] // _NANOS_PER_DAY)
        if integration is not None and last_day is not None and day != last_day:
            integration.begin_session()
        last_day = day

        dqn_action = int(policy.act(obs))
        mu = float(mu_bps[t])
        sigma = float(sigma_bps[t])
        if integration is not None:
            screened = integration.screen(
                _ActionResult(dqn_action, ACTION_NAMES[dqn_action]),
                mu,
                sigma,
                timestamp_ns=int(ts_ns[t]),
                price=float(close[t]),
            )
            exec_action = int(screened.action)
            reason = screened.reason
            gate_series.append((int(ts_ns[t]), bool(screened.gate_active)))
        else:
            exec_action = dqn_action
            reason = "baseline"

        out = env.step(exec_action)
        records.append(
            StepRecord(
                timestamp_ns=int(ts_ns[t]),
                dqn_action=dqn_action,
                final_action=exec_action,
                reason=reason,
                mu=mu,
                sigma=sigma,
                reward=float(out.raw_pnl_delta),  # money == reward proxy in the prototype
                high_sigma=bool(sigma > config.variance_threshold),
                raw_pnl_delta=float(out.raw_pnl_delta),
                max_total_margin=float(abs(out.net_position)) * float(close[t]),
            )
        )
        done = out.done
    return records, gate_series


def run_arms(
    *,
    env_factory,
    policy,
    meta: dict,
    signals,
    close: np.ndarray,
    ts_ns: np.ndarray,
    config: IntegrationConfig,
) -> ArmResults:
    """Run combined / baseline / forecaster arms and validate Req-14 thresholds.

    A fresh ``IntegrationLayer`` (combined) and fresh envs are built here; the
    SAME trained ``policy`` is shared across arms and regimes so the screen's
    effect is isolated.
    """
    integration = IntegrationLayer(
        dqn=None,  # type: ignore[arg-type] - screen() does not call it
        forecaster_bridge=None,  # type: ignore[arg-type]
        signal_cache=None,  # type: ignore[arg-type]
        config=config,
    )

    combined, gate_series = _run_policy_arm(
        env_factory(), policy, signals.mu_bps, signals.sigma_bps, ts_ns, close, config, integration
    )
    baseline, _ = _run_policy_arm(
        env_factory(), policy, signals.mu_bps, signals.sigma_bps, ts_ns, close, config, None
    )
    comparison = compare_results(
        combined,
        baseline,
        pip_size=config.pip_size,
    )
    report = validate_thresholds(comparison)
    return ArmResults(
        regime=signals.regime,
        gate_enabled=True,
        combined=combined,
        baseline=baseline,
        gate_active_series=gate_series,
        comparison=comparison,
        report=report,
        meta=meta,
    )
