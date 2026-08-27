"""Run the DQN->PF integration / profitability-gate feature prototype.

Trains one compact Double-DQN, then evaluates the REAL integration layer under
three configurations and writes a metrics JSON + figures:

  C1  informative sigma, gate ON   -> feasibility: the screen can help
  C2  collapsed   sigma, gate ON   -> honest failure + the gate self-deactivates
  C3  collapsed   sigma, gate OFF  -> what the gate prevents (screen strictly worse)

Usage (from the forex repo root, so deepqnetwork/tradingmodel import):
    python -m preliminaryreport.prototype.run            # or:
    python preliminaryreport/prototype/run.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

# --- make both the forex packages and the sibling prototype modules importable
_THIS = Path(__file__).resolve()
_FOREX_ROOT = _THIS.parents[3]
_PROTO_DIR = _THIS.parent
for p in (str(_FOREX_ROOT), str(_PROTO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import data as data_mod  # noqa: E402
import signals as sig_mod  # noqa: E402
import stats_tests  # noqa: E402
from policy import InferencePolicy, ReplayTradingEnv  # noqa: E402
from engine import run_arms  # noqa: E402
from tradingmodel.intraday.dqnpf.config import IntegrationConfig  # noqa: E402

logger = logging.getLogger("prototype")

FIG_DIR = _THIS.parents[1] / "figures"
RESULTS_DIR = _PROTO_DIR / "results"
VARIANCE_THRESHOLD = 3.0
DIRECTIONAL_TOLERANCE = 1.0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_FOREX_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _cum_pips(records, pip_size: float) -> np.ndarray:
    return np.cumsum([r.raw_pnl_delta for r in records]) / pip_size


def _make_config(*, gate_enabled: bool, directional: bool) -> IntegrationConfig:
    return IntegrationConfig(
        symbol="USDJPY",
        variance_threshold=VARIANCE_THRESHOLD,
        max_risk_long_units=2,
        max_risk_short_units=1,
        forecaster_risk_aversion=0.1,
        forecaster_position_size=100_000.0,  # match VOLUME_PER_UNIT for comparability
        screen_profit_window_sessions=3,
        pip_size=data_mod.PIP_SIZE,
    )


def _summarise(arm) -> dict:
    c = arm.comparison
    return {
        "regime": arm.regime,
        "gate_enabled": arm.gate_enabled,
        "combined_sharpe_pnl": round(c.combined_sharpe_pnl, 4),
        "baseline_sharpe_pnl": round(c.baseline_sharpe_pnl, 4),
        "combined_pnl_pips": round(c.combined_pnl_pips, 1),
        "baseline_pnl_pips": round(c.baseline_pnl_pips, 1),
        "suppression_rate": round(c.suppression_rate, 4),
        "suppression_by_reason": c.suppression_by_reason,
        "high_sigma_time_fraction": round(c.high_sigma_time_fraction, 4),
        "trades_combined": c.trades_combined,
        "trades_baseline": c.trades_baseline,
        "high_sigma_neg_pnl_prop_combined": round(
            c.high_sigma_negative_raw_pnl_proportion_combined, 4
        ),
        "high_sigma_neg_pnl_prop_baseline": round(
            c.high_sigma_negative_raw_pnl_proportion_baseline, 4
        ),
        "sigma_distribution_combined": {
            k: round(v, 3) for k, v in c.sigma_distribution_combined.items()
        },
        "final_gate_active": arm.gate_active_series[-1][1] if arm.gate_active_series else None,
        "req14_passed": arm.report.passed,
        "req14_failures": arm.report.failures,
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_price_and_sigma(close, info_sig, coll_sig, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax1.plot(close, lw=0.8, color="#222")
    ax1.set_ylabel("USDJPY close")
    ax1.set_title("Real USDJPY M5 path (Dukascopy) and forecaster sigma")
    ax2.plot(info_sig.sigma_bps, lw=0.8, color="#1f77b4", label="informative sigma (EWMA vol)")
    ax2.plot(coll_sig.sigma_bps, lw=1.0, color="#d62728", ls="--", label="collapsed sigma (constant)")
    ax2.axhline(VARIANCE_THRESHOLD, color="#666", ls=":", label=f"variance_threshold={VARIANCE_THRESHOLD}")
    ax2.set_ylabel("sigma (bps)")
    ax2.set_xlabel("M5 bar index")
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_equity(results: dict, pip_size, path):
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.0), sharey=False)
    if len(results) == 1:
        axes = [axes]
    for ax, (label, arm) in zip(axes, results.items()):
        ax.plot(_cum_pips(arm.baseline, pip_size), label="DQN-only", color="#1f77b4", lw=1.1)
        ax.plot(_cum_pips(arm.combined, pip_size), label="combined (DQN->screen)", color="#d62728", lw=1.1)
        ax.axhline(0, color="#999", lw=0.6)
        verdict = "PASS" if arm.report.passed else "FAIL"
        ax.set_title(f"{label}\nReq-14 gate: {verdict}", fontsize=9)
        ax.set_xlabel("M5 bar index")
        ax.set_ylabel("cumulative PnL (pips)")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Three-arm equity curves", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_suppression_by_sigma(arm, path):
    sig = np.array([r.sigma for r in arm.combined])
    screened = np.array([r.reason not in ("pass", "baseline") for r in arm.combined])
    bins = np.linspace(sig.min(), max(sig.max(), VARIANCE_THRESHOLD + 1), 11)
    idx = np.digitize(sig, bins)
    centres, rates = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() >= 5:
            centres.append(0.5 * (bins[b - 1] + bins[b]))
            rates.append(screened[m].mean())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(centres, rates, width=(bins[1] - bins[0]) * 0.85, color="#9467bd", alpha=0.85)
    ax.axvline(VARIANCE_THRESHOLD, color="#d62728", ls="--", label=f"variance_threshold={VARIANCE_THRESHOLD}")
    ax.set_xlabel("forecaster sigma (bps)")
    ax.set_ylabel("fraction of DQN actions screened")
    ax.set_title(f"Screen suppression vs sigma ({arm.regime}, gate {'on' if arm.gate_enabled else 'off'})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_gate_timeline(results: dict, path):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for label, arm in results.items():
        if not arm.gate_active_series:
            continue
        xs = np.arange(len(arm.gate_active_series))
        ys = np.array([1 if g else 0 for _, g in arm.gate_active_series])
        ax.plot(xs, ys, lw=1.1, label=label, alpha=0.85)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["screen OFF\n(gate bypassed)", "screen ON"])
    ax.set_xlabel("combined-arm step")
    ax.set_title("Profitability gate: screen active/bypassed over time")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_sigma_calibration(close, info_sig, coll_sig, warmup, path):
    n = len(close)
    fwd = np.zeros(n)
    fwd[:-1] = np.abs((close[1:] - close[:-1]) / close[:-1]) * 1e4  # |fwd ret| bps
    fig, ax = plt.subplots(figsize=(7, 4))
    for sig, name, col in ((info_sig, "informative", "#1f77b4"), (coll_sig, "collapsed", "#d62728")):
        s = sig.sigma_bps[warmup : n - 1]
        f = fwd[warmup : n - 1]
        if np.ptp(s) < 1e-6:  # constant sigma -> single point
            ax.scatter([s.mean()], [f.mean()], color=col, s=60, label=f"{name} (flat)")
            continue
        qs = np.quantile(s, np.linspace(0, 1, 9))
        cx, cy = [], []
        for i in range(len(qs) - 1):
            m = (s >= qs[i]) & (s < qs[i + 1])
            if m.sum() >= 5:
                cx.append(s[m].mean())
                cy.append(f[m].mean())
        ax.plot(cx, cy, "o-", color=col, label=name)
    ax.axvline(VARIANCE_THRESHOLD, color="#666", ls=":", label=f"variance_threshold={VARIANCE_THRESHOLD}")
    ax.set_xlabel("forecaster sigma (bps)")
    ax.set_ylabel("mean |next-bar return| (bps)")
    ax.set_title("Sigma calibration: does sigma predict realised volatility?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic-bars", type=int, default=6000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # The integration layer logs every budget-exhausted suppression at INFO; quiet
    # it so the prototype's own progress is readable.
    logging.getLogger("tradingmodel.intraday.dqnpf.integration").setLevel(logging.WARNING)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    price = data_mod.load_usdjpy_m5(
        cache_path=_PROTO_DIR / "cache" / "usdjpy_m5.parquet",
        synthetic_bars=args.synthetic_bars,
        seed=7,
    )
    close = price.frame["close"].to_numpy()
    ts_ns = price.frame["timestamp_ns"].to_numpy()
    logger.info("data: source=%s n_bars=%d span=%s", price.source, price.n_bars, price.span)

    # A compact replay env over the real M5 slice, rebuilt fresh per arm.
    def env_factory():
        return ReplayTradingEnv(close)

    # Inference-only DQN stand-in (no training), shared across every config so the
    # screen's effect is isolated.
    policy = InferencePolicy()
    policy_info = {"type": "inference-only momentum/mean-reversion stand-in (no training)"}

    # Inference-only forecaster signals (no training); the real causal-Transformer
    # PF is a production component (Ch3). `informative` = EWMA-vol sigma;
    # `collapsed` = constant sigma (the production failure mode).
    info_sig = sig_mod.informative_signals(close)
    coll_sig = sig_mod.collapsed_signals(close)

    configs = {
        "C1 informative + gate": (_make_config(gate_enabled=True, directional=True), info_sig),
        "C2 collapsed + gate": (_make_config(gate_enabled=True, directional=True), coll_sig),
        "C3 collapsed + gate OFF": (_make_config(gate_enabled=False, directional=True), coll_sig),
    }

    results = {}
    for label, (cfg, sig) in configs.items():
        logger.info("running %s ...", label)
        results[label] = run_arms(
            env_factory=env_factory,
            policy=policy,
            meta=policy_info,
            signals=sig,
            close=close,
            ts_ns=ts_ns,
            config=cfg,
        )

    # --- figures ----------------------------------------------------------
    warmup = env_factory().warmup
    fig_price_and_sigma(close, info_sig, coll_sig, FIG_DIR / "fig1_price_sigma.png")
    fig_equity(results, data_mod.PIP_SIZE, FIG_DIR / "fig2_equity_curves.png")
    fig_suppression_by_sigma(results["C1 informative + gate"], FIG_DIR / "fig3_suppression_by_sigma.png")
    fig_gate_timeline(
        {k: v for k, v in results.items() if v.gate_enabled},
        FIG_DIR / "fig4_gate_timeline.png",
    )
    fig_sigma_calibration(close, info_sig, coll_sig, warmup, FIG_DIR / "fig5_sigma_calibration.png")

    # --- formal evaluation tests on the prototype's own outputs -----------
    nbar = len(close)
    fwd = np.zeros(nbar)
    fwd[:-1] = (close[1:] - close[:-1]) / close[:-1]
    sl = slice(warmup, nbar - 1)
    headline = "C1 informative + gate"
    head_pnl = [r.raw_pnl_delta for r in results[headline].combined]
    trial_sharpes = [v.comparison.combined_sharpe_pnl for v in results.values()]
    eval_tests = {
        "forecaster_pesaran_timmermann": stats_tests.pesaran_timmermann(
            info_sig.mu_bps[sl], fwd[sl]
        ),
        "headline_arm": headline,
        "headline_psr_vs_zero": stats_tests.probabilistic_sharpe_ratio(head_pnl, 0.0),
        "headline_dsr_over_n_trials": stats_tests.deflated_sharpe_ratio(head_pnl, trial_sharpes),
    }

    # --- metrics JSON -----------------------------------------------------
    out = {
        "meta": {
            "git_sha": _git_sha(),
            "data_source": price.source,
            "n_bars": price.n_bars,
            "span": price.span,
            "seed": args.seed,
            "variance_threshold": VARIANCE_THRESHOLD,
            "policy": policy_info,
            "evaluation_tests": eval_tests,
        },
        "configs": {label: _summarise(arm) for label, arm in results.items()},
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", metrics_path)

    # --- console summary --------------------------------------------------
    print("\n================ PROTOTYPE SUMMARY ================")
    print(f"data={price.source}  bars={price.n_bars}  span={price.span}")
    print(f"policy: {policy_info['type']}")
    hdr = f"{'config':26s} {'comb_Sharpe':>11s} {'base_Sharpe':>11s} {'suppr':>7s} {'gate_end':>8s} {'Req14':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for label, arm in results.items():
        s = _summarise(arm)
        print(f"{label:26s} {s['combined_sharpe_pnl']:>11.4f} {s['baseline_sharpe_pnl']:>11.4f} "
              f"{s['suppression_rate']*100:>6.1f}% {str(s['final_gate_active']):>8s} "
              f"{'PASS' if s['req14_passed'] else 'FAIL':>6s}")
    pt = eval_tests["forecaster_pesaran_timmermann"]
    psr = eval_tests["headline_psr_vs_zero"]
    dsr = eval_tests["headline_dsr_over_n_trials"]
    print("--- formal tests ---")
    print(f"Pesaran-Timmermann (forecaster): DA={pt['directional_accuracy']} "
          f"stat={pt['statistic']} p={pt['p_value_one_sided']}")
    print(f"{headline}: PSR(>0)={psr['psr']}  "
          f"DSR(N={dsr['n_trials']})={dsr['dsr']}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
