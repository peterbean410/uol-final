"""Formal evaluation tests applied to the prototype's own outputs.

Turns the citations in Chapter 2 into computed results:

* **Pesaran & Timmermann (1992)** directional/market-timing test, does the
  forecaster's sign predict the next move better than chance?
* **Probabilistic / Deflated Sharpe Ratio** (Bailey & López de Prado, 2014),
  is an arm's Sharpe statistically distinguishable from zero given its sample
  length and non-normality (PSR), and does it survive the selection bias of
  having tried N configurations (DSR)?
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, norm, skew

_EULER = 0.5772156649015329  # Euler–Mascheroni constant (gamma)


def pesaran_timmermann(pred: np.ndarray, actual: np.ndarray) -> dict:
    """PT (1992) test of directional predictability. One-sided p (skill = +)."""
    pred = np.asarray(pred, float)
    actual = np.asarray(actual, float)
    mask = actual != 0.0
    sp = np.sign(pred[mask])
    sa = np.sign(actual[mask])
    n = int(sa.size)
    p = float(np.mean(sp == sa))            # directional accuracy
    px = float(np.mean(sp > 0))             # P(predict up)
    py = float(np.mean(sa > 0))             # P(actual up)
    pstar = py * px + (1 - py) * (1 - px)   # accuracy expected under independence
    var_p = pstar * (1 - pstar) / n
    var_pstar = (
        ((2 * py - 1) ** 2) * px * (1 - px) / n
        + ((2 * px - 1) ** 2) * py * (1 - py) / n
        + 4 * py * px * (1 - py) * (1 - px) / n ** 2
    )
    denom = var_p - var_pstar
    stat = (p - pstar) / np.sqrt(denom) if denom > 0 else 0.0
    return {
        "n": n,
        "directional_accuracy": round(p, 4),
        "expected_under_chance": round(pstar, 4),
        "statistic": round(float(stat), 4),
        "p_value_one_sided": round(float(1.0 - norm.cdf(stat)), 4),
    }


def _sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def probabilistic_sharpe_ratio(returns: np.ndarray, sr_star: float = 0.0) -> dict:
    """PSR: probability the true (per-observation) Sharpe exceeds ``sr_star``."""
    r = np.asarray(returns, float)
    n = int(r.size)
    sr = _sharpe(r)
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))  # non-excess (3 for normal)
    denom = np.sqrt(max(1e-12, 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2))
    z = (sr - sr_star) * np.sqrt(max(1, n - 1)) / denom
    return {
        "sharpe_per_bar": round(sr, 5),
        "skew": round(g3, 3),
        "kurtosis": round(g4, 2),
        "n": n,
        "sr_benchmark": round(float(sr_star), 5),
        "psr": round(float(norm.cdf(z)), 4),
    }


def deflated_sharpe_ratio(returns: np.ndarray, trial_sharpes) -> dict:
    """DSR: PSR with the benchmark raised to the expected max Sharpe of N trials."""
    ts = np.asarray(trial_sharpes, float)
    n_trials = int(ts.size)
    var_trials = float(ts.var(ddof=1)) if n_trials > 1 else 0.0
    if n_trials > 1 and var_trials > 0:
        sr_star = np.sqrt(var_trials) * (
            (1 - _EULER) * norm.ppf(1 - 1.0 / n_trials)
            + _EULER * norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    else:
        sr_star = 0.0
    out = probabilistic_sharpe_ratio(returns, sr_star=float(sr_star))
    out.update({"n_trials": n_trials, "dsr": out.pop("psr"), "trial_sharpe_var": round(var_trials, 8)})
    return out
