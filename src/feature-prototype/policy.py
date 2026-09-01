"""Inference-only trading components for the feature prototype (no training).

The feature under test is the DQN->PF integration layer and profitability gate.
The DQN and the forecaster around it are therefore **inference-only stand-ins**;
there is deliberately no training code in the prototype:

* the **policy** is a fixed momentum/mean-reversion rule over a compact market
  observation (a stand-in for the production DQN, which in serving is a loaded
  `QNetwork` checkpoint, Chapter 3); and
* the **forecaster** signal is computed, not trained (see `signals.py`).

`ReplayTradingEnv` is a lightweight stand-in for the Rust `modelenv` gRPC service:
it replays the real M5 close path, exposes the observation the policy reads, and
books next-bar mark-to-market PnL (the same money convention the gate uses).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dqnpf.action_mapper import Direction, map_action

ACTION_DIM = 5
HOLD, BUY_1, BUY_2, SELL_1, SELL_2 = 0, 1, 2, 3, 4
MAX_POSITION = 2
VOLUME_PER_UNIT = 10_000.0


@dataclass
class StepOutcome:
    obs: np.ndarray
    raw_pnl_delta: float
    net_position: int
    done: bool


class ReplayTradingEnv:
    """Replay env over a fixed M5 close path (a stand-in for the Rust modelenv)."""

    OBS_DIM = 9

    def __init__(self, close: np.ndarray, *, warmup: int = 16):
        self._close = close.astype("float64")
        self._ret = np.zeros_like(self._close)
        self._ret[1:] = (self._close[1:] - self._close[:-1]) / self._close[:-1]
        self._vol = (
            pd.Series(self._ret).ewm(halflife=24, adjust=False).std().fillna(0.0).to_numpy()
        )
        self._rsi = self._compute_rsi(self._close, period=14)
        self._warmup = warmup
        self._n = len(self._close)
        self.reset()

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.zeros_like(close)
        delta[1:] = close[1:] - close[:-1]
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        ag = pd.Series(gain).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
        al = pd.Series(loss).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
        rs = ag / np.where(al < 1e-12, 1e-12, al)
        return 100.0 - 100.0 / (1.0 + rs)

    def reset(self) -> np.ndarray:
        self._t = self._warmup
        self._pos = 0
        return self._observe()

    @property
    def t(self) -> int:
        return self._t

    @property
    def warmup(self) -> int:
        return self._warmup

    @property
    def n(self) -> int:
        return self._n

    def position(self) -> int:
        return self._pos

    def _observe(self) -> np.ndarray:
        t = self._t
        last = self._ret[t]
        feats = np.array(
            [
                self._ret[t] * 1e4 / 10.0,
                self._ret[t - 1] * 1e4 / 10.0,
                self._ret[t - 2] * 1e4 / 10.0,
                self._ret[t - 4] * 1e4 / 10.0,
                self._ret[t - 9] * 1e4 / 10.0,
                self._vol[t] * 1e4 / 10.0,
                (self._rsi[t] - 50.0) / 50.0,
                self._pos / MAX_POSITION,
                np.sign(self._pos * last),
            ],
            dtype="float32",
        )
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    def observe(self) -> np.ndarray:
        return self._observe()

    def observe_at(self, t: int, pos: int = 0) -> np.ndarray:
        """Observation at bar ``t`` with net position ``pos`` (for one-shot advice)."""
        self._t = int(t)
        self._pos = int(pos)
        return self._observe()

    def step(self, action: int) -> StepOutcome:
        unit = map_action(action)
        delta = 0
        if unit.direction == Direction.LONG:
            delta = unit.risk_units
        elif unit.direction == Direction.SHORT:
            delta = -unit.risk_units
        self._pos = int(np.clip(self._pos + delta, -MAX_POSITION, MAX_POSITION))

        t = self._t
        done = t + 1 >= self._n
        raw_pnl = 0.0 if done else self._pos * VOLUME_PER_UNIT * (self._close[t + 1] - self._close[t])
        self._t += 1
        obs = self._observe() if not done else np.zeros(self.OBS_DIM, dtype="float32")
        return StepOutcome(obs=obs, raw_pnl_delta=float(raw_pnl), net_position=int(self._pos), done=done)


class InferencePolicy:
    """A fixed, inference-only momentum/mean-reversion policy (no learning).

    Reads the env observation and emits an action in {HOLD, BUY_1, BUY_2, SELL_1,
    SELL_2}. Deterministic; a transparent stand-in for the production DQN's
    inference, sufficient to produce a realistic action stream for the screen.
    """

    def __init__(self, threshold: float = 0.5):
        self._t = threshold

    def act(self, obs: np.ndarray) -> int:
        mom = float(obs[0] + obs[1])
        pos_norm = float(obs[7])
        if mom > self._t and pos_norm < 1.0:
            return BUY_2 if mom > 2.0 * self._t else BUY_1
        if mom < -self._t and pos_norm > -1.0:
            return SELL_2 if mom < -2.0 * self._t else SELL_1
        return HOLD
