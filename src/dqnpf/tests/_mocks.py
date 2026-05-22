"""Mock collaborators for dqnpf integration tests.

Lightweight duck-typed fakes for modelenv gRPC messages, EnvironmentClient,
DQN advisor, state preprocessor, and forecaster bridge.

These are deliberately permissive so callers can mix-and-match, e.g. scripted
DQN actions with a real bridge, or scripted bridge signals with a real cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from tradingmodel.intraday.dqnpf.action_mapper import ACTION_NAMES


# ---------------------------------------------------------------------------
# Protobuf-shaped messages
# ---------------------------------------------------------------------------


@dataclass
class MockBar:
    timestamp_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class MockBarList:
    bars: list[MockBar]


@dataclass
class MockRecentBarsResponse:
    bars: dict[str, MockBarList]


@dataclass
class MockStateRow:
    values: list[float]


@dataclass
class MockObservation:
    state_columns: list[str] = field(default_factory=lambda: ["x"])
    state_data: list[MockStateRow] = field(
        default_factory=lambda: [MockStateRow(values=[0.0])]
    )
    reward: float = 0.0
    done: bool = False


@dataclass
class MockStepResponse:
    data: MockObservation
    info: str = ""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


_START_TS_NS = 1_700_000_000_000_000_000  # 2023-11-14 UTC
_M5_NS = 5 * 60 * 1_000_000_000


def make_m5_bars(n: int, start_ts_ns: int = _START_TS_NS) -> list[MockBar]:
    """Construct ``n`` synthetic M5 bars with strictly positive OHLC.

    Uses a deterministic tiny walk so rolling std is non-zero. Suitable for
    feeding into the real ``compute_features``.
    """
    import numpy as np

    rng = np.random.default_rng(seed=20260515)
    base = 100.0
    bars: list[MockBar] = []
    for i in range(n):
        drift = float(rng.normal(0.0, 0.05))
        open_p = base + i * 0.01 + drift
        close_p = open_p + float(rng.normal(0.0, 0.02))
        high_p = max(open_p, close_p) + abs(float(rng.normal(0.0, 0.01))) + 1e-3
        low_p = min(open_p, close_p) - abs(float(rng.normal(0.0, 0.01))) - 1e-3
        bars.append(
            MockBar(
                timestamp_ns=start_ts_ns + i * _M5_NS,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=1000.0 + i,
            )
        )
    return bars


def make_response(m5_bars: list[MockBar]) -> MockRecentBarsResponse:
    return MockRecentBarsResponse(bars={"M5": MockBarList(bars=m5_bars)})


def make_observation(reward: float = 0.0, done: bool = False) -> MockObservation:
    return MockObservation(reward=reward, done=done)


# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------


@dataclass
class FakeActionResult:
    action: int
    action_name: str = ""


class MockEnvClient:
    """Scripted env client used by ``_run_episode`` and direct tests.

    Args:
        observations: list of MockObservation, one per step *after reset*. The
            last entry should have ``done=True`` to terminate the loop.
        bars_responses: list or callable producing a MockRecentBarsResponse
            for each ``recent_bars`` call.
    """

    def __init__(
        self,
        *,
        observations: list[MockObservation],
        bars_responses: (
            list[MockRecentBarsResponse]
            | Callable[[int], MockRecentBarsResponse]
        ),
    ) -> None:
        self._observations = list(observations)
        self._bars_responses = bars_responses
        self.reset_calls: list[tuple] = []
        self.step_calls: list[tuple[int, str]] = []
        self.recent_bars_calls: int = 0

    def reset(
        self,
        symbol: str,
        episode_start_ts: int,
        episode_end_ts: int,
        step_size_seconds: int,
    ) -> MockObservation:
        self.reset_calls.append((symbol, episode_start_ts, episode_end_ts, step_size_seconds))
        return self._observations[0]

    def step(self, action: int, client_order_id: str) -> MockStepResponse:
        self.step_calls.append((action, client_order_id))
        idx = len(self.step_calls)  # next observation index
        if idx < len(self._observations):
            obs = self._observations[idx]
        else:
            obs = MockObservation(reward=0.0, done=True)
        return MockStepResponse(data=obs)

    def recent_bars(self, symbol: str) -> MockRecentBarsResponse:
        idx = self.recent_bars_calls
        self.recent_bars_calls += 1
        if callable(self._bars_responses):
            return self._bars_responses(idx)
        if idx < len(self._bars_responses):
            return self._bars_responses[idx]
        return self._bars_responses[-1]


class MockDQN:
    """Returns scripted action indices in order."""

    def __init__(self, action_script: list[int]) -> None:
        self.action_script = list(action_script)
        self.call_count = 0

    def recommend_action(self, state) -> FakeActionResult:
        idx = self.call_count
        self.call_count += 1
        action = (
            self.action_script[idx] if idx < len(self.action_script) else 0
        )
        return FakeActionResult(action=action, action_name=ACTION_NAMES[action])


class MockPreprocessor:
    """No-op preprocessor; returns a zero tensor and records call count."""

    def __init__(self) -> None:
        self.call_count = 0

    def process(self, obs) -> object:
        self.call_count += 1
        import torch

        return torch.zeros(1)


class MockBridge:
    """Scripted ForecasterBridge replacement returning fixed (mu, sigma)."""

    def __init__(self, signal_script: list[tuple[float, float]]) -> None:
        self.signal_script = list(signal_script)
        self.call_count = 0

    def compute_signal(self) -> tuple[float, float]:
        idx = self.call_count
        self.call_count += 1
        return (
            self.signal_script[idx]
            if idx < len(self.signal_script)
            else self.signal_script[-1]
        )
