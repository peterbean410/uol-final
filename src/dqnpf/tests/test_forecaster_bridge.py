"""Unit tests for ForecasterBridge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import torch

from tradingmodel.intraday.dqnpf.forecaster_bridge import (
    FEATURE_DIM,
    LOOKBACK_WINDOW,
    ForecasterBridge,
    _bars_to_dataframe,
)


@dataclass
class _FakeBar:
    timestamp_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class _FakeBarList:
    bars: list[_FakeBar]


@dataclass
class _FakeResponse:
    bars: dict[str, _FakeBarList]


class _FakeEnvClient:
    """Minimal env client capturing recent_bars calls."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[str] = []

    def recent_bars(self, symbol: str) -> _FakeResponse:
        self.calls.append(symbol)
        return self._response


class _FakeForecaster:
    """Records predict() inputs and returns a constant (mu, sigma)."""

    def __init__(self, mu: float = 0.25, sigma: float = 3.5) -> None:
        self.mu = mu
        self.sigma = sigma
        self.received: list[torch.Tensor] = []

    def predict(self, features: torch.Tensor) -> tuple[float, float]:
        self.received.append(features)
        return self.mu, self.sigma


def _make_bars(n: int, base_price: float = 100.0) -> list[_FakeBar]:
    """Build n synthetic M5 bars with valid (non-zero, slightly varying) OHLC."""
    rng = np.random.default_rng(seed=42)
    bars = []
    start_ns = 1_700_000_000_000_000_000  # 2023-11-14 UTC, in ns
    step_ns = 5 * 60 * 1_000_000_000
    for i in range(n):
        # Tiny random walk so std isn't zero
        drift = rng.normal(0.0, 0.05)
        open_p = base_price + i * 0.01 + drift
        close_p = open_p + rng.normal(0.0, 0.02)
        high_p = max(open_p, close_p) + abs(rng.normal(0.0, 0.01)) + 1e-3
        low_p = min(open_p, close_p) - abs(rng.normal(0.0, 0.01)) - 1e-3
        bars.append(
            _FakeBar(
                timestamp_ns=start_ns + i * step_ns,
                open=float(open_p),
                high=float(high_p),
                low=float(low_p),
                close=float(close_p),
                volume=float(1000 + i),
            )
        )
    return bars


def test_bars_to_dataframe_columns_and_values() -> None:
    bars = _make_bars(3)
    bar_list = _FakeBarList(bars=bars)
    df = _bars_to_dataframe(bar_list)
    assert list(df.columns) == ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 3
    assert df.loc[0, "Open"] == pytest.approx(bars[0].open)
    assert df.loc[2, "Volume"] == pytest.approx(bars[2].volume)


def test_compute_signal_pipeline_with_mock_grpc_client() -> None:
    bars = _make_bars(1500)
    response = _FakeResponse(bars={"M5": _FakeBarList(bars=bars)})
    env = _FakeEnvClient(response=response)
    forecaster = _FakeForecaster(mu=0.42, sigma=2.71)

    bridge = ForecasterBridge(forecaster=forecaster, symbol="USDJPY", env_client=env)
    mu, sigma = bridge.compute_signal()

    assert (mu, sigma) == (0.42, 2.71)
    assert env.calls == ["USDJPY"]
    assert len(forecaster.received) == 1
    received = forecaster.received[0]
    assert received.shape == (LOOKBACK_WINDOW, FEATURE_DIM)
    assert received.dtype == torch.float32


def test_compute_signal_missing_m5_raises_key_error() -> None:
    response = _FakeResponse(bars={"M1": _FakeBarList(bars=_make_bars(1500))})
    env = _FakeEnvClient(response=response)
    bridge = ForecasterBridge(
        forecaster=_FakeForecaster(), symbol="USDJPY", env_client=env
    )

    with pytest.raises(KeyError, match="M5"):
        bridge.compute_signal()


def test_compute_signal_few_bars_raises_value_error() -> None:
    bars = _make_bars(LOOKBACK_WINDOW - 1)
    response = _FakeResponse(bars={"M5": _FakeBarList(bars=bars)})
    env = _FakeEnvClient(response=response)
    bridge = ForecasterBridge(
        forecaster=_FakeForecaster(), symbol="USDJPY", env_client=env
    )

    with pytest.raises(ValueError, match=str(LOOKBACK_WINDOW - 1)):
        bridge.compute_signal()


def test_compute_signal_from_bars_with_mock_dataframe() -> None:
    bars = _make_bars(1500)
    bars_df = pd.DataFrame(
        [(b.timestamp_ns, b.open, b.high, b.low, b.close, b.volume) for b in bars],
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )
    forecaster = _FakeForecaster(mu=-0.1, sigma=1.2)

    bridge = ForecasterBridge(
        forecaster=forecaster, symbol="USDJPY", env_client=_FakeEnvClient(_FakeResponse({}))
    )
    mu, sigma = bridge.compute_signal_from_bars(bars_df)

    assert (mu, sigma) == (-0.1, 1.2)
    assert forecaster.received[0].shape == (LOOKBACK_WINDOW, FEATURE_DIM)
    assert forecaster.received[0].dtype == torch.float32


def test_compute_signal_from_bars_few_rows_raises_value_error() -> None:
    bars_df = pd.DataFrame(
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    bridge = ForecasterBridge(
        forecaster=_FakeForecaster(),
        symbol="USDJPY",
        env_client=_FakeEnvClient(_FakeResponse({})),
    )
    with pytest.raises(ValueError, match="0 bars"):
        bridge.compute_signal_from_bars(bars_df)
