"""Forecaster bridge: RecentBars → compute_features → predict pipeline.

Encapsulates the data path from the modelenv `RecentBars` RPC through the
probabilistic forecaster's feature engine and `predict()` call. Keeps
collaborators duck-typed so unit tests can substitute mocks for the gRPC
client and the forecaster.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch

from probabilisticforecaster.features import compute_features

logger = logging.getLogger(__name__)


_BAR_COLUMNS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
LOOKBACK_WINDOW = 36
FEATURE_DIM = 16
HISTORICAL_WINDOW = 1440
RECENT_BARS_COUNT = HISTORICAL_WINDOW + LOOKBACK_WINDOW
BPS_PER_UNIT = 10_000.0


class _ForecasterLike(Protocol):
    def predict(self, features: torch.Tensor) -> tuple[float, float]: ...


class _EnvClientLike(Protocol):
    def recent_bars(self, symbol: str, count: int = 0) -> Any: ...


def _bars_to_dataframe(bar_list: Any) -> pd.DataFrame:
    """Convert a protobuf BarList to the OHLC DataFrame compute_features expects.

    Args:
        bar_list: protobuf message with a repeated `bars` field whose items
            expose `timestamp_ns`, `open`, `high`, `low`, `close`, `volume`.
    """
    rows = [
        (
            bar.timestamp_ns,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
        )
        for bar in bar_list.bars
    ]
    return pd.DataFrame(rows, columns=_BAR_COLUMNS)


def _features_to_tensor(features_df: pd.DataFrame) -> torch.Tensor:
    """Take the most recent LOOKBACK_WINDOW rows as a (36, 16) float32 tensor."""
    window = features_df.iloc[-LOOKBACK_WINDOW:]
    array = window.to_numpy(dtype=np.float32, copy=False)
    return torch.from_numpy(array)


class ForecasterBridge:
    """Pipeline: RecentBars M5 bars → features → (mu, sigma).

    Args:
        forecaster: Loaded ForecasterInference (or any object exposing
            ``predict(tensor) -> (mu, sigma)``).
        symbol: Currency pair used in RecentBars requests.
        env_client: modelenv gRPC client exposing
            ``recent_bars(symbol) -> RecentBarsResponse``.
    """

    def __init__(
        self,
        forecaster: _ForecasterLike,
        symbol: str,
        env_client: _EnvClientLike,
    ) -> None:
        self._forecaster = forecaster
        self._symbol = symbol
        self._env_client = env_client

    @property
    def symbol(self) -> str:
        return self._symbol

    def compute_signal(self) -> tuple[float, float]:
        """Call RecentBars, extract M5 bars, compute features, predict.

        Returns:
            (mu, sigma) as Python floats, in basis points (the forecaster's
            raw fractional output scaled by BPS_PER_UNIT).

        Raises:
            KeyError: If "M5" is missing from the RecentBarsResponse.
            ValueError: If the M5 series has fewer than 36 bars.
        """
        response = self._env_client.recent_bars(
            self._symbol, count=RECENT_BARS_COUNT
        )
        bars = getattr(response, "bars", None)
        if bars is None or "M5" not in bars:
            available = list(bars.keys()) if bars is not None else []
            raise KeyError(
                f"RecentBarsResponse missing 'M5' key; available={available}"
            )

        bars_df = _bars_to_dataframe(bars["M5"])
        return self.compute_signal_from_bars(bars_df)

    def compute_signal_from_bars(self, bars_df: pd.DataFrame) -> tuple[float, float]:
        """Compute (mu, sigma) from a pre-loaded M5 bar DataFrame.

        Used during warm-up and testing to bypass the gRPC call. Returns
        (mu, sigma) in basis points (raw fractional output scaled by
        BPS_PER_UNIT).

        Raises:
            ValueError: If the bar series has fewer than 36 rows.
        """
        if len(bars_df) < LOOKBACK_WINDOW:
            raise ValueError(
                f"M5 bar series has {len(bars_df)} bars, "
                f"need at least {LOOKBACK_WINDOW}"
            )

        features_df = compute_features(bars_df, historical_window=HISTORICAL_WINDOW)
        if features_df.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"compute_features returned {features_df.shape[1]} columns, "
                f"expected {FEATURE_DIM}"
            )
        if len(features_df) < LOOKBACK_WINDOW:
            raise ValueError(
                f"Feature DataFrame has {len(features_df)} rows after "
                f"compute_features, need at least {LOOKBACK_WINDOW}"
            )

        tensor = _features_to_tensor(features_df)
        mu, sigma = self._forecaster.predict(tensor)
        return float(mu) * BPS_PER_UNIT, float(sigma) * BPS_PER_UNIT
