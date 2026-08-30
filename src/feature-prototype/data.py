"""USD/JPY M5 data for the integration-layer prototype: download + aggregate.

The prototype uses **real** market data. Two stages, both first-class parts of
the prototype (mirroring the project's own ticks -> interval-price pipeline):

1. **Download** real USD/JPY ticks from Dukascopy (free, no auth), `dukascopy.py`.
2. **Aggregate** those ticks into M5 OHLCV bars, `aggregate_m5` below.

Resolution order: a cached aggregated parquet (deterministic re-runs) -> a live
Dukascopy download+aggregate -> a seeded synthetic fallback (only if the network
is unavailable, so the prototype still runs offline). The chosen source is
recorded so the report can state exactly which data backed a run.

Schema returned everywhere: ``[timestamp_ns, open, high, low, close, volume]``
(UTC, ascending).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `import dukascopy` whether imported as a sibling or run standalone.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dukascopy  # noqa: E402

logger = logging.getLogger(__name__)

SYMBOL = "USDJPY"
PIP_SIZE = 0.01  # one pip for JPY-quote pairs
BAR_SECONDS = 300  # M5
POINT_DIVISOR = 1000.0  # Dukascopy integer price -> price for 3-decimal JPY pairs

# Default reproducible window: 3 weeks of Jan-2024 USD/JPY (~4k M5 bars).
DEFAULT_START = "2024-01-08"
DEFAULT_END = "2024-01-29"


@dataclass
class PriceData:
    """A loaded price slice plus provenance."""

    frame: pd.DataFrame  # [timestamp_ns, open, high, low, close, volume]
    source: str  # "cache:<path>" | "dukascopy:<range>" | "synthetic"

    @property
    def n_bars(self) -> int:
        return len(self.frame)

    @property
    def span(self) -> str:
        ts = self.frame["timestamp_ns"]
        a = pd.Timestamp(ts.iloc[0], unit="ns", tz="UTC")
        b = pd.Timestamp(ts.iloc[-1], unit="ns", tz="UTC")
        return f"{a:%Y-%m-%d %H:%M} -> {b:%Y-%m-%d %H:%M} UTC"


_REQUIRED = ["timestamp_ns", "open", "high", "low", "close", "volume"]


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[_REQUIRED].copy()
    frame = frame.dropna()
    frame = frame.sort_values("timestamp_ns").reset_index(drop=True)
    frame["timestamp_ns"] = frame["timestamp_ns"].astype("int64")
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = frame[col].astype("float64")
    frame = frame[frame["close"] > 0.0].reset_index(drop=True)
    return frame


def aggregate_m5(ticks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a tick DataFrame (``timestamp_ns, mid, volume``) into M5 OHLCV.

    Standard time-bar aggregation: OHLC of the mid price within each 5-minute
    (left-closed) bucket, volume summed; empty buckets (e.g. weekends) dropped.
    ``timestamp_ns`` is the bar's open (left edge), matching the rest of the
    pipeline.
    """
    idx = pd.to_datetime(ticks["timestamp_ns"], unit="ns", utc=True)
    mid = pd.Series(ticks["mid"].to_numpy(), index=idx)
    vol = pd.Series(ticks["volume"].to_numpy(), index=idx)
    ohlc = mid.resample("5min", label="left", closed="left").ohlc().dropna()
    v = vol.resample("5min", label="left", closed="left").sum().reindex(ohlc.index).fillna(0.0)
    out = pd.DataFrame(
        {
            "timestamp_ns": ohlc.index.asi8,
            "open": ohlc["open"].to_numpy(),
            "high": ohlc["high"].to_numpy(),
            "low": ohlc["low"].to_numpy(),
            "close": ohlc["close"].to_numpy(),
            "volume": v.to_numpy(),
        }
    )
    logger.info("aggregated %d ticks -> %d M5 bars", len(ticks), len(out))
    return _normalise(out)


def _from_dukascopy(start_date: str, end_date: str, cache_dir: Path) -> pd.DataFrame:
    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
    ticks = dukascopy.fetch_ticks(
        SYMBOL, start, end, point_divisor=POINT_DIVISOR, cache_dir=cache_dir / "dukascopy"
    )
    return aggregate_m5(ticks)


def _synthetic(n_bars: int, seed: int) -> pd.DataFrame:
    """Seeded USD/JPY-like M5 path with volatility clustering (offline fallback)."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 1.0e-9, 0.08, 0.90
    var = np.empty(n_bars)
    eps = np.empty(n_bars)
    var[0] = omega / max(1e-12, (1.0 - alpha - beta))
    eps[0] = rng.normal(0.0, np.sqrt(var[0]))
    for t in range(1, n_bars):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
        eps[t] = rng.normal(0.0, np.sqrt(var[t]))
    close = np.exp(np.log(150.0) + np.cumsum(1.0e-6 + eps))
    bar_rng = np.abs(rng.normal(0.0, np.sqrt(var)) * close)
    open_ = np.empty(n_bars)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + 0.5 * bar_rng
    low = np.minimum(open_, close) - 0.5 * bar_rng
    volume = rng.integers(50, 500, size=n_bars).astype("float64")
    start_ns = pd.Timestamp("2024-01-01 00:00", tz="UTC").value
    ts = start_ns + np.arange(n_bars, dtype="int64") * (BAR_SECONDS * 1_000_000_000)
    return _normalise(
        pd.DataFrame(
            {"timestamp_ns": ts, "open": open_, "high": high, "low": low,
             "close": close, "volume": volume}
        )
    )


def load_usdjpy_m5(
    *,
    cache_path: str | Path = "feature-prototype/cache/usdjpy_m5.parquet",
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    synthetic_bars: int = 6000,
    seed: int = 7,
) -> PriceData:
    """Load a USD/JPY M5 slice: cache -> Dukascopy download+aggregate -> synthetic."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        frame = _normalise(pd.read_parquet(cache_path))
        logger.info("loaded %d M5 bars from cache %s", len(frame), cache_path)
        return PriceData(frame=frame, source=f"cache:{cache_path.name}")

    try:
        frame = _from_dukascopy(start_date, end_date, cache_path.parent)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)
        logger.info("downloaded+aggregated %d real M5 bars -> cached", len(frame))
        return PriceData(frame=frame, source=f"dukascopy:{start_date}..{end_date}")
    except Exception as exc:  # noqa: BLE001 - fall back so the prototype still runs
        logger.warning("Dukascopy unavailable (%s); using synthetic fallback", exc)

    frame = _synthetic(synthetic_bars, seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    logger.info("generated %d synthetic M5 bars -> cached", len(frame))
    return PriceData(frame=frame, source="synthetic")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_usdjpy_m5(cache_path=Path(__file__).resolve().parent / "cache" / "usdjpy_m5.parquet")
    print(f"source={data.source} n_bars={data.n_bars} span={data.span}")
    print(data.frame.head())
    print(data.frame.describe())
