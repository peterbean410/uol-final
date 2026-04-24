"""
Unit tests for aggregate-interval-price.aggregate.

Tests mock S3 interactions and verify resample correctness, S3 path
construction, and edge-case handling.

Run:
    python -m pytest marketdata/usecases/test_aggregate_interval_price.py -v
"""

import importlib
import io
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_mod = importlib.import_module("marketdata.usecases.aggregate-interval-price")
aggregate = _mod.aggregate
_iter_source_prefixes = _mod._iter_source_prefixes
_aggregate = _mod._aggregate
_build_output_key = _mod._build_output_key

BUCKET = "prod-fintech-forex-sg-731833471586"
FX_SYMBOL = "USDJPY"
SOURCE_INTERVAL = "M1"
EXECUTION_DT = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


def _make_m1_bars(start: datetime, count: int, symbol: str = FX_SYMBOL) -> pd.DataFrame:
    """Build a synthetic M1 OHLCV DataFrame starting at `start`."""
    timestamps = pd.date_range(start, periods=count, freq="1min", tz="UTC")
    return pd.DataFrame({
        "Timestamp": timestamps,
        "Symbol": [symbol] * count,
        "Open": [float(i) for i in range(count)],
        "High": [float(i) + 0.5 for i in range(count)],
        "Low": [float(i) - 0.5 for i in range(count)],
        "Close": [float(i) + 0.1 for i in range(count)],
        "Volume": [10] * count,
    })


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    return buf.getvalue()


def _mock_s3_with_partitions(partitions: dict) -> MagicMock:
    """Return a mock S3 client whose prefixes map to DataFrames.

    `partitions`: dict of {prefix: DataFrame}. Prefixes absent from the dict
    return an empty listing.
    """
    s3 = MagicMock()

    def list_objects_v2(Bucket, Prefix):
        if Prefix in partitions:
            return {"Contents": [{"Key": Prefix + "bars.parquet"}]}
        return {}

    def get_object(Bucket, Key):
        for prefix, df in partitions.items():
            if Key.startswith(prefix):
                return {"Body": io.BytesIO(_parquet_bytes(df))}
        raise AssertionError(f"Unexpected get_object Key={Key}")

    s3.list_objects_v2.side_effect = list_objects_v2
    s3.get_object.side_effect = get_object
    return s3


# ── _iter_source_prefixes ───────────────────────────────────────────

def test_iter_source_prefixes_one_hour_window():
    start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    prefixes = list(_iter_source_prefixes(FX_SYMBOL, "M1", start, end))
    assert len(prefixes) == 1
    assert prefixes[0].endswith("year=2026/month=01/day=01/hour=09/")


def test_iter_source_prefixes_day_window_yields_24():
    start = datetime(2025, 12, 31, 10, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    prefixes = list(_iter_source_prefixes(FX_SYMBOL, "M1", start, end))
    assert len(prefixes) == 24


# ── M1 → M5 over a 1-hour window ────────────────────────────────────

def test_m1_to_m5_one_hour_window_aggregates_to_12_bars():
    start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    df = _make_m1_bars(start, 60)
    prefix = (
        f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=M1"
        f"/year=2026/month=01/day=01/hour=09/"
    )
    s3 = _mock_s3_with_partitions({prefix: df})

    result = aggregate(FX_SYMBOL, "M1", "M5", EXECUTION_DT, 60, s3, BUCKET)

    assert len(result) == 12
    assert list(result.columns) == [
        "Timestamp", "Symbol", "Open", "High", "Low", "Close", "Volume",
    ]
    # First M5 bucket: minutes 0..4 → Open=0, High=4.5, Low=-0.5, Close=4.1, Volume=50
    first = result.iloc[0]
    assert first["Open"] == 0.0
    assert first["High"] == 4.5
    assert first["Low"] == -0.5
    assert first["Close"] == 4.1
    assert first["Volume"] == 50
    assert (result["Symbol"] == FX_SYMBOL).all()

    # Upload called once with expected hour-tier key
    s3.put_object.assert_called_once()
    key = s3.put_object.call_args.kwargs["Key"]
    assert "interval=M5" in key
    assert "year=2026/month=01/day=01/hour=10" in key


# ── M1 → H1 over a 24-hour window (day tier) ────────────────────────

def test_m1_to_h1_daily_window_writes_to_day_tier():
    partitions = {}
    for h in range(10, 24):
        start = datetime(2025, 12, 31, h, 0, tzinfo=timezone.utc)
        partitions[
            f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=M1"
            f"/year=2025/month=12/day=31/hour={h:02d}/"
        ] = _make_m1_bars(start, 60)
    for h in range(0, 10):
        start = datetime(2026, 1, 1, h, 0, tzinfo=timezone.utc)
        partitions[
            f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=M1"
            f"/year=2026/month=01/day=01/hour={h:02d}/"
        ] = _make_m1_bars(start, 60)

    s3 = _mock_s3_with_partitions(partitions)

    result = aggregate(FX_SYMBOL, "M1", "H1", EXECUTION_DT, 1440, s3, BUCKET)

    # 24 hourly bars expected
    assert len(result) == 24
    key = s3.put_object.call_args.kwargs["Key"]
    assert "interval=H1" in key
    assert "year=2026/month=01/day=01/" in key
    assert "hour=" not in key


# ── M1 → D1 over a 30-day window (year/month tier) ──────────────────

def test_m1_to_d1_monthly_window_writes_to_year_month_tier():
    # Single partition is enough to exercise the path construction
    start = datetime(2025, 12, 31, 23, 0, tzinfo=timezone.utc)
    prefix = (
        f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=M1"
        f"/year=2025/month=12/day=31/hour=23/"
    )
    s3 = _mock_s3_with_partitions({prefix: _make_m1_bars(start, 60)})

    result = aggregate(FX_SYMBOL, "M1", "D1", EXECUTION_DT, 43200, s3, BUCKET)

    assert len(result) >= 1
    key = s3.put_object.call_args.kwargs["Key"]
    assert "interval=D1" in key
    assert "year=2026/month=01" in key
    assert "day=" not in key
    assert "hour=" not in key


# ── Target must be a multiple of source ─────────────────────────────

def test_target_not_multiple_of_source_raises():
    with pytest.raises(ValueError, match="must be a multiple"):
        aggregate(
            FX_SYMBOL, "M3", "M5", EXECUTION_DT, 60, MagicMock(), BUCKET,
        )


def test_unknown_target_interval_raises():
    with pytest.raises(ValueError, match="Unknown TARGET_INTERVAL"):
        aggregate(
            FX_SYMBOL, "M1", "X9", EXECUTION_DT, 60, MagicMock(), BUCKET,
        )


def test_unknown_source_interval_raises():
    with pytest.raises(ValueError, match="Unknown SOURCE_INTERVAL"):
        aggregate(
            FX_SYMBOL, "Z1", "M5", EXECUTION_DT, 60, MagicMock(), BUCKET,
        )


# ── Missing source partitions ───────────────────────────────────────

def test_some_source_partitions_empty_still_succeeds():
    # Only 12 of 24 hours present
    partitions = {}
    for h in range(10, 22):
        start = datetime(2025, 12, 31, h, 0, tzinfo=timezone.utc)
        partitions[
            f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=M1"
            f"/year=2025/month=12/day=31/hour={h:02d}/"
        ] = _make_m1_bars(start, 60)
    s3 = _mock_s3_with_partitions(partitions)

    result = aggregate(FX_SYMBOL, "M1", "H1", EXECUTION_DT, 1440, s3, BUCKET)

    assert len(result) == 12
    s3.put_object.assert_called_once()


def test_all_partitions_empty_does_not_upload():
    s3 = _mock_s3_with_partitions({})

    result = aggregate(FX_SYMBOL, "M1", "M5", EXECUTION_DT, 60, s3, BUCKET)

    assert result.empty
    s3.put_object.assert_not_called()


# ── Boundary filter ─────────────────────────────────────────────────

def test_rows_outside_window_are_excluded():
    # Source file returns 60 M1 bars starting 08:00; window is [09:00, 10:00).
    # Entire source file is outside window → no aggregation.
    start = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    prefix = (
        f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=M1"
        f"/year=2026/month=01/day=01/hour=09/"
    )
    s3 = _mock_s3_with_partitions({prefix: _make_m1_bars(start, 60)})

    result = aggregate(FX_SYMBOL, "M1", "M5", EXECUTION_DT, 60, s3, BUCKET)

    assert result.empty
    s3.put_object.assert_not_called()


# ── Cross-partition dedup ───────────────────────────────────────────

def test_duplicate_rows_across_partitions_are_deduped():
    start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    df = _make_m1_bars(start, 60)
    # Same prefix served twice via two parquet objects would duplicate rows;
    # emulate by returning two keys from list_objects_v2 for the same prefix.
    prefix = (
        f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=M1"
        f"/year=2026/month=01/day=01/hour=09/"
    )
    s3 = MagicMock()
    s3.list_objects_v2.return_value = {
        "Contents": [
            {"Key": prefix + "a.parquet"},
            {"Key": prefix + "b.parquet"},
        ],
    }
    s3.get_object.side_effect = lambda Bucket, Key: {
        "Body": io.BytesIO(_parquet_bytes(df)),
    }

    result = aggregate(FX_SYMBOL, "M1", "M5", EXECUTION_DT, 60, s3, BUCKET)

    # Dedup collapses duplicates → same 12 M5 bars, Volume is still 50 per bucket
    assert len(result) == 12
    assert (result["Volume"] == 50).all()
