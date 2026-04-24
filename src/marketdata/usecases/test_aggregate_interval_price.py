"""
Unit tests for aggregate-interval-price.aggregate.

Tests mock S3 interactions and verify resample correctness, S3 path
construction, and edge-case handling across hour / day / month tiers.

Run:
    python -m pytest marketdata/usecases/test_aggregate_interval_price.py -v
"""

import importlib
import io
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

_mod = importlib.import_module("marketdata.usecases.aggregate-interval-price")
aggregate = _mod.aggregate
_iter_source_prefixes = _mod._iter_source_prefixes
_build_output_key = _mod._build_output_key

BUCKET = "prod-fintech-forex-sg-731833471586"
FX_SYMBOL = "USDJPY"
EXECUTION_DT = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


def _make_bars(
    start: datetime, count: int, freq: str, symbol: str = FX_SYMBOL,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame at the given frequency."""
    timestamps = pd.date_range(start, periods=count, freq=freq, tz="UTC")
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
    """Return a mock S3 client whose prefixes map to DataFrames."""
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


# ── _iter_source_prefixes ────────────────────────────────────────────

def test_iter_source_prefixes_hour_tier():
    start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    prefixes = list(_iter_source_prefixes(FX_SYMBOL, "M1", start, end, 60))
    # Inclusive end: covers start-hour and end-hour (end-keyed storage)
    assert len(prefixes) == 2
    assert prefixes[0].endswith("year=2026/month=01/day=01/hour=09/")
    assert prefixes[-1].endswith("year=2026/month=01/day=01/hour=10/")


def test_iter_source_prefixes_day_tier():
    start = datetime(2025, 12, 29, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    prefixes = list(_iter_source_prefixes(FX_SYMBOL, "D1", start, end, 1440))
    # 7 days in window + 1 end-keyed day
    assert len(prefixes) == 8
    assert prefixes[0].endswith("year=2025/month=12/day=29/")
    assert prefixes[-1].endswith("year=2026/month=01/day=05/")


def test_iter_source_prefixes_month_tier():
    start = datetime(2025, 11, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    prefixes = list(_iter_source_prefixes(FX_SYMBOL, "W1", start, end, 10080))
    # Nov, Dec, Jan + end-keyed Feb
    assert len(prefixes) == 4
    assert prefixes[0].endswith("year=2025/month=11/")
    assert prefixes[-1].endswith("year=2026/month=02/")


# ── M1 → M5 (hour tier source) ──────────────────────────────────────

def test_m1_to_m5_one_hour_window():
    start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    df = _make_bars(start, 60, "1min")
    prefix = (
        f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=M1"
        f"/year=2026/month=01/day=01/hour=09/"
    )
    s3 = _mock_s3_with_partitions({prefix: df})

    result = aggregate(
        FX_SYMBOL, "M1", "M5", EXECUTION_DT, 60, 60, s3, BUCKET,
    )

    assert len(result) == 12
    first = result.iloc[0]
    assert first["Open"] == 0.0
    assert first["High"] == 4.5
    assert first["Low"] == -0.5
    assert first["Close"] == 4.1
    assert first["Volume"] == 50

    key = s3.put_object.call_args.kwargs["Key"]
    assert "interval=M5" in key
    assert "year=2026/month=01/day=01/hour=10" in key


# ── M1 → H1 over daily window (day-tier output) ─────────────────────

def test_m1_to_h1_daily_window():
    partitions = {}
    for h in range(10, 24):
        start = datetime(2025, 12, 31, h, 0, tzinfo=timezone.utc)
        partitions[
            f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=M1"
            f"/year=2025/month=12/day=31/hour={h:02d}/"
        ] = _make_bars(start, 60, "1min")
    for h in range(0, 10):
        start = datetime(2026, 1, 1, h, 0, tzinfo=timezone.utc)
        partitions[
            f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=M1"
            f"/year=2026/month=01/day=01/hour={h:02d}/"
        ] = _make_bars(start, 60, "1min")

    s3 = _mock_s3_with_partitions(partitions)
    result = aggregate(
        FX_SYMBOL, "M1", "H1", EXECUTION_DT, 1440, 60, s3, BUCKET,
    )

    assert len(result) == 24
    key = s3.put_object.call_args.kwargs["Key"]
    assert "interval=H1" in key
    assert "year=2026/month=01/day=01/" in key
    assert "hour=" not in key


# ── M1 → D1 over 30-day window (year/month-tier output) ─────────────

def test_m1_to_d1_monthly_window():
    start = datetime(2025, 12, 31, 23, 0, tzinfo=timezone.utc)
    prefix = (
        f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=M1"
        f"/year=2025/month=12/day=31/hour=23/"
    )
    s3 = _mock_s3_with_partitions({prefix: _make_bars(start, 60, "1min")})

    result = aggregate(
        FX_SYMBOL, "M1", "D1", EXECUTION_DT, 43200, 60, s3, BUCKET,
    )

    assert not result.empty
    key = s3.put_object.call_args.kwargs["Key"]
    assert "interval=D1" in key
    assert "year=2026/month=01" in key
    assert "day=" not in key
    assert "hour=" not in key


# ── Boundary: data stored at end-of-range partition key ────────────

def test_last_hour_stored_at_end_keyed_partition_is_captured():
    """Real pipeline stores [23:00, 00:00) data under the 00:00 partition.

    Without the inclusive-end iteration fix, the last hour of each day/month
    would be silently dropped. This test stores data at the *end*-keyed
    prefix (hour=00 of next day) matching the real download layout.
    """
    # Window: [Jan 31 00:00, Feb 1 00:00)
    end = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    # Bars timestamped Jan 31 23:00 – 23:59 (last hour of Jan 31)
    last_hour_start = datetime(2026, 1, 31, 23, 0, tzinfo=timezone.utc)
    # Stored at Feb 1 00:00 hour partition (end-keyed, as real pipeline does)
    prefix = (
        f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=M1"
        f"/year=2026/month=02/day=01/hour=00/"
    )
    s3 = _mock_s3_with_partitions({prefix: _make_bars(last_hour_start, 60, "1min")})

    result = aggregate(FX_SYMBOL, "M1", "H1", end, 1440, 60, s3, BUCKET)

    # One H1 bar for the captured hour
    assert len(result) == 1
    assert result.iloc[0]["Timestamp"] == pd.Timestamp("2026-01-31 23:00", tz="UTC")
    assert result.iloc[0]["Volume"] == 600  # 60 mins × 10


# ── D1 → W1 over 7-day window (day-tier source → year/month output) ─

def test_d1_to_w1_weekly_window():
    # End at Monday 2026-01-05 00:00 → window [2025-12-29 Mon, 2026-01-05 Mon)
    end = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    partitions = {}
    for i in range(7):
        day_dt = datetime(2025, 12, 29, tzinfo=timezone.utc)
        day_dt = day_dt.replace(day=29 + i) if (29 + i) <= 31 else datetime(2026, 1, i - 2, tzinfo=timezone.utc)
        prefix = (
            f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=D1"
            f"/year={day_dt.year}/month={day_dt.month:02d}/day={day_dt.day:02d}/"
        )
        partitions[prefix] = _make_bars(day_dt, 1, "1D")

    s3 = _mock_s3_with_partitions(partitions)
    result = aggregate(
        FX_SYMBOL, "D1", "W1", end, 10080, 1440, s3, BUCKET,
    )

    # One Monday-anchored weekly bar
    assert len(result) == 1
    week_start = result.iloc[0]["Timestamp"]
    assert week_start == pd.Timestamp("2025-12-29", tz="UTC")
    assert result.iloc[0]["Volume"] == 70  # 7 days × 10

    key = s3.put_object.call_args.kwargs["Key"]
    assert "interval=W1" in key
    assert "year=2026/month=01" in key
    assert "day=" not in key


# ── D1 → MN1 over a calendar month ──────────────────────────────────

def test_d1_to_mn1_calendar_month():
    # Window [2025-12-01, 2026-01-01), 31 days = 44640 minutes
    end = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    partitions = {}
    for day in range(1, 32):
        day_dt = datetime(2025, 12, day, tzinfo=timezone.utc)
        prefix = (
            f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=D1"
            f"/year=2025/month=12/day={day:02d}/"
        )
        partitions[prefix] = _make_bars(day_dt, 1, "1D")

    s3 = _mock_s3_with_partitions(partitions)
    result = aggregate(
        FX_SYMBOL, "D1", "MN1", end, 44640, 1440, s3, BUCKET,
    )

    assert len(result) == 1
    month_start = result.iloc[0]["Timestamp"]
    assert month_start == pd.Timestamp("2025-12-01", tz="UTC")
    assert result.iloc[0]["Volume"] == 310  # 31 days × 10

    key = s3.put_object.call_args.kwargs["Key"]
    assert "interval=MN1" in key
    assert "year=2026/month=01" in key
    assert "day=" not in key


# ── Interval validation ─────────────────────────────────────────────

def test_target_not_multiple_of_source_raises():
    with pytest.raises(ValueError, match="must be a multiple"):
        aggregate(FX_SYMBOL, "M3", "M5", EXECUTION_DT, 60, 60, MagicMock(), BUCKET)


def test_unknown_target_interval_raises():
    with pytest.raises(ValueError, match="Unknown TARGET_INTERVAL"):
        aggregate(FX_SYMBOL, "M1", "X9", EXECUTION_DT, 60, 60, MagicMock(), BUCKET)


def test_unknown_source_interval_raises():
    with pytest.raises(ValueError, match="Unknown SOURCE_INTERVAL"):
        aggregate(FX_SYMBOL, "Z1", "M5", EXECUTION_DT, 60, 60, MagicMock(), BUCKET)


def test_mn1_from_weekly_source_rejected():
    with pytest.raises(ValueError, match="D1 or finer"):
        aggregate(FX_SYMBOL, "W1", "MN1", EXECUTION_DT, 44640, 10080, MagicMock(), BUCKET)


def test_mn1_from_daily_source_accepted():
    # Validation passes; empty partitions → no upload but no error.
    s3 = _mock_s3_with_partitions({})
    result = aggregate(
        FX_SYMBOL, "D1", "MN1", EXECUTION_DT, 44640, 1440, s3, BUCKET,
    )
    assert result.empty


# ── Missing / empty partitions ──────────────────────────────────────

def test_some_source_partitions_empty_still_succeeds():
    partitions = {}
    for h in range(10, 22):
        start = datetime(2025, 12, 31, h, 0, tzinfo=timezone.utc)
        partitions[
            f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=M1"
            f"/year=2025/month=12/day=31/hour={h:02d}/"
        ] = _make_bars(start, 60, "1min")
    s3 = _mock_s3_with_partitions(partitions)

    result = aggregate(
        FX_SYMBOL, "M1", "H1", EXECUTION_DT, 1440, 60, s3, BUCKET,
    )

    assert len(result) == 12
    s3.put_object.assert_called_once()


def test_all_partitions_empty_does_not_upload():
    s3 = _mock_s3_with_partitions({})
    result = aggregate(
        FX_SYMBOL, "M1", "M5", EXECUTION_DT, 60, 60, s3, BUCKET,
    )
    assert result.empty
    s3.put_object.assert_not_called()


# ── Window boundary / dedup ─────────────────────────────────────────

def test_rows_outside_window_are_excluded():
    start = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    prefix = (
        f"marketdata/interval-price/symbol={FX_SYMBOL}/interval=M1"
        f"/year=2026/month=01/day=01/hour=09/"
    )
    s3 = _mock_s3_with_partitions({prefix: _make_bars(start, 60, "1min")})

    result = aggregate(
        FX_SYMBOL, "M1", "M5", EXECUTION_DT, 60, 60, s3, BUCKET,
    )

    assert result.empty
    s3.put_object.assert_not_called()


def test_duplicate_rows_across_partitions_are_deduped():
    start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    df = _make_bars(start, 60, "1min")
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

    result = aggregate(
        FX_SYMBOL, "M1", "M5", EXECUTION_DT, 60, 60, s3, BUCKET,
    )

    assert len(result) == 12
    assert (result["Volume"] == 50).all()
