"""
Unit tests for create-eoi-price-snapshot.create_snapshot.

These tests mock S3 interactions and verify the merge logic,
S3 path construction, and edge-case handling.

Run:
    python -m pytest marketdata/usecases/test_create_eoi_price_snapshot.py -v
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# The module uses a hyphenated filename, so import via importlib
import importlib

_mod = importlib.import_module("marketdata.usecases.create-eoi-price-snapshot")
create_snapshot = _mod.create_snapshot
_load_partition = _mod._load_partition
_load_snapshot_file = _mod._load_snapshot_file
_upload_to_s3 = _mod._upload_to_s3

BUCKET = "prod-fintech-forex-sg-731833471586"
FX_SYMBOL = "EURUSD"
INTERVAL = "1h"
EXECUTION_DT = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
TS = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")


def _make_df(rows):
    """Build a DataFrame with Timestamp and Symbol columns from simplified rows."""
    if rows is None:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ── Fixtures for partition data ──────────────────────────────────────

CURRENT_DATA = [
    {"Timestamp": TS, "Symbol": "EURUSD", "price": 1.10},
    {"Timestamp": TS, "Symbol": "USDJPY", "price": 110},
]

PREVIOUS_DATA = [
    {"Timestamp": TS, "Symbol": "EURUSD", "price": 1.00},
    {"Timestamp": TS, "Symbol": "GBPUSD", "price": 1.30},
]


# ── Test Case 1: Daily snapshot (24h window) ────────────────────────

@patch.object(_mod, "_upload_to_s3")
@patch.object(_mod, "_load_snapshot_file")
@patch.object(_mod, "_load_partition")
def test_daily_snapshot_merge_and_path(mock_load, mock_load_prev, mock_upload):
    """24h window: merges current raw partition with previous snapshot, correct S3 path."""
    mock_load.return_value = _make_df(CURRENT_DATA)
    mock_load_prev.return_value = _make_df(PREVIOUS_DATA)

    df = create_snapshot(FX_SYMBOL, INTERVAL, EXECUTION_DT, 1440, MagicMock(), BUCKET)

    # upload called exactly once
    assert mock_upload.call_count == 1

    uploaded_df = mock_upload.call_args[0][0]
    s3_key = mock_upload.call_args[0][2]

    # S3 path assertions
    assert s3_key.startswith("marketdata/eod-snapshot/")
    assert f"symbol={FX_SYMBOL}" in s3_key
    assert f"interval={INTERVAL}" in s3_key
    assert "year=2026/month=01/day=01" in s3_key
    assert "hour=" not in s3_key

    # DataFrame assertions
    assert len(uploaded_df) == 3
    symbols = set(uploaded_df["Symbol"].tolist())
    assert symbols == {"EURUSD", "USDJPY", "GBPUSD"}

    # Current partition takes precedence for EURUSD
    eurusd_price = uploaded_df.loc[uploaded_df["Symbol"] == "EURUSD", "price"].iloc[0]
    assert eurusd_price == 1.10


# ── Test Case 2: Hourly snapshot (1h window) ────────────────────────

@patch.object(_mod, "_upload_to_s3")
@patch.object(_mod, "_load_snapshot_file")
@patch.object(_mod, "_load_partition")
def test_hourly_snapshot_path_contains_hour(mock_load, mock_load_prev, mock_upload):
    """1h window: S3 path includes hour component and uses eod-snapshot prefix."""
    mock_load.return_value = _make_df(CURRENT_DATA)
    mock_load_prev.return_value = _make_df(PREVIOUS_DATA)

    df = create_snapshot(FX_SYMBOL, INTERVAL, EXECUTION_DT, 60, MagicMock(), BUCKET)

    assert mock_upload.call_count == 1

    uploaded_df = mock_upload.call_args[0][0]
    s3_key = mock_upload.call_args[0][2]

    assert s3_key.startswith("marketdata/eoh-snapshot/")
    assert "year=2026/month=01/day=01/hour=10" in s3_key
    assert len(uploaded_df) == 3


# ── Test Case 2b: Monthly snapshot (43200-min window) ───────────────

@patch.object(_mod, "_upload_to_s3")
@patch.object(_mod, "_load_snapshot_file")
@patch.object(_mod, "_load_partition")
def test_monthly_snapshot_uses_eom_prefix(mock_load, mock_load_prev, mock_upload):
    """43200-min window: S3 path uses eom-snapshot prefix, year/month only."""
    mock_load.return_value = _make_df(CURRENT_DATA)
    mock_load_prev.return_value = _make_df(PREVIOUS_DATA)

    df = create_snapshot(FX_SYMBOL, INTERVAL, EXECUTION_DT, 43200, MagicMock(), BUCKET)

    assert mock_upload.call_count == 1
    s3_key = mock_upload.call_args[0][2]

    assert s3_key.startswith("marketdata/eom-snapshot/")
    assert "year=2026/month=01" in s3_key
    assert "day=" not in s3_key
    assert "hour=" not in s3_key


# ── Test Case 2c: Weekly snapshot (10080-min window) ────────────────

@patch.object(_mod, "_upload_to_s3")
@patch.object(_mod, "_load_snapshot_file")
@patch.object(_mod, "_load_partition")
def test_weekly_snapshot_uses_eow_prefix(mock_load, mock_load_prev, mock_upload):
    """10080-min window: S3 path uses eow-snapshot prefix, year/month only."""
    mock_load.return_value = _make_df(CURRENT_DATA)
    mock_load_prev.return_value = _make_df(PREVIOUS_DATA)

    df = create_snapshot(FX_SYMBOL, INTERVAL, EXECUTION_DT, 10080, MagicMock(), BUCKET)

    assert mock_upload.call_count == 1
    s3_key = mock_upload.call_args[0][2]

    assert s3_key.startswith("marketdata/eow-snapshot/")
    assert "year=2026/month=01" in s3_key
    assert "day=" not in s3_key
    assert "hour=" not in s3_key


# ── Test Case 2d: Monthly lookback steps one calendar month ─────────

@patch.object(_mod, "_upload_to_s3")
@patch.object(_mod, "_load_snapshot_file")
@patch.object(_mod, "_load_partition")
def test_monthly_previous_snapshot_is_prior_calendar_month(mock_load, mock_load_prev, mock_upload):
    """43200-min window: previous snapshot key resolves to the prior month.

    With end_dt = 2026-03-01, the previous snapshot must land in Feb 2026
    (not Jan, which a naive 30-day subtraction from March would produce).
    """
    end_dt = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    mock_load.return_value = _make_df(CURRENT_DATA)
    mock_load_prev.return_value = _make_df(PREVIOUS_DATA)

    create_snapshot(FX_SYMBOL, INTERVAL, end_dt, 43200, MagicMock(), BUCKET)

    current_prefix = mock_load.call_args[0][2]
    previous_snapshot_key = mock_load_prev.call_args[0][2]

    assert "year=2026/month=03" in current_prefix
    assert previous_snapshot_key.startswith("marketdata/eom-snapshot/")
    assert "year=2026/month=02" in previous_snapshot_key


# ── Test Case 3: Missing previous snapshot (first run in chain) ─────

@patch.object(_mod, "_upload_to_s3")
@patch.object(_mod, "_load_snapshot_file")
@patch.object(_mod, "_load_partition")
def test_missing_previous_snapshot(mock_load, mock_load_prev, mock_upload):
    """When previous snapshot is empty (first run), output contains only current data."""
    current_only = [{"Timestamp": TS, "Symbol": "EURUSD", "price": 1.10}]
    mock_load.return_value = _make_df(current_only)
    mock_load_prev.return_value = _make_df(None)

    df = create_snapshot(FX_SYMBOL, INTERVAL, EXECUTION_DT, 60, MagicMock(), BUCKET)

    assert mock_upload.call_count == 1
    uploaded_df = mock_upload.call_args[0][0]
    assert len(uploaded_df) == 1


# ── Test Case 4: Empty current data ─────────────────────────────────

@patch.object(_mod, "_upload_to_s3")
@patch.object(_mod, "_load_snapshot_file")
@patch.object(_mod, "_load_partition")
def test_empty_current_falls_back_to_previous(mock_load, mock_load_prev, mock_upload):
    """When current partition is empty, falls back to previous snapshot."""
    previous_only = [
        {"Timestamp": TS, "Symbol": "EURUSD", "price": 1.00},
        {"Timestamp": TS, "Symbol": "GBPUSD", "price": 1.30},
    ]
    mock_load.return_value = _make_df(None)
    mock_load_prev.return_value = _make_df(previous_only)

    df = create_snapshot(FX_SYMBOL, INTERVAL, EXECUTION_DT, 60, MagicMock(), BUCKET)

    assert mock_upload.call_count == 1
    uploaded_df = mock_upload.call_args[0][0]
    assert len(uploaded_df) == 2
    assert not uploaded_df.empty


# ── Test Case 5: Invalid TIME_WINDOW_IN_MINUTES ─────────────────────

def test_invalid_time_window_raises_value_error():
    """TIME_WINDOW_IN_MINUTES=30 should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid TIME_WINDOW_IN_MINUTES=30"):
        create_snapshot(FX_SYMBOL, INTERVAL, EXECUTION_DT, 30, MagicMock(), BUCKET)


# ── Test Case 6: Correct merge of current and previous partitions ────

@patch.object(_mod, "_upload_to_s3")
@patch.object(_mod, "_load_snapshot_file")
@patch.object(_mod, "_load_partition")
def test_merge_current_overrides_previous_preserves_unique(mock_load, mock_load_prev, mock_upload):
    """Verifies merge: current overrides previous for duplicates, unique rows kept."""
    prev = [
        {"Timestamp": TS, "Symbol": "EURUSD", "price": 1.00},
        {"Timestamp": TS, "Symbol": "GBPUSD", "price": 1.30},
        {"Timestamp": TS, "Symbol": "AUDUSD", "price": 0.65},
    ]
    curr = [
        {"Timestamp": TS, "Symbol": "EURUSD", "price": 1.10},
        {"Timestamp": TS, "Symbol": "USDJPY", "price": 110},
        {"Timestamp": TS, "Symbol": "GBPUSD", "price": 1.35},
    ]
    mock_load.return_value = _make_df(curr)
    mock_load_prev.return_value = _make_df(prev)

    df = create_snapshot(FX_SYMBOL, INTERVAL, EXECUTION_DT, 60, MagicMock(), BUCKET)

    uploaded_df = mock_upload.call_args[0][0]

    # 4 unique symbols after merge
    assert len(uploaded_df) == 4
    symbols = set(uploaded_df["Symbol"].tolist())
    assert symbols == {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD"}

    # Current partition values win for overlapping symbols
    eurusd = uploaded_df.loc[uploaded_df["Symbol"] == "EURUSD", "price"].iloc[0]
    gbpusd = uploaded_df.loc[uploaded_df["Symbol"] == "GBPUSD", "price"].iloc[0]
    assert eurusd == 1.10
    assert gbpusd == 1.35

    # Previous-only symbol preserved
    audusd = uploaded_df.loc[uploaded_df["Symbol"] == "AUDUSD", "price"].iloc[0]
    assert audusd == 0.65


# ── Test Case 7: Large data volume and memory management ─────────────

@patch.object(_mod, "_upload_to_s3")
@patch.object(_mod, "_load_snapshot_file")
@patch.object(_mod, "_load_partition")
def test_large_data_volume_merge(mock_load, mock_load_prev, mock_upload):
    """Verifies correct handling of large DataFrames during merge."""
    n = 50_000
    rng = np.random.default_rng(42)

    timestamps = pd.date_range("2026-01-01 09:00", periods=n, freq="s", tz="UTC")
    symbols = [f"SYM{i}" for i in range(n)]

    prev_df = pd.DataFrame({
        "Timestamp": timestamps,
        "Symbol": symbols,
        "price": rng.uniform(1.0, 200.0, size=n),
    })

    # Current partition: overlaps first 10k symbols with new prices, plus 10k new symbols
    overlap = 10_000
    new_count = 10_000
    curr_timestamps = list(timestamps[:overlap]) + list(
        pd.date_range("2026-01-01 10:00", periods=new_count, freq="s", tz="UTC")
    )
    curr_symbols = symbols[:overlap] + [f"NEW{i}" for i in range(new_count)]
    curr_df = pd.DataFrame({
        "Timestamp": curr_timestamps,
        "Symbol": curr_symbols,
        "price": rng.uniform(1.0, 200.0, size=overlap + new_count),
    })

    mock_load.return_value = curr_df
    mock_load_prev.return_value = prev_df

    df = create_snapshot(FX_SYMBOL, INTERVAL, EXECUTION_DT, 60, MagicMock(), BUCKET)

    assert mock_upload.call_count == 1
    uploaded_df = mock_upload.call_args[0][0]

    # 50k from previous - 10k overlapping + 10k overlapping from current + 10k new = 60k
    expected_rows = (n - overlap) + overlap + new_count
    assert len(uploaded_df) == expected_rows

    # Verify no duplicate (Timestamp, Symbol) pairs
    dupes = uploaded_df.duplicated(subset=["Timestamp", "Symbol"])
    assert not dupes.any()

    # Timestamps should be sorted
    assert uploaded_df["Timestamp"].is_monotonic_increasing
