"""
Create an end-of-interval (EOI) TA snapshot by loading the current price
snapshot, computing all technical indicators and patterns, and uploading
the result to S3.

Usage:
    python ta/usecases/create-eoi-ta-snapshot.py

Environment variables:
    FX_SYMBOL: The FX symbol (e.g. 'USDJPY')
    INTERVAL: The interval string (e.g. 'M15')
    EXECUTION_TS: ISO-8601 timestamp marking the end of the scheduled window
    TIME_WINDOW_IN_MINUTES: The partition granularity in minutes (e.g. 60, 1440, 43200)
"""

import io
import os
from datetime import datetime, timezone

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from ta.momentum import cci, rsi
from ta.patterns import doublebottom, doubletop
from ta.support import fr
from ta.trend import adx, ic, macd, movingavg
from ta.volatility import bb

# ── Time-window constants (mirrored from create-eoi-price-snapshot) ──
HOUR_MINUTES = 60
DAILY_MINUTES = 1440
WEEK_MINUTES = 10080
MONTH_MINUTES = 43200
VALID_TIME_WINDOWS = {HOUR_MINUTES, DAILY_MINUTES, WEEK_MINUTES, MONTH_MINUTES}


def _price_snapshot_root(time_window_minutes: int) -> str:
    if time_window_minutes == MONTH_MINUTES:
        return "marketdata/eom-snapshot"
    if time_window_minutes == WEEK_MINUTES:
        return "marketdata/eow-snapshot"
    if time_window_minutes == HOUR_MINUTES:
        return "marketdata/eoh-snapshot"
    return "marketdata/eod-snapshot"


def _ta_snapshot_root(time_window_minutes: int) -> str:
    if time_window_minutes == MONTH_MINUTES:
        return "ta/eom-ta-snapshot"
    if time_window_minutes == WEEK_MINUTES:
        return "ta/eow-ta-snapshot"
    if time_window_minutes == HOUR_MINUTES:
        return "ta/eoh-ta-snapshot"
    return "ta/eod-ta-snapshot"


def _build_snapshot_key(root: str, fx_symbol: str, interval: str, dt: datetime,
                        time_window_minutes: int) -> str:
    ts = dt.strftime("%Y%m%dT%H%M%SZ")
    base = f"{root}/symbol={fx_symbol}/interval={interval}"
    key = f"{base}/year={dt.year}/month={dt.month:02d}"
    if time_window_minutes > DAILY_MINUTES:
        return f"{key}/{ts}.parquet"
    key += f"/day={dt.day:02d}"
    if time_window_minutes < DAILY_MINUTES:
        key += f"/hour={dt.hour:02d}"
    return f"{key}/{ts}.parquet"


def _load_snapshot_file(s3, bucket: str, key: str) -> pd.DataFrame:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            print(f"No snapshot at s3://{bucket}/{key}")
            return pd.DataFrame()
        raise
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _upload_to_s3(df: pd.DataFrame, bucket: str, key: str, s3) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"Uploaded {len(df)} rows to s3://{bucket}/{key}")


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Chain all TA indicators, appending columns to a copy of *df*."""
    df = adx.compute(df)
    df = cci.compute(df)
    df = macd.compute(df)
    df = movingavg.compute(df)
    df = rsi.compute(df)
    df = bb.compute(df)
    df = ic.compute(df)
    df = fr.compute(df)
    return df


def create_ta_snapshot(fx_symbol: str, interval: str, end_dt: datetime,
                       time_window_minutes: int, s3, bucket: str) -> pd.DataFrame:
    """Load the current price snapshot, compute indicators and patterns, and
    upload the TA snapshot to S3.

    Raises
    ------
    ValueError
        If *time_window_minutes* is not one of the valid window sizes.
    """
    if time_window_minutes not in VALID_TIME_WINDOWS:
        raise ValueError(
            f"Invalid TIME_WINDOW_IN_MINUTES={time_window_minutes}. "
            f"Must be one of {sorted(VALID_TIME_WINDOWS)}."
        )

    price_root = _price_snapshot_root(time_window_minutes)
    price_key = _build_snapshot_key(price_root, fx_symbol, interval, end_dt, time_window_minutes)
    print(f"Loading price snapshot: {price_key}")
    df = _load_snapshot_file(s3, bucket, price_key)

    if df.empty:
        print(f"No price snapshot found at s3://{bucket}/{price_key}")
        return df

    df = _compute_indicators(df)

    ta_root = _ta_snapshot_root(time_window_minutes)
    ta_key = _build_snapshot_key(ta_root, fx_symbol, interval, end_dt, time_window_minutes)
    _upload_to_s3(df, bucket, ta_key, s3)

    # ── Patterns (separate parquet files alongside the main snapshot) ──
    prefix = ta_key.rsplit("/", 1)[0]
    ts = end_dt.strftime("%Y%m%dT%H%M%SZ")

    db_patterns, _, _ = doublebottom.detect_double_bottoms(df)
    if not db_patterns.empty:
        _upload_to_s3(db_patterns, bucket, f"{prefix}/{ts}_doublebottom.parquet", s3)

    dt_patterns, _, _ = doubletop.detect_double_tops(df)
    if not dt_patterns.empty:
        _upload_to_s3(dt_patterns, bucket, f"{prefix}/{ts}_doubletop.parquet", s3)

    print(df.tail(10).to_string(index=False))
    return df


if __name__ == "__main__":
    fx_symbol = os.environ.get("FX_SYMBOL", "USDJPY")
    interval = os.environ.get("INTERVAL", "M15")
    execution_ts = os.environ.get("EXECUTION_TS")
    time_window = int(os.environ.get("TIME_WINDOW_IN_MINUTES", "60"))
    bucket = os.environ.get("S3_BUCKET", "")

    end_dt = (
        datetime.fromisoformat(execution_ts).astimezone(timezone.utc)
        if execution_ts
        else datetime.now(tz=timezone.utc)
    )

    s3 = boto3.client("s3")

    create_ta_snapshot(fx_symbol, interval, end_dt, time_window, s3, bucket)
