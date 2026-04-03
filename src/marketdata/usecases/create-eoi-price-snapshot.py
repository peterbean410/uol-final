"""
Create an end-of-interval (EOI) price snapshot by merging two consecutive
interval-price partitions and uploading the result to S3.

Usage:
    python marketdata/usecases/create-eoi-price-snapshot.py

Environment variables:
    FX_SYMBOL: The FX symbol (e.g. 'USDJPY')
    INTERVAL: The interval string (e.g. 'M15', 'H1', 'D1')
    EXECUTION_TS: ISO-8601 timestamp marking the end of the scheduled window
    TIME_WINDOW_IN_MINUTES: The partition granularity in minutes (e.g. 60, 1440)
"""

import io
import os
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd

from commons.python.appconfig import AppConfig

DAILY_MINUTES = 1440


def _partition_prefix(fx_symbol: str, interval: str, dt: datetime, time_window_minutes: int) -> str:
    """Build the S3 prefix for an interval-price partition."""
    base = f"marketdata/interval-price/symbol={fx_symbol}/interval={interval}"
    prefix = f"{base}/year={dt.year}/month={dt.month:02d}/day={dt.day:02d}"
    if time_window_minutes < DAILY_MINUTES:
        prefix += f"/hour={dt.hour:02d}"
    return prefix + "/"


def _previous_partition_dt(dt: datetime, time_window_minutes: int) -> datetime:
    """Return the datetime representing the previous partition."""
    return dt - timedelta(minutes=time_window_minutes)


def _load_partition(s3, bucket: str, prefix: str) -> pd.DataFrame:
    """Load all parquet files under an S3 prefix into a single DataFrame."""
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = resp.get("Contents", [])
    parquet_keys = [obj["Key"] for obj in contents if obj["Key"].endswith(".parquet")]

    if not parquet_keys:
        print(f"No parquet files found under s3://{bucket}/{prefix}")
        return pd.DataFrame()

    frames = []
    for key in parquet_keys:
        obj = s3.get_object(Bucket=bucket, Key=key)
        frames.append(pd.read_parquet(io.BytesIO(obj["Body"].read())))

    return pd.concat(frames, ignore_index=True)


def _build_snapshot_key(fx_symbol: str, interval: str, dt: datetime, time_window_minutes: int) -> str:
    """Build the S3 key for the EOI snapshot."""
    ts = dt.strftime("%Y%m%dT%H%M%SZ")
    base = f"marketdata/eod-snapshot/symbol={fx_symbol}/interval={interval}"
    key = f"{base}/year={dt.year}/month={dt.month:02d}/day={dt.day:02d}"
    if time_window_minutes < DAILY_MINUTES:
        key += f"/hour={dt.hour:02d}"
    return f"{key}/{ts}.parquet"


def _upload_to_s3(df: pd.DataFrame, bucket: str, key: str, s3) -> None:
    """Upload a DataFrame as a Parquet file to S3."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"Uploaded to s3://{bucket}/{key}")


if __name__ == "__main__":
    config = AppConfig()
    fx_symbol = os.environ.get("FX_SYMBOL", "USDJPY")
    interval = os.environ.get("INTERVAL", "M15")
    execution_ts = os.environ.get("EXECUTION_TS")
    time_window = int(os.environ.get("TIME_WINDOW_IN_MINUTES", "60"))

    end_dt = (
        datetime.fromisoformat(execution_ts).astimezone(timezone.utc)
        if execution_ts
        else datetime.now(tz=timezone.utc)
    )

    s3 = boto3.client("s3")
    bucket = config.s3_bucket

    current_prefix = _partition_prefix(fx_symbol, interval, end_dt, time_window)
    prev_dt = _previous_partition_dt(end_dt, time_window)
    prev_prefix = _partition_prefix(fx_symbol, interval, prev_dt, time_window)

    print(f"Loading current partition: {current_prefix}")
    df_current = _load_partition(s3, bucket, current_prefix)

    print(f"Loading previous partition: {prev_prefix}")
    df_previous = _load_partition(s3, bucket, prev_prefix)

    df = pd.concat([df_previous, df_current], ignore_index=True)

    if df.empty:
        print("No data found in either partition.")
    else:
        df.drop_duplicates(subset=["Timestamp", "Symbol"], inplace=True)
        df.sort_values("Timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        snapshot_key = _build_snapshot_key(fx_symbol, interval, end_dt, time_window)
        _upload_to_s3(df, bucket, snapshot_key, s3)
        print(f"Snapshot contains {len(df)} rows")
        print(df.tail(15).to_string(index=False))
