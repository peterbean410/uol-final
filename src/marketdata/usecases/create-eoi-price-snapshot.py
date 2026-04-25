"""
Create an end-of-interval (EOI) price snapshot by merging two consecutive
interval-price partitions and uploading the result to S3.

Usage:
    python marketdata/usecases/create-eoi-price-snapshot.py

Environment variables:
    FX_SYMBOL: The FX symbol (e.g. 'USDJPY')
    INTERVAL: The interval string (e.g. 'M15', 'H1', 'D1')
    EXECUTION_TS: ISO-8601 timestamp marking the end of the scheduled window
    TIME_WINDOW_IN_MINUTES: The partition granularity in minutes (e.g. 60, 1440, 43200)
"""

import io
import os
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from dateutil.relativedelta import relativedelta

from commons.python.appconfig import AppConfig

HOUR_MINUTES = 60
DAILY_MINUTES = 1440
WEEK_MINUTES = 10080
MONTH_MINUTES = 43200
VALID_TIME_WINDOWS = {HOUR_MINUTES, DAILY_MINUTES, WEEK_MINUTES, MONTH_MINUTES}


def _partition_prefix(fx_symbol: str, interval: str, dt: datetime, time_window_minutes: int) -> str:
    """Build the S3 prefix for an interval-price partition.

    Matches the three-tier layout produced by download-interval-price-data.py:
      - > 1440 min: year/month
      - == 1440 min: year/month/day
      - < 1440 min: year/month/day/hour
    """
    base = f"marketdata/interval-price/symbol={fx_symbol}/interval={interval}"
    prefix = f"{base}/year={dt.year}/month={dt.month:02d}"
    if time_window_minutes > DAILY_MINUTES:
        return prefix + "/"
    prefix += f"/day={dt.day:02d}"
    if time_window_minutes < DAILY_MINUTES:
        prefix += f"/hour={dt.hour:02d}"
    return prefix + "/"


def _previous_partition_dt(dt: datetime, time_window_minutes: int) -> datetime:
    """Return the datetime representing the previous partition.

    For MONTH_MINUTES we step back one calendar month instead of a fixed
    30-day timedelta, so the prefix resolves to the prior month's
    `year=/month=` tier (which is how MN1 aggregates are keyed).
    """
    if time_window_minutes == MONTH_MINUTES:
        return dt - relativedelta(months=1)
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


def _load_snapshot_file(s3, bucket: str, key: str) -> pd.DataFrame:
    """Load a single snapshot Parquet file by exact key.

    Returns an empty DataFrame if the object does not exist (e.g. very first
    run in the chain). Any other S3 error is re-raised.
    """
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            print(f"No previous snapshot at s3://{bucket}/{key}")
            return pd.DataFrame()
        raise
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _snapshot_root(time_window_minutes: int) -> str:
    """Return the snapshot tree root keyed by partition granularity."""
    if time_window_minutes == MONTH_MINUTES:
        return "marketdata/eom-snapshot"
    if time_window_minutes == WEEK_MINUTES:
        return "marketdata/eow-snapshot"
    return "marketdata/eod-snapshot"


def _build_snapshot_key(fx_symbol: str, interval: str, dt: datetime, time_window_minutes: int) -> str:
    """Build the S3 key for the EOI snapshot.

    - MONTH_MINUTES: marketdata/eom-snapshot/.../year/month/{ts}.parquet
    - WEEK_MINUTES:  marketdata/eow-snapshot/.../year/month/{ts}.parquet
    - DAILY_MINUTES: marketdata/eod-snapshot/.../year/month/day/{ts}.parquet
    - HOUR_MINUTES:  marketdata/eod-snapshot/.../year/month/day/hour/{ts}.parquet
    """
    ts = dt.strftime("%Y%m%dT%H%M%SZ")
    base = f"{_snapshot_root(time_window_minutes)}/symbol={fx_symbol}/interval={interval}"
    key = f"{base}/year={dt.year}/month={dt.month:02d}"

    if time_window_minutes > DAILY_MINUTES:
        return f"{key}/{ts}.parquet"

    key += f"/day={dt.day:02d}"
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


def create_snapshot(fx_symbol: str, interval: str, end_dt: datetime,
                    time_window_minutes: int, s3, bucket: str) -> pd.DataFrame:
    """Create an EOI snapshot by merging the current raw partition with the
    previously-written snapshot, so each snapshot accumulates history.

    Raises:
        ValueError: If time_window_minutes is not 60 or 1440 or 10080 or 43200.
    """
    if time_window_minutes not in VALID_TIME_WINDOWS:
        raise ValueError(
            f"Invalid TIME_WINDOW_IN_MINUTES={time_window_minutes}. Must be one of {sorted(VALID_TIME_WINDOWS)}."
        )

    current_prefix = _partition_prefix(fx_symbol, interval, end_dt, time_window_minutes)
    print(f"Loading current partition: {current_prefix}")
    df_current = _load_partition(s3, bucket, current_prefix)

    prev_dt = _previous_partition_dt(end_dt, time_window_minutes)
    prev_snapshot_key = _build_snapshot_key(fx_symbol, interval, prev_dt, time_window_minutes)
    print(f"Loading previous snapshot: {prev_snapshot_key}")
    df_previous = _load_snapshot_file(s3, bucket, prev_snapshot_key)

    df = pd.concat([df_previous, df_current], ignore_index=True)

    if df.empty:
        print("No data found in either partition.")
        return df

    df.drop_duplicates(subset=["Timestamp", "Symbol"], keep="last", inplace=True)
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    snapshot_key = _build_snapshot_key(fx_symbol, interval, end_dt, time_window_minutes)
    _upload_to_s3(df, bucket, snapshot_key, s3)
    print(f"Snapshot contains {len(df)} rows")
    print(df.tail(15).to_string(index=False))

    return df


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

    create_snapshot(fx_symbol, interval, end_dt, time_window, s3, bucket)
