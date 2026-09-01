"""
Create an end-of-day (EOD) tick snapshot by merging the previous trading
day's tick partitions with the current day's (so-far) tick partitions and
uploading the cumulative result to S3.

Usage:
    python marketdata/usecases/create-eod-tick-snapshot.py

Environment variables:
    FX_SYMBOL: The FX symbol (e.g. 'USDJPY')
    EXECUTION_TS: ISO-8601 timestamp marking the end of the scheduled window (UTC)
"""

import io
import os
import tracemalloc
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from commons.python.appconfig import AppConfig

tracemalloc.start()


def _log_memory(label: str) -> None:
    current, peak = tracemalloc.get_traced_memory()
    print(f"[MEM] {label}: current={current / 1024 / 1024:.1f}MB peak={peak / 1024 / 1024:.1f}MB")

TICK_DEDUP_KEYS = ["Timestamp", "Symbol", "Type"]


def _partition_prefix(fx_symbol: str, dt: datetime) -> str:
    """S3 prefix covering every hour partition of ticks for the given calendar day."""
    return (
        f"marketdata/interval-price/symbol={fx_symbol}/interval=ticks"
        f"/year={dt.year}/month={dt.month:02d}/day={dt.day:02d}/"
    )


def _load_partition(s3, bucket: str, prefix: str) -> pd.DataFrame:
    """Load every parquet file under an S3 prefix (paginated) into one DataFrame."""
    frames = []
    paginator = s3.get_paginator("list_objects_v2")
    empty = True
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            empty = False
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            frames.append(pd.read_parquet(io.BytesIO(body)))

    if empty:
        print(f"No parquet files found under s3://{bucket}/{prefix}")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_snapshot_file(s3, bucket: str, key: str) -> pd.DataFrame:
    """Load a single snapshot Parquet file by exact key.

    Returns an empty DataFrame if the object does not exist (e.g. very
    first run in the chain). Any other S3 error is re-raised.
    """
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            print(f"No previous snapshot at s3://{bucket}/{key}")
            return pd.DataFrame()
        raise
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _build_snapshot_key(fx_symbol: str, dt: datetime) -> str:
    """Snapshot key at `marketdata/eod-tick-snapshot/.../year/month/day/{ts}.parquet`."""
    ts = dt.strftime("%Y%m%dT%H%M%SZ")
    return (
        f"marketdata/eod-tick-snapshot/symbol={fx_symbol}"
        f"/year={dt.year}/month={dt.month:02d}/day={dt.day:02d}/{ts}.parquet"
    )


def _upload_to_s3(df: pd.DataFrame, bucket: str, key: str, s3) -> None:
    """Upload a DataFrame as a Parquet file to S3."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"Uploaded to s3://{bucket}/{key}")


def create_snapshot(fx_symbol: str, end_dt: datetime, s3, bucket: str) -> pd.DataFrame:
    """Merge the current day's raw tick partition with the previous EOD tick
    snapshot so each snapshot accumulates history (snapshot[N] = raw[N] ∪ snapshot[N-1])."""
    current_prefix = _partition_prefix(fx_symbol, end_dt)
    print(f"Loading current partition: {current_prefix}")
    df_current = _load_partition(s3, bucket, current_prefix)
    _log_memory("after loading current partition")

    prev_snapshot_key = _build_snapshot_key(fx_symbol, end_dt - timedelta(days=1))
    print(f"Loading previous snapshot: {prev_snapshot_key}")
    df_previous = _load_snapshot_file(s3, bucket, prev_snapshot_key)
    _log_memory("after loading previous snapshot")

    df = pd.concat([df_previous, df_current], ignore_index=True)
    _log_memory("after concat")

    if df.empty:
        print("No data found in either partition.")
        return df

    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).astype("int64")

    df.drop_duplicates(subset=TICK_DEDUP_KEYS, keep="last", inplace=True)
    _log_memory("after dedup")

    cutoff = pd.Timestamp(end_dt) - pd.DateOffset(months=24)
    before = len(df)
    df = df[df["Timestamp"] >= cutoff.value]
    _log_memory(f"after 24m trim (dropped {before - len(df)} rows)")

    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    _log_memory("after sort + reset_index")

    key = _build_snapshot_key(fx_symbol, end_dt)
    _upload_to_s3(df, bucket, key, s3)
    print(f"Snapshot contains {len(df)} rows")
    print(df.tail(15).to_string(index=False))
    return df


if __name__ == "__main__":
    config = AppConfig()
    fx_symbol = os.environ.get("FX_SYMBOL", "USDJPY")
    execution_ts = os.environ.get("EXECUTION_TS")

    print(f"FX_SYMBOL={fx_symbol}")
    print(f"EXECUTION_TS={execution_ts}")

    end_dt = (
        datetime.fromisoformat(execution_ts).astimezone(timezone.utc)
        if execution_ts
        else datetime.now(tz=timezone.utc)
    )

    s3 = boto3.client("s3")
    create_snapshot(fx_symbol, end_dt, s3, config.s3_bucket)
