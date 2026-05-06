"""
Create an end-of-day (EOD) news snapshot by merging the previous trading
day's news snapshot with the current day's raw news partition and
uploading the cumulative result to S3.

Equivalent to create-eod-tick-snapshot.py but for interval-news data.

Usage:
    python marketdata/usecases/create-eod-news-snapshot.py

Environment variables:
    FX_PAIR: The FX pair in dash format (e.g. 'USD-JPY')
    EXECUTION_TS: ISO-8601 timestamp marking the end of the scheduled window (UTC)
"""

import io
import os
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from commons.python.appconfig import AppConfig

NEWS_DEDUP_KEYS = ["news_url"]


def _raw_partition_key(fx_pair: str, end_dt: datetime) -> str:
    """S3 key for the daily raw news partition."""
    ts = end_dt.strftime("%Y%m%dT%H%M%SZ")
    return (
        f"marketdata/interval-news/symbol={fx_pair}/interval=D1"
        f"/year={end_dt.year}/month={end_dt.month:02d}/day={end_dt.day:02d}/{ts}.parquet"
    )


def _load_parquet_file(s3, bucket: str, key: str) -> pd.DataFrame:
    """Load a single parquet file by exact key.

    Returns an empty DataFrame if the object does not exist.
    """
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            print(f"No file at s3://{bucket}/{key}")
            return pd.DataFrame()
        raise
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _snapshot_key(fx_pair: str, dt: datetime) -> str:
    """EOD news snapshot key."""
    ts = dt.strftime("%Y%m%dT%H%M%SZ")
    return (
        f"marketdata/eod-news-snapshot/symbol={fx_pair}"
        f"/year={dt.year}/month={dt.month:02d}/day={dt.day:02d}/{ts}.parquet"
    )


def _upload_to_s3(df: pd.DataFrame, bucket: str, key: str, s3) -> None:
    """Upload a DataFrame as a Parquet file to S3."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"Uploaded to s3://{bucket}/{key}")


def create_snapshot(fx_pair: str, end_dt: datetime, s3, bucket: str) -> pd.DataFrame:
    """Merge the current day's raw news partition with the previous EOD news
    snapshot so each snapshot accumulates history (N = raw[N] + snapshot[N-1])."""
    current_key = _raw_partition_key(fx_pair, end_dt)
    print(f"Loading current partition: {current_key}")
    df_current = _load_parquet_file(s3, bucket, current_key)

    prev_key = _snapshot_key(fx_pair, end_dt - timedelta(days=1))
    print(f"Loading previous snapshot: {prev_key}")
    df_previous = _load_parquet_file(s3, bucket, prev_key)

    df = pd.concat([df_previous, df_current], ignore_index=True)

    if df.empty:
        print("No news data found in either source.")
        return df

    df.drop_duplicates(subset=NEWS_DEDUP_KEYS, keep="last", inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    key = _snapshot_key(fx_pair, end_dt)
    _upload_to_s3(df, bucket, key, s3)
    print(f"Snapshot contains {len(df)} rows")
    print(df.tail(15).to_string(index=False))
    return df


if __name__ == "__main__":
    config = AppConfig()
    fx_pair = os.environ.get("FX_PAIR", "USD-JPY")
    execution_ts = os.environ.get("EXECUTION_TS")

    print(f"FX_PAIR={fx_pair}")
    print(f"EXECUTION_TS={execution_ts}")

    end_dt = (
        datetime.fromisoformat(execution_ts).astimezone(timezone.utc)
        if execution_ts
        else datetime.now(tz=timezone.utc)
    )

    s3 = boto3.client("s3")
    create_snapshot(fx_pair, end_dt, s3, config.s3_bucket)
