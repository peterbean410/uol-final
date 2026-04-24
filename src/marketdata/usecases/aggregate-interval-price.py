"""
Aggregate source interval price bars (default M1) stored under
marketdata/interval-price/... into a larger target interval and upload
the result back to the same tree keyed by interval={TARGET_INTERVAL}.

Usage:
    python marketdata/usecases/aggregate-interval-price.py

Environment variables:
    FX_SYMBOL: The FX symbol (e.g. 'USDJPY')
    SOURCE_INTERVAL: Source interval string. Defaults to 'M1'.
    TARGET_INTERVAL: Target interval string (e.g. 'M5', 'M15', 'H1', 'D1')
    EXECUTION_TS: ISO-8601 end timestamp of the scheduled window (UTC)
    TIME_WINDOW_IN_MINUTES: Output partition granularity (also the lookback window)
"""

import io
import os
from datetime import datetime, timedelta, timezone
from typing import Iterable

import boto3
import pandas as pd

from commons.python.appconfig import AppConfig

INTERVAL_TO_MINUTES = {
    "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M10": 10,
    "M15": 15, "M30": 30, "H1": 60, "H4": 240, "H12": 720, "D1": 1440,
    "W1": 10080,
}

DAILY_MINUTES = 1440
SOURCE_PARTITION_MINUTES = 60  # source bars are partitioned per 1-hour folder


def _iter_source_prefixes(
    fx_symbol: str, source_interval: str, start_dt: datetime, end_dt: datetime,
) -> Iterable[str]:
    """Yield every hourly source S3 prefix that overlaps [start_dt, end_dt)."""
    base = f"marketdata/interval-price/symbol={fx_symbol}/interval={source_interval}"
    # Floor start to the hour so we catch the partition containing start_dt.
    cursor = start_dt.replace(minute=0, second=0, microsecond=0)
    while cursor < end_dt:
        yield (
            f"{base}/year={cursor.year}/month={cursor.month:02d}"
            f"/day={cursor.day:02d}/hour={cursor.hour:02d}/"
        )
        cursor += timedelta(minutes=SOURCE_PARTITION_MINUTES)


def _load_partitions(s3, bucket: str, prefixes: Iterable[str]) -> pd.DataFrame:
    """Load all parquet files under the given S3 prefixes into one DataFrame."""
    frames = []
    for prefix in prefixes:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        keys = [
            obj["Key"] for obj in resp.get("Contents", [])
            if obj["Key"].endswith(".parquet")
        ]
        if not keys:
            print(f"No parquet files under s3://{bucket}/{prefix}")
            continue
        for key in keys:
            obj = s3.get_object(Bucket=bucket, Key=key)
            frames.append(pd.read_parquet(io.BytesIO(obj["Body"].read())))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _aggregate(df: pd.DataFrame, fx_symbol: str, target_minutes: int) -> pd.DataFrame:
    """Resample OHLCV bars into `target_minutes` buckets on the epoch grid."""
    if df.empty:
        return df

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df = df.set_index("Timestamp").sort_index()

    agg = df.resample(
        f"{target_minutes}min",
        label="left",
        closed="left",
        origin="epoch",
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })

    agg = agg.dropna(subset=["Open"])
    agg["Symbol"] = fx_symbol
    agg = agg.reset_index()
    return agg[["Timestamp", "Symbol", "Open", "High", "Low", "Close", "Volume"]]


def _build_output_key(
    fx_symbol: str, target_interval: str, end_dt: datetime, time_window_minutes: int,
) -> str:
    """Build the output S3 key using the three-tier interval-price layout."""
    ts = end_dt.strftime("%Y%m%dT%H%M%SZ")
    base = f"marketdata/interval-price/symbol={fx_symbol}/interval={target_interval}"
    year_month = f"year={end_dt.year}/month={end_dt.month:02d}"

    if time_window_minutes > DAILY_MINUTES:
        return f"{base}/{year_month}/{ts}.parquet"
    if time_window_minutes == DAILY_MINUTES:
        return f"{base}/{year_month}/day={end_dt.day:02d}/{ts}.parquet"
    return (
        f"{base}/{year_month}/day={end_dt.day:02d}"
        f"/hour={end_dt.hour:02d}/{ts}.parquet"
    )


def _upload_to_s3(df: pd.DataFrame, bucket: str, key: str, s3) -> None:
    """Upload a DataFrame as a Parquet file to S3."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"Uploaded to s3://{bucket}/{key}")


def aggregate(
    fx_symbol: str,
    source_interval: str,
    target_interval: str,
    end_dt: datetime,
    time_window_minutes: int,
    s3,
    bucket: str,
) -> pd.DataFrame:
    """Aggregate source bars into target interval and upload to S3."""
    if source_interval not in INTERVAL_TO_MINUTES:
        raise ValueError(
            f"Unknown SOURCE_INTERVAL='{source_interval}'. "
            f"Choose from: {sorted(INTERVAL_TO_MINUTES.keys())}"
        )
    if target_interval not in INTERVAL_TO_MINUTES:
        raise ValueError(
            f"Unknown TARGET_INTERVAL='{target_interval}'. "
            f"Choose from: {sorted(INTERVAL_TO_MINUTES.keys())}"
        )

    source_minutes = INTERVAL_TO_MINUTES[source_interval]
    target_minutes = INTERVAL_TO_MINUTES[target_interval]
    if target_minutes % source_minutes != 0 or target_minutes < source_minutes:
        raise ValueError(
            f"TARGET_INTERVAL='{target_interval}' ({target_minutes}m) must be a "
            f"multiple of SOURCE_INTERVAL='{source_interval}' ({source_minutes}m)."
        )

    start_dt = end_dt - timedelta(minutes=time_window_minutes)
    prefixes = list(_iter_source_prefixes(fx_symbol, source_interval, start_dt, end_dt))
    df = _load_partitions(s3, bucket, prefixes)

    if not df.empty:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        mask = (df["Timestamp"] >= pd.Timestamp(start_dt)) & (df["Timestamp"] < pd.Timestamp(end_dt))
        df = df.loc[mask]
        df = df.drop_duplicates(subset=["Timestamp", "Symbol"], keep="last")

    aggregated = _aggregate(df, fx_symbol, target_minutes)

    if aggregated.empty:
        print("No data to aggregate.")
        return aggregated

    key = _build_output_key(fx_symbol, target_interval, end_dt, time_window_minutes)
    _upload_to_s3(aggregated, bucket, key, s3)
    print(f"Aggregated {len(aggregated)} rows")
    print(aggregated.tail(15).to_string(index=False))
    return aggregated


if __name__ == "__main__":
    config = AppConfig()
    fx_symbol = os.environ.get("FX_SYMBOL", "USDJPY")
    source_interval = os.environ.get("SOURCE_INTERVAL", "M1")
    target_interval = os.environ.get("TARGET_INTERVAL", "M15")
    execution_ts = os.environ.get("EXECUTION_TS")
    time_window = int(os.environ.get("TIME_WINDOW_IN_MINUTES", "60"))

    print(f"FX_SYMBOL={fx_symbol}")
    print(f"SOURCE_INTERVAL={source_interval}")
    print(f"TARGET_INTERVAL={target_interval}")
    print(f"EXECUTION_TS={execution_ts}")
    print(f"TIME_WINDOW_IN_MINUTES={time_window}")

    end_dt = (
        datetime.fromisoformat(execution_ts).astimezone(timezone.utc)
        if execution_ts
        else datetime.now(tz=timezone.utc)
    )

    s3 = boto3.client("s3")
    aggregate(
        fx_symbol, source_interval, target_interval, end_dt, time_window,
        s3, config.s3_bucket,
    )
