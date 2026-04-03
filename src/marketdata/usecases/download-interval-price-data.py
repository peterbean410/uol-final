"""
Download interval price data for an FX symbol, upload to S3, and print the latest rows.

Usage:
    python marketdata/usecases/download-interval-price-data.py

Environment variables:
    FX_SYMBOL: The FX symbol (e.g. 'USDJPY')
    INTERVAL: The interval string (e.g. 'M15', 'D1', 'ticks')
    EXECUTION_TS: ISO-8601 end timestamp for the data window
    TIME_WINDOW_IN_MINUTES: How far back from EXECUTION_TS to fetch
"""

import io
import os
from datetime import datetime, timezone

import boto3
import pandas as pd

from commons.python.appconfig import AppConfig
from marketdata.pricedata.price_bars import get_price_bars
from marketdata.pricedata.price_ticks import get_price_ticks

INTERVAL_TO_MINUTES = {
    "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M10": 10,
    "M15": 15, "M30": 30, "H1": 60, "H4": 240, "H12": 720, "D1": 1440,
}


def _build_s3_key(
    fx_symbol: str, interval: str, end_dt: datetime, time_window_minutes: int,
) -> str:
    """Build the S3 object key based on the time window granularity."""
    ts = end_dt.strftime("%Y%m%dT%H%M%SZ")
    base = f"marketdata/interval-price/symbol={fx_symbol}/interval={interval}"

    if time_window_minutes >= 1440:
        return f"{base}/year={end_dt.year}/month={end_dt.month:02d}/day={end_dt.day:02d}/{ts}.parquet"

    return f"{base}/year={end_dt.year}/month={end_dt.month:02d}/day={end_dt.day:02d}/hour={end_dt.hour:02d}/{ts}.parquet"


def _upload_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    """Upload a DataFrame as a Parquet file to S3."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)

    s3 = boto3.client("s3")
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

    if interval == "ticks":
        df = get_price_ticks(fx_symbol)
    else:
        minutes = INTERVAL_TO_MINUTES.get(interval)
        if minutes is None:
            raise ValueError(
                f"Unsupported INTERVAL='{interval}'. "
                f"Choose from: {sorted(INTERVAL_TO_MINUTES.keys())} or 'ticks'"
            )
        df = get_price_bars(
            fx_symbol,
            time_interval=minutes,
            end_dt=end_dt,
            time_window_minutes=time_window,
        )

    if df.empty:
        print("No data returned.")
    else:
        s3_key = _build_s3_key(fx_symbol, interval, end_dt, time_window)
        _upload_to_s3(df, config.s3_bucket, s3_key)
        print(df.tail(15).to_string(index=False))
