"""
Download interval news data for an FX currency pair, upload to S3, and print the latest rows.

Usage:
    python marketdata/usecases/download-interval-news-data.py

Environment variables:
    FX_CURRENCY_PAIR: The FX pair (e.g. 'USD-JPY')
    INTERVAL: The interval string (e.g. 'D1')
    EXECUTION_TS: ISO-8601 end timestamp for the scheduled window
    FXNEWS_API_KEY: forexnewsapi.com API key (read via AppConfig)
"""

import io
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd
import requests

from commons.python.appconfig import AppConfig
from marketdata.newsdata.fxnews import get_news_data

logger = logging.getLogger(__name__)

INTERVAL_TO_MINUTES = {
    "D1": 1440,
}

DATE_FORMAT = "%m%d%Y"
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1
DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 100


def _build_s3_key(fx_currency_pair: str, interval: str, end_dt: datetime) -> str:
    """Build the S3 object key for the news partition."""
    ts = end_dt.strftime("%Y%m%dT%H%M%SZ")
    return (
        f"marketdata/interval-news/symbol={fx_currency_pair}/interval={interval}/"
        f"year={end_dt.year}/month={end_dt.month:02d}/day={end_dt.day:02d}/{ts}.parquet"
    )


def _upload_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    """Upload a DataFrame as a Parquet file to S3."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)

    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"Uploaded to s3://{bucket}/{key}")


def _fetch_page_with_retry(
    fx_currency_pair: str,
    start_date: str,
    end_date: str,
    api_key: str,
    page: int,
    items: int,
    max_retries: int = MAX_RETRIES,
) -> list[dict]:
    """Call get_news_data with exponential-backoff retries on network/API errors."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return get_news_data(
                fx_currency_pair,
                start_date=start_date,
                end_date=end_date,
                api_key=api_key,
                page=page,
                items=items,
            )
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
            logger.warning(
                "News API call failed (attempt %d/%d, page=%d): %s",
                attempt + 1, max_retries, page, exc,
            )
            if attempt < max_retries - 1:
                time.sleep(backoff)

    logger.error(
        "News API call failed after %d retries (page=%d): %s",
        max_retries, page, last_error,
    )
    raise last_error


def _fetch_all_pages(
    fx_currency_pair: str,
    start_date: str,
    end_date: str,
    api_key: str,
    items: int = DEFAULT_PAGE_SIZE,
) -> list[dict]:
    """Fetch every page of news until the API returns fewer than `items`."""
    all_items: list[dict] = []
    page = DEFAULT_PAGE
    while True:
        page_items = _fetch_page_with_retry(
            fx_currency_pair, start_date, end_date, api_key, page, items
        )
        print(f"Fetched page {page} with {len(page_items)} items")
        all_items.extend(page_items)
        if len(page_items) < items:
            break
        page += 1
    print(f"Total news items fetched: {len(all_items)}")
    return all_items


def download_news_data(
    fx_currency_pair: str,
    interval: str,
    end_dt: datetime,
    api_key: str,
    bucket: str,
) -> pd.DataFrame:
    """Fetch news for [end_dt - 1 day, end_dt], upload to S3, return DataFrame."""
    start_dt = end_dt - timedelta(days=1)
    end_str = end_dt.strftime(DATE_FORMAT)
    start_str = start_dt.strftime(DATE_FORMAT)

    news_items = _fetch_all_pages(
        fx_currency_pair, start_str, end_str, api_key, items=DEFAULT_PAGE_SIZE
    )

    df = pd.DataFrame(news_items)

    if df.empty:
        print("No news data returned.")
        return df

    s3_key = _build_s3_key(fx_currency_pair, interval, end_dt)
    _upload_to_s3(df, bucket, s3_key)
    print(df.tail(15).to_string(index=False))
    return df


if __name__ == "__main__":
    config = AppConfig()
    fx_currency_pair = os.environ.get("FX_CURRENCY_PAIR", "USD-JPY")
    interval = os.environ.get("INTERVAL", "D1")
    execution_ts = os.environ.get("EXECUTION_TS")

    end_dt = (
        datetime.fromisoformat(execution_ts).astimezone(timezone.utc)
        if execution_ts
        else datetime.now(tz=timezone.utc)
    )

    download_news_data(
        fx_currency_pair=fx_currency_pair,
        interval=interval,
        end_dt=end_dt,
        api_key=config.fxnews_api_key,
        bucket=config.s3_bucket,
    )
