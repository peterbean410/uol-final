"""
Create an end-of-day (EOD) news snapshot by merging the previous trading
day's news snapshot with the current day's raw news partition and
uploading the cumulative result to S3.

Equivalent to create-eod-tick-snapshot.py but for interval-news data.

Rows are enriched with `boj_policy` and `jpy_intervention` flags from an LLM
endpoint. Enrichment is incremental and idempotent: only rows whose labels are
still null are sent, so the first run backfills the accumulated history and
every later run labels just that day's new articles. It is also best-effort -
if the endpoint is unset or unreachable the snapshot is still written, with
null labels for the rows that could not be classified.

Usage:
    python marketdata/usecases/create-eod-news-snapshot.py

Environment variables:
    FX_PAIR: The FX pair in dash format (e.g. 'USD-JPY')
    EXECUTION_TS: ISO-8601 timestamp marking the end of the scheduled window (UTC)
    LLM_ENDPOINT: OpenAI-compatible base URL; unset disables labelling
    LLM_MODEL: Model name (default gemma-4-31b-it)
    LLM_API_KEY: Bearer token for LLM_ENDPOINT, if required
    NEWS_LABEL_MAX_ROWS: Cap on rows labelled per run (default 20000)
"""

import io
import os
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from commons.python.appconfig import AppConfig
from marketdata.newsdata.news_labeller import LABEL_COLUMNS, label_headlines

NEWS_DEDUP_KEYS = ["news_url"]

# Safety valve for the first backfill run: bounds pod runtime if the whole
# cumulative history arrives unlabelled. Leftovers are picked up next run.
DEFAULT_MAX_LABEL_ROWS = 20000


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


def _unlabelled_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that still need labelling: any label column null, and a title present."""
    missing = df[list(LABEL_COLUMNS)].isna().any(axis=1)
    return missing & df["title"].notna() & (df["title"].str.strip() != "")


def add_labels(df: pd.DataFrame, config: AppConfig, max_rows: int) -> pd.DataFrame:
    """Fill in missing news labels in place, best-effort.

    Ensures the label columns exist, sends only unlabelled rows to the LLM, and
    swallows any failure so a labelling outage never blocks the snapshot.
    """
    for column in LABEL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = df[column].astype("boolean")

    if not config.llm_endpoint:
        print("LLM_ENDPOINT not set - skipping news labelling")
        return df

    pending = df.index[_unlabelled_mask(df)]
    if pending.empty:
        print("All news rows already labelled")
        return df

    if len(pending) > max_rows:
        print(f"Labelling first {max_rows} of {len(pending)} unlabelled rows this run")
        pending = pending[:max_rows]

    print(f"Labelling {len(pending)} rows via {config.llm_endpoint} ({config.llm_model})")
    try:
        labels = label_headlines(
            df.loc[pending, "title"].tolist(),
            endpoint=config.llm_endpoint,
            model=config.llm_model,
            api_key=config.llm_api_key or None,
        )
    except Exception as exc:  # noqa: BLE001 - enrichment must never fail the snapshot
        print(f"WARNING: news labelling failed, writing snapshot unlabelled: {exc}")
        return df

    for column in LABEL_COLUMNS:
        values = [label[column] if label else pd.NA for label in labels]
        df.loc[pending, column] = pd.array(values, dtype="boolean")

    labelled = sum(1 for label in labels if label)
    print(f"Labelled {labelled}/{len(pending)} rows "
          f"({len(pending) - labelled} unresolved, left null for a later run)")
    for column in LABEL_COLUMNS:
        print(f"  {column}: {int(df[column].fillna(False).sum())} true of {int(df[column].notna().sum())} labelled")
    return df


def create_snapshot(fx_pair: str, end_dt: datetime, s3, bucket: str,
                    config: AppConfig, max_label_rows: int = DEFAULT_MAX_LABEL_ROWS) -> pd.DataFrame:
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

    df = add_labels(df, config, max_label_rows)

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

    max_label_rows = int(os.environ.get("NEWS_LABEL_MAX_ROWS", DEFAULT_MAX_LABEL_ROWS))

    s3 = boto3.client("s3")
    create_snapshot(fx_pair, end_dt, s3, config.s3_bucket, config, max_label_rows)
