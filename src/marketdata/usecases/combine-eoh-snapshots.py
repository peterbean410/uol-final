"""
Combine and deduplicate the latest end-of-hour (EOH) price snapshots from
several open-ended backfill lanes into a single consolidated snapshot, and
overwrite the target lane's latest snapshot with the result.

Why this exists
---------------
The `create_eoh_snapshot_<year>` DAGs are open-ended, overlapping catch-up
lanes that all write to the same S3 snapshot tree
(`marketdata/eoh-snapshot/symbol=.../interval=.../year/month/day/hour/{ts}.parquet`),
keyed by timestamp, so a given key is last-written by whichever lane most
recently passed through it. Each lane accumulates history from its own start,
so at its *current frontier* a lane holds `[lane_start .. frontier]` and, being
the most-recent writer there, its frontier snapshot is intact. No single lane
holds the full history: 2012 reaches ~2017, 2017 reaches ~2018, and only 2026
holds the current tail. Concatenating every lane's frontier snapshot and
deduplicating on (Timestamp, Symbol) reconstructs the complete series, which we
then write back to the target lane's latest partition.

This is the manual counterpart to `create-eoi-price-snapshot.py`; it reuses the
identical S3 layout and the identical dedup key so the consolidated file is
byte-compatible with what the scheduled lanes produce.

Environment variables:
    FX_SYMBOL:         The FX symbol (e.g. 'USDJPY'). Default 'USDJPY'.
    INTERVALS:         Comma-separated intervals to consolidate (e.g. 'M1,M5,M15').
                       Default 'M1,M5,M15'.
    TARGET_DAG:        The lane whose latest snapshot is overwritten with the
                       combined result. Default 'create_eoh_snapshot_2026'.
    SOURCE_FRONTIERS:  JSON object mapping each source lane's dag_id to the
                       ISO-8601 timestamp of its latest snapshot (its frontier),
                       e.g. {"create_eoh_snapshot_2012": "2017-05-21T20:00:00", ...}.
                       Naive timestamps are treated as UTC. Resolved at run time
                       by the upstream `resolve_frontiers` task from the Airflow
                       metadata DB; null values (a lane with no success) are
                       skipped.
"""

import gc
import io
import json
import os
from datetime import datetime, timezone

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from commons.python.appconfig import AppConfig

HOUR_MINUTES = 60
DAILY_MINUTES = 1440

# Dedup key, MUST match create-eoi-price-snapshot.py so consolidated snapshots
# stay drop-in compatible with what the scheduled lanes write.
DEDUP_SUBSET = ["Timestamp", "Symbol"]

DEFAULT_INTERVALS = ["M1", "M5", "M15"]
DEFAULT_TARGET_DAG = "create_eoh_snapshot_2026"


# --- S3 layout helpers: mirrors create-eoi-price-snapshot.py (EOH / 60-min) ---

def _snapshot_root(time_window_minutes: int) -> str:
    """Return the snapshot tree root keyed by partition granularity."""
    if time_window_minutes == HOUR_MINUTES:
        return "marketdata/eoh-snapshot"
    if time_window_minutes == DAILY_MINUTES:
        return "marketdata/eod-snapshot"
    raise ValueError(f"Unsupported time_window_minutes={time_window_minutes} for EOH combine.")


def _build_snapshot_key(fx_symbol: str, interval: str, dt: datetime, time_window_minutes: int) -> str:
    """Build the S3 key for an EOH snapshot, identical to create-eoi-price-snapshot.py."""
    ts = dt.strftime("%Y%m%dT%H%M%SZ")
    base = f"{_snapshot_root(time_window_minutes)}/symbol={fx_symbol}/interval={interval}"
    key = f"{base}/year={dt.year}/month={dt.month:02d}"
    key += f"/day={dt.day:02d}"
    if time_window_minutes < DAILY_MINUTES:
        key += f"/hour={dt.hour:02d}"
    return f"{key}/{ts}.parquet"


def _load_snapshot_file(s3, bucket: str, key: str) -> pd.DataFrame:
    """Load a single snapshot Parquet file by exact key; empty DataFrame if absent."""
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            print(f"  MISSING s3://{bucket}/{key}")
            return pd.DataFrame()
        raise
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _upload_to_s3(df: pd.DataFrame, bucket: str, key: str, s3) -> None:
    """Upload a DataFrame as a Parquet file to S3."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"  Uploaded {len(df)} rows -> s3://{bucket}/{key}")


# --- combine ------------------------------------------------------------------

def _parse_frontier(value: str) -> datetime:
    """Parse an ISO-8601 frontier timestamp; treat a naive value as UTC."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def combine_interval(fx_symbol: str, interval: str, sources: list[tuple[str, datetime]],
                     target_dt: datetime, s3, bucket: str) -> pd.DataFrame:
    """Combine one interval's per-lane frontier snapshots and overwrite the target.

    `sources` is a list of (dag_id, frontier_dt) ordered oldest-frontier-first, so
    that under drop_duplicates(keep='last') the most recent lane wins any tie.
    """
    print(f"[{interval}] combining {len(sources)} lane snapshots")
    frames = []
    for dag_id, dt in sources:
        key = _build_snapshot_key(fx_symbol, interval, dt, HOUR_MINUTES)
        df = _load_snapshot_file(s3, bucket, key)
        if not df.empty:
            print(f"  {dag_id}: {len(df)} rows from {key}")
            frames.append(df)

    if not frames:
        print(f"[{interval}] no source rows found, skipping.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    before = len(combined)
    combined.drop_duplicates(subset=DEDUP_SUBSET, keep="last", inplace=True)
    combined.sort_values("Timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    print(f"[{interval}] {before} rows -> {len(combined)} after dedup "
          f"(span {combined['Timestamp'].min()} .. {combined['Timestamp'].max()})")

    target_key = _build_snapshot_key(fx_symbol, interval, target_dt, HOUR_MINUTES)
    _upload_to_s3(combined, bucket, target_key, s3)
    return combined


def main() -> None:
    config = AppConfig()
    fx_symbol = os.environ.get("FX_SYMBOL", "USDJPY")
    intervals = [s.strip() for s in os.environ.get("INTERVALS", ",".join(DEFAULT_INTERVALS)).split(",") if s.strip()]
    target_dag = os.environ.get("TARGET_DAG", DEFAULT_TARGET_DAG)

    raw = os.environ.get("SOURCE_FRONTIERS")
    if not raw:
        raise SystemExit("SOURCE_FRONTIERS env var is required (JSON dag_id -> ISO ts).")
    frontiers_raw = json.loads(raw)

    # dag_id -> frontier datetime, dropping lanes with no successful run.
    frontiers = {dag: _parse_frontier(ts) for dag, ts in frontiers_raw.items() if ts}
    if target_dag not in frontiers:
        raise SystemExit(f"Target lane {target_dag} has no resolved frontier; cannot pick write partition.")

    target_dt = frontiers[target_dag]
    # Oldest frontier first so the newest lane wins ties under keep='last'.
    sources = sorted(frontiers.items(), key=lambda kv: kv[1])

    s3 = boto3.client("s3")
    bucket = config.s3_bucket

    print(f"Symbol={fx_symbol}  intervals={intervals}")
    print(f"Target lane={target_dag}  frontier={target_dt.isoformat()}")
    print("Source lanes (oldest->newest frontier):")
    for dag, dt in sources:
        print(f"  {dag}: {dt.isoformat()}")

    for interval in intervals:
        combine_interval(fx_symbol, interval, sources, target_dt, s3, bucket)
        gc.collect()

    print("Combine complete.")


if __name__ == "__main__":
    main()
