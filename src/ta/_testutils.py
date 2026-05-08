"""
Shared test utilities for loading EOH snapshot data from S3.
"""

import io

import boto3
import pandas as pd
from rich import print

BUCKET = "prod-fintech-forex-sg-731833471586"
SNAPSHOT_ROOT = "marketdata/eoh-snapshot"


def build_snapshot_prefix(symbol: str, interval: str, year: int, month: int,
                          day: int, hour: int) -> str:
    return (
        f"{SNAPSHOT_ROOT}/symbol={symbol}/interval={interval}"
        f"/year={year}/month={month:02d}/day={day:02d}/hour={hour:02d}/"
    )


def load_snapshot_from_s3(prefix: str, bucket: str = BUCKET) -> pd.DataFrame:
    """Load the latest parquet file found under *prefix* from S3."""
    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=10)
    keys = [obj["Key"] for obj in resp.get("Contents", []) if obj["Key"].endswith(".parquet")]

    if not keys:
        raise FileNotFoundError(f"No parquet files found under s3://{bucket}/{prefix}")

    key = sorted(keys)[-1]
    print(f"[cyan]Loading:[/cyan] s3://{bucket}/{key}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    print(f"[green]Loaded {len(df)} rows[/green]")
    return df


def load_data(args) -> pd.DataFrame:
    """Load a DataFrame based on CLI args (--local or S3)."""
    if args.local:
        print(f"[cyan]Loading local file:[/cyan] {args.local}")
        df = pd.read_parquet(args.local)
        print(f"[green]Loaded {len(df)} rows[/green]")
        return df
    prefix = build_snapshot_prefix(
        args.symbol, args.interval,
        args.year, args.month, args.day, args.hour,
    )
    return load_snapshot_from_s3(prefix)
