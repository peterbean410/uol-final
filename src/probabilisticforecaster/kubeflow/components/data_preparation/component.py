"""Data Preparation KFP component.

Loads 5-minute OHLC parquet data from S3, computes 16 engineered features
(z-score, return, volatility, time), and builds train/test ForexDataset splits
using the configured lookback_window, forecast_horizon, and date boundaries.

Outputs train and test datasets as KFP artifacts to S3.

Requirements: 1.3, 2.1, 2.2, 2.3, 2.4, 2.5
"""

import argparse
import io
import os
import pickle
import sys
import tempfile
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd

# Add parent paths so we can import the probabilisticforecaster package
sys.path.insert(0, "/app")

from probabilisticforecaster.config import S3_BUCKET
from probabilisticforecaster.dataset import ForexDataset
from probabilisticforecaster.features import compute_features
from probabilisticforecaster.kubeflow.monitoring.metrics import get_logger

logger = get_logger(__name__, component="data_preparation")


def load_data_from_s3(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    bucket: str = S3_BUCKET,
    snapshot_date: datetime | None = None,
) -> pd.DataFrame:
    """Load 5-minute OHLC data from S3 by reading one cumulative hour=23 snapshot.

    EOH snapshots are fully cumulative; the hour=23 snapshot of a given day
    contains ALL bars from the start of the chain through that day. We read
    that one file and filter to [start_date, end_date]. By default the file is
    the snapshot of end_date; pass ``snapshot_date`` to read a different
    (typically later) snapshot, needed when end_date's own snapshot predates
    a history consolidation and so does not actually hold the full chain.

    Args:
        symbol: Currency pair (e.g., "USDJPY", "AUDJPY").
        start_date: Start of the date range (inclusive).
        end_date: End of the date range (inclusive).
        bucket: S3 bucket name.
        snapshot_date: Day whose hour=23 snapshot file to read (defaults to
            end_date). Must be >= end_date to cover the requested range.

    Returns:
        DataFrame with columns [Timestamp, Symbol, Open, High, Low, Close, Volume]
        sorted by Timestamp.

    Raises:
        FileNotFoundError: If the expected snapshot file does not exist in S3.
        RuntimeError: If no data exists in the requested range.
    """
    s3 = boto3.client("s3")

    key = _build_eoh_snapshot_key(symbol, snapshot_date or end_date)

    logger.info(
        "Loading data from S3",
        extra={"s3_uri": f"s3://{bucket}/{key}", "symbol": symbol},
    )

    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except Exception as e:
        if "NoSuchKey" in str(type(e).__name__) or "NoSuchKey" in str(e) or "404" in str(e):
            raise FileNotFoundError(
                f"Snapshot not found: s3://{bucket}/{key}"
            ) from e
        raise

    data = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], utc=True)
    data.sort_values("Timestamp", inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Filter to requested date range
    mask = (data["Timestamp"] >= pd.Timestamp(start_date)) & (
        data["Timestamp"] <= pd.Timestamp(end_date)
    )
    data = data[mask].reset_index(drop=True)

    if data.empty:
        raise RuntimeError(
            f"No data in requested range [{start_date.strftime('%Y-%m-%d')}, "
            f"{end_date.strftime('%Y-%m-%d')}] from snapshot {key}"
        )

    logger.info(
        "Data loaded successfully",
        extra={"num_bars": len(data), "date_range": f"{start_date} to {end_date}"},
    )
    return data


def _build_eoh_snapshot_key(symbol: str, dt: datetime) -> str:
    """Build the S3 key for the hour=23 EOH snapshot of a given date.

    Key structure:
        marketdata/eoh-snapshot/symbol={SYMBOL}/interval=M5/
        year={YYYY}/month={MM}/day={DD}/hour=23/{YYYYMMDD}T230000Z.parquet
    """
    return (
        f"marketdata/eoh-snapshot/symbol={symbol}/interval=M5"
        f"/year={dt.year}/month={dt.month:02d}/day={dt.day:02d}"
        f"/hour=23/{dt.strftime('%Y%m%d')}T230000Z.parquet"
    )


def build_datasets(
    data: pd.DataFrame,
    lookback_window: int,
    forecast_horizon: int,
    historical_window: int,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple[ForexDataset, ForexDataset]:
    """Compute features and build train/test ForexDataset splits.

    Args:
        data: Raw OHLC DataFrame with Timestamp column.
        lookback_window: Number of bars in the lookback window.
        forecast_horizon: Number of bars ahead for forward return label.
        historical_window: Rolling window size for z-score and volatility.
        train_start: ISO date string for training start.
        train_end: ISO date string for training end.
        test_start: ISO date string for test start.
        test_end: ISO date string for test end.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    logger.info(
        "Computing features",
        extra={"historical_window": historical_window, "num_bars": len(data)},
    )
    features_df = compute_features(data, historical_window=historical_window)
    logger.info(
        "Features computed",
        extra={"rows": features_df.shape[0], "columns": features_df.shape[1]},
    )

    # Align close prices with features index
    data_indexed = data.set_index(pd.to_datetime(data["Timestamp"], utc=True))
    close_prices = data_indexed["Close"].reindex(features_df.index)

    # Parse date boundaries
    train_start_dt = pd.Timestamp(train_start, tz="UTC")
    train_end_dt = pd.Timestamp(train_end + " 23:59:59", tz="UTC")
    test_start_dt = pd.Timestamp(test_start, tz="UTC")
    test_end_dt = pd.Timestamp(test_end + " 23:59:59", tz="UTC")

    # Split by date boundaries
    train_mask = (features_df.index >= train_start_dt) & (features_df.index <= train_end_dt)
    test_mask = (features_df.index >= test_start_dt) & (features_df.index <= test_end_dt)

    train_features = features_df[train_mask]
    test_features = features_df[test_mask]
    train_close = close_prices[train_mask]
    test_close = close_prices[test_mask]

    logger.info(
        "Train set prepared",
        extra={
            "num_bars": len(train_features),
            "start": str(train_features.index.min()) if len(train_features) > 0 else "N/A",
            "end": str(train_features.index.max()) if len(train_features) > 0 else "N/A",
        },
    )
    logger.info(
        "Test set prepared",
        extra={
            "num_bars": len(test_features),
            "start": str(test_features.index.min()) if len(test_features) > 0 else "N/A",
            "end": str(test_features.index.max()) if len(test_features) > 0 else "N/A",
        },
    )

    train_dataset = ForexDataset(
        train_features,
        train_close,
        lookback=lookback_window,
        horizon=forecast_horizon,
    )
    test_dataset = ForexDataset(
        test_features,
        test_close,
        lookback=lookback_window,
        horizon=forecast_horizon,
    )

    logger.info(
        "Datasets constructed",
        extra={
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "feature_shape": f"({lookback_window}, 16)",
        },
    )

    return train_dataset, test_dataset


def upload_dataset_to_s3(
    dataset: ForexDataset,
    s3_uri: str,
    bucket: str = S3_BUCKET,
) -> None:
    """Serialize and upload a ForexDataset to the artifact store named by ``s3_uri``.

    Routing follows the URI scheme (see ``probabilisticforecaster.artifact_io``):
    ``minio://...`` lands in the cluster MinIO; ``s3://...`` lands in AWS S3;
    a bare key lands in ``bucket`` (default ``S3_BUCKET``).
    """
    from probabilisticforecaster.artifact_io import put_object_bytes

    buffer = io.BytesIO()
    pickle.dump(dataset, buffer)
    data = buffer.getvalue()

    put_object_bytes(s3_uri, data, default_bucket=bucket)

    logger.info(
        "Dataset uploaded",
        extra={"uri": s3_uri, "size_bytes": len(data)},
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the data preparation component."""
    parser = argparse.ArgumentParser(
        description="Data Preparation KFP Component",
    )
    parser.add_argument("--symbol", type=str, required=True, help="Currency pair symbol")
    parser.add_argument("--train-start", type=str, required=True, help="Training start date (ISO)")
    parser.add_argument("--train-end", type=str, required=True, help="Training end date (ISO)")
    parser.add_argument("--test-start", type=str, required=True, help="Test start date (ISO)")
    parser.add_argument("--test-end", type=str, required=True, help="Test end date (ISO)")
    parser.add_argument("--lookback-window", type=int, default=36, help="Lookback window in bars")
    parser.add_argument("--forecast-horizon", type=int, default=1, help="Forecast horizon in bars")
    parser.add_argument("--historical-window", type=int, default=1440, help="Historical window for features")
    parser.add_argument(
        "--train-dataset-path",
        type=str,
        required=True,
        help="S3 key path for the output train dataset artifact",
    )
    parser.add_argument(
        "--test-dataset-path",
        type=str,
        required=True,
        help="S3 key path for the output test dataset artifact",
    )
    parser.add_argument("--bucket", type=str, default=S3_BUCKET, help="S3 bucket name")
    parser.add_argument(
        "--data-snapshot-date", type=str, default="",
        help="Day whose hour=23 EOH snapshot file to read (ISO; defaults to test-end)")
    return parser.parse_args()


def main() -> None:
    """Main entry point for the data preparation component."""
    args = parse_args()

    logger.info(
        "Data preparation component started",
        extra={
            "symbol": args.symbol,
            "train_start": args.train_start,
            "train_end": args.train_end,
            "test_start": args.test_start,
            "test_end": args.test_end,
            "lookback_window": args.lookback_window,
            "forecast_horizon": args.forecast_horizon,
            "historical_window": args.historical_window,
        },
    )

    # Step 1: Load data from S3
    # We need data from train_start through test_end
    # The EOH snapshot of test_end contains all historical data
    start_date = datetime.fromisoformat(args.train_start).replace(tzinfo=timezone.utc)
    end_date = datetime.fromisoformat(args.test_end).replace(tzinfo=timezone.utc)
    snapshot_date = (
        datetime.fromisoformat(args.data_snapshot_date).replace(tzinfo=timezone.utc)
        if args.data_snapshot_date
        else None
    )

    data = load_data_from_s3(
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date,
        bucket=args.bucket,
        snapshot_date=snapshot_date,
    )

    # Step 2: Compute features and build train/test datasets
    train_dataset, test_dataset = build_datasets(
        data=data,
        lookback_window=args.lookback_window,
        forecast_horizon=args.forecast_horizon,
        historical_window=args.historical_window,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    # Step 3: Upload datasets to S3 as KFP artifacts
    upload_dataset_to_s3(train_dataset, args.train_dataset_path, bucket=args.bucket)
    upload_dataset_to_s3(test_dataset, args.test_dataset_path, bucket=args.bucket)

    logger.info(
        "Data preparation component completed successfully",
        extra={
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "train_artifact": args.train_dataset_path,
            "test_artifact": args.test_dataset_path,
        },
    )


if __name__ == "__main__":
    main()
