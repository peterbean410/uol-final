"""End-to-end pipeline for the Transformer Probabilistic Forex Forecaster.

Orchestrates the full workflow: data loading from S3 parquet → feature computation
→ dataset construction → model training → evaluation → backtesting for all strategies.

Usage:
    python -m probabilisticforecaster.pipeline --symbol USDJPY --horizon 1

Environment:
    Requires AWS credentials configured for S3 access to the
    prod-fintech-forex-sg-731833471586 bucket.
"""

import argparse
import io
import logging
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd
import torch

from probabilisticforecaster.backtest import BacktestResult, run_backtest
from probabilisticforecaster.config import ForecasterConfig, S3_BUCKET
from probabilisticforecaster.dataset import ForexDataset
from probabilisticforecaster.evaluation import (
    EvaluationMetrics,
    evaluate_by_hour,
    evaluate_model,
)
from probabilisticforecaster.features import compute_features
from probabilisticforecaster.model import ProbabilisticTransformer
from probabilisticforecaster.strategy import (
    BuyAndHoldBenchmark,
    DirectionalStrategy,
    MeanVarianceStrategy,
    MovingAverageBenchmark,
)
from probabilisticforecaster.training import train_model

logger = logging.getLogger(__name__)

TRAIN_START = datetime(2012, 1, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2022, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
TEST_END = datetime(2026, 4, 30, 23, 59, 59, tzinfo=timezone.utc)


def load_data_from_s3(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    bucket: str = S3_BUCKET,
) -> pd.DataFrame:
    """Load 5-minute OHLC data from S3 by reading the hour=23 snapshot of end_date.

    Delegates to load_data_from_s3_efficient.
    """
    return load_data_from_s3_efficient(symbol, start_date, end_date, bucket)


def load_data_from_s3_efficient(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    bucket: str = S3_BUCKET,
) -> pd.DataFrame:
    """Load 5-minute OHLC data from S3 by reading the single hour=23 snapshot of end_date.

    EOH snapshots are fully cumulative; the DAG runs every hour, every day
    (including weekends), so hour=23 always exists for any date in the chain.
    The hour=23 snapshot of end_date contains ALL bars from the start of the
    chain through that day. We read that one file and filter to [start_date, end_date].

    Args:
        symbol: Currency pair (e.g., "USDJPY", "AUDJPY").
        start_date: Start of the date range (inclusive).
        end_date: End of the date range (inclusive).
        bucket: S3 bucket name.

    Returns:
        DataFrame with columns [Timestamp, Symbol, Open, High, Low, Close, Volume]
        sorted by Timestamp.

    Raises:
        FileNotFoundError: If the expected snapshot file does not exist in S3.
    """
    s3 = boto3.client("s3")

    key = _build_eoh_snapshot_key(symbol, end_date)

    logger.info(
        "Loading data from S3: s3://%s/%s",
        bucket,
        key,
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

    mask = (data["Timestamp"] >= pd.Timestamp(start_date)) & (
        data["Timestamp"] <= pd.Timestamp(end_date)
    )
    data = data[mask].reset_index(drop=True)

    if data.empty:
        raise RuntimeError(
            f"No data in requested range [{start_date.strftime('%Y-%m-%d')}, "
            f"{end_date.strftime('%Y-%m-%d')}] from snapshot {key}"
        )

    logger.info("Loaded %d bars from snapshot", len(data))
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
    config: ForecasterConfig,
) -> tuple[ForexDataset, ForexDataset, pd.DataFrame]:
    """Compute features and build train/test datasets with date split.

    Args:
        data: Raw OHLC DataFrame with Timestamp column.
        config: ForecasterConfig with lookback, horizon, and historical_window.

    Returns:
        Tuple of (train_dataset, test_dataset, features_df).
    """
    logger.info("Computing features (historical_window=%d)...", config.historical_window)
    features_df = compute_features(data, historical_window=config.historical_window)
    logger.info("Features computed: %d rows × %d columns", *features_df.shape)

    data_indexed = data.set_index(pd.to_datetime(data["Timestamp"], utc=True))
    close_prices = data_indexed["Close"].reindex(features_df.index)

    train_mask = features_df.index < TEST_START
    test_mask = features_df.index >= TEST_START

    train_features = features_df[train_mask]
    test_features = features_df[test_mask]
    train_close = close_prices[train_mask]
    test_close = close_prices[test_mask]

    logger.info(
        "Train set: %d bars (%s to %s)",
        len(train_features),
        train_features.index.min().strftime("%Y-%m-%d") if len(train_features) > 0 else "N/A",
        train_features.index.max().strftime("%Y-%m-%d") if len(train_features) > 0 else "N/A",
    )
    logger.info(
        "Test set: %d bars (%s to %s)",
        len(test_features),
        test_features.index.min().strftime("%Y-%m-%d") if len(test_features) > 0 else "N/A",
        test_features.index.max().strftime("%Y-%m-%d") if len(test_features) > 0 else "N/A",
    )

    train_dataset = ForexDataset(
        train_features,
        train_close,
        lookback=config.lookback_window,
        horizon=config.forecast_horizon,
    )
    test_dataset = ForexDataset(
        test_features,
        test_close,
        lookback=config.lookback_window,
        horizon=config.forecast_horizon,
    )

    logger.info("Train dataset: %d samples", len(train_dataset))
    logger.info("Test dataset: %d samples", len(test_dataset))

    return train_dataset, test_dataset, features_df


def run_evaluation(
    model: ProbabilisticTransformer,
    test_dataset: ForexDataset,
    config: ForecasterConfig,
    test_timestamps: pd.Series,
) -> tuple[EvaluationMetrics, pd.DataFrame]:
    """Evaluate the trained model on the test set.

    Args:
        model: Trained ProbabilisticTransformer model.
        test_dataset: Test ForexDataset.
        config: ForecasterConfig.
        test_timestamps: Timestamps aligned with test dataset samples.

    Returns:
        Tuple of (overall_metrics, hourly_metrics_df).
    """
    logger.info("Evaluating model on test set...")
    metrics = evaluate_model(model, test_dataset, config)
    logger.info("Overall Metrics:")
    logger.info("  NLL:                  %.4f", metrics.nll)
    logger.info("  Directional Accuracy: %.4f", metrics.directional_accuracy)
    logger.info("  95%% Covered Ratio:    %.4f", metrics.covered_ratio_95)
    logger.info("  RMSE:                 %.6f", metrics.rmse)

    hourly_metrics = evaluate_by_hour(model, test_dataset, test_timestamps, config)
    logger.info("Hourly Metrics:\n%s", hourly_metrics.to_string())

    return metrics, hourly_metrics


def run_all_backtests(
    model: ProbabilisticTransformer,
    test_dataset: ForexDataset,
    test_features: pd.DataFrame,
    test_close: pd.Series,
    config: ForecasterConfig,
) -> dict[str, BacktestResult]:
    """Run backtests for all strategies.

    Args:
        model: Trained ProbabilisticTransformer model.
        test_dataset: Test ForexDataset.
        test_features: Test features DataFrame.
        test_close: Test close prices Series.
        config: ForecasterConfig.

    Returns:
        Dictionary mapping strategy name to BacktestResult.
    """
    logger.info("Generating predictions for backtest...")

    model.eval()
    device = next(model.parameters()).device
    predictions_list = []

    from torch.utils.data import DataLoader

    loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    sample_idx = 0
    with torch.no_grad():
        for features_batch, _ in loader:
            features_batch = features_batch.to(device)
            mu, sigma = model(features_batch)
            mu_last = mu[:, -1, 0].cpu().numpy()
            sigma_last = sigma[:, -1, 0].cpu().numpy()

            for i in range(len(mu_last)):
                if sample_idx + i < len(test_dataset):
                    start_idx = test_dataset.valid_indices[sample_idx + i]
                    ts_idx = start_idx + test_dataset.lookback - 1
                    ts = test_dataset.timestamps[ts_idx]
                    predictions_list.append(
                        {"timestamp": ts, "mu": mu_last[i], "sigma": sigma_last[i]}
                    )
            sample_idx += len(mu_last)

    predictions_df = pd.DataFrame(predictions_list)
    logger.info("Generated %d predictions", len(predictions_df))

    prices_df = pd.DataFrame(
        {"timestamp": test_features.index, "close": test_close.values}
    )

    strategies: dict[str, object] = {
        "Directional": DirectionalStrategy(),
        "MeanVariance_gamma_0.01": MeanVarianceStrategy(risk_aversion=0.01),
        "MeanVariance_gamma_0.05": MeanVarianceStrategy(risk_aversion=0.05),
        "MeanVariance_gamma_0.10": MeanVarianceStrategy(risk_aversion=0.1),
        "BuyAndHold": BuyAndHoldBenchmark(),
    }

    results: dict[str, BacktestResult] = {}

    for name, strategy in strategies.items():
        logger.info("Running backtest: %s", name)
        result = run_backtest(predictions_df, prices_df, strategy, config)
        results[name] = result
        logger.info(
            "  %s, Ann. Return: %.4f, Sharpe: %.4f, Max DD: %.4f",
            name,
            result.annualised_return,
            result.sharpe_ratio,
            result.max_drawdown,
        )

    logger.info("Running backtest: MovingAverageBenchmark")
    ma_strategy = MovingAverageBenchmark()

    ma20 = test_close.rolling(window=20).mean()
    ma_predictions = predictions_df.copy()

    ma_preds_list = []
    for _, row in predictions_df.iterrows():
        ts = row["timestamp"]
        if ts in test_close.index and ts in ma20.index:
            close_val = test_close.loc[ts]
            ma_val = ma20.loc[ts]
            if pd.notna(ma_val):
                mu_override = 1.0 if close_val >= ma_val else -1.0
                ma_preds_list.append(
                    {"timestamp": ts, "mu": mu_override, "sigma": 1.0}
                )
            else:
                ma_preds_list.append(
                    {"timestamp": ts, "mu": 1.0, "sigma": 1.0}
                )
        else:
            ma_preds_list.append(
                {"timestamp": ts, "mu": row["mu"], "sigma": row["sigma"]}
            )

    if ma_preds_list:
        ma_preds_df = pd.DataFrame(ma_preds_list)
        ma_result = run_backtest(
            ma_preds_df, prices_df, DirectionalStrategy(), config
        )
        results["MovingAverage_MA20"] = ma_result
        logger.info(
            "  MovingAverage_MA20, Ann. Return: %.4f, Sharpe: %.4f, Max DD: %.4f",
            ma_result.annualised_return,
            ma_result.sharpe_ratio,
            ma_result.max_drawdown,
        )

    return results


def print_backtest_summary(results: dict[str, BacktestResult]) -> None:
    """Print a formatted summary table of all backtest results.

    Args:
        results: Dictionary mapping strategy name to BacktestResult.
    """
    logger.info("=" * 80)
    logger.info("BACKTEST RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "%-30s %12s %10s %10s",
        "Strategy", "Ann. Return", "Sharpe", "Max DD",
    )
    logger.info("-" * 80)
    for name, result in results.items():
        logger.info(
            "%-30s %12.6f %10.4f %10.6f",
            name,
            result.annualised_return,
            result.sharpe_ratio,
            result.max_drawdown,
        )
    logger.info("=" * 80)


def run_pipeline(config: ForecasterConfig, upload_to_s3: bool = True) -> None:
    """Execute the full pipeline: data → features → dataset → train → evaluate → backtest.

    Args:
        config: ForecasterConfig with all parameters.
        upload_to_s3: Whether to upload the trained model to S3.
    """
    logger.info("=" * 80)
    logger.info("TRANSFORMER PROBABILISTIC FOREX FORECASTER PIPELINE")
    logger.info("=" * 80)
    logger.info("Symbol: %s", config.symbol)
    logger.info("Forecast Horizon: %d bars (%d min)", config.forecast_horizon, config.forecast_horizon * 5)
    logger.info("Lookback Window: %d bars (%d min)", config.lookback_window, config.lookback_window * 5)
    logger.info("=" * 80)

    logger.info("Step 1: Loading data from S3...")
    data = load_data_from_s3_efficient(
        symbol=config.symbol,
        start_date=TRAIN_START,
        end_date=TEST_END,
        bucket=S3_BUCKET,
    )
    logger.info("Total data loaded: %d bars", len(data))

    logger.info("Step 2: Computing features and building datasets...")
    train_dataset, test_dataset, features_df = build_datasets(data, config)

    logger.info("Step 3: Training model...")
    model = ProbabilisticTransformer(config)
    logger.info(
        "Model parameters: %d",
        sum(p.numel() for p in model.parameters()),
    )

    training_history = train_model(
        model=model,
        train_dataset=train_dataset,
        config=config,
        val_dataset=None,
        version=1,
        upload_to_s3=upload_to_s3,
    )

    logger.info("Training complete. Final epoch loss: %.4f", training_history["epoch_loss"][-1])
    logger.info("Model saved to: %s", config.model_path)

    logger.info("Step 4: Evaluating model...")
    test_timestamps = pd.Series(
        [
            test_dataset.timestamps[
                test_dataset.valid_indices[i] + test_dataset.lookback - 1
            ]
            for i in range(len(test_dataset))
        ]
    )

    metrics, hourly_metrics = run_evaluation(
        model, test_dataset, config, test_timestamps
    )

    logger.info("Step 5: Running backtests...")
    test_mask = features_df.index >= TEST_START
    test_features = features_df[test_mask]

    data_indexed = data.set_index(pd.to_datetime(data["Timestamp"], utc=True))
    test_close = data_indexed["Close"].reindex(test_features.index)

    backtest_results = run_all_backtests(
        model, test_dataset, test_features, test_close, config
    )

    print_backtest_summary(backtest_results)

    logger.info("Pipeline complete.")


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Transformer Probabilistic Forex Forecaster Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="USDJPY",
        help="Currency pair symbol (e.g., USDJPY, AUDJPY)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        choices=[1, 3, 6, 12],
        help="Forecast horizon in bars (1=5min, 3=15min, 6=30min, 12=60min)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local path to save the trained model",
    )
    parser.add_argument(
        "--no-s3-upload",
        action="store_true",
        help="Skip uploading model to S3",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    model_path = args.model_path or f"models/transformer_{args.symbol}_h{args.horizon}.pt"

    config = ForecasterConfig(
        symbol=args.symbol,
        forecast_horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_path=model_path,
    )

    run_pipeline(config, upload_to_s3=not args.no_s3_upload)


if __name__ == "__main__":
    main()
