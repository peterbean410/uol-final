"""Backtesting KFP component.

Loads a model checkpoint and test dataset from S3, runs all trading strategies
(directional, mean-variance with γ=0.01/0.05/0.1, buy-and-hold benchmark),
and outputs performance metrics (annualised return, Sharpe ratio, max drawdown)
as a JSON artifact to S3.

Requirements: 1.6, 2.1, 2.2, 2.3, 2.4, 2.5
"""

import argparse
import io
import json
import pickle
import sys
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd
import torch

# Add parent paths so we can import the probabilisticforecaster package
sys.path.insert(0, "/app")

from probabilisticforecaster.backtest import BacktestResult, run_backtest
from probabilisticforecaster.config import ForecasterConfig, S3_BUCKET
from probabilisticforecaster.dataset import ForexDataset
from probabilisticforecaster.kubeflow.monitoring.metrics import get_logger
from probabilisticforecaster.model import ProbabilisticTransformer
from probabilisticforecaster.strategy import (
    BuyAndHoldBenchmark,
    DirectionalStrategy,
    MeanVarianceStrategy,
    TradingStrategy,
)

logger = get_logger(__name__, component="backtesting")


def load_checkpoint_from_s3(
    checkpoint_path: str,
    bucket: str = S3_BUCKET,
) -> dict:
    """Download and load a model checkpoint from S3.

    Args:
        checkpoint_path: S3 key path for the model checkpoint.
        bucket: S3 bucket name.

    Returns:
        Checkpoint dictionary containing model_state_dict and config.

    Raises:
        FileNotFoundError: If the checkpoint does not exist in S3.
    """
    s3 = boto3.client("s3")

    logger.info(
        "Loading model checkpoint from S3",
        extra={"s3_key": checkpoint_path, "bucket": bucket},
    )

    try:
        obj = s3.get_object(Bucket=bucket, Key=checkpoint_path)
    except Exception as e:
        if "NoSuchKey" in str(type(e).__name__) or "NoSuchKey" in str(e) or "404" in str(e):
            raise FileNotFoundError(
                f"Checkpoint not found: s3://{bucket}/{checkpoint_path}"
            ) from e
        raise

    buffer = io.BytesIO(obj["Body"].read())
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=False)

    logger.info("Model checkpoint loaded successfully")
    return checkpoint


def load_dataset_from_s3(
    dataset_path: str,
    bucket: str = S3_BUCKET,
) -> ForexDataset:
    """Download and deserialize a ForexDataset from S3.

    Args:
        dataset_path: S3 key path for the pickled dataset.
        bucket: S3 bucket name.

    Returns:
        Deserialized ForexDataset instance.

    Raises:
        FileNotFoundError: If the dataset does not exist in S3.
    """
    s3 = boto3.client("s3")

    logger.info(
        "Loading test dataset from S3",
        extra={"s3_key": dataset_path, "bucket": bucket},
    )

    try:
        obj = s3.get_object(Bucket=bucket, Key=dataset_path)
    except Exception as e:
        if "NoSuchKey" in str(type(e).__name__) or "NoSuchKey" in str(e) or "404" in str(e):
            raise FileNotFoundError(
                f"Dataset not found: s3://{bucket}/{dataset_path}"
            ) from e
        raise

    buffer = io.BytesIO(obj["Body"].read())
    dataset = pickle.loads(buffer.getvalue())

    logger.info(
        "Test dataset loaded successfully",
        extra={"num_samples": len(dataset)},
    )
    return dataset


def generate_predictions(
    model: ProbabilisticTransformer,
    dataset: ForexDataset,
    config: ForecasterConfig,
) -> pd.DataFrame:
    """Run model inference on the test dataset to generate predictions.

    Iterates over the dataset, runs each sample through the model, and
    collects (timestamp, mu, sigma) predictions for the last position
    in each lookback window.

    Args:
        model: Loaded ProbabilisticTransformer in eval mode.
        dataset: Test ForexDataset.
        config: ForecasterConfig for model parameters.

    Returns:
        DataFrame with columns ['timestamp', 'mu', 'sigma'].
    """
    model.eval()
    predictions = []

    logger.info(
        "Generating predictions",
        extra={"num_samples": len(dataset)},
    )

    with torch.no_grad():
        for i in range(len(dataset)):
            features, _ = dataset[i]

            # features shape: (lookback_window, 16)
            if isinstance(features, np.ndarray):
                input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            else:
                input_tensor = features.unsqueeze(0).float()

            mu, sigma = model(input_tensor)

            # Extract last position prediction
            mu_val = mu[0, -1, 0].item()
            sigma_val = sigma[0, -1, 0].item()

            # Get the timestamp for this sample (last bar in the lookback window)
            # valid_indices[i] is the start index; the last bar is at start + lookback - 1
            start_idx = dataset.valid_indices[i]
            t = start_idx + dataset.lookback - 1
            timestamp = dataset.timestamps[t]

            predictions.append({
                "timestamp": timestamp,
                "mu": mu_val,
                "sigma": sigma_val,
            })

    pred_df = pd.DataFrame(predictions)

    logger.info(
        "Predictions generated",
        extra={
            "num_predictions": len(pred_df),
            "mu_mean": float(pred_df["mu"].mean()),
            "sigma_mean": float(pred_df["sigma"].mean()),
        },
    )

    return pred_df


def get_prices_from_dataset(dataset: ForexDataset) -> pd.DataFrame:
    """Extract price data from the ForexDataset for backtesting.

    The backtest engine needs a DataFrame with ['timestamp', 'close'] columns.
    We extract all timestamps and close prices that are referenced by the
    valid samples, including the next-bar close needed for PnL calculation.

    Args:
        dataset: Test ForexDataset containing close prices.

    Returns:
        DataFrame with columns ['timestamp', 'close'].
    """
    # Collect all bar indices referenced by valid samples
    # Each sample uses bars [start, start+lookback-1] for features
    # and bar start+lookback-1+horizon for the label (next close)
    referenced_indices = set()

    for start_idx in dataset.valid_indices:
        t = start_idx + dataset.lookback - 1  # last bar in lookback window
        t_next = t + 1  # next bar for PnL calculation
        referenced_indices.add(t)
        if t_next < len(dataset.timestamps):
            referenced_indices.add(t_next)

    # Sort indices and build the prices DataFrame
    sorted_indices = sorted(referenced_indices)

    timestamps = [dataset.timestamps[i] for i in sorted_indices]
    closes = [float(dataset.close_prices[i]) for i in sorted_indices]

    return pd.DataFrame({"timestamp": timestamps, "close": closes})


def get_all_strategies() -> dict[str, TradingStrategy]:
    """Return all trading strategies to run in the backtest.

    Returns:
        Dictionary mapping strategy name to TradingStrategy instance.
    """
    return {
        "directional": DirectionalStrategy(),
        "mean_variance_gamma_0.01": MeanVarianceStrategy(risk_aversion=0.01),
        "mean_variance_gamma_0.05": MeanVarianceStrategy(risk_aversion=0.05),
        "mean_variance_gamma_0.1": MeanVarianceStrategy(risk_aversion=0.1),
        "buy_and_hold": BuyAndHoldBenchmark(),
    }


def backtest_result_to_dict(result: BacktestResult) -> dict:
    """Convert a BacktestResult to a serializable dictionary.

    Args:
        result: BacktestResult from run_backtest.

    Returns:
        Dictionary with annualised_return, sharpe_ratio, max_drawdown.
    """
    return {
        "annualised_return": float(result.annualised_return),
        "sharpe_ratio": float(result.sharpe_ratio),
        "max_drawdown": float(result.max_drawdown),
    }


def run_all_strategies(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    config: ForecasterConfig,
) -> dict:
    """Run backtests for all trading strategies.

    Args:
        predictions: DataFrame with columns ['timestamp', 'mu', 'sigma'].
        prices: DataFrame with columns ['timestamp', 'close'].
        config: ForecasterConfig with position_size and risk_aversion.

    Returns:
        Dictionary mapping strategy name to performance metrics dict.
    """
    strategies = get_all_strategies()
    results = {}

    for name, strategy in strategies.items():
        logger.info(
            "Running backtest",
            extra={"strategy": name},
        )

        try:
            bt_result = run_backtest(predictions, prices, strategy, config)
            metrics = backtest_result_to_dict(bt_result)
            results[name] = metrics

            logger.info(
                "Backtest completed",
                extra={
                    "strategy": name,
                    "annualised_return": metrics["annualised_return"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                },
            )
        except Exception as e:
            logger.error(
                "Backtest failed for strategy",
                extra={"strategy": name, "error": str(e)},
                exc_info=True,
            )
            results[name] = {
                "annualised_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "error": str(e),
            }

    return results


def upload_results_to_s3(
    results: dict,
    output_path: str,
    bucket: str = S3_BUCKET,
) -> None:
    """Upload backtest results as a JSON artifact to S3.

    Args:
        results: Dictionary of strategy results to serialize.
        output_path: S3 key path for the output JSON artifact.
        bucket: S3 bucket name.
    """
    s3 = boto3.client("s3")

    payload = json.dumps(results, indent=2)

    s3.put_object(
        Bucket=bucket,
        Key=output_path,
        Body=payload.encode("utf-8"),
        ContentType="application/json",
    )

    logger.info(
        "Backtest results uploaded to S3",
        extra={"s3_key": output_path, "size_bytes": len(payload)},
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the backtesting component."""
    parser = argparse.ArgumentParser(
        description="Backtesting KFP Component",
    )
    parser.add_argument(
        "--model-checkpoint-path",
        type=str,
        required=True,
        help="S3 key path for the model checkpoint artifact",
    )
    parser.add_argument(
        "--test-dataset-path",
        type=str,
        required=True,
        help="S3 key path for the test dataset artifact",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="S3 key path for the output backtest results JSON artifact",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="USDJPY",
        help="Currency pair symbol",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=1,
        help="Forecast horizon in bars",
    )
    parser.add_argument(
        "--lookback-window",
        type=int,
        default=36,
        help="Lookback window in bars",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=10_000_000,
        help="Position size in base currency units",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=0.05,
        help="Risk aversion parameter (gamma) for mean-variance strategy",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=S3_BUCKET,
        help="S3 bucket name",
    )
    parser.add_argument("--config-json", type=str, default="{}", help="JSON config blob from pipeline")
    return parser.parse_args()


def main() -> None:
    """Main entry point for the backtesting component."""
    args = parse_args()
    import json as _json
    _cfg = _json.loads(args.config_json)
    for _key, _val in _cfg.items():
        if hasattr(args, _key.replace("-", "_")):
            setattr(args, _key.replace("-", "_"), _val)

    logger.info(
        "Backtesting component started",
        extra={
            "model_checkpoint_path": args.model_checkpoint_path,
            "test_dataset_path": args.test_dataset_path,
            "output_path": args.output_path,
            "symbol": args.symbol,
            "forecast_horizon": args.forecast_horizon,
            "lookback_window": args.lookback_window,
            "position_size": args.position_size,
            "risk_aversion": args.risk_aversion,
        },
    )

    # Step 1: Load model checkpoint from S3
    checkpoint = load_checkpoint_from_s3(
        args.model_checkpoint_path, bucket=args.bucket
    )

    # Reconstruct model config from checkpoint
    if "config" in checkpoint:
        config = ForecasterConfig(**checkpoint["config"])
    else:
        config = ForecasterConfig(
            symbol=args.symbol,
            forecast_horizon=args.forecast_horizon,
            lookback_window=args.lookback_window,
            position_size=args.position_size,
            risk_aversion=args.risk_aversion,
        )

    # Override position_size and risk_aversion from CLI args
    config.position_size = args.position_size
    config.risk_aversion = args.risk_aversion

    # Instantiate and load model
    model = ProbabilisticTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(
        "Model loaded",
        extra={
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "num_features": config.num_features,
        },
    )

    # Step 2: Load test dataset from S3
    dataset = load_dataset_from_s3(args.test_dataset_path, bucket=args.bucket)

    # Step 3: Generate predictions from model
    predictions = generate_predictions(model, dataset, config)

    # Step 4: Extract prices from dataset for backtesting
    prices = get_prices_from_dataset(dataset)

    # Step 5: Run all trading strategies
    strategy_results = run_all_strategies(predictions, prices, config)

    # Step 6: Assemble final output
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": config.symbol,
        "forecast_horizon": config.forecast_horizon,
        "lookback_window": config.lookback_window,
        "position_size": config.position_size,
        "risk_aversion": config.risk_aversion,
        "num_test_samples": len(dataset),
        "num_predictions": len(predictions),
        "strategies": strategy_results,
    }

    # Step 7: Upload results to S3
    upload_results_to_s3(output, args.output_path, bucket=args.bucket)

    logger.info(
        "Backtesting component completed successfully",
        extra={
            "num_strategies": len(strategy_results),
            "output_artifact": args.output_path,
        },
    )


if __name__ == "__main__":
    main()
