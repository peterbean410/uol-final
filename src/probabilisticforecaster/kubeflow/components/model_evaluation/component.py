"""Model Evaluation KFP component.

Loads a model checkpoint and test dataset from S3, computes evaluation metrics
(NLL, directional accuracy, 95% coverage ratio, RMSE), performs a forgetting
check on a 10% sample of the training data, and runs a degradation gate
comparing against production model metrics.

Handles initial deployment bootstrap: if no production metrics are available,
the gate is skipped and the model is auto-promoted.

Requirements: 1.5, 9.4
"""

import argparse
import io
import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import boto3
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Add parent paths so we can import the probabilisticforecaster package
sys.path.insert(0, "/app")

from probabilisticforecaster.config import ForecasterConfig, S3_BUCKET
from probabilisticforecaster.dataset import ForexDataset
from probabilisticforecaster.model import ProbabilisticTransformer
from probabilisticforecaster.kubeflow.monitoring.metrics import get_logger

logger = get_logger(__name__, component="model_evaluation")

# Degradation thresholds (from pipeline config)
DEFAULT_NLL_DEGRADATION_THRESHOLD = 0.1
DEFAULT_DA_DEGRADATION_THRESHOLD = 0.05

# Fixed seed for forgetting check sampling
FORGETTING_CHECK_SEED = 42
FORGETTING_CHECK_SAMPLE_RATIO = 0.10


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics."""

    nll: float
    directional_accuracy: float
    coverage_ratio_95: float
    rmse: float


@dataclass
class EvaluationResult:
    """Complete evaluation output including gate decision."""

    test_metrics: EvaluationMetrics
    forgetting_metrics: EvaluationMetrics | None
    gate_passed: bool
    gate_skipped: bool
    gate_reason: str
    timestamp: str


def compute_nll(mu: np.ndarray, sigma: np.ndarray, actual: np.ndarray) -> float:
    """Compute mean Gaussian negative log-likelihood.

    NLL = mean(0.5 * (log(σ²) + ((x - μ) / σ)² + log(2π)))

    Args:
        mu: Predicted means, shape (N,).
        sigma: Predicted std devs (> 0), shape (N,).
        actual: Realized values, shape (N,).

    Returns:
        Mean NLL as a float.
    """
    variance = sigma**2
    log_variance = np.log(variance)
    squared_error = ((actual - mu) / sigma) ** 2
    log_2pi = math.log(2 * math.pi)

    nll = 0.5 * (log_variance + squared_error + log_2pi)
    return float(np.mean(nll))


def compute_directional_accuracy(mu: np.ndarray, actual: np.ndarray) -> float:
    """Compute directional accuracy.

    DA = count(sign(μ̂) == sign(actual)) / N

    Args:
        mu: Predicted means, shape (N,).
        actual: Realized values, shape (N,).

    Returns:
        Directional accuracy as a float in [0, 1].
    """
    return float(np.mean(np.sign(mu) == np.sign(actual)))


def compute_coverage_ratio_95(
    mu: np.ndarray, sigma: np.ndarray, actual: np.ndarray
) -> float:
    """Compute 95% coverage ratio.

    CR95 = count(|actual - μ̂| ≤ 1.96 * σ̂) / N

    Args:
        mu: Predicted means, shape (N,).
        sigma: Predicted std devs (> 0), shape (N,).
        actual: Realized values, shape (N,).

    Returns:
        Coverage ratio as a float in [0, 1].
    """
    return float(np.mean(np.abs(actual - mu) <= 1.96 * sigma))


def compute_rmse(mu: np.ndarray, actual: np.ndarray) -> float:
    """Compute root mean squared error.

    RMSE = sqrt(mean((μ̂ - actual)²))

    Args:
        mu: Predicted means, shape (N,).
        actual: Realized values, shape (N,).

    Returns:
        RMSE as a float.
    """
    return float(np.sqrt(np.mean((mu - actual) ** 2)))


def compute_all_metrics(
    mu: np.ndarray, sigma: np.ndarray, actual: np.ndarray
) -> EvaluationMetrics:
    """Compute all four evaluation metrics from prediction arrays.

    Args:
        mu: Predicted means, shape (N,).
        sigma: Predicted std devs (> 0), shape (N,).
        actual: Realized values, shape (N,).

    Returns:
        EvaluationMetrics with nll, directional_accuracy, coverage_ratio_95, rmse.
    """
    return EvaluationMetrics(
        nll=compute_nll(mu, sigma, actual),
        directional_accuracy=compute_directional_accuracy(mu, actual),
        coverage_ratio_95=compute_coverage_ratio_95(mu, sigma, actual),
        rmse=compute_rmse(mu, actual),
    )


def collect_predictions(
    model: ProbabilisticTransformer,
    dataset: ForexDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference on a dataset and collect predictions.

    Args:
        model: Trained ProbabilisticTransformer model in eval mode.
        dataset: ForexDataset to evaluate on.
        batch_size: Batch size for inference.
        device: Device to run inference on.

    Returns:
        Tuple of (mu_array, sigma_array, actual_array), each shape (N,).
    """
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    all_mu: list[np.ndarray] = []
    all_sigma: list[np.ndarray] = []
    all_actual: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)  # (batch, lookback, 16)

            mu, sigma = model(features)

            # Use only the last position for evaluation
            mu_last = mu[:, -1, 0]  # (batch,)
            sigma_last = sigma[:, -1, 0]  # (batch,)
            actual = labels[:, 0]  # (batch,)

            all_mu.append(mu_last.cpu().numpy())
            all_sigma.append(sigma_last.cpu().numpy())
            all_actual.append(actual.numpy())

    mu_array = np.concatenate(all_mu)
    sigma_array = np.concatenate(all_sigma)
    actual_array = np.concatenate(all_actual)

    return mu_array, sigma_array, actual_array


def collect_predictions_subset(
    model: ProbabilisticTransformer,
    dataset: ForexDataset,
    indices: list[int],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference on a subset of a dataset.

    Args:
        model: Trained ProbabilisticTransformer model in eval mode.
        dataset: Full ForexDataset.
        indices: List of sample indices to evaluate.
        batch_size: Batch size for inference.
        device: Device to run inference on.

    Returns:
        Tuple of (mu_array, sigma_array, actual_array), each shape (N,).
    """
    subset = Subset(dataset, indices)

    model.eval()
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    all_mu: list[np.ndarray] = []
    all_sigma: list[np.ndarray] = []
    all_actual: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)

            mu, sigma = model(features)

            mu_last = mu[:, -1, 0]
            sigma_last = sigma[:, -1, 0]
            actual = labels[:, 0]

            all_mu.append(mu_last.cpu().numpy())
            all_sigma.append(sigma_last.cpu().numpy())
            all_actual.append(actual.numpy())

    mu_array = np.concatenate(all_mu)
    sigma_array = np.concatenate(all_sigma)
    actual_array = np.concatenate(all_actual)

    return mu_array, sigma_array, actual_array


def evaluate_on_dataset(
    model: ProbabilisticTransformer,
    dataset: ForexDataset,
    batch_size: int,
    device: torch.device,
) -> EvaluationMetrics:
    """Evaluate model on a full dataset.

    Args:
        model: Trained ProbabilisticTransformer model.
        dataset: ForexDataset to evaluate on.
        batch_size: Batch size for inference.
        device: Device to run inference on.

    Returns:
        EvaluationMetrics computed over the full dataset.
    """
    mu, sigma, actual = collect_predictions(model, dataset, batch_size, device)
    return compute_all_metrics(mu, sigma, actual)


def forgetting_check(
    model: ProbabilisticTransformer,
    train_dataset: ForexDataset,
    batch_size: int,
    device: torch.device,
    sample_ratio: float = FORGETTING_CHECK_SAMPLE_RATIO,
    seed: int = FORGETTING_CHECK_SEED,
) -> EvaluationMetrics:
    """Evaluate model on a random 10% sample of the training data.

    Uses a fixed seed for reproducibility across runs. This checks whether
    the model has "forgotten" patterns from the training period.

    Args:
        model: Trained ProbabilisticTransformer model.
        train_dataset: Full training ForexDataset.
        batch_size: Batch size for inference.
        device: Device to run inference on.
        sample_ratio: Fraction of training data to sample (default 0.10).
        seed: Fixed random seed for reproducible sampling.

    Returns:
        EvaluationMetrics computed over the sampled training subset.
    """
    n_total = len(train_dataset)
    n_sample = max(1, int(n_total * sample_ratio))

    # Use fixed seed for reproducible sampling
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n_sample, replace=False).tolist()

    logger.info(
        "Forgetting check: sampling training data",
        extra={
            "total_samples": n_total,
            "sampled_samples": n_sample,
            "sample_ratio": sample_ratio,
            "seed": seed,
        },
    )

    mu, sigma, actual = collect_predictions_subset(
        model, train_dataset, indices, batch_size, device
    )
    metrics = compute_all_metrics(mu, sigma, actual)

    logger.info(
        "Forgetting check completed",
        extra={
            "nll": round(metrics.nll, 6),
            "directional_accuracy": round(metrics.directional_accuracy, 4),
            "coverage_ratio_95": round(metrics.coverage_ratio_95, 4),
            "rmse": round(metrics.rmse, 6),
        },
    )

    return metrics


def degradation_gate(
    current_metrics: EvaluationMetrics,
    production_metrics: dict,
    nll_threshold: float = DEFAULT_NLL_DEGRADATION_THRESHOLD,
    da_threshold: float = DEFAULT_DA_DEGRADATION_THRESHOLD,
) -> tuple[bool, str]:
    """Compare current model metrics against production model metrics.

    The gate fails (returns False) if:
    - Current NLL exceeds production NLL by more than nll_threshold
    - Current DA drops below production DA by more than da_threshold

    Args:
        current_metrics: Metrics from the newly trained model.
        production_metrics: Dictionary with production model metrics
            (keys: "nll", "directional_accuracy").
        nll_threshold: Maximum allowed NLL increase over production.
        da_threshold: Maximum allowed DA decrease from production.

    Returns:
        Tuple of (gate_passed: bool, reason: str).
    """
    prod_nll = production_metrics.get("nll", 0.0)
    prod_da = production_metrics.get("directional_accuracy", 0.0)

    nll_delta = current_metrics.nll - prod_nll
    da_delta = prod_da - current_metrics.directional_accuracy

    reasons = []

    if nll_delta > nll_threshold:
        reasons.append(
            f"NLL degraded: current={current_metrics.nll:.6f}, "
            f"production={prod_nll:.6f}, delta={nll_delta:.6f} > threshold={nll_threshold}"
        )

    if da_delta > da_threshold:
        reasons.append(
            f"DA degraded: current={current_metrics.directional_accuracy:.4f}, "
            f"production={prod_da:.4f}, delta={da_delta:.4f} > threshold={da_threshold}"
        )

    if reasons:
        return False, "; ".join(reasons)

    return True, "All metrics within acceptable thresholds"


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
        "Loading dataset from S3",
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
        "Dataset loaded successfully",
        extra={"num_samples": len(dataset)},
    )
    return dataset


def load_production_metrics_from_s3(
    metrics_path: str,
    bucket: str = S3_BUCKET,
) -> dict:
    """Load production model metrics JSON from S3.

    Args:
        metrics_path: S3 key path for the production metrics JSON.
        bucket: S3 bucket name.

    Returns:
        Dictionary with production model metrics.

    Raises:
        FileNotFoundError: If the metrics file does not exist in S3.
    """
    s3 = boto3.client("s3")

    logger.info(
        "Loading production metrics from S3",
        extra={"s3_key": metrics_path, "bucket": bucket},
    )

    try:
        obj = s3.get_object(Bucket=bucket, Key=metrics_path)
    except Exception as e:
        if "NoSuchKey" in str(type(e).__name__) or "NoSuchKey" in str(e) or "404" in str(e):
            raise FileNotFoundError(
                f"Production metrics not found: s3://{bucket}/{metrics_path}"
            ) from e
        raise

    data = json.loads(obj["Body"].read().decode("utf-8"))

    logger.info(
        "Production metrics loaded",
        extra={
            "prod_nll": data.get("nll"),
            "prod_da": data.get("directional_accuracy"),
        },
    )
    return data


def upload_metrics_to_s3(
    metrics: dict,
    output_path: str,
    bucket: str = S3_BUCKET,
) -> None:
    """Upload evaluation metrics as a JSON artifact to S3.

    Args:
        metrics: Dictionary of evaluation results to serialize.
        output_path: S3 key path for the output JSON artifact.
        bucket: S3 bucket name.
    """
    s3 = boto3.client("s3")

    payload = json.dumps(metrics, indent=2)

    s3.put_object(
        Bucket=bucket,
        Key=output_path,
        Body=payload.encode("utf-8"),
        ContentType="application/json",
    )

    logger.info(
        "Evaluation metrics uploaded to S3",
        extra={"s3_key": output_path, "size_bytes": len(payload)},
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the model evaluation component."""
    parser = argparse.ArgumentParser(
        description="Model Evaluation KFP Component",
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
        help="S3 key path for the test dataset artifact (pickle)",
    )
    parser.add_argument(
        "--train-dataset-path",
        type=str,
        default="",
        help="S3 key path for the train dataset artifact (pickle) for forgetting check",
    )
    parser.add_argument(
        "--production-metrics-path",
        type=str,
        default="",
        help="S3 key path for production model metrics JSON (empty = initial deployment, skip gate)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="S3 key path for the output evaluation metrics JSON artifact",
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
        "--num-features",
        type=int,
        default=16,
        help="Number of input features",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of Transformer layers",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--nll-degradation-threshold",
        type=float,
        default=DEFAULT_NLL_DEGRADATION_THRESHOLD,
        help="Maximum allowed NLL increase over production model",
    )
    parser.add_argument(
        "--da-degradation-threshold",
        type=float,
        default=DEFAULT_DA_DEGRADATION_THRESHOLD,
        help="Maximum allowed DA decrease from production model",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=S3_BUCKET,
        help="S3 bucket name",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the model evaluation component."""
    args = parse_args()

    logger.info(
        "Model evaluation component started",
        extra={
            "model_checkpoint_path": args.model_checkpoint_path,
            "test_dataset_path": args.test_dataset_path,
            "train_dataset_path": args.train_dataset_path,
            "production_metrics_path": args.production_metrics_path,
            "output_path": args.output_path,
            "symbol": args.symbol,
            "forecast_horizon": args.forecast_horizon,
            "nll_degradation_threshold": args.nll_degradation_threshold,
            "da_degradation_threshold": args.da_degradation_threshold,
        },
    )

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device selected", extra={"device": str(device)})

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
            lookback_window=args.lookback_window,
            forecast_horizon=args.forecast_horizon,
            num_features=args.num_features,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )

    # Instantiate and load model
    model = ProbabilisticTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(
        "Model loaded for evaluation",
        extra={
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "num_features": config.num_features,
            "lookback_window": config.lookback_window,
        },
    )

    # Step 2: Load test dataset from S3
    test_dataset = load_dataset_from_s3(args.test_dataset_path, bucket=args.bucket)

    # Step 3: Compute evaluation metrics on test dataset
    logger.info(
        "Evaluating model on test dataset",
        extra={"num_test_samples": len(test_dataset)},
    )
    test_metrics = evaluate_on_dataset(
        model, test_dataset, args.batch_size, device
    )

    logger.info(
        "Test evaluation completed",
        extra={
            "nll": round(test_metrics.nll, 6),
            "directional_accuracy": round(test_metrics.directional_accuracy, 4),
            "coverage_ratio_95": round(test_metrics.coverage_ratio_95, 4),
            "rmse": round(test_metrics.rmse, 6),
        },
    )

    # Step 4: Forgetting check (if train dataset path provided)
    forgetting_metrics = None
    if args.train_dataset_path:
        logger.info("Loading train dataset for forgetting check")
        train_dataset = load_dataset_from_s3(
            args.train_dataset_path, bucket=args.bucket
        )
        forgetting_metrics = forgetting_check(
            model, train_dataset, args.batch_size, device
        )

    # Step 5: Degradation gate
    gate_passed = True
    gate_skipped = False
    gate_reason = ""

    if not args.production_metrics_path:
        # Initial deployment bootstrap: no production model → skip gate, auto-promote
        gate_passed = True
        gate_skipped = True
        gate_reason = "Initial deployment: no production model exists, auto-promoting"
        logger.info(
            "Degradation gate skipped (initial deployment bootstrap)",
            extra={"gate_passed": True, "gate_skipped": True},
        )
    else:
        # Load production metrics and compare
        try:
            production_metrics = load_production_metrics_from_s3(
                args.production_metrics_path, bucket=args.bucket
            )
            gate_passed, gate_reason = degradation_gate(
                current_metrics=test_metrics,
                production_metrics=production_metrics,
                nll_threshold=args.nll_degradation_threshold,
                da_threshold=args.da_degradation_threshold,
            )
            logger.info(
                "Degradation gate evaluated",
                extra={
                    "gate_passed": gate_passed,
                    "gate_reason": gate_reason,
                },
            )
        except FileNotFoundError:
            # Production metrics file doesn't exist, treat as initial deployment
            gate_passed = True
            gate_skipped = True
            gate_reason = (
                "Production metrics file not found, treating as initial deployment"
            )
            logger.warning(
                "Production metrics not found, skipping gate",
                extra={
                    "production_metrics_path": args.production_metrics_path,
                    "gate_passed": True,
                    "gate_skipped": True,
                },
            )

    # Step 6: Assemble output
    timestamp = datetime.now(timezone.utc).isoformat()

    output = {
        "timestamp": timestamp,
        "symbol": config.symbol,
        "forecast_horizon": config.forecast_horizon,
        "lookback_window": config.lookback_window,
        "num_test_samples": len(test_dataset),
        "test_metrics": {
            "nll": test_metrics.nll,
            "directional_accuracy": test_metrics.directional_accuracy,
            "coverage_ratio_95": test_metrics.coverage_ratio_95,
            "rmse": test_metrics.rmse,
        },
        "forgetting_check": None,
        "degradation_gate": {
            "gate_passed": gate_passed,
            "gate_skipped": gate_skipped,
            "reason": gate_reason,
            "nll_degradation_threshold": args.nll_degradation_threshold,
            "da_degradation_threshold": args.da_degradation_threshold,
        },
    }

    if forgetting_metrics is not None:
        output["forgetting_check"] = {
            "nll": forgetting_metrics.nll,
            "directional_accuracy": forgetting_metrics.directional_accuracy,
            "coverage_ratio_95": forgetting_metrics.coverage_ratio_95,
            "rmse": forgetting_metrics.rmse,
            "sample_ratio": FORGETTING_CHECK_SAMPLE_RATIO,
            "seed": FORGETTING_CHECK_SEED,
        }

    # Step 7: Upload metrics to S3
    upload_metrics_to_s3(output, args.output_path, bucket=args.bucket)

    # Log final summary
    if not gate_passed:
        logger.warning(
            "Model evaluation completed, GATE FAILED (flagged for manual review)",
            extra={
                "gate_passed": False,
                "gate_reason": gate_reason,
                "output_artifact": args.output_path,
            },
        )
    else:
        logger.info(
            "Model evaluation component completed successfully",
            extra={
                "gate_passed": gate_passed,
                "gate_skipped": gate_skipped,
                "output_artifact": args.output_path,
                "test_nll": round(test_metrics.nll, 6),
                "test_da": round(test_metrics.directional_accuracy, 4),
            },
        )


if __name__ == "__main__":
    main()
