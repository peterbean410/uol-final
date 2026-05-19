"""Model Training KFP component.

Loads the train dataset (pickle) from S3, instantiates ProbabilisticTransformer,
trains with Gaussian NLL loss, and uploads the model checkpoint artifact to S3.

Supports two training modes:
  - scratch: Random initialisation, full epochs (5), standard LR (0.001)
  - finetune: Load production model weights from S3, reduced LR (0.0001), fewer epochs (2)

Requirements: 1.4, 2.1, 2.2, 2.3, 2.4, 2.5
"""

import argparse
import io
import math
import os
import pickle
import sys
from dataclasses import asdict
from datetime import datetime, timezone

import boto3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add parent paths so we can import the probabilisticforecaster package
sys.path.insert(0, "/app")

from probabilisticforecaster.config import ForecasterConfig, S3_BUCKET
from probabilisticforecaster.dataset import ForexDataset
from probabilisticforecaster.model import ProbabilisticTransformer
from probabilisticforecaster.kubeflow.monitoring.metrics import get_logger

logger = get_logger(__name__, component="model_training")


def load_dataset_from_s3(s3_key: str, bucket: str = S3_BUCKET) -> ForexDataset:
    """Download and deserialize a ForexDataset pickle from S3.

    Args:
        s3_key: S3 object key for the dataset artifact.
        bucket: S3 bucket name.

    Returns:
        Deserialized ForexDataset instance.

    Raises:
        FileNotFoundError: If the S3 key does not exist.
    """
    s3 = boto3.client("s3")

    logger.info(
        "Loading dataset from S3",
        extra={"s3_key": s3_key, "bucket": bucket},
    )

    try:
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
    except Exception as e:
        if "NoSuchKey" in str(type(e).__name__) or "NoSuchKey" in str(e) or "404" in str(e):
            raise FileNotFoundError(
                f"Dataset not found: s3://{bucket}/{s3_key}"
            ) from e
        raise

    data = obj["Body"].read()
    dataset = pickle.loads(data)

    logger.info(
        "Dataset loaded successfully",
        extra={"s3_key": s3_key, "num_samples": len(dataset)},
    )
    return dataset


def load_model_weights_from_s3(
    s3_key: str, bucket: str = S3_BUCKET
) -> dict:
    """Download a model checkpoint from S3 and return the state dict.

    Args:
        s3_key: S3 object key for the model checkpoint.
        bucket: S3 bucket name.

    Returns:
        The checkpoint dictionary containing model_state_dict and config.

    Raises:
        FileNotFoundError: If the S3 key does not exist.
    """
    s3 = boto3.client("s3")

    logger.info(
        "Loading model weights from S3 for fine-tuning",
        extra={"s3_key": s3_key, "bucket": bucket},
    )

    try:
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
    except Exception as e:
        if "NoSuchKey" in str(type(e).__name__) or "NoSuchKey" in str(e) or "404" in str(e):
            raise FileNotFoundError(
                f"Model checkpoint not found: s3://{bucket}/{s3_key}"
            ) from e
        raise

    buffer = io.BytesIO(obj["Body"].read())
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=False)

    logger.info(
        "Model weights loaded successfully",
        extra={"s3_key": s3_key},
    )
    return checkpoint


def gaussian_nll_loss(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Compute Gaussian Negative Log-Likelihood loss.

    L = 0.5 * (log(sigma^2) + ((target - mu) / sigma)^2 + log(2π))
    Averaged over the batch.

    Args:
        mu: Predicted mean, shape (batch, ...).
        sigma: Predicted std dev (must be > 0), shape (batch, ...).
        target: Ground truth values, shape (batch, ...).

    Returns:
        Scalar tensor with the mean NLL loss over the batch.
    """
    variance = sigma**2
    log_variance = torch.log(variance)
    squared_error = ((target - mu) / sigma) ** 2
    log_2pi = math.log(2 * math.pi)

    nll = 0.5 * (log_variance + squared_error + log_2pi)
    return nll.mean()


def train_model(
    model: ProbabilisticTransformer,
    train_dataset: ForexDataset,
    config: ForecasterConfig,
    device: torch.device,
) -> dict:
    """Train the ProbabilisticTransformer using Gaussian NLL loss.

    Implements the training loop with Adam optimizer, validation-based
    checkpointing, and structured logging of per-epoch metrics.

    Args:
        model: The ProbabilisticTransformer model to train.
        train_dataset: Training dataset (ForexDataset).
        config: ForecasterConfig with training hyperparameters.
        device: Device to run training on (cpu or cuda).

    Returns:
        Dictionary with training history and best validation NLL:
            {
                "epoch_loss": [float, ...],
                "val_nll": [float, ...],
                "best_val_nll": float,
            }

    Raises:
        ValueError: If training dataset is empty.
        RuntimeError: If NaN loss is detected during training.
    """
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")

    # Set fixed random seed for reproducibility
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    # Split off 10% for validation
    val_size = max(1, int(len(train_dataset) * 0.1))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.random_seed),
    )

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training state
    best_val_nll = float("inf")
    best_state_dict = None
    training_history: dict = {
        "epoch_loss": [],
        "val_nll": [],
        "best_val_nll": float("inf"),
    }

    logger.info(
        "Training started",
        extra={
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "train_samples": train_size,
            "val_samples": val_size,
            "device": str(device),
        },
    )

    # Training loop
    model.train()
    for epoch in range(config.epochs):
        epoch_losses: list[float] = []

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)  # (batch, lookback, 16)
            labels = labels.to(device)  # (batch, 1)

            optimizer.zero_grad()

            # Forward pass, model outputs (batch, seq_len, 1) for mu and sigma
            mu, sigma = model(features)

            # Use only the last position prediction
            mu_last = mu[:, -1, :]  # (batch, 1)
            sigma_last = sigma[:, -1, :]  # (batch, 1)

            # Compute Gaussian NLL loss
            loss = gaussian_nll_loss(mu_last, sigma_last, labels)

            # Check for NaN loss
            if torch.isnan(loss):
                raise RuntimeError(
                    f"NaN loss at epoch {epoch + 1}, batch {batch_idx + 1}. "
                    "Check learning rate or input data."
                )

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        # Compute mean epoch loss
        mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        training_history["epoch_loss"].append(mean_epoch_loss)

        # Validation pass
        val_nll = _compute_validation_nll(model, val_loader, device)
        training_history["val_nll"].append(val_nll)

        logger.info(
            "Epoch completed",
            extra={
                "epoch": epoch + 1,
                "total_epochs": config.epochs,
                "train_nll": round(mean_epoch_loss, 6),
                "val_nll": round(val_nll, 6),
            },
        )

        # Checkpoint best model by validation NLL
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(
                "New best model checkpoint",
                extra={"best_val_nll": round(best_val_nll, 6), "epoch": epoch + 1},
            )

    # Restore best model weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    training_history["best_val_nll"] = best_val_nll

    logger.info(
        "Training completed",
        extra={
            "best_val_nll": round(best_val_nll, 6),
            "final_train_nll": round(training_history["epoch_loss"][-1], 6),
        },
    )

    return training_history


def _compute_validation_nll(
    model: ProbabilisticTransformer,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean NLL on the validation set.

    Args:
        model: The model to evaluate.
        val_loader: DataLoader for validation data.
        device: Device to run computation on.

    Returns:
        Mean validation NLL as a float.
    """
    model.eval()
    total_nll = 0.0
    num_batches = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            mu, sigma = model(features)
            mu_last = mu[:, -1, :]
            sigma_last = sigma[:, -1, :]

            nll = gaussian_nll_loss(mu_last, sigma_last, labels)
            total_nll += nll.item()
            num_batches += 1

    model.train()

    if num_batches == 0:
        return float("inf")

    return total_nll / num_batches


def upload_checkpoint_to_s3(
    checkpoint: dict,
    s3_key: str,
    bucket: str = S3_BUCKET,
) -> None:
    """Serialize and upload a model checkpoint to S3.

    Args:
        checkpoint: Dictionary containing model state, config, history, metadata.
        s3_key: S3 object key path for the checkpoint artifact.
        bucket: S3 bucket name.
    """
    s3 = boto3.client("s3")

    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)

    s3.put_object(Bucket=bucket, Key=s3_key, Body=buffer.getvalue())

    logger.info(
        "Model checkpoint uploaded to S3",
        extra={"s3_key": s3_key, "size_bytes": buffer.getbuffer().nbytes},
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the model training component."""
    parser = argparse.ArgumentParser(
        description="Model Training KFP Component",
    )
    parser.add_argument(
        "--train-dataset-path",
        type=str,
        required=True,
        help="S3 key path for the input train dataset artifact (pickle)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="S3 key path for the output model checkpoint artifact",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["scratch", "finetune"],
        default="scratch",
        help="Training mode: scratch (random init) or finetune (load production weights)",
    )
    parser.add_argument(
        "--production-model-path",
        type=str,
        default="",
        help="S3 key path for the production model weights (required for finetune mode)",
    )
    parser.add_argument("--symbol", type=str, default="USDJPY", help="Currency pair symbol")
    parser.add_argument("--forecast-horizon", type=int, default=1, help="Forecast horizon in bars")
    parser.add_argument("--lookback-window", type=int, default=36, help="Lookback window in bars")
    parser.add_argument("--num-features", type=int, default=16, help="Number of input features")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of Transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bucket", type=str, default=S3_BUCKET, help="S3 bucket name")
    parser.add_argument("--config-json", type=str, default="{}", help="JSON config blob from pipeline")
    return parser.parse_args()


def main() -> None:
    """Main entry point for the model training component."""
    args = parse_args()
    # Merge --config-json overrides into default argparse values
    import json as _json
    _cfg = _json.loads(args.config_json)
    for _key, _val in _cfg.items():
        if hasattr(args, _key.replace("-", "_")):
            setattr(args, _key.replace("-", "_"), _val)

    logger.info(
        "Model training component started",
        extra={
            "training_mode": args.training_mode,
            "symbol": args.symbol,
            "forecast_horizon": args.forecast_horizon,
            "lookback_window": args.lookback_window,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        },
    )

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device selected", extra={"device": str(device)})

    # Step 1: Load train dataset from S3
    train_dataset = load_dataset_from_s3(args.train_dataset_path, bucket=args.bucket)

    # Step 2: Build ForecasterConfig from args
    config = ForecasterConfig(
        symbol=args.symbol,
        lookback_window=args.lookback_window,
        forecast_horizon=args.forecast_horizon,
        num_features=args.num_features,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        random_seed=args.random_seed,
    )

    # Step 3: Apply training mode settings
    if args.training_mode == "finetune":
        # Finetune mode: reduced LR (0.0001) and fewer epochs (2)
        config = ForecasterConfig(
            symbol=config.symbol,
            lookback_window=config.lookback_window,
            forecast_horizon=config.forecast_horizon,
            num_features=config.num_features,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            learning_rate=0.0001,
            batch_size=config.batch_size,
            epochs=2,
            random_seed=config.random_seed,
        )
        logger.info(
            "Finetune mode: using reduced LR and fewer epochs",
            extra={"learning_rate": 0.0001, "epochs": 2},
        )
    else:
        # Scratch mode: standard LR (0.001) and full epochs (5)
        logger.info(
            "Scratch mode: random initialisation with full training",
            extra={"learning_rate": config.learning_rate, "epochs": config.epochs},
        )

    # Step 4: Instantiate model
    model = ProbabilisticTransformer(config)
    model = model.to(device)

    # Step 5: For finetune mode, load production model weights
    if args.training_mode == "finetune":
        if not args.production_model_path:
            raise ValueError(
                "Finetune mode requires --production-model-path to be specified"
            )
        production_checkpoint = load_model_weights_from_s3(
            args.production_model_path, bucket=args.bucket
        )
        model.load_state_dict(production_checkpoint["model_state_dict"])
        logger.info(
            "Production model weights loaded for fine-tuning",
            extra={"production_model_path": args.production_model_path},
        )

    # Step 6: Train the model
    training_history = train_model(model, train_dataset, config, device)

    # Step 7: Build checkpoint artifact
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "training_history": training_history,
        "metadata": {
            "symbol": config.symbol,
            "horizon": config.forecast_horizon,
            "trained_at": timestamp,
            "training_mode": args.training_mode,
            "train_nll": training_history["epoch_loss"][-1],
            "best_val_nll": training_history["best_val_nll"],
            "epochs_trained": config.epochs,
            "learning_rate": config.learning_rate,
        },
    }

    # Step 8: Upload checkpoint to S3
    upload_checkpoint_to_s3(checkpoint, args.checkpoint_path, bucket=args.bucket)

    logger.info(
        "Model training component completed successfully",
        extra={
            "training_mode": args.training_mode,
            "checkpoint_path": args.checkpoint_path,
            "best_val_nll": round(training_history["best_val_nll"], 6),
            "final_train_nll": round(training_history["epoch_loss"][-1], 6),
            "epochs_trained": config.epochs,
        },
    )


if __name__ == "__main__":
    main()
