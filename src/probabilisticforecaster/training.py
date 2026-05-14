"""Training loop for the Probabilistic Transformer Forecaster.

Implements Gaussian NLL loss, Adam optimizer with checkpointing,
and S3 model upload for persistence.
"""

import io
import math
from dataclasses import asdict
from datetime import datetime, timezone

import boto3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from probabilisticforecaster.config import ForecasterConfig, S3_BUCKET
from probabilisticforecaster.dataset import ForexDataset
from probabilisticforecaster.model import ProbabilisticTransformer


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
    val_dataset: ForexDataset | None = None,
    version: int = 1,
    upload_to_s3: bool = True,
) -> dict[str, list[float]]:
    """Train the Probabilistic Transformer model using Gaussian NLL loss.

    Uses Adam optimizer with the learning rate, batch size, and epochs
    specified in the config. Sets a fixed random seed for reproducibility.
    Saves the best model by validation NLL with checkpointing.

    Args:
        model: The ProbabilisticTransformer model to train.
        train_dataset: Training dataset (ForexDataset).
        config: ForecasterConfig with training hyperparameters.
        val_dataset: Optional validation dataset. If None, 10% of train_dataset
                     is split off for validation.
        version: Model version number for S3 path.
        upload_to_s3: Whether to upload the trained model to S3.

    Returns:
        Dictionary with training history:
            {"epoch_loss": [float, ...], "batch_losses": [float, ...]}

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

    # Split off validation set if not provided
    if val_dataset is None:
        val_size = max(1, int(len(train_dataset) * 0.1))
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.random_seed),
        )
    else:
        train_subset = train_dataset
        val_subset = val_dataset

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
    device = next(model.parameters()).device
    best_val_nll = float("inf")
    best_state_dict = None
    training_history: dict[str, list[float]] = {
        "epoch_loss": [],
        "batch_losses": [],
    }

    # Training loop
    model.train()
    for epoch in range(config.epochs):
        epoch_losses: list[float] = []

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)  # (batch, 36, 16)
            labels = labels.to(device)  # (batch, 1)

            optimizer.zero_grad()

            # Forward pass, model outputs (batch, seq_len, 1) for mu and sigma
            mu, sigma = model(features)

            # Use only the last position prediction (sequence-to-one for loss)
            mu_last = mu[:, -1, :]  # (batch, 1)
            sigma_last = sigma[:, -1, :]  # (batch, 1)

            # Compute Gaussian NLL loss
            loss = gaussian_nll_loss(mu_last, sigma_last, labels)

            # Check for NaN loss
            if torch.isnan(loss):
                raise RuntimeError(
                    f"NaN loss at epoch {epoch + 1}, batch {batch_idx + 1}. "
                    "Check learning rate."
                )

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_losses.append(batch_loss)
            training_history["batch_losses"].append(batch_loss)

        # Compute mean epoch loss
        mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        training_history["epoch_loss"].append(mean_epoch_loss)

        # Validation pass
        val_nll = _compute_validation_nll(model, val_loader, device)

        # Checkpoint best model by validation NLL
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best model weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Save model with metadata
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final_train_nll = training_history["epoch_loss"][-1] if training_history["epoch_loss"] else 0.0

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "training_history": training_history,
        "metadata": {
            "symbol": config.symbol,
            "horizon": config.forecast_horizon,
            "trained_at": timestamp,
            "train_nll": final_train_nll,
        },
    }

    # Save locally
    torch.save(checkpoint, config.model_path)

    # Upload to S3
    if upload_to_s3:
        s3_key = config.get_s3_model_path(version=version, timestamp=timestamp)
        _upload_model_to_s3(checkpoint, s3_key)

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


def _upload_model_to_s3(checkpoint: dict, s3_key: str) -> None:
    """Upload model checkpoint to S3.

    Args:
        checkpoint: Dictionary containing model state, config, history, metadata.
        s3_key: S3 object key path.
    """
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)

    s3 = boto3.client("s3")
    s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buffer.getvalue())
