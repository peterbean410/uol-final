"""Distributed training script for ProbabilisticTransformer using PyTorch DDP.

Implements data-parallel distributed training orchestrated by the Kubeflow
Training Operator (PyTorchJob). Uses NCCL backend for GPU communication and
DistributedSampler for non-overlapping data partitioning across workers.

Environment variables set by the Training Operator:
  - RANK: Global rank of this process (0 = master)
  - WORLD_SIZE: Total number of processes
  - MASTER_ADDR: Address of the master node
  - MASTER_PORT: Port for distributed communication

Graceful fallback: If DDP initialization fails (e.g., single-node without
distributed env vars), falls back to single-GPU training on the local device.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
"""

import io
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

import boto3
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, "/app")

from probabilisticforecaster.config import ForecasterConfig, S3_BUCKET
from probabilisticforecaster.dataset import ForexDataset
from probabilisticforecaster.model import ProbabilisticTransformer
from probabilisticforecaster.kubeflow.monitoring.metrics import get_logger

logger = get_logger(__name__, component="distributed_training")


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


def setup_distributed() -> bool:
    """Initialize DDP process group using env vars set by Training Operator.

    Uses NCCL backend for GPU communication. The Training Operator sets
    RANK, WORLD_SIZE, MASTER_ADDR, and MASTER_PORT environment variables.

    Returns:
        True if distributed setup succeeded, False if fallback to single-GPU
        is needed (e.g., env vars not set or init fails).
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "")
    master_port = os.environ.get("MASTER_PORT", "29500")

    if world_size <= 1 or not master_addr:
        logger.info(
            "Skipping DDP setup: single worker or MASTER_ADDR not set",
            extra={"rank": rank, "world_size": world_size, "master_addr": master_addr},
        )
        return False

    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        logger.info(
            "DDP process group initialized",
            extra={
                "rank": rank,
                "world_size": world_size,
                "master_addr": master_addr,
                "master_port": master_port,
                "backend": "nccl",
            },
        )
        return True
    except Exception as e:
        logger.warning(
            "DDP initialization failed, falling back to single-GPU training",
            extra={
                "error": str(e),
                "rank": rank,
                "world_size": world_size,
                "master_addr": master_addr,
            },
        )
        return False


def _save_checkpoint_to_s3(
    model: ProbabilisticTransformer,
    config: ForecasterConfig,
    epoch: int,
    epoch_loss: float,
    checkpoint_dir: str,
    bucket: str = S3_BUCKET,
) -> str:
    """Save an epoch checkpoint to S3.

    Args:
        model: The unwrapped model (not DDP-wrapped).
        config: Training configuration.
        epoch: Current epoch number (0-indexed).
        epoch_loss: Mean training loss for this epoch.
        checkpoint_dir: S3 key prefix for checkpoints.
        bucket: S3 bucket name.

    Returns:
        The S3 key where the checkpoint was saved.
    """
    s3 = boto3.client("s3")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "epoch_loss": epoch_loss,
        "timestamp": timestamp,
    }

    s3_key = f"{checkpoint_dir}/epoch_{epoch:03d}_{timestamp}.pt"

    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)

    s3.put_object(Bucket=bucket, Key=s3_key, Body=buffer.getvalue())

    logger.info(
        "Epoch checkpoint saved to S3",
        extra={
            "s3_key": s3_key,
            "epoch": epoch,
            "epoch_loss": round(epoch_loss, 6),
            "size_bytes": buffer.getbuffer().nbytes,
        },
    )
    return s3_key


def _save_final_checkpoint_to_s3(
    model: ProbabilisticTransformer,
    config: ForecasterConfig,
    training_history: dict,
    checkpoint_dir: str,
    bucket: str = S3_BUCKET,
) -> str:
    """Save the final consolidated checkpoint to S3 (rank 0 only).

    This is the checkpoint used for evaluation and serving. It contains
    the model state dict, config, and full training history.

    Args:
        model: The unwrapped model (not DDP-wrapped).
        config: Training configuration.
        training_history: Dict with epoch losses and metadata.
        checkpoint_dir: S3 key prefix for checkpoints.
        bucket: S3 bucket name.

    Returns:
        The S3 key where the final checkpoint was saved.
    """
    s3 = boto3.client("s3")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "training_history": training_history,
        "metadata": {
            "symbol": config.symbol,
            "horizon": config.forecast_horizon,
            "trained_at": timestamp,
            "training_mode": "distributed",
            "epochs_trained": config.epochs,
            "learning_rate": config.learning_rate,
            "final_train_nll": training_history["epoch_losses"][-1]
            if training_history["epoch_losses"]
            else None,
        },
    }

    s3_key = f"{checkpoint_dir}/final_model_{timestamp}.pt"

    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)

    s3.put_object(Bucket=bucket, Key=s3_key, Body=buffer.getvalue())

    logger.info(
        "Final consolidated checkpoint saved to S3",
        extra={
            "s3_key": s3_key,
            "size_bytes": buffer.getbuffer().nbytes,
            "epochs_trained": config.epochs,
        },
    )
    return s3_key


def train_distributed(
    model: ProbabilisticTransformer,
    dataset: ForexDataset,
    config: ForecasterConfig,
    checkpoint_dir: str,
    bucket: str = S3_BUCKET,
) -> dict:
    """DDP training loop with epoch checkpointing to S3.

    Handles the full distributed training lifecycle:
    1. Initialize DDP process group (with graceful fallback to single-GPU)
    2. Wrap model in DistributedDataParallel
    3. Create DistributedSampler for non-overlapping data partitioning
    4. Train for configured epochs with gradient synchronization
    5. Save checkpoint to S3 at end of each epoch (rank 0 only)
    6. Save consolidated final checkpoint on rank 0

    Args:
        model: The ProbabilisticTransformer model to train.
        dataset: Training dataset (ForexDataset).
        config: ForecasterConfig with training hyperparameters.
        checkpoint_dir: S3 key prefix for saving checkpoints.
        bucket: S3 bucket name.

    Returns:
        Dictionary with training history:
            {
                "epoch_losses": [float, ...],
                "distributed": bool,
                "world_size": int,
                "rank": int,
                "final_checkpoint_key": str (rank 0 only),
            }

    Raises:
        ValueError: If dataset is empty.
        RuntimeError: If NaN loss is detected during training.
    """
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty")

    is_distributed = setup_distributed()

    rank = int(os.environ.get("RANK", "0")) if is_distributed else 0
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if is_distributed else 1

    if torch.cuda.is_available():
        if is_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    logger.info(
        "Training device configured",
        extra={
            "device": str(device),
            "rank": rank,
            "world_size": world_size,
            "distributed": is_distributed,
        },
    )

    model = model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
        logger.info(
            "Model wrapped in DistributedDataParallel",
            extra={"rank": rank, "device_ids": [device.index] if device.type == "cuda" else None},
        )

    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            drop_last=False,
        )
    else:
        sampler = None
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    torch.manual_seed(config.random_seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed + rank)

    training_history: dict = {
        "epoch_losses": [],
        "distributed": is_distributed,
        "world_size": world_size,
        "rank": rank,
        "final_checkpoint_key": "",
    }

    logger.info(
        "Distributed training started",
        extra={
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "dataset_size": len(dataset),
            "world_size": world_size,
            "rank": rank,
            "distributed": is_distributed,
        },
    )

    model.train()
    for epoch in range(config.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_losses: list[float] = []

        for batch_idx, (features, labels) in enumerate(loader):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            mu, sigma = model(features)

            mu_last = mu[:, -1, :]
            sigma_last = sigma[:, -1, :]

            loss = gaussian_nll_loss(mu_last, sigma_last, labels)

            if torch.isnan(loss):
                raise RuntimeError(
                    f"NaN loss at epoch {epoch + 1}, batch {batch_idx + 1} "
                    f"on rank {rank}. Check learning rate or input data."
                )

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        training_history["epoch_losses"].append(mean_epoch_loss)

        logger.info(
            "Epoch completed",
            extra={
                "epoch": epoch + 1,
                "total_epochs": config.epochs,
                "train_nll": round(mean_epoch_loss, 6),
                "rank": rank,
                "world_size": world_size,
            },
        )

        if rank == 0:
            unwrapped_model = model.module if is_distributed else model
            _save_checkpoint_to_s3(
                model=unwrapped_model,
                config=config,
                epoch=epoch,
                epoch_loss=mean_epoch_loss,
                checkpoint_dir=checkpoint_dir,
                bucket=bucket,
            )

        if is_distributed:
            dist.barrier()

    if rank == 0:
        unwrapped_model = model.module if is_distributed else model
        final_key = _save_final_checkpoint_to_s3(
            model=unwrapped_model,
            config=config,
            training_history=training_history,
            checkpoint_dir=checkpoint_dir,
            bucket=bucket,
        )
        training_history["final_checkpoint_key"] = final_key

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()
        logger.info(
            "DDP process group destroyed",
            extra={"rank": rank},
        )

    logger.info(
        "Distributed training completed",
        extra={
            "rank": rank,
            "world_size": world_size,
            "epochs_trained": config.epochs,
            "final_train_nll": round(training_history["epoch_losses"][-1], 6)
            if training_history["epoch_losses"]
            else None,
            "distributed": is_distributed,
        },
    )

    return training_history
