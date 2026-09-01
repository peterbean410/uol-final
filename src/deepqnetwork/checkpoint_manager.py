"""Checkpoint management for the DQN trading agent.

Handles local save/load and S3 upload/download of model checkpoints.
S3 uploads use exponential backoff retry and never abort training on failure.
"""

from __future__ import annotations

import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import torch
import torch.nn as nn
from botocore.exceptions import ClientError
from torch.optim import Optimizer

logger = logging.getLogger(__name__)

_S3_MAX_RETRIES = 3
_S3_BASE_DELAY = 1.0
_S3_BACKOFF_FACTOR = 2.0


class CheckpointManager:
    """Manages saving, loading, and S3 synchronisation of DQN checkpoints.

    Checkpoints are saved locally with the pattern `dqn_episode_{N}.pt` and
    uploaded to S3 with the pattern `{YYYYMMDD}T{HH}0000Z.pt` (UTC, minutes
    and seconds zeroed to the hour).

    S3 path structure:
        s3://{bucket}/deepqnetwork/model/symbol={SYMBOL}/horizon={HORIZON}/version={VERSION}/{filename}
    """

    def __init__(
        self,
        checkpoint_dir: str = "deepqnetwork/checkpoints/",
        s3_prefix: str | None = None,
        symbol: str = "",
        horizon: str = "",
        version: str = "",
    ) -> None:
        """Initialise the checkpoint manager.

        Args:
            checkpoint_dir: Local directory for saving checkpoint files.
            s3_prefix: S3 prefix for uploads (e.g. "s3://bucket/deepqnetwork/model/").
                If None, S3 upload is disabled.
            symbol: Trading symbol (e.g. "USDJPY") for S3 path construction.
            horizon: Time horizon identifier for S3 path construction.
            version: Model version identifier for S3 path construction.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.s3_prefix = s3_prefix
        self.symbol = symbol
        self.horizon = horizon
        self.version = version

        self._s3_client = None

    @property
    def s3_client(self):
        """Lazily create and return the boto3 S3 client."""
        if self._s3_client is None:
            self._s3_client = boto3.client("s3")
        return self._s3_client

    def save(
        self,
        episode: int,
        q_network: nn.Module,
        target_network: nn.Module,
        optimizer: Optimizer,
        epsilon: float,
        step_count: int,
        config: dict | None = None,
    ) -> str:
        """Save checkpoint locally and upload to S3.

        Saves the full training state as `dqn_episode_{episode}.pt` in the
        local checkpoint directory. If S3 is configured, uploads with the
        timestamp-based filename `{YYYYMMDD}T{HH}0000Z.pt`.

        Args:
            episode: Current episode number.
            q_network: The online Q-Network.
            target_network: The target Q-Network.
            optimizer: The optimiser instance.
            epsilon: Current exploration epsilon value.
            step_count: Total training steps completed.
            config: Optional full resolved config dict for reproducibility.

        Returns:
            The local file path of the saved checkpoint.
        """
        checkpoint_dict = {
            "q_network_state_dict": q_network.state_dict(),
            "target_network_state_dict": target_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epsilon": epsilon,
            "step_count": step_count,
            "episode_count": episode,
            "config": config or {},
        }

        local_filename = f"dqn_episode_{episode}.pt"
        local_path = self.checkpoint_dir / local_filename
        torch.save(checkpoint_dict, str(local_path))
        logger.info("Checkpoint saved locally: %s", local_path)

        if self.s3_prefix is not None:
            s3_filename = self._generate_s3_filename()
            self._upload_to_s3(str(local_path), s3_filename)

        return str(local_path)

    def load(self, path: str) -> dict:
        """Load a checkpoint from a local path or S3.

        If the path starts with `s3://`, the checkpoint is downloaded to a
        local temporary file first. Load failures are handled gracefully by
        logging a warning and returning an empty dict.

        Args:
            path: Local file path or S3 URI (s3://bucket/key).

        Returns:
            The checkpoint dictionary, or an empty dict if loading fails.
        """
        local_path = path

        if path.startswith("s3://"):
            local_path = self._download_from_s3(path)
            if local_path is None:
                logger.warning(
                    "Failed to download checkpoint from S3: %s. "
                    "Starting from scratch.",
                    path,
                )
                return {}

        try:
            checkpoint = torch.load(
                local_path, map_location="cpu", weights_only=False
            )
            logger.info("Checkpoint loaded from: %s", path)
            return checkpoint
        except FileNotFoundError:
            logger.warning(
                "Checkpoint file not found: %s. Starting from scratch.", path
            )
            return {}
        except Exception as e:
            logger.warning(
                "Failed to load checkpoint from %s: %s. Starting from scratch.",
                path,
                e,
            )
            return {}

    def _generate_s3_filename(self) -> str:
        """Generate S3 filename from current UTC time with zeroed minutes/seconds.

        Returns:
            Filename in the format `{YYYYMMDD}T{HH}0000Z.pt`.
        """
        now = datetime.now(timezone.utc)
        return now.strftime("%Y%m%dT%H") + "0000Z.pt"

    def _build_s3_key(self, filename: str) -> tuple[str, str]:
        """Build the full S3 bucket and key from the configured prefix.

        The S3 path is constructed as:
            {s3_prefix}/symbol={SYMBOL}/horizon={HORIZON}/version={VERSION}/{filename}

        Args:
            filename: The checkpoint filename (e.g. "20240101T120000Z.pt").

        Returns:
            Tuple of (bucket_name, object_key).
        """
        prefix = self.s3_prefix.rstrip("/")
        if prefix.startswith("s3://"):
            prefix = prefix[5:]

        parts = prefix.split("/", 1)
        bucket = parts[0]
        key_prefix = parts[1] if len(parts) > 1 else ""

        partition = (
            f"symbol={self.symbol}/horizon={self.horizon}/version={self.version}"
        )
        if key_prefix:
            object_key = f"{key_prefix}/{partition}/{filename}"
        else:
            object_key = f"{partition}/{filename}"

        return bucket, object_key

    def _upload_to_s3(self, local_path: str, s3_filename: str) -> None:
        """Upload a checkpoint file to S3 with retry and exponential backoff.

        Retries up to 3 times on failure. Logs a warning on exhaustion but
        does not raise, training continues with the local checkpoint.

        Args:
            local_path: Path to the local checkpoint file.
            s3_filename: The filename to use in S3.
        """
        bucket, key = self._build_s3_key(s3_filename)
        delay = _S3_BASE_DELAY

        for attempt in range(1, _S3_MAX_RETRIES + 1):
            try:
                self.s3_client.upload_file(local_path, bucket, key)
                logger.info(
                    "Checkpoint uploaded to S3: s3://%s/%s", bucket, key
                )
                return
            except (ClientError, Exception) as e:
                if attempt < _S3_MAX_RETRIES:
                    logger.warning(
                        "S3 upload attempt %d/%d failed: %s. "
                        "Retrying in %.1fs...",
                        attempt,
                        _S3_MAX_RETRIES,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= _S3_BACKOFF_FACTOR
                else:
                    logger.warning(
                        "S3 upload failed after %d attempts: %s. "
                        "Local checkpoint remains valid at %s.",
                        _S3_MAX_RETRIES,
                        e,
                        local_path,
                    )

    def _download_from_s3(self, s3_uri: str) -> str | None:
        """Download a checkpoint from S3 to a local temporary file.

        Args:
            s3_uri: Full S3 URI (e.g. "s3://bucket/path/to/checkpoint.pt").

        Returns:
            Path to the downloaded local file, or None on failure.
        """
        uri = s3_uri[5:]
        parts = uri.split("/", 1)
        if len(parts) < 2:
            logger.warning("Invalid S3 URI: %s", s3_uri)
            return None

        bucket = parts[0]
        key = parts[1]

        local_path = self.checkpoint_dir / Path(key).name
        try:
            self.s3_client.download_file(bucket, key, str(local_path))
            logger.info("Checkpoint downloaded from S3: %s → %s", s3_uri, local_path)
            return str(local_path)
        except (ClientError, Exception) as e:
            logger.warning("Failed to download from S3 %s: %s", s3_uri, e)
            return None
