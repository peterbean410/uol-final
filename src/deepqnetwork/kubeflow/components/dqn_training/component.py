"""DQN Training KFP component.

Loads DQNPipelineConfig, launches the modelenv gRPC server as a subprocess
sidecar, runs DQN training via `deepqnetwork.train.train(config)`, and uploads
the final checkpoint artifact to S3.

Supports two training modes:
  - scratch: Random initialisation, full episodes, standard LR
  - finetune: Load production checkpoint, reduced LR, fewer episodes

The modelenv sidecar lifecycle is managed entirely within this component:
  - Start modelenv-server before training begins
  - Health check localhost:50051 before proceeding
  - Terminate on exit (SIGTERM → wait → SIGKILL)
  - Capture stderr for debugging

Requirements: DQN-R5, DQN-R6, DQN-R7, DQN-R8
"""

import argparse
import io
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone

import boto3
import torch

# Add parent paths so we can import the deepqnetwork package
sys.path.insert(0, "/app")

from deepqnetwork.config import DQNConfig
from deepqnetwork.train import train
from deepqnetwork.kubeflow.pipeline.config_schema import DQNPipelineConfig
from deepqnetwork.swap_rates import resolve_swap_rates
from probabilisticforecaster.kubeflow.monitoring.metrics import get_logger

logger = get_logger(__name__, component="dqn_training")

# S3 bucket for artifacts
S3_BUCKET = os.environ.get("S3_BUCKET", "prod-fintech-forex-sg-731833471586")

# Modelenv sidecar configuration
MODELENV_BINARY = "/usr/local/bin/modelenv-server"
MODELENV_HOST = "localhost"
MODELENV_PORT = 50051
# Seconds to wait for modelenv's startup data preload. It downloads the full
# training window's bars + per-day news snapshots from S3 (~1 file/s cold);
# a 14-year window (adhoc20260720: 2012->2026) is thousands of files, so a
# cold/partially-cold cache PVC can legitimately take over an hour, 600s
# killed a healthy sidecar mid-preload. Overridable via env for odd cases.
MODELENV_HEALTH_CHECK_TIMEOUT = int(
    os.environ.get("MODELENV_HEALTH_CHECK_TIMEOUT", "7200")
)
MODELENV_HEALTH_CHECK_INTERVAL = 1   # seconds
MODELENV_SHUTDOWN_TIMEOUT = 10  # seconds


class ModelenvSidecar:
    """Manages the modelenv gRPC server subprocess lifecycle.

    Starts the modelenv-server binary as a subprocess, waits for the gRPC
    port to become available, and handles graceful shutdown on exit.
    """

    def __init__(
        self,
        symbol: str | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
        hour_start: int | None = None,
        hour_end: int | None = None,
        price_snapshot_date: str | None = None,
    ) -> None:
        self._process: subprocess.Popen | None = None
        self._stderr_lines: list[str] = []
        self._stderr_pump: "threading.Thread | None" = None
        self._symbol = symbol
        # When all four are set, modelenv scopes its startup tick preload to
        # this window (date_start..=date_end × [hour_start..hour_end]) instead
        # of the full M1 reference span. Out-of-window tick files are still
        # downloaded lazily on Reset().
        self._date_start = date_start
        self._date_end = date_end
        self._hour_start = hour_start
        self._hour_end = hour_end
        # When set (ISO date/datetime, UTC assumed if naive), passed to modelenv
        # as --price-snapshot-ts so ALL interval snapshots resolve "latest at or
        # before" this instant instead of at the training window's date_end.
        # Needed when date_end's own snapshot partitions predate a history
        # consolidation and hold only lane-local bars (adhoc20260720: the
        # 2026-04-17 partitions were 2026-only, so 2012 episodes found no bars).
        self._price_snapshot_date = price_snapshot_date

    def start(self) -> None:
        """Start the modelenv-server subprocess.

        Raises:
            FileNotFoundError: If the modelenv binary is not found.
            RuntimeError: If the process fails to start.
        """
        cli_args: list[str] = [MODELENV_BINARY]
        if self._symbol:
            cli_args.extend(["--symbol", self._symbol])
        if (
            self._date_start
            and self._date_end
            and self._hour_start is not None
            and self._hour_end is not None
        ):
            cli_args.extend(
                [
                    "--training-date-start", self._date_start,
                    "--training-date-end",   self._date_end,
                    "--training-hour-start", str(self._hour_start),
                    "--training-hour-end",   str(self._hour_end),
                ]
            )
        if self._price_snapshot_date:
            from datetime import datetime, timezone

            dt = datetime.fromisoformat(self._price_snapshot_date)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            cli_args.extend(
                ["--price-snapshot-ts", str(int(dt.timestamp() * 1_000_000_000))]
            )

        logger.info(
            "Starting modelenv sidecar",
            extra={
                "binary": MODELENV_BINARY,
                "port": MODELENV_PORT,
                "training_window": (
                    f"{self._date_start}..{self._date_end} "
                    f"[{self._hour_start}..{self._hour_end}]"
                    if self._date_start and self._date_end
                    else None
                ),
            },
        )

        try:
            self._process = subprocess.Popen(
                cli_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "MODELENV_PORT": str(MODELENV_PORT)},
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"modelenv binary not found at {MODELENV_BINARY}. "
                "Ensure the DQN base image bundles the binary."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start modelenv sidecar: {e}") from e

        logger.info(
            "modelenv sidecar process started",
            extra={"pid": self._process.pid},
        )

        # Stream modelenv stderr to the parent's stderr live (visible in
        # `kubectl logs`) AND retain it in self._stderr_lines for the
        # post-mortem dump on graceful shutdown. Without the pump, the
        # pipe buffers silently and the only chance to see modelenv logs
        # is a graceful shutdown drain, useless when Reset hangs.
        def _pump_stderr() -> None:
            assert self._process is not None and self._process.stderr is not None
            try:
                for raw_line in iter(self._process.stderr.readline, b""):
                    line = raw_line.decode("utf-8", errors="replace").rstrip()
                    if not line:
                        continue
                    self._stderr_lines.append(line)
                    print(f"[modelenv] {line}", file=sys.stderr, flush=True)
            except (OSError, ValueError):
                pass

        self._stderr_pump = threading.Thread(
            target=_pump_stderr, name="modelenv-stderr-pump", daemon=True,
        )
        self._stderr_pump.start()

    def wait_for_ready(self) -> None:
        """Wait for the modelenv gRPC server to accept connections.

        Polls localhost:50051 until a TCP connection succeeds or the
        timeout is reached.

        Raises:
            RuntimeError: If the server does not become ready within the timeout,
                or if the process exits prematurely.
        """
        logger.info(
            "Waiting for modelenv health check",
            extra={
                "host": MODELENV_HOST,
                "port": MODELENV_PORT,
                "timeout_seconds": MODELENV_HEALTH_CHECK_TIMEOUT,
            },
        )

        deadline = time.time() + MODELENV_HEALTH_CHECK_TIMEOUT

        while time.time() < deadline:
            # Check if process has exited prematurely
            if self._process is not None and self._process.poll() is not None:
                stderr_output = self._capture_stderr()
                raise RuntimeError(
                    f"modelenv sidecar exited prematurely with code "
                    f"{self._process.returncode}. stderr: {stderr_output}"
                )

            # Try TCP connection to the gRPC port
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex((MODELENV_HOST, MODELENV_PORT))
                sock.close()

                if result == 0:
                    logger.info(
                        "modelenv sidecar is ready",
                        extra={"host": MODELENV_HOST, "port": MODELENV_PORT},
                    )
                    return
            except OSError:
                pass

            time.sleep(MODELENV_HEALTH_CHECK_INTERVAL)

        # Timeout reached
        stderr_output = self._capture_stderr()
        raise RuntimeError(
            f"modelenv sidecar did not become ready within "
            f"{MODELENV_HEALTH_CHECK_TIMEOUT}s. stderr: {stderr_output}"
        )

    def stop(self) -> None:
        """Gracefully stop the modelenv sidecar.

        Sends SIGTERM, waits for graceful shutdown, then SIGKILL if needed.
        Captures stderr for debugging.
        """
        if self._process is None or self._process.poll() is not None:
            logger.info("modelenv sidecar already stopped")
            return

        pid = self._process.pid
        logger.info("Stopping modelenv sidecar", extra={"pid": pid})

        # Send SIGTERM for graceful shutdown
        try:
            self._process.send_signal(signal.SIGTERM)
        except OSError as e:
            logger.warning(
                "Failed to send SIGTERM to modelenv",
                extra={"pid": pid, "error": str(e)},
            )
            return

        # Wait for graceful shutdown
        try:
            self._process.wait(timeout=MODELENV_SHUTDOWN_TIMEOUT)
            logger.info(
                "modelenv sidecar stopped gracefully",
                extra={"pid": pid, "returncode": self._process.returncode},
            )
        except subprocess.TimeoutExpired:
            # Force kill if graceful shutdown fails
            logger.warning(
                "modelenv sidecar did not stop gracefully, sending SIGKILL",
                extra={"pid": pid},
            )
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass

        # Capture stderr for debugging
        stderr_output = self._capture_stderr()
        if stderr_output:
            logger.info(
                "modelenv sidecar stderr output",
                extra={"stderr": stderr_output},
            )

    def _capture_stderr(self) -> str:
        """Return all stderr lines accumulated by the pump thread.

        The pump thread continuously drains the pipe and appends each line
        to ``self._stderr_lines``. We briefly join the pump so any final
        lines emitted after SIGTERM but before process exit get flushed
        into the buffer.
        """
        if self._stderr_pump is not None and self._stderr_pump.is_alive():
            self._stderr_pump.join(timeout=2.0)
        return "\n".join(self._stderr_lines)


def build_dqn_config(pipeline_config: DQNPipelineConfig, args: argparse.Namespace) -> DQNConfig:
    """Build a DQNConfig from the pipeline config and CLI args.

    Applies training mode adjustments:
      - scratch: uses base learning_rate and full num_episodes_per_range
      - finetune: uses finetune_learning_rate and finetune_num_episodes_per_range

    Args:
        pipeline_config: The loaded DQNPipelineConfig.
        args: Parsed CLI arguments with overrides.

    Returns:
        A DQNConfig instance ready for training.
    """
    # Determine effective LR and episodes based on training mode
    training_mode = args.training_mode or pipeline_config.training_mode

    if training_mode == "finetune":
        effective_lr = pipeline_config.finetune_learning_rate
    else:
        effective_lr = pipeline_config.learning_rate

    # Episode-count knobs are mode-specific; route only the one that applies so
    # the resulting DQNConfig never trips train()'s mis-set guard. The compiled
    # KFP graph always passes --num-episodes, so in date-range mode (where
    # repeats_per_date governs) we deliberately drop it here. num_episodes_per_range
    # (including the finetune override) applies to fixed-window mode only.
    if pipeline_config.date_start and pipeline_config.date_end:
        eff_num_episodes = None
        eff_repeats = pipeline_config.repeats_per_date
    else:
        eff_num_episodes = (
            pipeline_config.finetune_num_episodes_per_range
            if training_mode == "finetune"
            else pipeline_config.num_episodes_per_range
        )
        eff_repeats = None

    config = DQNConfig(
        # Environment
        grpc_address=pipeline_config.grpc_address,
        symbol=pipeline_config.symbol,
        episode_start_ts=pipeline_config.episode_start_ts,
        episode_end_ts=pipeline_config.episode_end_ts,
        step_size_seconds=pipeline_config.step_size_seconds,
        date_start=pipeline_config.date_start,
        date_end=pipeline_config.date_end,
        hour_of_day_start=pipeline_config.hour_of_day_start,
        hour_of_day_end=pipeline_config.hour_of_day_end,
        # Agent
        gamma=pipeline_config.gamma,
        epsilon_start=pipeline_config.epsilon_start,
        epsilon_end=pipeline_config.epsilon_end,
        epsilon_decay_steps=pipeline_config.epsilon_decay_steps,
        batch_size=pipeline_config.batch_size,
        replay_buffer_size=pipeline_config.replay_buffer_size,
        target_update_freq=pipeline_config.target_update_freq,
        train_freq=pipeline_config.train_freq,
        tau=pipeline_config.tau,
        # Network
        hidden_dims=pipeline_config.hidden_dims,
        activation=pipeline_config.activation,
        layer_norm=pipeline_config.layer_norm,
        dropout=pipeline_config.dropout,
        dueling=pipeline_config.dueling,
        learning_rate=effective_lr,
        betas=pipeline_config.betas,
        eps=pipeline_config.eps,
        weight_decay=pipeline_config.weight_decay,
        grad_clip_norm=pipeline_config.grad_clip_norm,
        loss_function=pipeline_config.loss_function,
        # Training
        num_episodes_per_range=eff_num_episodes,
        repeats_per_date=eff_repeats,
        max_steps_per_episode=pipeline_config.max_steps_per_episode,
        checkpoint_interval=pipeline_config.checkpoint_interval,
        checkpoint_dir="/tmp/dqn_checkpoints",
        # Mode
        mode="train",
        checkpoint=args.production_checkpoint_path if training_mode == "finetune" else None,
        device="auto",
        # S3
        s3_checkpoint_prefix=None,  # We handle S3 upload separately in this component
        model_version=None,
    )

    return config


def upload_checkpoint_to_s3(
    checkpoint_dir: str,
    output_uri: str,
    bucket: str = S3_BUCKET,
) -> None:
    """Find the final checkpoint in the directory and upload to the URI's store.

    URI-aware: routes ``minio://`` to in-cluster MinIO, ``s3://`` to AWS S3,
    bare keys to the default bucket. See ``deepqnetwork.artifact_io``.

    Args:
        checkpoint_dir: Local directory containing checkpoint files.
        output_uri: KFP artifact URI for the checkpoint (one of
            ``minio://b/k``, ``s3://b/k``, or a bare key).
        bucket: Fallback bucket for bare-key URIs.
    """
    from pathlib import Path

    from deepqnetwork.artifact_io import put_file

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_files = sorted(checkpoint_path.glob("dqn_episode_*.pt"))

    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files found in {checkpoint_dir}"
        )

    # Use the last (highest episode) checkpoint
    final_checkpoint_path = checkpoint_files[-1]

    logger.info(
        "Uploading checkpoint",
        extra={
            "local_path": str(final_checkpoint_path),
            "output_uri": output_uri,
        },
    )

    put_file(output_uri, str(final_checkpoint_path), default_bucket=bucket)

    logger.info(
        "Checkpoint uploaded",
        extra={
            "output_uri": output_uri,
            "size_bytes": final_checkpoint_path.stat().st_size,
        },
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the DQN training component."""
    parser = argparse.ArgumentParser(
        description="DQN Training KFP Component",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="/app/deepqnetwork/kubeflow/config/dqn_pipeline_config.yaml",
        help="Path to the DQN pipeline config YAML file",
    )
    parser.add_argument(
        "--checkpoint-output-path",
        type=str,
        required=True,
        help="S3 key path for the output checkpoint artifact",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["scratch", "finetune"],
        default=None,
        help="Training mode: scratch (random init) or finetune (load production weights)",
    )
    parser.add_argument(
        "--production-checkpoint-path",
        type=str,
        default="",
        help="S3 key path for the production checkpoint (required for finetune mode)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Currency pair symbol override",
    )
    parser.add_argument(
        "--episode-start-ts",
        type=int,
        default=None,
        help="Episode start timestamp override",
    )
    parser.add_argument(
        "--episode-end-ts",
        type=int,
        default=None,
        help="Episode end timestamp override",
    )
    parser.add_argument(
        "--step-size-seconds",
        type=int,
        default=None,
        help="Step size in seconds for the modelenv cursor (override)",
    )
    parser.add_argument(
        "--date-start",
        type=str,
        default=None,
        help="ISO date for first episode (e.g. 2012-01-01)",
    )
    parser.add_argument(
        "--date-end",
        type=str,
        default=None,
        help="ISO date for last episode (e.g. 2022-12-31)",
    )
    parser.add_argument(
        "--hour-start",
        type=int,
        default=None,
        dest="hour_of_day_start",
        help="Hour of day to start each episode (0-23)",
    )
    parser.add_argument(
        "--hour-end",
        type=int,
        default=None,
        dest="hour_of_day_end",
        help="Hour of day to end each episode (0-23)",
    )
    parser.add_argument(
        "--num-episodes-per-range",
        type=int,
        default=None,
        help="Fixed-window mode: number of training episodes override",
    )
    parser.add_argument(
        "--repeats-per-date",
        type=int,
        default=None,
        help="Date-range mode: times each date window is replayed",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate override",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size override",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=S3_BUCKET,
        help="S3 bucket name",
    )
    parser.add_argument(
        "--price-snapshot-date",
        type=str,
        default="",
        help=(
            "ISO date/datetime (UTC if naive) at which modelenv resolves ALL "
            "interval snapshots (latest at-or-before), instead of the training "
            "window's date_end. Use a post-consolidation instant when date_end's "
            "own snapshot partitions predate a history consolidation."
        ),
    )
    # Reward-shaping overrides for the decision-persistence experiments
    # (specs/experiments/dqn-decision-persistence.md). Empty = modelenv default
    # (action_penalty 0.001, holding_penalty 1e-6, lambda 0.5). Set as env before
    # the sidecar starts; modelenv reads MODELENV_REWARD_* and the sidecar Popen
    # inherits os.environ. These shape TRAINING only, backtest PnL is fill-based.
    parser.add_argument("--reward-action-penalty", type=str, default="",
                        help="Override MODELENV_REWARD_ACTION_PENALTY (turnover penalty)")
    parser.add_argument("--reward-holding-penalty", type=str, default="",
                        help="Override MODELENV_REWARD_HOLDING_PENALTY")
    parser.add_argument("--reward-lambda", type=str, default="",
                        help="Override MODELENV_REWARD_LAMBDA (loss-aversion)")
    return parser.parse_args()


def download_production_checkpoint(s3_key: str, bucket: str = S3_BUCKET) -> str:
    """Download the production checkpoint from S3 for fine-tuning.

    Args:
        s3_key: S3 object key for the production checkpoint.
        bucket: S3 bucket name.

    Returns:
        Local file path of the downloaded checkpoint.

    Raises:
        FileNotFoundError: If the S3 key does not exist.
    """
    s3 = boto3.client("s3")
    local_path = "/tmp/production_checkpoint.pt"

    logger.info(
        "Downloading production checkpoint for fine-tuning",
        extra={"s3_key": s3_key, "bucket": bucket},
    )

    try:
        s3.download_file(bucket, s3_key, local_path)
    except Exception as e:
        if "NoSuchKey" in str(type(e).__name__) or "404" in str(e):
            raise FileNotFoundError(
                f"Production checkpoint not found: s3://{bucket}/{s3_key}"
            ) from e
        raise

    logger.info(
        "Production checkpoint downloaded",
        extra={"s3_key": s3_key, "local_path": local_path},
    )
    return local_path


def main() -> None:
    """Main entry point for the DQN training component."""
    args = parse_args()

    # Apply reward-shaping overrides as env BEFORE the sidecar starts (it inherits
    # os.environ). modelenv reads these; empty string = keep modelenv's default.
    for cli_val, env_key in (
        (args.reward_action_penalty, "MODELENV_REWARD_ACTION_PENALTY"),
        (args.reward_holding_penalty, "MODELENV_REWARD_HOLDING_PENALTY"),
        (args.reward_lambda, "MODELENV_REWARD_LAMBDA"),
    ):
        if cli_val:
            os.environ[env_key] = cli_val
            logger.info("Reward-shaping override", extra={env_key: cli_val})

    logger.info(
        "DQN training component started",
        extra={
            "config_path": args.config_path,
            "training_mode": args.training_mode,
            "checkpoint_output_path": args.checkpoint_output_path,
        },
    )

    # Step 1: Load pipeline configuration
    pipeline_config = DQNPipelineConfig.from_yaml(args.config_path)

    # Apply CLI overrides to pipeline config
    overrides = {}
    if args.symbol is not None:
        overrides["symbol"] = args.symbol
    if args.episode_start_ts is not None:
        overrides["episode_start_ts"] = args.episode_start_ts
    if args.episode_end_ts is not None:
        overrides["episode_end_ts"] = args.episode_end_ts
    if args.step_size_seconds is not None:
        overrides["step_size_seconds"] = args.step_size_seconds
    if args.date_start is not None:
        overrides["date_start"] = args.date_start
    if args.date_end is not None:
        overrides["date_end"] = args.date_end
    if args.hour_of_day_start is not None:
        overrides["hour_of_day_start"] = args.hour_of_day_start
    if args.hour_of_day_end is not None:
        overrides["hour_of_day_end"] = args.hour_of_day_end
    # The compiled KFP graph always passes --num-episodes, but it only applies to
    # fixed-window mode. In date-range mode repeats_per_date governs, so dropping
    # it here keeps the always-present arg from tripping the mode-mismatch guard.
    eff_date_start = overrides.get("date_start", pipeline_config.date_start)
    eff_date_end = overrides.get("date_end", pipeline_config.date_end)
    if args.num_episodes_per_range is not None and not (eff_date_start and eff_date_end):
        overrides["num_episodes_per_range"] = args.num_episodes_per_range
    # Mirror image: repeats_per_date applies only in date-range mode.
    if args.repeats_per_date is not None and (eff_date_start and eff_date_end):
        overrides["repeats_per_date"] = args.repeats_per_date
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.training_mode is not None:
        overrides["training_mode"] = args.training_mode

    if overrides:
        pipeline_config = pipeline_config.override(**overrides)
        logger.info(
            "Pipeline config overrides applied",
            extra={"overrides": overrides},
        )

    # Validate configuration
    errors = pipeline_config.validate()
    if errors:
        logger.error(
            "Pipeline configuration validation failed",
            extra={"errors": errors},
        )
        raise ValueError(f"Invalid pipeline configuration: {errors}")

    # Step 2: Determine training mode
    training_mode = args.training_mode or pipeline_config.training_mode
    logger.info(
        "Training mode resolved",
        extra={"training_mode": training_mode},
    )

    # Step 3: For finetune mode, download production checkpoint
    if training_mode == "finetune":
        if not args.production_checkpoint_path:
            raise ValueError(
                "Finetune mode requires --production-checkpoint-path to be specified"
            )
        local_checkpoint_path = download_production_checkpoint(
            args.production_checkpoint_path, bucket=args.bucket
        )
        # Update args so build_dqn_config can reference it
        args.production_checkpoint_path = local_checkpoint_path
        logger.info(
            "Finetune mode: production checkpoint ready",
            extra={
                "learning_rate": pipeline_config.finetune_learning_rate,
                "num_episodes_per_range": pipeline_config.finetune_num_episodes_per_range,
            },
        )
    else:
        logger.info(
            "Scratch mode: random initialisation with full training",
            extra={
                "learning_rate": pipeline_config.learning_rate,
                "num_episodes_per_range": pipeline_config.num_episodes_per_range,
            },
        )

    # Step 4: Build DQNConfig for training
    dqn_config = build_dqn_config(pipeline_config, args)

    logger.info(
        "DQN config built",
        extra={
            "symbol": dqn_config.symbol,
            "num_episodes_per_range": dqn_config.num_episodes_per_range,
            "learning_rate": dqn_config.learning_rate,
            "batch_size": dqn_config.batch_size,
            "grpc_address": dqn_config.grpc_address,
        },
    )

    # Step 5: Start modelenv sidecar. Pass the trainer's per-date episode
    # window into modelenv so its startup preload only fetches ticks in that
    # range (instead of the full M1 reference span, which can be many months).
    # modelenv still lazily downloads any out-of-window tick file on Reset().
    sidecar = ModelenvSidecar(
        symbol=dqn_config.symbol,
        date_start=dqn_config.date_start or None,
        date_end=dqn_config.date_end or None,
        hour_start=dqn_config.hour_of_day_start,
        hour_end=dqn_config.hour_of_day_end,
        price_snapshot_date=args.price_snapshot_date or None,
    )
    try:
        sidecar.start()
        sidecar.wait_for_ready()

        # Step 6: Run DQN training
        logger.info("Starting DQN training loop")
        train(dqn_config)
        logger.info("DQN training loop completed")

    except Exception as e:
        logger.error(
            "DQN training failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        raise
    finally:
        # Step 7: Always stop the sidecar on exit
        sidecar.stop()

    # Step 8: Upload final checkpoint to the KFP-advertised store
    upload_checkpoint_to_s3(
        checkpoint_dir=dqn_config.checkpoint_dir,
        output_uri=args.checkpoint_output_path,
        bucket=args.bucket,
    )

    # Step 9: Log completion
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger.info(
        "DQN training component completed successfully",
        extra={
            "training_mode": training_mode,
            "checkpoint_output_path": args.checkpoint_output_path,
            "symbol": dqn_config.symbol,
            "num_episodes_per_range": dqn_config.num_episodes_per_range,
            "learning_rate": dqn_config.learning_rate,
            # Overnight financing the modelenv sidecar applied during training.
            # Sidecar runs in Training mode without swap CLI flags, so these are
            # the built-in default-table rates unless overridden via
            # MODELENV_SWAP_* / MODELENV_NO_SWAP env. See deepqnetwork/swap_rates.py.
            "swap_rates": resolve_swap_rates(dqn_config.symbol),
            "completed_at": timestamp,
        },
    )


if __name__ == "__main__":
    main()
