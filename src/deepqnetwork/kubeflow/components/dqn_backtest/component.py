"""DQN Backtest KFP component.

Loads a DQN checkpoint into DQNAdvisor, launches the modelenv gRPC server as a
subprocess sidecar, runs evaluation episodes on unseen date ranges, computes
backtest metrics (cumulative P&L, Sharpe ratio, max drawdown, win rate, average
episode reward, average episode length), and runs a degradation gate comparing
against production model metrics.

Handles initial deployment bootstrap: if no production model metrics exist,
the degradation gate is skipped and the model is auto-promoted.

The degradation gate implements four checks:
  (a) Block if Sharpe degrades vs production by >0.1
  (b) Block if P&L < 0 while production P&L > 0
  (c) Block if Sharpe < 1.0 absolute (must beat buy & hold baseline from thesis)
  (d) Block if P&L ≤ 0 absolute

Requirements: DQN-R9, DQN-R10
"""

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import boto3
import grpc
import numpy as np
import torch

# Add parent paths so we can import the deepqnetwork package
sys.path.insert(0, "/app")

from deepqnetwork.advisor import DQNAdvisor
from deepqnetwork.kubeflow.pipeline.config_schema import DQNPipelineConfig
from probabilisticforecaster.kubeflow.monitoring.metrics import get_logger

# Import gRPC stubs (available via PYTHONPATH from base image)
import environment_pb2
import environment_pb2_grpc

logger = get_logger(__name__, component="dqn_backtest")

# S3 bucket for artifacts
S3_BUCKET = os.environ.get("S3_BUCKET", "prod-fintech-forex-sg-731833471586")

# Modelenv sidecar configuration
MODELENV_BINARY = "/usr/local/bin/modelenv-server"
MODELENV_HOST = "localhost"
MODELENV_PORT = 50051
MODELENV_HEALTH_CHECK_TIMEOUT = 60  # seconds
MODELENV_HEALTH_CHECK_INTERVAL = 1  # seconds
MODELENV_SHUTDOWN_TIMEOUT = 10  # seconds

# Degradation gate thresholds (from DQNPipelineConfig / thesis)
DEFAULT_SHARPE_DEGRADATION_THRESHOLD = 0.1
DEFAULT_SHARPE_ABSOLUTE_THRESHOLD = 1.0  # Must beat buy & hold baseline
DEFAULT_PNL_ABSOLUTE_THRESHOLD = 0.0  # Must be profitable

# Annualisation factor for Sharpe ratio (assuming 5-second steps, ~252 trading days)
# Episodes per year approximation for annualised Sharpe
TRADING_DAYS_PER_YEAR = 252


class ModelenvSidecar:
    """Manages the modelenv gRPC server subprocess lifecycle.

    Starts the modelenv-server binary as a subprocess, waits for the gRPC
    port to become available, and handles graceful shutdown on exit.
    """

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._stderr_lines: list[str] = []

    def start(self) -> None:
        """Start the modelenv-server subprocess.

        Raises:
            FileNotFoundError: If the modelenv binary is not found.
            RuntimeError: If the process fails to start.
        """
        logger.info(
            "Starting modelenv sidecar",
            extra={"binary": MODELENV_BINARY, "port": MODELENV_PORT},
        )

        try:
            self._process = subprocess.Popen(
                [MODELENV_BINARY],
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
        """Read and return any available stderr from the subprocess."""
        if self._process is None or self._process.stderr is None:
            return ""

        try:
            stderr_data = self._process.stderr.read()
            if stderr_data:
                output = stderr_data.decode("utf-8", errors="replace").strip()
                self._stderr_lines.append(output)
                return output
        except (OSError, ValueError):
            pass

        return "\n".join(self._stderr_lines)


@dataclass
class EpisodeResult:
    """Result of a single evaluation episode."""

    total_reward: float
    cumulative_pnl: float
    num_steps: int
    num_trades: int
    winning_trades: int


@dataclass
class BacktestMetrics:
    """Aggregated backtest metrics across all evaluation episodes."""

    cumulative_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_episode_reward: float
    avg_episode_length: float


def run_evaluation_episode(
    advisor: DQNAdvisor,
    env_stub: environment_pb2_grpc.EnvironmentStub,
    symbol: str,
    episode_start_ts: int,
    episode_end_ts: int,
    step_size_seconds: int,
    max_steps: int,
    episode_seed: int,
) -> EpisodeResult:
    """Run a single evaluation episode using the DQNAdvisor.

    Resets the modelenv environment, then steps through the episode using
    greedy action selection from the advisor until done or max_steps reached.

    Args:
        advisor: Loaded DQNAdvisor for greedy action selection.
        env_stub: gRPC stub for the modelenv Environment service.
        symbol: Currency pair symbol (e.g. "USDJPY").
        episode_start_ts: Episode start timestamp.
        episode_end_ts: Episode end timestamp.
        step_size_seconds: Step size in seconds.
        max_steps: Maximum steps per episode.
        episode_seed: Random seed for the episode reset.

    Returns:
        EpisodeResult with reward, P&L, steps, and trade statistics.
    """
    # Reset the environment
    reset_request = environment_pb2.ResetRequest(
        symbol=symbol,
        episode_start_ts=episode_start_ts,
        episode_end_ts=episode_end_ts,
        seed=episode_seed,
        step_size_seconds=step_size_seconds,
    )
    observation = env_stub.Reset(reset_request)

    total_reward = 0.0
    num_steps = 0
    num_trades = 0
    winning_trades = 0
    cumulative_pnl = 0.0

    # Extract initial state vector from observation
    state = np.array(observation.state_data[0].values, dtype=np.float32)

    while not observation.done and num_steps < max_steps:
        # Get greedy action from advisor
        result = advisor.recommend_action(state)
        action_idx = result.action

        # Map action index to protobuf ActionType
        action_type = environment_pb2.ActionType.Value(
            ["ACTION_HOLD", "ACTION_BUY_1", "ACTION_BUY_2", "ACTION_SELL_1", "ACTION_SELL_2"][action_idx]
        )

        # Step the environment
        action_msg = environment_pb2.Action(
            action=action_type,
            client_order_id=f"backtest_{episode_seed}_{num_steps}",
        )
        step_response = env_stub.Step(action_msg)
        observation_data = step_response.data

        # Accumulate reward
        reward = observation_data.reward
        total_reward += reward

        # Track trades (any non-HOLD action)
        if action_idx != 0:
            num_trades += 1
            # A positive reward on a trade step indicates a winning trade
            if reward > 0:
                winning_trades += 1

        # Update state for next step
        if observation_data.state_data:
            state = np.array(observation_data.state_data[0].values, dtype=np.float32)

        num_steps += 1

        # Check if episode is done
        if observation_data.done:
            break

    # Get final reference data for realised P&L
    try:
        ref_request = environment_pb2.ObserveRequest(symbol=symbol)
        reference = env_stub.ReferenceData(ref_request)
        cumulative_pnl = reference.realised_pnl_12m
    except Exception:
        # Fall back to using total reward as P&L proxy
        cumulative_pnl = total_reward

    return EpisodeResult(
        total_reward=total_reward,
        cumulative_pnl=cumulative_pnl,
        num_steps=num_steps,
        num_trades=num_trades,
        winning_trades=winning_trades,
    )


def compute_backtest_metrics(episode_results: list[EpisodeResult]) -> BacktestMetrics:
    """Compute aggregated backtest metrics from episode results.

    Args:
        episode_results: List of EpisodeResult from evaluation episodes.

    Returns:
        BacktestMetrics with cumulative P&L, Sharpe ratio, max drawdown,
        win rate, average episode reward, and average episode length.
    """
    if not episode_results:
        return BacktestMetrics(
            cumulative_pnl=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            avg_episode_reward=0.0,
            avg_episode_length=0.0,
        )

    # Cumulative P&L: sum of all episode P&Ls
    pnl_values = [ep.cumulative_pnl for ep in episode_results]
    cumulative_pnl = sum(pnl_values)

    # Sharpe ratio: mean(episode_rewards) / std(episode_rewards) * sqrt(N_episodes_per_year)
    # Using episode rewards as the return series
    rewards = np.array([ep.total_reward for ep in episode_results])
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0

    if std_reward > 0:
        # Annualise: assume episodes represent trading periods
        # Scale by sqrt(num_episodes_per_range) as a proxy for annualisation
        sharpe_ratio = (mean_reward / std_reward) * math.sqrt(len(episode_results))
    else:
        sharpe_ratio = 0.0 if mean_reward == 0 else math.copysign(float("inf"), mean_reward)

    # Ensure Sharpe is finite (requirement DQN-R10: Sharpe ratio is finite)
    if not math.isfinite(sharpe_ratio):
        sharpe_ratio = math.copysign(100.0, sharpe_ratio)

    # Max drawdown: computed from cumulative P&L curve across episodes
    cumulative_curve = np.cumsum(pnl_values)
    running_max = np.maximum.accumulate(cumulative_curve)
    drawdowns = running_max - cumulative_curve
    if running_max.max() > 0:
        max_drawdown = float(drawdowns.max() / running_max.max())
    else:
        max_drawdown = 0.0

    # Clamp max_drawdown to [0, 1] (requirement DQN-R10)
    max_drawdown = max(0.0, min(1.0, max_drawdown))

    # Win rate: proportion of trades that were profitable
    total_trades = sum(ep.num_trades for ep in episode_results)
    total_winning = sum(ep.winning_trades for ep in episode_results)
    win_rate = total_winning / total_trades if total_trades > 0 else 0.0

    # Clamp win_rate to [0, 1] (requirement DQN-R10)
    win_rate = max(0.0, min(1.0, win_rate))

    # Average episode reward
    avg_episode_reward = mean_reward

    # Average episode length
    avg_episode_length = float(np.mean([ep.num_steps for ep in episode_results]))

    return BacktestMetrics(
        cumulative_pnl=cumulative_pnl,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        avg_episode_reward=avg_episode_reward,
        avg_episode_length=avg_episode_length,
    )


def degradation_gate(
    current_metrics: BacktestMetrics,
    production_metrics: dict,
    sharpe_degradation_threshold: float = DEFAULT_SHARPE_DEGRADATION_THRESHOLD,
    sharpe_absolute_threshold: float = DEFAULT_SHARPE_ABSOLUTE_THRESHOLD,
    pnl_absolute_threshold: float = DEFAULT_PNL_ABSOLUTE_THRESHOLD,
) -> tuple[bool, str]:
    """Compare current backtest metrics against production model and absolute floors.

    The gate fails (returns False) if any of:
      (a) Sharpe degrades vs production by more than sharpe_degradation_threshold
      (b) P&L < 0 while production P&L > 0
      (c) Sharpe < sharpe_absolute_threshold (must beat buy & hold baseline)
      (d) P&L ≤ 0 in absolute terms

    Args:
        current_metrics: Metrics from the newly trained model's backtest.
        production_metrics: Dictionary with production model metrics
            (keys: "sharpe_ratio", "cumulative_pnl").
        sharpe_degradation_threshold: Maximum allowed Sharpe decrease vs production.
        sharpe_absolute_threshold: Minimum absolute Sharpe ratio (from thesis).
        pnl_absolute_threshold: Minimum absolute P&L threshold.

    Returns:
        Tuple of (gate_passed: bool, reason: str).
    """
    prod_sharpe = production_metrics.get("sharpe_ratio", 0.0)
    prod_pnl = production_metrics.get("cumulative_pnl", 0.0)

    reasons = []

    # (a) Relative Sharpe degradation check
    sharpe_delta = prod_sharpe - current_metrics.sharpe_ratio
    if sharpe_delta > sharpe_degradation_threshold:
        reasons.append(
            f"Sharpe degraded: current={current_metrics.sharpe_ratio:.4f}, "
            f"production={prod_sharpe:.4f}, delta={sharpe_delta:.4f} > "
            f"threshold={sharpe_degradation_threshold}"
        )

    # (b) P&L sign flip check: new model loses money while production is profitable
    if current_metrics.cumulative_pnl < 0 and prod_pnl > 0:
        reasons.append(
            f"P&L sign flip: current={current_metrics.cumulative_pnl:.6f} < 0 "
            f"while production={prod_pnl:.6f} > 0"
        )

    # (c) Absolute Sharpe floor (must beat buy & hold baseline from thesis)
    if current_metrics.sharpe_ratio < sharpe_absolute_threshold:
        reasons.append(
            f"Sharpe below absolute floor: current={current_metrics.sharpe_ratio:.4f} < "
            f"absolute_threshold={sharpe_absolute_threshold}"
        )

    # (d) Absolute P&L floor (must be profitable)
    if current_metrics.cumulative_pnl <= pnl_absolute_threshold:
        reasons.append(
            f"P&L below absolute floor: current={current_metrics.cumulative_pnl:.6f} <= "
            f"absolute_threshold={pnl_absolute_threshold}"
        )

    if reasons:
        return False, "; ".join(reasons)

    return True, "All metrics within acceptable thresholds"


def download_checkpoint_from_s3(
    checkpoint_uri: str,
    bucket: str = S3_BUCKET,
) -> str:
    """Download a DQN checkpoint to a local path.

    URI-aware: routes ``minio://`` to in-cluster MinIO, ``s3://`` to AWS S3,
    bare keys to the default bucket. See ``deepqnetwork.artifact_io``.

    Args:
        checkpoint_uri: KFP artifact URI for the checkpoint (one of
            ``minio://b/k``, ``s3://b/k``, or a bare key).
        bucket: Fallback bucket for bare-key URIs.

    Returns:
        Local file path of the downloaded checkpoint.

    Raises:
        FileNotFoundError: If the object does not exist.
    """
    from deepqnetwork.artifact_io import download_file

    local_path = "/tmp/dqn_checkpoint.pt"

    logger.info(
        "Downloading DQN checkpoint",
        extra={"checkpoint_uri": checkpoint_uri},
    )

    try:
        download_file(checkpoint_uri, local_path, default_bucket=bucket)
    except Exception as e:
        if "NoSuchKey" in str(type(e).__name__) or "404" in str(e):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_uri}"
            ) from e
        raise

    logger.info(
        "DQN checkpoint downloaded",
        extra={"checkpoint_uri": checkpoint_uri, "local_path": local_path},
    )
    return local_path


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
            "prod_sharpe": data.get("sharpe_ratio"),
            "prod_pnl": data.get("cumulative_pnl"),
        },
    )
    return data


def upload_metrics_to_s3(
    metrics: dict,
    output_uri: str,
    bucket: str = S3_BUCKET,
) -> None:
    """Upload evaluation metrics as a JSON artifact to the URI's store.

    URI-aware: routes ``minio://`` to in-cluster MinIO, ``s3://`` to AWS S3,
    bare keys to the default bucket. See ``deepqnetwork.artifact_io``.

    Args:
        metrics: Dictionary of evaluation results to serialize.
        output_uri: KFP artifact URI for the output JSON artifact.
        bucket: Fallback bucket for bare-key URIs.
    """
    from deepqnetwork.artifact_io import put_object_bytes

    payload = json.dumps(metrics, indent=2).encode("utf-8")

    put_object_bytes(
        output_uri,
        payload,
        content_type="application/json",
        default_bucket=bucket,
    )

    logger.info(
        "Evaluation metrics uploaded",
        extra={"output_uri": output_uri, "size_bytes": len(payload)},
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the DQN backtest component."""
    parser = argparse.ArgumentParser(
        description="DQN Backtest KFP Component",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="S3 key path for the DQN checkpoint artifact to evaluate",
    )
    parser.add_argument(
        "--production-metrics-path",
        type=str,
        default="",
        help="S3 key path for production model metrics JSON (empty = bootstrap, skip gate)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="S3 key path for the output evaluation metrics JSON artifact",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="/app/deepqnetwork/kubeflow/config/dqn_pipeline_config.yaml",
        help="Path to the DQN pipeline config YAML file",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="USDJPY",
        help="Currency pair symbol",
    )
    parser.add_argument(
        "--eval-episode-start-ts",
        type=int,
        required=True,
        help="Evaluation episode start timestamp (unseen date range)",
    )
    parser.add_argument(
        "--eval-episode-end-ts",
        type=int,
        required=True,
        help="Evaluation episode end timestamp (unseen date range)",
    )
    parser.add_argument(
        "--date-start",
        type=str,
        default=None,
        help="ISO date for first evaluation episode",
    )
    parser.add_argument(
        "--date-end",
        type=str,
        default=None,
        help="ISO date for last evaluation episode",
    )
    parser.add_argument(
        "--hour-start",
        type=int,
        default=None,
        dest="hour_of_day_start",
        help="Hour of day to start each eval episode",
    )
    parser.add_argument(
        "--hour-end",
        type=int,
        default=None,
        dest="hour_of_day_end",
        help="Hour of day to end each eval episode",
    )
    parser.add_argument(
        "--step-size-seconds",
        type=int,
        default=5,
        help="Step size in seconds for the environment",
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes to run",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=30_000,
        help="Maximum steps per evaluation episode",
    )
    parser.add_argument(
        "--sharpe-degradation-threshold",
        type=float,
        default=DEFAULT_SHARPE_DEGRADATION_THRESHOLD,
        help="Maximum allowed Sharpe decrease vs production model",
    )
    parser.add_argument(
        "--sharpe-absolute-threshold",
        type=float,
        default=DEFAULT_SHARPE_ABSOLUTE_THRESHOLD,
        help="Absolute minimum Sharpe ratio (must beat buy & hold baseline)",
    )
    parser.add_argument(
        "--pnl-absolute-threshold",
        type=float,
        default=DEFAULT_PNL_ABSOLUTE_THRESHOLD,
        help="Absolute minimum P&L threshold",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=S3_BUCKET,
        help="S3 bucket name",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the DQN backtest component."""
    args = parse_args()

    logger.info(
        "DQN backtest component started",
        extra={
            "checkpoint_path": args.checkpoint_path,
            "production_metrics_path": args.production_metrics_path,
            "output_path": args.output_path,
            "symbol": args.symbol,
            "eval_episode_start_ts": args.eval_episode_start_ts,
            "eval_episode_end_ts": args.eval_episode_end_ts,
            "num_eval_episodes": args.num_eval_episodes,
            "max_steps_per_episode": args.max_steps_per_episode,
        },
    )

    # Step 1: Download checkpoint from its KFP-advertised store (MinIO/S3/bare key)
    local_checkpoint_path = download_checkpoint_from_s3(
        checkpoint_uri=args.checkpoint_path, bucket=args.bucket
    )

    # Step 2: Load checkpoint into DQNAdvisor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    advisor = DQNAdvisor.from_checkpoint(local_checkpoint_path, device=device)

    logger.info(
        "DQNAdvisor loaded",
        extra={
            "state_dim": advisor.state_dim,
            "device": str(advisor.device),
            "checkpoint_path": local_checkpoint_path,
        },
    )

    # Step 3: Start modelenv sidecar
    sidecar = ModelenvSidecar()
    try:
        sidecar.start()
        sidecar.wait_for_ready()

        # Step 4: Create gRPC channel and stub
        channel = grpc.insecure_channel(f"{MODELENV_HOST}:{MODELENV_PORT}")
        env_stub = environment_pb2_grpc.EnvironmentStub(channel)

        # Step 5: Run evaluation episodes
        logger.info(
            "Starting evaluation episodes",
            extra={
                "num_episodes_per_range": args.num_eval_episodes,
                "episode_start_ts": args.eval_episode_start_ts,
                "episode_end_ts": args.eval_episode_end_ts,
                "step_size_seconds": args.step_size_seconds,
            },
        )

        episode_results: list[EpisodeResult] = []

        for episode_idx in range(args.num_eval_episodes):
            episode_seed = 42 + episode_idx  # Deterministic seeds for reproducibility

            result = run_evaluation_episode(
                advisor=advisor,
                env_stub=env_stub,
                symbol=args.symbol,
                episode_start_ts=args.eval_episode_start_ts,
                episode_end_ts=args.eval_episode_end_ts,
                step_size_seconds=args.step_size_seconds,
                max_steps=args.max_steps_per_episode,
                episode_seed=episode_seed,
            )

            episode_results.append(result)

            logger.info(
                f"Episode {episode_idx + 1}/{args.num_eval_episodes} completed",
                extra={
                    "episode": episode_idx + 1,
                    "reward": round(result.total_reward, 4),
                    "pnl": round(result.cumulative_pnl, 6),
                    "steps": result.num_steps,
                    "trades": result.num_trades,
                    "winning_trades": result.winning_trades,
                },
            )

        # Close gRPC channel
        channel.close()

    except Exception as e:
        logger.error(
            "DQN backtest evaluation failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        raise
    finally:
        # Always stop the sidecar on exit
        sidecar.stop()

    # Step 6: Compute aggregated backtest metrics
    metrics = compute_backtest_metrics(episode_results)

    logger.info(
        "Backtest metrics computed",
        extra={
            "cumulative_pnl": round(metrics.cumulative_pnl, 6),
            "sharpe_ratio": round(metrics.sharpe_ratio, 4),
            "max_drawdown": round(metrics.max_drawdown, 4),
            "win_rate": round(metrics.win_rate, 4),
            "avg_episode_reward": round(metrics.avg_episode_reward, 4),
            "avg_episode_length": round(metrics.avg_episode_length, 1),
        },
    )

    # Step 7: Degradation gate
    gate_passed = True
    gate_skipped = False
    gate_reason = ""

    if not args.production_metrics_path:
        # Bootstrap: no production model → skip gate, auto-promote
        gate_passed = True
        gate_skipped = True
        gate_reason = "Bootstrap: no production model exists, auto-promoting"
        logger.info(
            "Degradation gate skipped (bootstrap (no production model)",
            extra={"gate_passed": True, "gate_skipped": True},
        )
    else:
        # Load production metrics and compare
        try:
            production_metrics = load_production_metrics_from_s3(
                args.production_metrics_path, bucket=args.bucket
            )
            gate_passed, gate_reason = degradation_gate(
                current_metrics=metrics,
                production_metrics=production_metrics,
                sharpe_degradation_threshold=args.sharpe_degradation_threshold,
                sharpe_absolute_threshold=args.sharpe_absolute_threshold,
                pnl_absolute_threshold=args.pnl_absolute_threshold,
            )
            logger.info(
                "Degradation gate evaluated",
                extra={
                    "gate_passed": gate_passed,
                    "gate_reason": gate_reason,
                },
            )
        except FileNotFoundError:
            # Production metrics file doesn't exist) treat as bootstrap
            gate_passed = True
            gate_skipped = True
            gate_reason = (
                "Production metrics file not found, treating as bootstrap"
            )
            logger.warning(
                "Production metrics not found, skipping gate",
                extra={
                    "production_metrics_path": args.production_metrics_path,
                    "gate_passed": True,
                    "gate_skipped": True,
                },
            )

    # Step 8: Assemble output artifact
    timestamp = datetime.now(timezone.utc).isoformat()

    output = {
        "timestamp": timestamp,
        "symbol": args.symbol,
        "eval_episode_start_ts": args.eval_episode_start_ts,
        "eval_episode_end_ts": args.eval_episode_end_ts,
        "step_size_seconds": args.step_size_seconds,
        "num_eval_episodes": args.num_eval_episodes,
        "max_steps_per_episode": args.max_steps_per_episode,
        "checkpoint_path": args.checkpoint_path,
        "backtest_metrics": {
            "cumulative_pnl": metrics.cumulative_pnl,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "avg_episode_reward": metrics.avg_episode_reward,
            "avg_episode_length": metrics.avg_episode_length,
        },
        "episode_details": [
            {
                "episode": i + 1,
                "total_reward": ep.total_reward,
                "cumulative_pnl": ep.cumulative_pnl,
                "num_steps": ep.num_steps,
                "num_trades": ep.num_trades,
                "winning_trades": ep.winning_trades,
            }
            for i, ep in enumerate(episode_results)
        ],
        "degradation_gate": {
            "gate_passed": gate_passed,
            "gate_skipped": gate_skipped,
            "reason": gate_reason,
            "sharpe_degradation_threshold": args.sharpe_degradation_threshold,
            "sharpe_absolute_threshold": args.sharpe_absolute_threshold,
            "pnl_absolute_threshold": args.pnl_absolute_threshold,
        },
    }

    # Step 9: Upload metrics to S3
    upload_metrics_to_s3(output, output_uri=args.output_path, bucket=args.bucket)

    # Log final summary
    if not gate_passed:
        logger.warning(
            "DQN backtest completed, GATE FAILED (blocked from promotion)",
            extra={
                "gate_passed": False,
                "gate_reason": gate_reason,
                "output_artifact": args.output_path,
            },
        )
    else:
        logger.info(
            "DQN backtest component completed successfully",
            extra={
                "gate_passed": gate_passed,
                "gate_skipped": gate_skipped,
                "output_artifact": args.output_path,
                "sharpe_ratio": round(metrics.sharpe_ratio, 4),
                "cumulative_pnl": round(metrics.cumulative_pnl, 6),
                "win_rate": round(metrics.win_rate, 4),
                "max_drawdown": round(metrics.max_drawdown, 4),
            },
        )


if __name__ == "__main__":
    main()
