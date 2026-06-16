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
from deepqnetwork.episode_windows import iter_date_episodes
from deepqnetwork.kubeflow.pipeline.config_schema import DQNPipelineConfig
from deepqnetwork.swap_rates import resolve_swap_rates
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

# Emit the per-step progress/diagnostic logs only every Nth step to bound log
# volume. A hang is still localised to the [k*N, (k+1)*N) step window, and the
# in-flight operation is named by whichever marker (advisor vs Step) fired last.
STEP_LOG_INTERVAL = 500  # steps

# Per-RPC deadlines (seconds) on the modelenv gRPC calls. Defense-in-depth: a
# call with no deadline blocks forever if the server stalls (this is what turned
# a transient modelenv hiccup into a 29h hang, see T-14.1-07). With a deadline
# the call fails fast with DeadlineExceeded, which fails the component and lets
# KFP retry it. Reset is generous because the first episode does an S3/parquet
# load that can be slow on a cold cache; Step/ReferenceData are sub-second in
# practice so their ceilings are loose.
GRPC_RESET_TIMEOUT_S = 600
GRPC_STEP_TIMEOUT_S = 120
GRPC_REFERENCE_TIMEOUT_S = 120

# ReferenceData carries session_realised_pnl; the real money figure the
# degradation gate promotes on. On a transient RpcError, retry a few times with
# linear backoff; if it still fails, FAIL the episode (raise) rather than fall
# back to a reward-derived proxy. total_reward is a clipped, penalty-laden,
# reward_scale'd function of PnL that can differ from realised P&L in magnitude
# AND sign, so substituting it would silently feed the gate a wrong number (the
# same class of hazard as the gate bug that auto-promoted a degenerate model).
GRPC_REFERENCE_MAX_ATTEMPTS = 3
GRPC_REFERENCE_RETRY_BACKOFF_S = 2

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
            # Inherit the parent's stdout/stderr (do NOT use subprocess.PIPE).
            # modelenv logs continuously to stderr; the parent does not drain
            # those pipes during the run (it only read them on stop), so once
            # ~64 KB accumulated the kernel pipe buffer filled and modelenv's
            # next log write blocked forever, deadlocking the server mid-Reset
            # and hanging the whole backtest at 0 CPU. Inheriting the parent fds
            # streams modelenv's logs straight to the pod console (visible in
            # `kubectl logs`) and removes the deadlock entirely.
            self._process = subprocess.Popen(
                [MODELENV_BINARY],
                stdout=None,
                stderr=None,
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
                raise RuntimeError(
                    f"modelenv sidecar exited prematurely with code "
                    f"{self._process.returncode}. See pod logs for modelenv output."
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
        raise RuntimeError(
            f"modelenv sidecar did not become ready within "
            f"{MODELENV_HEALTH_CHECK_TIMEOUT}s. See pod logs for modelenv output."
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

        # modelenv's own stdout/stderr stream directly to the pod console
        # (it inherits the parent fds), so there is nothing to capture here.


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


def resolve_eval_windows(
    *,
    date_start: str | None,
    date_end: str | None,
    hour_of_day_start: int | None,
    hour_of_day_end: int | None,
    eval_episode_start_ts: int,
    eval_episode_end_ts: int,
    num_eval_episodes: int,
) -> list[tuple[int, int]]:
    """Resolve the list of ``(start_ts, end_ts)`` evaluation episode windows.

    Two modes:

    * **Date-range (preferred):** when ``date_start``/``date_end`` and the
      hour-of-day bounds are all set, produce one window per calendar date via
      :func:`iter_date_episodes`, i.e. evaluate the policy on a *series of
      distinct sessions*, the same windows DQN training and the dqnpf backtest
      use. This is what makes ``cumulative_pnl = sum(per-episode P&L)`` a genuine
      cumulative P&L over the eval span, and makes the cross-episode Sharpe
      finite/meaningful (distinct sessions have real return variance).

    * **Legacy fixed-window:** when the dates are unset, repeat the single
      ``[eval_episode_start_ts, eval_episode_end_ts]`` window
      ``num_eval_episodes`` times. NOTE: repeating one window only samples the
      env's seed noise under a deterministic greedy policy; its near-zero
      reward variance makes the Sharpe ``mean/std`` blow up (the source of the
      absurd ~1e16 Sharpe values), and summing the repeats inflates P&L. Kept
      only for backward compatibility / explicit single-window probes.
    """
    if (
        date_start
        and date_end
        and hour_of_day_start is not None
        and hour_of_day_end is not None
    ):
        return iter_date_episodes(
            date_start, date_end, hour_of_day_start, hour_of_day_end
        )
    return [(eval_episode_start_ts, eval_episode_end_ts)] * num_eval_episodes


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
    # Log before/after every blocking gRPC call so that, if the component
    # hangs, the last emitted log line deterministically identifies which
    # operation is stuck (Reset vs Step vs ReferenceData) and at which step.
    logger.info(
        "Reset: calling env_stub.Reset",
        extra={
            "seed": episode_seed,
            "symbol": symbol,
            "episode_start_ts": episode_start_ts,
            "episode_end_ts": episode_end_ts,
            "step_size_seconds": step_size_seconds,
        },
    )
    _reset_t0 = time.monotonic()
    observation = env_stub.Reset(reset_request, timeout=GRPC_RESET_TIMEOUT_S)
    logger.info(
        "Reset: env_stub.Reset returned",
        extra={
            "seed": episode_seed,
            "done": observation.done,
            "elapsed_ms": round((time.monotonic() - _reset_t0) * 1000, 1),
        },
    )

    total_reward = 0.0
    num_steps = 0
    num_trades = 0
    winning_trades = 0
    cumulative_pnl = 0.0

    # Extract initial state vector from observation
    state = np.array(observation.state_data[0].values, dtype=np.float32)

    while not observation.done and num_steps < max_steps:
        # Throttle the per-step diagnostic logs to every Nth step.
        log_step = num_steps % STEP_LOG_INTERVAL == 0
        # Marker before advisor inference: a hang here points at the model,
        # a hang at the next marker points at the Step RPC / modelenv.
        if log_step:
            logger.info(
                "Step: advisor.recommend_action",
                extra={"seed": episode_seed, "step": num_steps},
            )
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
        if log_step:
            logger.info(
                "Step: calling env_stub.Step",
                extra={"seed": episode_seed, "step": num_steps, "action": action_idx},
            )
        _step_t0 = time.monotonic()
        step_response = env_stub.Step(action_msg, timeout=GRPC_STEP_TIMEOUT_S)
        if log_step:
            logger.info(
                "Step: env_stub.Step returned",
                extra={
                    "seed": episode_seed,
                    "step": num_steps,
                    "elapsed_ms": round((time.monotonic() - _step_t0) * 1000, 1),
                },
            )
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

    # Get final reference data for realised P&L. This is the money figure the
    # degradation gate promotes on, so it MUST be the real session_realised_pnl
    # from modelenv, never a reward-derived proxy. Retry on transient RpcError;
    # if it still cannot be read, raise so the component fails and KFP's retry(3)
    # re-runs the backtest, rather than registering a bogus (reward-as-P&L)
    # number. (See GRPC_REFERENCE_MAX_ATTEMPTS note above.)
    ref_request = environment_pb2.ObserveRequest(symbol=symbol)
    reference = None
    last_exc: grpc.RpcError | None = None
    for attempt in range(1, GRPC_REFERENCE_MAX_ATTEMPTS + 1):
        try:
            logger.info(
                "ReferenceData: calling env_stub.ReferenceData",
                extra={
                    "seed": episode_seed,
                    "symbol": symbol,
                    "steps": num_steps,
                    "attempt": attempt,
                },
            )
            reference = env_stub.ReferenceData(
                ref_request, timeout=GRPC_REFERENCE_TIMEOUT_S
            )
            logger.info(
                "ReferenceData: env_stub.ReferenceData returned",
                extra={"seed": episode_seed},
            )
            break
        except grpc.RpcError as exc:
            last_exc = exc
            logger.error(
                "ReferenceData RPC failed (attempt %d/%d): %s",
                attempt,
                GRPC_REFERENCE_MAX_ATTEMPTS,
                exc,
                extra={"seed": episode_seed},
            )
            if attempt < GRPC_REFERENCE_MAX_ATTEMPTS:
                time.sleep(GRPC_REFERENCE_RETRY_BACKOFF_S * attempt)

    if reference is None:
        raise RuntimeError(
            f"ReferenceData failed after {GRPC_REFERENCE_MAX_ATTEMPTS} attempts "
            f"(seed={episode_seed}): cannot read session_realised_pnl. Refusing "
            "to substitute total_reward as P&L; it would corrupt the "
            "degradation gate. Failing the episode so KFP can retry."
        ) from last_exc

    cumulative_pnl = reference.session_realised_pnl

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

    # Sharpe ratio across episodes, using per-episode total_reward as the return
    # series. This is only meaningful when the episodes are DISTINCT sessions
    # (date-range mode): then std() reflects real session-to-session variation.
    # In the legacy single-window-repeated mode the rollouts are near-identical
    # (deterministic greedy policy, one window), std()->~0 and this ratio blows
    # up, which is exactly why that mode is deprecated (see resolve_eval_windows).
    rewards = np.array([ep.total_reward for ep in episode_results])
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0

    if std_reward > 0:
        # Scale by sqrt(num sessions) as a proxy for annualisation. (Caveat: a
        # true daily annualisation would use sqrt(252); kept as-is to preserve
        # the existing gate-threshold calibration.)
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

    # (c)+(d) Absolute floors, candidate-only, ALSO enforced at bootstrap.
    reasons.extend(
        absolute_floor_reasons(
            current_metrics,
            sharpe_absolute_threshold=sharpe_absolute_threshold,
            pnl_absolute_threshold=pnl_absolute_threshold,
        )
    )

    if reasons:
        return False, "; ".join(reasons)

    return True, "All metrics within acceptable thresholds"


def absolute_floor_reasons(
    metrics: BacktestMetrics,
    sharpe_absolute_threshold: float = DEFAULT_SHARPE_ABSOLUTE_THRESHOLD,
    pnl_absolute_threshold: float = DEFAULT_PNL_ABSOLUTE_THRESHOLD,
) -> list[str]:
    """Candidate-only absolute-floor failures (independent of any production model).

    A model must clear these regardless of whether a production baseline exists,
    so they are enforced both by :func:`degradation_gate` (relative path) and by
    :func:`absolute_floor_gate` (bootstrap path). Empty list = both floors cleared.
    """
    reasons = []
    # Sharpe floor (must beat buy & hold baseline from thesis)
    if metrics.sharpe_ratio < sharpe_absolute_threshold:
        reasons.append(
            f"Sharpe below absolute floor: current={metrics.sharpe_ratio:.4f} < "
            f"absolute_threshold={sharpe_absolute_threshold}"
        )
    # P&L floor (must be profitable)
    if metrics.cumulative_pnl <= pnl_absolute_threshold:
        reasons.append(
            f"P&L below absolute floor: current={metrics.cumulative_pnl:.6f} <= "
            f"absolute_threshold={pnl_absolute_threshold}"
        )
    return reasons


def absolute_floor_gate(
    metrics: BacktestMetrics,
    sharpe_absolute_threshold: float = DEFAULT_SHARPE_ABSOLUTE_THRESHOLD,
    pnl_absolute_threshold: float = DEFAULT_PNL_ABSOLUTE_THRESHOLD,
) -> tuple[bool, str]:
    """The gate applied when there is no production baseline (bootstrap / first
    deployment): run the absolute floors only.

    Without this, the bootstrap path promoted ANY model, including degenerate
    ones (e.g. Sharpe ≈ -2e16, P&L 0), because the absolute floors lived only
    inside the relative comparison that bootstrap skipped.
    """
    reasons = absolute_floor_reasons(
        metrics,
        sharpe_absolute_threshold=sharpe_absolute_threshold,
        pnl_absolute_threshold=pnl_absolute_threshold,
    )
    if reasons:
        return False, "; ".join(reasons)
    return True, "Absolute floors passed (no production baseline to compare)"


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

        # Resolve evaluation windows: one episode per calendar date when a date
        # range is supplied (distinct sessions → true cumulative P&L + finite
        # cross-session Sharpe), else the legacy single-window-repeated mode.
        eval_windows = resolve_eval_windows(
            date_start=args.date_start,
            date_end=args.date_end,
            hour_of_day_start=args.hour_of_day_start,
            hour_of_day_end=args.hour_of_day_end,
            eval_episode_start_ts=args.eval_episode_start_ts,
            eval_episode_end_ts=args.eval_episode_end_ts,
            num_eval_episodes=args.num_eval_episodes,
        )
        date_range_mode = bool(
            args.date_start
            and args.date_end
            and args.hour_of_day_start is not None
            and args.hour_of_day_end is not None
        )
        logger.info(
            "Resolved evaluation windows",
            extra={
                "mode": "date-range" if date_range_mode else "fixed-window",
                "num_windows": len(eval_windows),
                "date_start": args.date_start,
                "date_end": args.date_end,
            },
        )

        episode_results: list[EpisodeResult] = []

        for episode_idx, (win_start, win_end) in enumerate(eval_windows):
            episode_seed = 42 + episode_idx  # Deterministic seeds for reproducibility

            logger.info(
                f"Episode {episode_idx + 1}/{len(eval_windows)} starting",
                extra={
                    "episode": episode_idx + 1,
                    "seed": episode_seed,
                    "episode_start_ts": win_start,
                    "episode_end_ts": win_end,
                },
            )

            result = run_evaluation_episode(
                advisor=advisor,
                env_stub=env_stub,
                symbol=args.symbol,
                episode_start_ts=win_start,
                episode_end_ts=win_end,
                step_size_seconds=args.step_size_seconds,
                max_steps=args.max_steps_per_episode,
                episode_seed=episode_seed,
            )

            episode_results.append(result)

            logger.info(
                f"Episode {episode_idx + 1}/{len(eval_windows)} completed",
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
    #
    # The absolute floors (Sharpe >= threshold, P&L > 0) ALWAYS apply; a model
    # with sub-floor metrics must never be promoted, production baseline or not.
    # "Bootstrap" only means "no production baseline to compare against", so it
    # skips the RELATIVE checks; it does NOT skip the absolute floors. (Before
    # this, bootstrap skipped the whole gate, so a degenerate model, e.g.
    # Sharpe ~ -2e16, P&L 0, auto-promoted.) gate_skipped now means "relative
    # checks skipped", and is only meaningful when the floors passed.
    def _bootstrap_gate(note: str) -> tuple[bool, bool, str]:
        passed, reason = absolute_floor_gate(
            metrics,
            sharpe_absolute_threshold=args.sharpe_absolute_threshold,
            pnl_absolute_threshold=args.pnl_absolute_threshold,
        )
        if passed:
            return True, True, f"{note}; absolute floors passed, auto-promoting"
        return False, False, f"{note}; blocked by absolute floor: {reason}"

    gate_passed = True
    gate_skipped = False
    gate_reason = ""

    if not args.production_metrics_path:
        gate_passed, gate_skipped, gate_reason = _bootstrap_gate(
            "Bootstrap: no production model"
        )
        logger.info(
            "Degradation gate (bootstrap (absolute floors only)",
            extra={
                "gate_passed": gate_passed,
                "gate_skipped": gate_skipped,
                "gate_reason": gate_reason,
            },
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
            # Production metrics file doesn't exist) treat as bootstrap, but
            # still enforce the absolute floors.
            gate_passed, gate_skipped, gate_reason = _bootstrap_gate(
                "Production metrics file not found"
            )
            logger.warning(
                "Production metrics not found; applying absolute floors only",
                extra={
                    "production_metrics_path": args.production_metrics_path,
                    "gate_passed": gate_passed,
                    "gate_skipped": gate_skipped,
                    "gate_reason": gate_reason,
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
        # Overnight financing the modelenv sidecar applied this backtest. The
        # sidecar runs in Training mode without swap CLI flags, so these are the
        # built-in default-table rates unless overridden via MODELENV_SWAP_* /
        # MODELENV_NO_SWAP env. See deepqnetwork/swap_rates.py.
        "swap_rates": resolve_swap_rates(args.symbol),
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
                "swap_rates": output["swap_rates"],
            },
        )


if __name__ == "__main__":
    main()
