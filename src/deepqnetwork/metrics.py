"""Logging and metrics for the DQN trading agent.

Provides structured logging of per-episode and per-step metrics to console,
CSV files, and optionally TensorBoard.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import deque
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


logger = logging.getLogger(__name__)


class MetricsLogger:
    """Handles training metrics logging to console, CSV, and TensorBoard.

    Tracks per-episode and per-step metrics, maintains rolling statistics,
    and writes structured CSV files for offline analysis.

    Args:
        checkpoint_dir: Directory for CSV output files.
        log_interval: Step interval for per-step metric logging.
    """

    def __init__(
        self,
        checkpoint_dir: str = "deepqnetwork/checkpoints/",
        log_interval: int = 10,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        self._episode_csv_path = self.checkpoint_dir / "episode_metrics.csv"
        self._step_csv_path = self.checkpoint_dir / "step_metrics.csv"

        self._episode_csv_initialized = False
        self._step_csv_initialized = False

        self._reward_history: deque[float] = deque(maxlen=100)
        self._best_reward: float = float("-inf")

        self._tb_writer: SummaryWriter | None = None
        if TENSORBOARD_AVAILABLE:
            tb_log_dir = self.checkpoint_dir / "tensorboard"
            try:
                self._tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
                logger.info("TensorBoard logging enabled at %s", tb_log_dir)
            except Exception as e:
                logger.warning("Failed to initialize TensorBoard: %s", e)
                self._tb_writer = None
        else:
            logger.info(
                "TensorBoard not available (torch.utils.tensorboard not installed). "
                "Skipping TensorBoard logging."
            )

    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        avg_loss: float,
        epsilon: float,
        duration: float,
    ) -> None:
        """Log per-episode metrics to console, CSV, and TensorBoard.

        Args:
            episode: Episode number.
            reward: Total undiscounted episode reward.
            length: Number of steps in the episode.
            avg_loss: Average training loss over the episode.
            epsilon: Final epsilon value at end of episode.
            duration: Wall-clock duration of the episode in seconds.
        """
        self._reward_history.append(reward)
        if reward > self._best_reward:
            self._best_reward = reward

        logger.info(
            "Episode %d | Reward: %.4f | Length: %d | Avg Loss: %.6f | "
            "Epsilon: %.4f | Duration: %.2fs",
            episode,
            reward,
            length,
            avg_loss,
            epsilon,
            duration,
        )

        self._write_episode_csv(episode, reward, length, avg_loss, epsilon, duration)

        if self._tb_writer is not None:
            self._tb_writer.add_scalar("episode/reward", reward, episode)
            self._tb_writer.add_scalar("episode/length", length, episode)
            self._tb_writer.add_scalar("episode/avg_loss", avg_loss, episode)
            self._tb_writer.add_scalar("episode/epsilon", epsilon, episode)
            self._tb_writer.add_scalar("episode/duration_seconds", duration, episode)

    def log_step(
        self,
        step: int,
        action: int,
        reward: float,
        q_value: float,
        epsilon: float,
        loss: float | None,
    ) -> None:
        """Log per-step metrics at configured intervals.

        Only logs when step is a multiple of log_interval.

        Args:
            step: Global step number.
            action: Action taken (0-4).
            reward: Reward received.
            q_value: Q-value of the selected action.
            epsilon: Current epsilon value.
            loss: Training loss (None if no update occurred).
        """
        if step % self.log_interval != 0:
            return

        loss_val = loss if loss is not None else 0.0

        logger.debug(
            "Step %d | Action: %d | Reward: %.4f | Q-value: %.4f | "
            "Epsilon: %.4f | Loss: %.6f",
            step,
            action,
            reward,
            q_value,
            epsilon,
            loss_val,
        )

        self._write_step_csv(step, action, reward, q_value, epsilon, loss_val)

        if self._tb_writer is not None:
            self._tb_writer.add_scalar("step/reward", reward, step)
            self._tb_writer.add_scalar("step/q_value", q_value, step)
            self._tb_writer.add_scalar("step/epsilon", epsilon, step)
            if loss is not None:
                self._tb_writer.add_scalar("step/loss", loss, step)

    def log_checkpoint_summary(
        self, episode: int, rewards_history: list[float] | None = None
    ) -> None:
        """Log summary statistics at checkpoint intervals.

        Reports best episode reward and average reward over the last 100 episodes.

        Args:
            episode: Current episode number.
            rewards_history: Optional external rewards list. If None, uses
                internal rolling history.
        """
        if rewards_history is not None:
            recent = rewards_history[-100:]
        else:
            recent = list(self._reward_history)

        if not recent:
            return

        avg_reward = sum(recent) / len(recent)
        best_reward = self._best_reward

        logger.info(
            "Checkpoint Summary (Episode %d) | Best Reward: %.4f | "
            "Avg Reward (last %d): %.4f",
            episode,
            best_reward,
            len(recent),
            avg_reward,
        )

        if self._tb_writer is not None:
            self._tb_writer.add_scalar("summary/best_reward", best_reward, episode)
            self._tb_writer.add_scalar("summary/avg_reward_100", avg_reward, episode)

    @property
    def best_reward(self) -> float:
        """Best episode reward seen so far."""
        return self._best_reward

    @property
    def avg_reward_100(self) -> float:
        """Average reward over the last 100 episodes."""
        if not self._reward_history:
            return 0.0
        return sum(self._reward_history) / len(self._reward_history)

    def flush(self) -> None:
        """Flush all pending writes (TensorBoard)."""
        if self._tb_writer is not None:
            self._tb_writer.flush()

    def close(self) -> None:
        """Close the metrics logger and release resources."""
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None

    def _write_episode_csv(
        self,
        episode: int,
        reward: float,
        length: int,
        avg_loss: float,
        epsilon: float,
        duration: float,
    ) -> None:
        """Write a row to the episode metrics CSV file."""
        write_header = not self._episode_csv_initialized

        with open(self._episode_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ["episode", "reward", "length", "avg_loss", "epsilon", "duration_seconds"]
                )
                self._episode_csv_initialized = True
            writer.writerow([episode, reward, length, avg_loss, epsilon, duration])

    def _write_step_csv(
        self,
        step: int,
        action: int,
        reward: float,
        q_value: float,
        epsilon: float,
        loss: float,
    ) -> None:
        """Write a row to the step metrics CSV file."""
        write_header = not self._step_csv_initialized

        with open(self._step_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ["step", "action", "reward", "q_value", "epsilon", "loss"]
                )
                self._step_csv_initialized = True
            writer.writerow([step, action, reward, q_value, epsilon, loss])
