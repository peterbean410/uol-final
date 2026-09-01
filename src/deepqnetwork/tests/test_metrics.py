"""Unit tests for deepqnetwork.metrics module."""

import csv
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from deepqnetwork.metrics import MetricsLogger, TENSORBOARD_AVAILABLE


class TestMetricsLoggerInit:
    """Tests for MetricsLogger initialization."""

    def test_creates_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, "new_subdir", "checkpoints")
            logger = MetricsLogger(checkpoint_dir=checkpoint_dir)
            assert os.path.isdir(checkpoint_dir)
            logger.close()

    def test_default_log_interval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)
            assert logger.log_interval == 10
            logger.close()

    def test_custom_log_interval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                checkpoint_dir=tmpdir, log_interval=5)
            assert logger.log_interval == 5
            logger.close()

    def test_initial_best_reward(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)
            assert logger.best_reward == float("-inf")
            logger.close()

    def test_initial_avg_reward(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)
            assert logger.avg_reward_100 == 0.0
            logger.close()


class TestLogEpisode:
    """Tests for MetricsLogger.log_episode."""

    def test_writes_csv_header_on_first_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)
            logger.log_episode(
                episode=1, reward=10.5, length=100, avg_loss=0.05, epsilon=0.9, duration=5.2
            )

            csv_path = os.path.join(tmpdir, "episode_metrics.csv")
            assert os.path.exists(csv_path)

            with open(csv_path) as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert rows[0] == ["episode", "reward", "length", "avg_loss", "epsilon", "duration_seconds"]
            assert rows[1][0] == "1"
            logger.close()

    def test_writes_correct_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)
            logger.log_episode(
                episode=5, reward=-2.3, length=50, avg_loss=0.123, epsilon=0.5, duration=3.7
            )

            csv_path = os.path.join(tmpdir, "episode_metrics.csv")
            with open(csv_path) as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert rows[1] == ["5", "-2.3", "50", "0.123", "0.5", "3.7"]
            logger.close()

    def test_appends_multiple_episodes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)
            for i in range(3):
                logger.log_episode(
                    episode=i, reward=float(i), length=10 * i,
                    avg_loss=0.01 * i, epsilon=1.0 - 0.1 * i, duration=1.0
                )

            csv_path = os.path.join(tmpdir, "episode_metrics.csv")
            with open(csv_path) as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 4
            logger.close()

    def test_updates_best_reward(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)
            logger.log_episode(episode=1, reward=5.0, length=10, avg_loss=0.0, epsilon=1.0, duration=1.0)
            logger.log_episode(episode=2, reward=10.0, length=10, avg_loss=0.0, epsilon=1.0, duration=1.0)
            logger.log_episode(episode=3, reward=3.0, length=10, avg_loss=0.0, epsilon=1.0, duration=1.0)

            assert logger.best_reward == 10.0
            logger.close()

    def test_updates_rolling_average(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)
            for i in range(10):
                logger.log_episode(
                    episode=i, reward=float(i), length=10,
                    avg_loss=0.0, epsilon=1.0, duration=1.0
                )

            assert abs(logger.avg_reward_100 - 4.5) < 1e-6
            logger.close()

    def test_rolling_average_caps_at_100(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)
            for i in range(150):
                logger.log_episode(
                    episode=i, reward=float(i), length=10,
                    avg_loss=0.0, epsilon=1.0, duration=1.0
                )

            assert abs(logger.avg_reward_100 - 99.5) < 1e-6
            logger.close()

    def test_console_logging(self, caplog):
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)
            with caplog.at_level(logging.INFO):
                logger.log_episode(
                    episode=1, reward=5.0, length=100,
                    avg_loss=0.01, epsilon=0.9, duration=2.5
                )

            assert "Episode 1" in caplog.text
            assert "Reward: 5.0000" in caplog.text
            logger.close()


class TestLogStep:
    """Tests for MetricsLogger.log_step."""

    def test_logs_at_interval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                checkpoint_dir=tmpdir, log_interval=10)
            logger.log_step(step=10, action=1, reward=0.5, q_value=1.2, epsilon=0.8, loss=0.01)

            csv_path = os.path.join(tmpdir, "step_metrics.csv")
            assert os.path.exists(csv_path)

            with open(csv_path) as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert rows[0] == ["step", "action", "reward", "q_value", "epsilon", "loss"]
            assert rows[1][0] == "10"
            logger.close()

    def test_skips_non_interval_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                checkpoint_dir=tmpdir, log_interval=10)
            for step in range(1, 10):
                logger.log_step(step=step, action=0, reward=0.0, q_value=0.0, epsilon=1.0, loss=None)

            csv_path = os.path.join(tmpdir, "step_metrics.csv")
            assert not os.path.exists(csv_path)
            logger.close()

    def test_writes_correct_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                checkpoint_dir=tmpdir, log_interval=5)
            logger.log_step(step=5, action=3, reward=-0.1, q_value=2.5, epsilon=0.7, loss=0.05)

            csv_path = os.path.join(tmpdir, "step_metrics.csv")
            with open(csv_path) as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert rows[1] == ["5", "3", "-0.1", "2.5", "0.7", "0.05"]
            logger.close()

    def test_none_loss_written_as_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                checkpoint_dir=tmpdir, log_interval=1)
            logger.log_step(step=1, action=0, reward=0.0, q_value=0.0, epsilon=1.0, loss=None)

            csv_path = os.path.join(tmpdir, "step_metrics.csv")
            with open(csv_path) as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert rows[1][5] == "0.0"
            logger.close()

    def test_step_zero_logged_when_interval_nonzero(self):
        """Step 0 is a multiple of any interval, so it should be logged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                checkpoint_dir=tmpdir, log_interval=10)
            logger.log_step(step=0, action=0, reward=0.0, q_value=0.0, epsilon=1.0, loss=None)

            csv_path = os.path.join(tmpdir, "step_metrics.csv")
            assert os.path.exists(csv_path)
            logger.close()


class TestLogCheckpointSummary:
    """Tests for MetricsLogger.log_checkpoint_summary."""

    def test_logs_summary_with_internal_history(self, caplog):
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)
            for i in range(10):
                logger.log_episode(
                    episode=i, reward=float(i), length=10,
                    avg_loss=0.0, epsilon=1.0, duration=1.0
                )

            with caplog.at_level(logging.INFO):
                logger.log_checkpoint_summary(episode=10)

            assert "Checkpoint Summary" in caplog.text
            assert "Best Reward: 9.0000" in caplog.text
            assert "Avg Reward" in caplog.text
            logger.close()

    def test_logs_summary_with_external_history(self, caplog):
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)
            rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
            logger._best_reward = 5.0

            with caplog.at_level(logging.INFO):
                logger.log_checkpoint_summary(episode=5, rewards_history=rewards)

            assert "Checkpoint Summary" in caplog.text
            assert "Best Reward: 5.0000" in caplog.text
            logger.close()

    def test_no_output_when_empty_history(self, caplog):
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)

            with caplog.at_level(logging.INFO):
                logger.log_checkpoint_summary(episode=0)

            assert "Checkpoint Summary" not in caplog.text
            logger.close()


class TestTensorBoardIntegration:
    """Tests for optional TensorBoard integration."""

    @pytest.mark.skipif(not TENSORBOARD_AVAILABLE, reason="TensorBoard not installed")
    def test_tensorboard_enabled_creates_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)
            assert logger._tb_writer is not None
            logger.close()

    @pytest.mark.skipif(not TENSORBOARD_AVAILABLE, reason="TensorBoard not installed")
    def test_tensorboard_episode_scalars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)
            logger.log_episode(
                episode=1, reward=5.0, length=100,
                avg_loss=0.01, epsilon=0.9, duration=2.5
            )
            logger.close()

    @pytest.mark.skipif(not TENSORBOARD_AVAILABLE, reason="TensorBoard not installed")
    def test_tensorboard_step_scalars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                checkpoint_dir=tmpdir, log_interval=1)
            logger.log_step(step=1, action=0, reward=0.5, q_value=1.0, epsilon=0.8, loss=0.01)
            logger.close()


class TestFlushAndClose:
    """Tests for flush and close methods."""

    def test_close_sets_writer_to_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(checkpoint_dir=tmpdir)
            logger.close()
            assert logger._tb_writer is None
