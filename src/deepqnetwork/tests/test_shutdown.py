"""Unit tests for the graceful shutdown module.

Tests signal handler registration, non-reentrant behaviour, checkpoint saving,
gRPC channel closure, and log flushing during shutdown.
"""

import logging
import signal
from unittest.mock import MagicMock, patch

import pytest

from deepqnetwork.shutdown import GracefulShutdown


class TestGracefulShutdown:
    """Tests for GracefulShutdown class."""

    def _make_mock_agent(self):
        """Create a mock DQNAgent with required attributes."""
        agent = MagicMock()
        agent.q_network = MagicMock()
        agent.target_network = MagicMock()
        agent.optimizer = MagicMock()
        agent.epsilon = 0.5
        agent.step_count = 1000
        return agent

    def _make_mock_checkpoint_mgr(self):
        """Create a mock CheckpointManager."""
        mgr = MagicMock()
        mgr.save.return_value = "/tmp/dqn_episode_42.pt"
        return mgr

    def _make_mock_env_client(self):
        """Create a mock EnvironmentClient."""
        return MagicMock()

    def test_initial_state(self):
        """Shutdown handler starts with shutdown_requested=False."""
        shutdown = GracefulShutdown()
        assert shutdown.shutdown_requested is False
        assert shutdown.check_shutdown() is False

    def test_register_installs_signal_handlers(self):
        """register() installs SIGINT and SIGTERM handlers."""
        shutdown = GracefulShutdown()
        agent = self._make_mock_agent()
        mgr = self._make_mock_checkpoint_mgr()
        client = self._make_mock_env_client()

        with patch("signal.signal") as mock_signal:
            shutdown.register(agent, mgr, client)

            mock_signal.assert_any_call(signal.SIGINT, shutdown.request_shutdown)
            mock_signal.assert_any_call(signal.SIGTERM, shutdown.request_shutdown)

    def test_request_shutdown_sets_flag(self):
        """request_shutdown sets shutdown_requested to True."""
        shutdown = GracefulShutdown()
        agent = self._make_mock_agent()
        mgr = self._make_mock_checkpoint_mgr()
        client = self._make_mock_env_client()

        shutdown.register(agent, mgr, client)
        shutdown.request_shutdown(signal.SIGINT, None)

        assert shutdown.shutdown_requested is True
        assert shutdown.check_shutdown() is True

    def test_request_shutdown_saves_checkpoint(self):
        """request_shutdown saves a final checkpoint via the checkpoint manager."""
        shutdown = GracefulShutdown()
        agent = self._make_mock_agent()
        mgr = self._make_mock_checkpoint_mgr()
        client = self._make_mock_env_client()

        shutdown.register(agent, mgr, client)
        shutdown.set_episode(42)
        shutdown.request_shutdown(signal.SIGTERM, None)

        mgr.save.assert_called_once_with(
            episode=42,
            q_network=agent.q_network,
            target_network=agent.target_network,
            optimizer=agent.optimizer,
            epsilon=agent.epsilon,
            step_count=agent.step_count,
        )

    def test_request_shutdown_closes_grpc_channel(self):
        """request_shutdown closes the gRPC environment client channel."""
        shutdown = GracefulShutdown()
        agent = self._make_mock_agent()
        mgr = self._make_mock_checkpoint_mgr()
        client = self._make_mock_env_client()

        shutdown.register(agent, mgr, client)
        shutdown.request_shutdown(signal.SIGINT, None)

        client.close.assert_called_once()

    def test_request_shutdown_flushes_logs(self):
        """request_shutdown flushes all root logger handlers."""
        shutdown = GracefulShutdown()
        agent = self._make_mock_agent()
        mgr = self._make_mock_checkpoint_mgr()
        client = self._make_mock_env_client()

        mock_handler = MagicMock()
        root_logger = logging.getLogger()
        root_logger.addHandler(mock_handler)

        try:
            shutdown.register(agent, mgr, client)
            shutdown.request_shutdown(signal.SIGINT, None)

            assert mock_handler.flush.call_count >= 1
        finally:
            root_logger.removeHandler(mock_handler)

    def test_non_reentrant_ignores_subsequent_signals(self):
        """Subsequent signals during shutdown are ignored (non-reentrant)."""
        shutdown = GracefulShutdown()
        agent = self._make_mock_agent()
        mgr = self._make_mock_checkpoint_mgr()
        client = self._make_mock_env_client()

        shutdown.register(agent, mgr, client)

        shutdown.request_shutdown(signal.SIGINT, None)
        assert mgr.save.call_count == 1
        assert client.close.call_count == 1

        shutdown.request_shutdown(signal.SIGTERM, None)
        assert mgr.save.call_count == 1
        assert client.close.call_count == 1

    def test_set_episode_updates_episode_number(self):
        """set_episode updates the episode used for checkpoint naming."""
        shutdown = GracefulShutdown()
        agent = self._make_mock_agent()
        mgr = self._make_mock_checkpoint_mgr()
        client = self._make_mock_env_client()

        shutdown.register(agent, mgr, client)
        shutdown.set_episode(100)
        shutdown.request_shutdown(signal.SIGINT, None)

        call_kwargs = mgr.save.call_args[1]
        assert call_kwargs["episode"] == 100

    def test_checkpoint_save_failure_does_not_crash(self):
        """If checkpoint save fails, shutdown continues without crashing."""
        shutdown = GracefulShutdown()
        agent = self._make_mock_agent()
        mgr = self._make_mock_checkpoint_mgr()
        mgr.save.side_effect = RuntimeError("Disk full")
        client = self._make_mock_env_client()

        shutdown.register(agent, mgr, client)
        shutdown.request_shutdown(signal.SIGINT, None)

        client.close.assert_called_once()

    def test_grpc_close_failure_does_not_crash(self):
        """If gRPC close fails, shutdown continues without crashing."""
        shutdown = GracefulShutdown()
        agent = self._make_mock_agent()
        mgr = self._make_mock_checkpoint_mgr()
        client = self._make_mock_env_client()
        client.close.side_effect = RuntimeError("Channel error")

        shutdown.register(agent, mgr, client)
        shutdown.request_shutdown(signal.SIGTERM, None)

        mgr.save.assert_called_once()

    def test_shutdown_with_sigint_logs_signal_name(self, caplog):
        """SIGINT shutdown logs the signal name."""
        shutdown = GracefulShutdown()
        agent = self._make_mock_agent()
        mgr = self._make_mock_checkpoint_mgr()
        client = self._make_mock_env_client()

        shutdown.register(agent, mgr, client)

        with caplog.at_level(logging.INFO):
            shutdown.request_shutdown(signal.SIGINT, None)

        assert "SIGINT" in caplog.text
        assert "graceful shutdown" in caplog.text.lower()

    def test_shutdown_with_sigterm_logs_signal_name(self, caplog):
        """SIGTERM shutdown logs the signal name."""
        shutdown = GracefulShutdown()
        agent = self._make_mock_agent()
        mgr = self._make_mock_checkpoint_mgr()
        client = self._make_mock_env_client()

        shutdown.register(agent, mgr, client)

        with caplog.at_level(logging.INFO):
            shutdown.request_shutdown(signal.SIGTERM, None)

        assert "SIGTERM" in caplog.text
        assert "Shutdown complete" in caplog.text

    def test_check_shutdown_returns_false_before_signal(self):
        """check_shutdown returns False when no signal has been received."""
        shutdown = GracefulShutdown()
        agent = self._make_mock_agent()
        mgr = self._make_mock_checkpoint_mgr()
        client = self._make_mock_env_client()

        shutdown.register(agent, mgr, client)
        assert shutdown.check_shutdown() is False

    def test_check_shutdown_returns_true_after_signal(self):
        """check_shutdown returns True after a signal has been received."""
        shutdown = GracefulShutdown()
        agent = self._make_mock_agent()
        mgr = self._make_mock_checkpoint_mgr()
        client = self._make_mock_env_client()

        shutdown.register(agent, mgr, client)
        shutdown.request_shutdown(signal.SIGINT, None)
        assert shutdown.check_shutdown() is True
