"""Graceful shutdown handling for the DQN trading agent.

Registers SIGINT/SIGTERM signal handlers to safely terminate training by
saving a final checkpoint, closing the gRPC channel, and flushing logs.

Requirements: 15.1, 15.2, 15.3, 15.4
"""

from __future__ import annotations

import logging
import signal
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepqnetwork.agent import DQNAgent
    from deepqnetwork.checkpoint_manager import CheckpointManager
    from deepqnetwork.environment_client import EnvironmentClient

logger = logging.getLogger(__name__)

_SIGNAL_NAMES = {
    signal.SIGINT: "SIGINT",
    signal.SIGTERM: "SIGTERM",
}


class GracefulShutdown:
    """Manages graceful shutdown on SIGINT/SIGTERM signals.

    Non-reentrant: subsequent signals during shutdown are ignored to prevent
    corruption from concurrent cleanup operations.

    Usage:
        shutdown = GracefulShutdown()
        shutdown.register(agent, checkpoint_mgr, env_client)

        # In training loop:
        if shutdown.check_shutdown():
            break
    """

    def __init__(self) -> None:
        self.shutdown_requested: bool = False
        self._shutting_down: bool = False
        self._agent: DQNAgent | None = None
        self._checkpoint_mgr: CheckpointManager | None = None
        self._env_client: EnvironmentClient | None = None
        self._episode: int = 0

    def register(
        self,
        agent: DQNAgent,
        checkpoint_mgr: CheckpointManager,
        env_client: EnvironmentClient,
    ) -> None:
        """Register components and install signal handlers.

        Args:
            agent: The DQN agent (provides networks, optimizer, epsilon, step_count).
            checkpoint_mgr: Checkpoint manager for saving final state.
            env_client: Environment client whose gRPC channel will be closed.
        """
        self._agent = agent
        self._checkpoint_mgr = checkpoint_mgr
        self._env_client = env_client

        signal.signal(signal.SIGINT, self.request_shutdown)
        signal.signal(signal.SIGTERM, self.request_shutdown)

        logger.info("Graceful shutdown handlers registered for SIGINT and SIGTERM")

    def set_episode(self, episode: int) -> None:
        """Update the current episode number for checkpoint naming.

        Args:
            episode: Current episode number in the training loop.
        """
        self._episode = episode

    def request_shutdown(self, signum: int, frame) -> None:
        """Signal handler that sets the shutdown flag and performs cleanup.

        This handler is non-reentrant, subsequent signals during shutdown
        are ignored to prevent concurrent cleanup.

        Args:
            signum: The signal number received (SIGINT or SIGTERM).
            frame: The current stack frame (unused).
        """
        if self._shutting_down:
            return

        self._shutting_down = True
        self.shutdown_requested = True

        signal_name = _SIGNAL_NAMES.get(signum, f"signal {signum}")
        logger.info("Received %s, initiating graceful shutdown...", signal_name)

        self._save_final_checkpoint()

        self._close_grpc_channel()

        self._flush_logs()

        logger.info("Shutdown complete (reason: %s)", signal_name)

        self._flush_logs()

    def check_shutdown(self) -> bool:
        """Check if shutdown was requested.

        Call this in the training loop at each iteration to allow clean exit.

        Returns:
            True if shutdown has been requested, False otherwise.
        """
        return self.shutdown_requested

    def _save_final_checkpoint(self) -> None:
        """Save a final checkpoint using the checkpoint manager."""
        if self._agent is None or self._checkpoint_mgr is None:
            logger.warning("Cannot save shutdown checkpoint: components not registered")
            return

        try:
            path = self._checkpoint_mgr.save(
                episode=self._episode,
                q_network=self._agent.q_network,
                target_network=self._agent.target_network,
                optimizer=self._agent.optimizer,
                epsilon=self._agent.epsilon,
                step_count=self._agent.step_count,
            )
            logger.info("Final shutdown checkpoint saved: %s", path)
        except Exception as e:
            logger.error("Failed to save shutdown checkpoint: %s", e)

    def _close_grpc_channel(self) -> None:
        """Close the gRPC environment client channel."""
        if self._env_client is None:
            return

        try:
            self._env_client.close()
        except Exception as e:
            logger.error("Error closing gRPC channel during shutdown: %s", e)

    def _flush_logs(self) -> None:
        """Flush all handlers on the root logger."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            try:
                handler.flush()
            except Exception:
                pass
