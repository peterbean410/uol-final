"""Training entry point for the DQN trading agent.

Wires together configuration, device resolution, agent, environment client,
checkpoint manager, and state preprocessor. Runs the training loop with
configurable episodes, steps, gradient updates, target syncs, and checkpointing.

Supports two modes:

- **Fixed timestamps**: episode_start_ts / episode_end_ts (legacy, all episodes
  train on the same time window).
- **Date-range**: date_start / date_end / hour_of_day_start / hour_of_day_end
  (one episode per calendar date, each with its own time window).

Usage:
    python -m deepqnetwork.train --config deepqnetwork/config.yaml
    python -m deepqnetwork.train --config deepqnetwork/config.yaml --checkpoint path/to/checkpoint.pt
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from datetime import date, datetime, timezone
from typing import Iterator

from deepqnetwork.agent import DQNAgent
from deepqnetwork.checkpoint_manager import CheckpointManager
from deepqnetwork.config import DQNConfig, load_config
from deepqnetwork.environment_client import EnvironmentClient
from deepqnetwork.preprocessor import StatePreprocessor
from deepqnetwork.utils import generate_order_id, resolve_device, setup_logging

logger = logging.getLogger(__name__)


def _restore_checkpoint(
    checkpoint_data: dict,
    agent: DQNAgent,
) -> int:
    """Restore agent state from a checkpoint dictionary.

    Args:
        checkpoint_data: Dictionary loaded from a checkpoint file.
        agent: The DQNAgent instance to restore state into.

    Returns:
        The episode count to resume from.
    """
    if not checkpoint_data:
        return 0

    # Restore network weights
    agent.q_network.load_state_dict(checkpoint_data["q_network_state_dict"])
    agent.target_network.load_state_dict(checkpoint_data["target_network_state_dict"])
    agent.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    # Restore training state
    agent.epsilon = checkpoint_data["epsilon"]
    agent.step_count = checkpoint_data["step_count"]
    start_episode = checkpoint_data.get("episode_count", 0)

    logger.info(
        "Resumed from checkpoint: episode=%d, step_count=%d, epsilon=%.4f",
        start_episode,
        agent.step_count,
        agent.epsilon,
    )
    return start_episode


def _iter_date_episodes(
    date_start: str,
    date_end: str,
    hour_start: int,
    hour_end: int,
) -> list[tuple[int, int]]:
    """Generate per-date (episode_start_ts, episode_end_ts) pairs.

    Each calendar date from ``date_start`` through ``date_end`` (inclusive)
    produces one pair.  The Unix timestamps are computed from the configured
    hour-of-day window.

    Args:
        date_start: ISO date string for the first date (e.g. ``"2012-01-01"``).
        date_end: ISO date string for the last date (e.g. ``"2022-12-31"``).
        hour_start: Hour (0-23) at which each episode begins.
        hour_end: Hour (0-23) at which each episode ends.

    Returns:
        List of ``(episode_start_ts, episode_end_ts)`` tuples, one per date.
    """
    ds = date.fromisoformat(date_start)
    de = date.fromisoformat(date_end)
    if ds > de:
        raise ValueError(
            f"date_start ({date_start}) must be <= date_end ({date_end})"
        )

    episodes: list[tuple[int, int]] = []
    current = ds
    while current <= de:
        dt_start = datetime(
            current.year, current.month, current.day,
            hour_start, 0, 0, tzinfo=timezone.utc,
        )
        dt_end = datetime(
            current.year, current.month, current.day,
            hour_end, 0, 0, tzinfo=timezone.utc,
        )
        episodes.append((int(dt_start.timestamp()), int(dt_end.timestamp())))
        current = date.fromordinal(current.toordinal() + 1)
    return episodes


def train(config: DQNConfig) -> None:
    """Run the DQN training loop.

    Args:
        config: Fully resolved DQN configuration.
    """
    # Resolve device
    device = resolve_device(config.device)
    logger.info("Device selected: %s", device)

    # Log full resolved config
    logger.info("Resolved configuration: %s", asdict(config))

    # Create state preprocessor
    preprocessor = StatePreprocessor(device)

    # Create DQN agent
    agent = DQNAgent(config, device)

    # Create environment client
    env_client = EnvironmentClient(
        address=config.grpc_address,
        timeout=30.0,
        max_retries=5,
    )

    # Create checkpoint manager
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=config.checkpoint_dir,
        s3_prefix=config.s3_checkpoint_prefix,
        symbol=config.symbol,
        horizon=str(config.step_size_seconds),
        version=config.model_version or "1",
    )

    # Optional resume from checkpoint
    start_episode = 0
    if config.checkpoint:
        checkpoint_data = checkpoint_mgr.load(config.checkpoint)
        start_episode = _restore_checkpoint(checkpoint_data, agent)

    # Training loop
    step_count = agent.step_count
    recent_rewards: list[float] = []
    best_reward = float("-inf")

    # Resolve episode windows: date-range mode takes precedence over fixed
    # episode_start_ts/end_ts.
    if config.date_start and config.date_end:
        episode_windows = _iter_date_episodes(
            config.date_start,
            config.date_end,
            config.hour_of_day_start,
            config.hour_of_day_end,
        )
        mode = "date-range"
    else:
        # Legacy fixed-timestamp mode, same window for every episode.
        episode_windows = [
            (config.episode_start_ts, config.episode_end_ts)
        ] * config.num_episodes
        mode = "fixed"

    logger.info(
        "Starting training: episodes=%d (mode=%s), max_steps=%d, "
        "train_freq=%d, target_update_freq=%d",
        len(episode_windows),
        mode,
        config.max_steps_per_episode,
        config.train_freq,
        config.target_update_freq,
    )

    repeats_per_date = 3 if mode == "date-range" else 1
    overall_episode = start_episode
    total_episodes = len(episode_windows) * repeats_per_date

    for date_idx, (ep_start, ep_end) in enumerate(episode_windows):
        for repeat in range(repeats_per_date):
            if overall_episode < start_episode:
                overall_episode += 1
                continue
            episode_start_time = time.time()

            # Reset environment for new episode
            obs = env_client.reset(
                symbol=config.symbol,
                episode_start_ts=ep_start,
                episode_end_ts=ep_end,
                step_size_seconds=config.step_size_seconds,
            )
            state = preprocessor.process(obs)
            episode_reward = 0.0
            episode_losses: list[float] = []

            for step in range(config.max_steps_per_episode):
                # Select action (epsilon-greedy)
                action = agent.select_action(state, training=True)

                # Step environment
                response = env_client.step(action, generate_order_id())
                next_state = preprocessor.process(response.data)
                reward = response.data.reward
                done = response.data.done

                # Store transition in replay buffer (numpy arrays)
                state_np = state.cpu().numpy()
                next_state_np = next_state.cpu().numpy()
                agent.replay_buffer.push(state_np, action, reward, next_state_np, done)

                # Gradient update at train_freq intervals
                loss = None
                if (
                    len(agent.replay_buffer) >= config.batch_size
                    and step_count % config.train_freq == 0
                ):
                    loss = agent.update()
                    if loss is not None:
                        episode_losses.append(loss)

                # Epsilon decay
                agent.step_epsilon()

                # Target network sync at target_update_freq intervals
                step_count += 1
                if step_count % config.target_update_freq == 0:
                    agent.sync_target()

                # Advance state
                state = next_state
                episode_reward += reward

                if done:
                    break

            # Episode complete, logging
            episode_duration = time.time() - episode_start_time
            avg_loss = (
                sum(episode_losses) / len(episode_losses) if episode_losses else 0.0
            )
            episode_length = step + 1

            # Track rewards for rolling average
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            best_reward = max(best_reward, episode_reward)

            # Log episode metrics
            if overall_episode % config.log_interval == 0:
                avg_recent = sum(recent_rewards) / len(recent_rewards)
                logger.info(
                    "Episode %d | date=%s | repeat=%d | reward=%.4f | length=%d | "
                    "avg_loss=%.6f | epsilon=%.4f | duration=%.1fs | "
                    "avg_100=%.4f | best=%.4f",
                    overall_episode,
                    ep_start,
                    repeat,
                    episode_reward,
                    episode_length,
                    avg_loss,
                    agent.epsilon,
                    episode_duration,
                    avg_recent,
                    best_reward,
                )

            # Checkpoint saving at configurable intervals
            if overall_episode > 0 and overall_episode % config.checkpoint_interval == 0:
                avg_recent = sum(recent_rewards) / len(recent_rewards)
                logger.info(
                    "Checkpoint interval: best_reward=%.4f, avg_100=%.4f",
                    best_reward,
                    avg_recent,
                )
                checkpoint_mgr.save(
                    episode=overall_episode,
                    q_network=agent.q_network,
                    target_network=agent.target_network,
                    optimizer=agent.optimizer,
                    epsilon=agent.epsilon,
                    step_count=step_count,
                    config=asdict(config),
                )

            overall_episode += 1

    # Final checkpoint at end of training
    logger.info("Training complete. Saving final checkpoint.")
    checkpoint_mgr.save(
        episode=overall_episode,
        q_network=agent.q_network,
        target_network=agent.target_network,
        optimizer=agent.optimizer,
        epsilon=agent.epsilon,
        step_count=step_count,
        config=asdict(config),
    )

    # Clean up
    env_client.close()
    logger.info(
        "Training finished: %d episodes, %d total steps, best_reward=%.4f",
        overall_episode - start_episode,
        step_count,
        best_reward,
    )


def main() -> None:
    """CLI entry point for DQN training."""
    config = load_config()

    # Setup logging
    setup_logging(level="INFO", csv_path=f"{config.checkpoint_dir}/training.log")

    # Log startup
    logger.info("DQN Training Agent starting...")
    logger.info("Mode: %s", config.mode)

    if config.mode != "train":
        logger.error("train.py is the training entry point. Use mode='train'.")
        raise SystemExit(1)

    train(config)


if __name__ == "__main__":
    main()
