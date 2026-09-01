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
from typing import Iterator

from deepqnetwork.agent import DQNAgent
from deepqnetwork.checkpoint_manager import CheckpointManager
from deepqnetwork.config import DQNConfig, load_config
from deepqnetwork.episode_windows import iter_date_episodes
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

    agent.q_network.load_state_dict(checkpoint_data["q_network_state_dict"])
    agent.target_network.load_state_dict(checkpoint_data["target_network_state_dict"])
    agent.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

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


_iter_date_episodes = iter_date_episodes

DEFAULT_NUM_EPISODES_PER_RANGE = 3000
DEFAULT_REPEATS_PER_DATE = 3


def train(config: DQNConfig) -> None:
    """Run the DQN training loop.

    Args:
        config: Fully resolved DQN configuration.
    """
    if not logging.getLogger().handlers:
        setup_logging(level="INFO", csv_path=f"{config.checkpoint_dir}/training.log")

    device = resolve_device(config.device)
    logger.info("Device selected: %s", device)

    logger.info("Resolved configuration: %s", asdict(config))

    preprocessor = StatePreprocessor(device)

    agent = DQNAgent(config, device)

    env_client = EnvironmentClient(
        address=config.grpc_address,
        timeout=30.0,
        max_retries=5,
    )

    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=config.checkpoint_dir,
        s3_prefix=config.s3_checkpoint_prefix,
        symbol=config.symbol,
        horizon=str(config.step_size_seconds),
        version=config.model_version or "1",
    )

    start_episode = 0
    if config.checkpoint:
        checkpoint_data = checkpoint_mgr.load(config.checkpoint)
        start_episode = _restore_checkpoint(checkpoint_data, agent)

    step_count = agent.step_count
    recent_rewards: list[float] = []
    best_reward = float("-inf")

    if config.date_start and config.date_end:
        mode = "date-range"
        if config.num_episodes_per_range is not None:
            raise ValueError(
                "num_episodes_per_range is set but has no effect in date-range "
                "mode; use repeats_per_date to control replays per date."
            )
        episode_windows = _iter_date_episodes(
            config.date_start,
            config.date_end,
            config.hour_of_day_start,
            config.hour_of_day_end,
        )
    else:
        mode = "fixed"
        if config.repeats_per_date is not None:
            raise ValueError(
                "repeats_per_date is set but has no effect in fixed-window mode; "
                "use num_episodes_per_range to control the episode count."
            )
        num_episodes = (
            config.num_episodes_per_range
            if config.num_episodes_per_range is not None
            else DEFAULT_NUM_EPISODES_PER_RANGE
        )
        episode_windows = [
            (config.episode_start_ts, config.episode_end_ts)
        ] * num_episodes

    logger.info(
        "Starting training: episodes=%d (mode=%s), max_steps=%d, "
        "train_freq=%d, target_update_freq=%d",
        len(episode_windows),
        mode,
        config.max_steps_per_episode,
        config.train_freq,
        config.target_update_freq,
    )

    if mode == "date-range":
        repeats_per_date = (
            config.repeats_per_date
            if config.repeats_per_date is not None
            else DEFAULT_REPEATS_PER_DATE
        )
    else:
        repeats_per_date = 1
    overall_episode = start_episode
    total_episodes = len(episode_windows) * repeats_per_date

    if mode == "date-range":
        logger.info(
            "date-range mode: %d date windows x %d repeats_per_date = %d episodes",
            len(episode_windows),
            repeats_per_date,
            total_episodes,
        )

    if config.epsilon_decay_steps <= 0:
        budget = 0
        for ep_start, ep_end in episode_windows:
            window_steps = (ep_end - ep_start) // config.step_size_seconds
            per_episode = (
                config.max_steps_per_episode
                if window_steps <= 0
                else min(config.max_steps_per_episode, window_steps)
            )
            budget += per_episode
        config.epsilon_decay_steps = max(1, budget * repeats_per_date)
        logger.info(
            "epsilon_decay_steps auto-derived from step budget: %d "
            "(total_episodes=%d, ~%d steps/episode)",
            config.epsilon_decay_steps,
            total_episodes,
            config.epsilon_decay_steps // max(1, total_episodes),
        )

    for date_idx, (ep_start, ep_end) in enumerate(episode_windows):
        for repeat in range(repeats_per_date):
            if overall_episode < start_episode:
                overall_episode += 1
                continue
            episode_start_time = time.time()

            obs = env_client.reset(
                symbol=config.symbol,
                episode_start_ts=ep_start,
                episode_end_ts=ep_end,
                step_size_seconds=config.step_size_seconds,
            )
            state = preprocessor.process(obs)
            episode_reward = 0.0
            episode_losses: list[float] = []
            last_loss: float | None = None

            t_action = t_env = t_prep = t_learn = 0.0

            for step in range(config.max_steps_per_episode):
                _t0 = time.perf_counter()
                action = agent.select_action(state, training=True)
                t_action += time.perf_counter() - _t0

                _t0 = time.perf_counter()
                response = env_client.step(action, generate_order_id())
                t_env += time.perf_counter() - _t0
                _t0 = time.perf_counter()
                next_state = preprocessor.process(response.data)
                t_prep += time.perf_counter() - _t0
                reward = response.data.reward
                done = response.data.done

                state_np = state.cpu().numpy()
                next_state_np = next_state.cpu().numpy()
                agent.replay_buffer.push(state_np, action, reward, next_state_np, done)

                loss = None
                if (
                    len(agent.replay_buffer) >= config.batch_size
                    and step_count % config.train_freq == 0
                ):
                    _t0 = time.perf_counter()
                    loss = agent.update()
                    t_learn += time.perf_counter() - _t0
                    if loss is not None:
                        episode_losses.append(loss)
                        last_loss = loss

                agent.step_epsilon()

                step_count += 1
                if step_count % config.target_update_freq == 0:
                    agent.sync_target()

                state = next_state
                episode_reward += reward

                if (
                    config.progress_log_interval
                    and step > 0
                    and step % config.progress_log_interval == 0
                ):
                    elapsed = time.time() - episode_start_time
                    steps_per_sec = step / elapsed if elapsed > 0 else 0.0
                    t_total = t_action + t_env + t_prep + t_learn
                    pct = (lambda v: 100.0 * v / t_total if t_total > 0 else 0.0)
                    logger.info(
                        "  [progress] ep=%d step=%d/%d global_step=%d epsilon=%.3f "
                        "buffer=%d reward=%.4f last_loss=%s %.1f steps/s | "
                        "profile: env=%.0f%% learn=%.0f%% action=%.0f%% prep=%.0f%% "
                        "(env=%.2fms learn=%.2fms / step)",
                        overall_episode,
                        step,
                        config.max_steps_per_episode,
                        step_count,
                        agent.epsilon,
                        len(agent.replay_buffer),
                        episode_reward,
                        f"{last_loss:.6f}" if last_loss is not None else "n/a",
                        steps_per_sec,
                        pct(t_env),
                        pct(t_learn),
                        pct(t_action),
                        pct(t_prep),
                        1000.0 * t_env / step,
                        1000.0 * t_learn / step,
                    )

                if done:
                    break

            episode_duration = time.time() - episode_start_time
            avg_loss = (
                sum(episode_losses) / len(episode_losses) if episode_losses else 0.0
            )
            episode_length = step + 1

            t_total = t_action + t_env + t_prep + t_learn
            if t_total > 0:
                logger.info(
                    "  [profile] ep=%d steps=%d | env=%.0f%% learn=%.0f%% "
                    "action=%.0f%% prep=%.0f%% | env=%.2fms learn=%.2fms "
                    "action=%.2fms prep=%.2fms per step | timed=%.1fs of %.1fs",
                    overall_episode,
                    episode_length,
                    100.0 * t_env / t_total,
                    100.0 * t_learn / t_total,
                    100.0 * t_action / t_total,
                    100.0 * t_prep / t_total,
                    1000.0 * t_env / episode_length,
                    1000.0 * t_learn / episode_length,
                    1000.0 * t_action / episode_length,
                    1000.0 * t_prep / episode_length,
                    t_total,
                    episode_duration,
                )

            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            best_reward = max(best_reward, episode_reward)

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

    setup_logging(level="INFO", csv_path=f"{config.checkpoint_dir}/training.log")

    logger.info("DQN Training Agent starting...")
    logger.info("Mode: %s", config.mode)

    if config.mode != "train":
        logger.error("train.py is the training entry point. Use mode='train'.")
        raise SystemExit(1)

    train(config)


if __name__ == "__main__":
    main()
