"""Katib custom metrics collector for the DQN trading agent.

Extracts average episode reward from structured JSON training component logs and
provides integration with the Model Registry for recording trial results.

This module can be used:
1. As a Katib sidecar metrics collector (reads logs from stdin or file)
2. As a standalone utility for extracting metrics from training logs

The DQN training component logs structured JSON (via StructuredJsonFormatter from
monitoring/metrics.py). The average episode reward is logged with a specific message
pattern that this collector parses.

Requirements: DQN-R14
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO


@dataclass
class DQNTrialResult:
    """Result of a single DQN Katib trial.

    Attributes:
        trial_name: Unique identifier for the trial.
        hyperparameters: Dict of hyperparameter name → value used in this trial.
        avg_episode_reward: The average episode reward metric (objective to maximise).
        episode: The episode at which the metric was recorded.
    """

    trial_name: str
    hyperparameters: dict[str, float | int | str]
    avg_episode_reward: float
    episode: int = 0


@dataclass
class DQNBestConfiguration:
    """The best hyperparameter configuration from a completed DQN Katib experiment.

    Attributes:
        hyperparameters: Dict of hyperparameter name → optimal value.
        avg_episode_reward: The best (highest) average episode reward achieved.
        trial_name: Name of the trial that achieved the best result.
    """

    hyperparameters: dict[str, float | int | str]
    avg_episode_reward: float
    trial_name: str


def parse_log_line(line: str) -> dict | None:
    """Parse a single structured JSON log line.

    Handles malformed lines gracefully by returning None for non-JSON lines
    or lines that cannot be parsed.

    Args:
        line: A single line from the DQN training component's stdout.

    Returns:
        Parsed JSON dict if the line is valid JSON, None otherwise.
    """
    stripped = line.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None


def extract_avg_episode_reward(log_entry: dict) -> float | None:
    """Extract avg_episode_reward value from a parsed log entry.

    The DQN training component logs average episode reward in one of these patterns:
    1. As a field in the log entry: {"avg_episode_reward": <value>}
    2. As a nested field in "extras"/"metrics"/"extra": {"metrics": {"avg_episode_reward": <value>}}
    3. In the message containing "avg_episode_reward" with a numeric value

    Args:
        log_entry: A parsed JSON log dict.

    Returns:
        The avg_episode_reward float value if found, None otherwise.
    """
    if "avg_episode_reward" in log_entry:
        try:
            return float(log_entry["avg_episode_reward"])
        except (TypeError, ValueError):
            pass

    for container_key in ("extras", "metrics", "extra"):
        container = log_entry.get(container_key)
        if isinstance(container, dict) and "avg_episode_reward" in container:
            try:
                return float(container["avg_episode_reward"])
            except (TypeError, ValueError):
                pass

    message = log_entry.get("message", "")
    if "avg_episode_reward" in message:
        for part in message.split():
            if part.startswith("avg_episode_reward="):
                try:
                    return float(part.split("=", 1)[1])
                except (ValueError, IndexError):
                    pass
        if "avg_episode_reward:" in message:
            try:
                idx = message.index("avg_episode_reward:")
                remainder = message[idx + len("avg_episode_reward:"):].strip()
                value_str = remainder.split()[0].rstrip(",;")
                return float(value_str)
            except (ValueError, IndexError):
                pass

    return None


def extract_episode(log_entry: dict) -> int | None:
    """Extract episode number from a parsed log entry.

    Args:
        log_entry: A parsed JSON log dict.

    Returns:
        The episode number if found, None otherwise.
    """
    if "episode" in log_entry:
        try:
            return int(log_entry["episode"])
        except (TypeError, ValueError):
            pass

    for container_key in ("extras", "metrics", "extra"):
        container = log_entry.get(container_key)
        if isinstance(container, dict) and "episode" in container:
            try:
                return int(container["episode"])
            except (TypeError, ValueError):
                pass

    message = log_entry.get("message", "")
    if "episode" in message.lower():
        for part in message.split():
            if part.startswith("episode="):
                try:
                    return int(part.split("=", 1)[1])
                except (ValueError, IndexError):
                    pass
        if "episode:" in message.lower():
            try:
                idx = message.lower().index("episode:")
                remainder = message[idx + len("episode:"):].strip()
                value_str = remainder.split()[0].rstrip(",;")
                return int(value_str)
            except (ValueError, IndexError):
                pass

    return None


def collect_metrics_from_stream(
    stream: TextIO,
) -> list[tuple[int, float]]:
    """Read log lines from a stream and extract all (episode, avg_episode_reward) pairs.

    Handles malformed lines gracefully by skipping them.

    Args:
        stream: A text stream (stdin, file handle) producing log lines.

    Returns:
        List of (episode, avg_episode_reward) tuples in order of appearance.
    """
    results: list[tuple[int, float]] = []
    current_episode = 0

    for line in stream:
        entry = parse_log_line(line)
        if entry is None:
            continue

        episode = extract_episode(entry)
        if episode is not None:
            current_episode = episode

        reward = extract_avg_episode_reward(entry)
        if reward is not None:
            results.append((current_episode, reward))

    return results


def get_final_avg_episode_reward(
    metrics: list[tuple[int, float]],
) -> tuple[int, float] | None:
    """Get the final average episode reward from collected metrics.

    Returns the last recorded avg_episode_reward value, which corresponds to
    the final episode window of training.

    Args:
        metrics: List of (episode, avg_episode_reward) tuples.

    Returns:
        Tuple of (episode, avg_episode_reward) for the last entry, or None if empty.
    """
    if not metrics:
        return None
    return metrics[-1]


def format_katib_metric(metric_name: str, value: float) -> str:
    """Format a metric value in Katib's expected stdout format.

    Katib's StdOut metrics collector expects lines in the format:
        metric_name=value

    Args:
        metric_name: Name of the metric (e.g. "avg_episode_reward").
        value: The metric value.

    Returns:
        Formatted metric string for Katib collection.
    """
    return f"{metric_name}={value}"


def record_trial_in_registry(
    trial_result: DQNTrialResult,
    registry_url: str = "http://model-registry-service.kubeflow.svc.cluster.local:8080",
) -> str | None:
    """Record trial hyperparameters and objective in the Model Registry.

    Uses a forward-compatible interface with the DQN registry client that will be
    implemented at `deepqnetwork/kubeflow/registry/dqn_registry_client.py`.

    Args:
        trial_result: The trial result containing hyperparameters and objective.
        registry_url: URL of the Kubeflow Model Registry service.

    Returns:
        The registered version ID if successful, None on failure.
    """
    try:
        from deepqnetwork.kubeflow.registry.dqn_registry_client import (
            DQNModelMetadata,
            DQNRegistryClient,
        )

        client = DQNRegistryClient(registry_url=registry_url)

        metadata = DQNModelMetadata(
            symbol=str(trial_result.hyperparameters.get("symbol", "USDJPY")),
            episode_start_ts=int(
                trial_result.hyperparameters.get("episode_start_ts", 0)
            ),
            episode_end_ts=int(
                trial_result.hyperparameters.get("episode_end_ts", 0)
            ),
            step_size_seconds=int(
                trial_result.hyperparameters.get("step_size_seconds", 5)
            ),
            training_timestamp="",
            cumulative_pnl=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            avg_episode_reward=trial_result.avg_episode_reward,
            hyperparameters=trial_result.hyperparameters,
            pipeline_run_id=trial_result.trial_name,
            lifecycle_stage="staging",
        )

        version_id = client.register_model(
            model_path=f"katib-trial/{trial_result.trial_name}",
            metadata=metadata,
        )
        return version_id

    except ImportError:
        print(
            json.dumps(
                {
                    "level": "WARNING",
                    "message": (
                        "DQN Model Registry client not available. "
                        "Trial result not recorded in registry."
                    ),
                    "trial_name": trial_result.trial_name,
                    "avg_episode_reward": trial_result.avg_episode_reward,
                }
            ),
            file=sys.stderr,
        )
        return None
    except Exception as exc:
        print(
            json.dumps(
                {
                    "level": "ERROR",
                    "message": f"Failed to record DQN trial in registry: {exc}",
                    "trial_name": trial_result.trial_name,
                }
            ),
            file=sys.stderr,
        )
        return None


def determine_best_configuration(
    trial_results: list[DQNTrialResult],
) -> DQNBestConfiguration | None:
    """Determine the best hyperparameter configuration from completed trials.

    Selects the trial with the highest average episode reward (the objective
    to maximise for DQN).

    Args:
        trial_results: List of completed trial results.

    Returns:
        DQNBestConfiguration for the trial with highest avg_episode_reward,
        or None if no trials provided.
    """
    if not trial_results:
        return None

    best_trial = max(trial_results, key=lambda t: t.avg_episode_reward)

    return DQNBestConfiguration(
        hyperparameters=best_trial.hyperparameters,
        avg_episode_reward=best_trial.avg_episode_reward,
        trial_name=best_trial.trial_name,
    )


def output_best_configuration(
    best_config: DQNBestConfiguration,
    output_path: str | None = None,
) -> str:
    """Output the best hyperparameter configuration as a structured JSON artifact.

    Writes to a file if output_path is provided, otherwise returns the JSON string.
    The output format is compatible with KFP artifact storage and can be consumed
    by downstream pipeline components.

    Args:
        best_config: The best configuration to output.
        output_path: Optional file path to write the artifact. If None, only
            returns the JSON string.

    Returns:
        JSON string of the best configuration artifact.
    """
    artifact = {
        "best_trial": best_config.trial_name,
        "objective_metric": {
            "name": "avg_episode_reward",
            "value": best_config.avg_episode_reward,
        },
        "hyperparameters": best_config.hyperparameters,
    }

    json_str = json.dumps(artifact, indent=2)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_str)

    return json_str


def main() -> None:
    """Run the DQN metrics collector as a standalone script.

    Reads structured JSON logs from stdin or a log file, extracts the final
    avg_episode_reward, and outputs it in Katib-compatible format.

    Usage:
        # From stdin (Katib sidecar mode):
        cat training.log | python dqn_metrics_collector.py

        # From a log file:
        python dqn_metrics_collector.py --log-file /var/log/training.log

        # With trial recording:
        python dqn_metrics_collector.py --log-file /var/log/training.log \\
            --trial-name trial-001 \\
            --hyperparameters '{"learning_rate": 0.0001, "gamma": 0.99}'

        # Output best config from multiple trial results:
        python dqn_metrics_collector.py --best-config \\
            --trials-file /tmp/trials.json \\
            --output-path /tmp/best_config.json
    """
    parser = argparse.ArgumentParser(
        description="Katib metrics collector for DQN trading agent"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to training log file. If not provided, reads from stdin.",
    )
    parser.add_argument(
        "--trial-name",
        type=str,
        default=None,
        help="Name of the current Katib trial.",
    )
    parser.add_argument(
        "--hyperparameters",
        type=str,
        default=None,
        help="JSON string of trial hyperparameters.",
    )
    parser.add_argument(
        "--registry-url",
        type=str,
        default="http://model-registry-service.kubeflow.svc.cluster.local:8080",
        help="URL of the Model Registry service.",
    )
    parser.add_argument(
        "--best-config",
        action="store_true",
        help="Output best configuration from trials file instead of collecting metrics.",
    )
    parser.add_argument(
        "--trials-file",
        type=str,
        default=None,
        help="Path to JSON file containing trial results (for --best-config mode).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to write the best configuration artifact.",
    )

    args = parser.parse_args()

    if args.best_config:
        if not args.trials_file:
            print("Error: --trials-file required with --best-config", file=sys.stderr)
            sys.exit(1)

        trials_path = Path(args.trials_file)
        if not trials_path.exists():
            print(f"Error: trials file not found: {args.trials_file}", file=sys.stderr)
            sys.exit(1)

        trials_data = json.loads(trials_path.read_text())
        trial_results = [
            DQNTrialResult(
                trial_name=t["trial_name"],
                hyperparameters=t["hyperparameters"],
                avg_episode_reward=t["avg_episode_reward"],
                episode=t.get("episode", 0),
            )
            for t in trials_data
        ]

        best = determine_best_configuration(trial_results)
        if best is None:
            print("Error: no trial results found", file=sys.stderr)
            sys.exit(1)

        json_output = output_best_configuration(best, args.output_path)
        print(json_output)
        return

    if args.log_file:
        log_path = Path(args.log_file)
        if not log_path.exists():
            print(f"Error: log file not found: {args.log_file}", file=sys.stderr)
            sys.exit(1)
        stream: TextIO = open(log_path)
    else:
        stream = sys.stdin

    try:
        metrics = collect_metrics_from_stream(stream)
    finally:
        if stream is not sys.stdin:
            stream.close()

    final = get_final_avg_episode_reward(metrics)
    if final is None:
        print(
            "Error: no avg_episode_reward found in logs",
            file=sys.stderr,
        )
        sys.exit(1)

    episode, reward = final

    print(format_katib_metric("avg_episode_reward", reward))

    if args.trial_name:
        hyperparams: dict[str, float | int | str] = {}
        if args.hyperparameters:
            try:
                hyperparams = json.loads(args.hyperparameters)
            except json.JSONDecodeError:
                print(
                    "Warning: could not parse --hyperparameters JSON",
                    file=sys.stderr,
                )

        trial_result = DQNTrialResult(
            trial_name=args.trial_name,
            hyperparameters=hyperparams,
            avg_episode_reward=reward,
            episode=episode,
        )

        record_trial_in_registry(trial_result, registry_url=args.registry_url)


if __name__ == "__main__":
    main()
