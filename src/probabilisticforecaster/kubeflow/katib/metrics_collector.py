"""Katib custom metrics collector for the Probabilistic Forex Forecaster.

Extracts validation NLL from structured JSON training component logs and
provides integration with the Model Registry for recording trial results.

This module can be used:
1. As a Katib sidecar metrics collector (reads logs from stdin or file)
2. As a standalone utility for extracting metrics from training logs

The training component logs structured JSON (via StructuredJsonFormatter from
monitoring/metrics.py). The validation NLL is logged with a specific message
pattern that this collector parses.

Requirements: 3.4, 3.5, 3.6
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TextIO


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Result of a single Katib trial.

    Attributes:
        trial_name: Unique identifier for the trial.
        hyperparameters: Dict of hyperparameter name → value used in this trial.
        validation_nll: The final validation NLL metric (objective to minimise).
        epoch: The epoch at which the final validation_nll was recorded.
    """

    trial_name: str
    hyperparameters: dict[str, float | int | str]
    validation_nll: float
    epoch: int = 0


@dataclass
class BestConfiguration:
    """The best hyperparameter configuration from a completed Katib experiment.

    Attributes:
        hyperparameters: Dict of hyperparameter name → optimal value.
        validation_nll: The best (lowest) validation NLL achieved.
        trial_name: Name of the trial that achieved the best result.
    """

    hyperparameters: dict[str, float | int | str]
    validation_nll: float
    trial_name: str


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def parse_log_line(line: str) -> dict | None:
    """Parse a single structured JSON log line.

    Handles malformed lines gracefully by returning None for non-JSON lines
    or lines that cannot be parsed.

    Args:
        line: A single line from the training component's stdout.

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


def extract_validation_nll(log_entry: dict) -> float | None:
    """Extract validation_nll value from a parsed log entry.

    The training component logs validation NLL in one of these patterns:
    1. As a field in the log message: "validation_nll=<value>"
    2. As an extra field in the JSON: {"validation_nll": <value>}
    3. In the message containing "validation_nll" with a numeric value

    Args:
        log_entry: A parsed JSON log dict.

    Returns:
        The validation_nll float value if found, None otherwise.
    """
    # Pattern 1: Direct field in the log entry (extra fields from logger)
    if "validation_nll" in log_entry:
        try:
            return float(log_entry["validation_nll"])
        except (TypeError, ValueError):
            pass

    # Pattern 2: Nested in an "extras" or "metrics" dict
    for container_key in ("extras", "metrics", "extra"):
        container = log_entry.get(container_key)
        if isinstance(container, dict) and "validation_nll" in container:
            try:
                return float(container["validation_nll"])
            except (TypeError, ValueError):
                pass

    # Pattern 3: Parse from the message string
    message = log_entry.get("message", "")
    if "validation_nll" in message:
        # Try pattern: "validation_nll=<float>"
        for part in message.split():
            if part.startswith("validation_nll="):
                try:
                    return float(part.split("=", 1)[1])
                except (ValueError, IndexError):
                    pass
        # Try pattern: "validation_nll: <float>"
        if "validation_nll:" in message:
            try:
                idx = message.index("validation_nll:")
                remainder = message[idx + len("validation_nll:"):].strip()
                # Take the first token as the value
                value_str = remainder.split()[0].rstrip(",;")
                return float(value_str)
            except (ValueError, IndexError):
                pass

    return None


def extract_epoch(log_entry: dict) -> int | None:
    """Extract epoch number from a parsed log entry.

    Args:
        log_entry: A parsed JSON log dict.

    Returns:
        The epoch number if found, None otherwise.
    """
    # Direct field
    if "epoch" in log_entry:
        try:
            return int(log_entry["epoch"])
        except (TypeError, ValueError):
            pass

    # Nested in extras/metrics
    for container_key in ("extras", "metrics", "extra"):
        container = log_entry.get(container_key)
        if isinstance(container, dict) and "epoch" in container:
            try:
                return int(container["epoch"])
            except (TypeError, ValueError):
                pass

    # Parse from message
    message = log_entry.get("message", "")
    if "epoch" in message.lower():
        for part in message.split():
            if part.startswith("epoch="):
                try:
                    return int(part.split("=", 1)[1])
                except (ValueError, IndexError):
                    pass
        if "epoch:" in message.lower():
            try:
                idx = message.lower().index("epoch:")
                remainder = message[idx + len("epoch:"):].strip()
                value_str = remainder.split()[0].rstrip(",;")
                return int(value_str)
            except (ValueError, IndexError):
                pass

    return None


def collect_metrics_from_stream(
    stream: TextIO,
) -> list[tuple[int, float]]:
    """Read log lines from a stream and extract all (epoch, validation_nll) pairs.

    Handles malformed lines gracefully by skipping them.

    Args:
        stream: A text stream (stdin, file handle) producing log lines.

    Returns:
        List of (epoch, validation_nll) tuples in order of appearance.
    """
    results: list[tuple[int, float]] = []
    current_epoch = 0

    for line in stream:
        entry = parse_log_line(line)
        if entry is None:
            continue

        # Update current epoch if present
        epoch = extract_epoch(entry)
        if epoch is not None:
            current_epoch = epoch

        # Extract validation NLL
        nll = extract_validation_nll(entry)
        if nll is not None:
            results.append((current_epoch, nll))

    return results


def get_final_validation_nll(
    metrics: list[tuple[int, float]],
) -> tuple[int, float] | None:
    """Get the final validation NLL from collected metrics.

    Returns the last recorded validation_nll value, which corresponds to
    the final epoch of training.

    Args:
        metrics: List of (epoch, validation_nll) tuples.

    Returns:
        Tuple of (epoch, validation_nll) for the last entry, or None if empty.
    """
    if not metrics:
        return None
    return metrics[-1]


# ---------------------------------------------------------------------------
# Katib-compatible metric output
# ---------------------------------------------------------------------------


def format_katib_metric(metric_name: str, value: float) -> str:
    """Format a metric value in Katib's expected stdout format.

    Katib's StdOut metrics collector expects lines in the format:
        metric_name=value

    Args:
        metric_name: Name of the metric (e.g. "validation_nll").
        value: The metric value.

    Returns:
        Formatted metric string for Katib collection.
    """
    return f"{metric_name}={value}"


# ---------------------------------------------------------------------------
# Model Registry integration
# ---------------------------------------------------------------------------


def record_trial_in_registry(
    trial_result: TrialResult,
    registry_url: str = "http://model-registry-service.kubeflow.svc.cluster.local:8080",
) -> str | None:
    """Record trial hyperparameters and objective in the Model Registry.

    Uses a forward-compatible interface with the registry client that will be
    implemented at `probabilisticforecaster/kubeflow/registry/registry_client.py`.

    Args:
        trial_result: The trial result containing hyperparameters and objective.
        registry_url: URL of the Kubeflow Model Registry service.

    Returns:
        The registered version ID if successful, None on failure.
    """
    try:
        from probabilisticforecaster.kubeflow.registry.registry_client import (
            ForecasterRegistryClient,
            ModelMetadata,
        )

        client = ForecasterRegistryClient(registry_url=registry_url)

        metadata = ModelMetadata(
            symbol=str(trial_result.hyperparameters.get("symbol", "USDJPY")),
            forecast_horizon=int(
                trial_result.hyperparameters.get("forecast_horizon", 1)
            ),
            training_timestamp="",  # Will be set by registry client
            training_nll=0.0,  # Not available at trial level
            validation_nll=trial_result.validation_nll,
            directional_accuracy=0.0,  # Not available at trial level
            hyperparameters=trial_result.hyperparameters,
            pipeline_run_id=trial_result.trial_name,
            data_snapshot_path="",
            lifecycle_stage="staging",
        )

        version_id = client.register_model(
            model_path=f"katib-trial/{trial_result.trial_name}",
            metadata=metadata,
        )
        return version_id

    except ImportError:
        # Registry client not yet implemented; log and continue
        print(
            json.dumps(
                {
                    "level": "WARNING",
                    "message": (
                        "Model Registry client not available. "
                        "Trial result not recorded in registry."
                    ),
                    "trial_name": trial_result.trial_name,
                    "validation_nll": trial_result.validation_nll,
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
                    "message": f"Failed to record trial in registry: {exc}",
                    "trial_name": trial_result.trial_name,
                }
            ),
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# Best configuration output
# ---------------------------------------------------------------------------


def determine_best_configuration(
    trial_results: list[TrialResult],
) -> BestConfiguration | None:
    """Determine the best hyperparameter configuration from completed trials.

    Selects the trial with the lowest validation NLL (the objective to minimise).

    Args:
        trial_results: List of completed trial results.

    Returns:
        BestConfiguration for the trial with lowest validation_nll,
        or None if no trials provided.
    """
    if not trial_results:
        return None

    best_trial = min(trial_results, key=lambda t: t.validation_nll)

    return BestConfiguration(
        hyperparameters=best_trial.hyperparameters,
        validation_nll=best_trial.validation_nll,
        trial_name=best_trial.trial_name,
    )


def output_best_configuration(
    best_config: BestConfiguration,
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
            "name": "validation_nll",
            "value": best_config.validation_nll,
        },
        "hyperparameters": best_config.hyperparameters,
    }

    json_str = json.dumps(artifact, indent=2)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_str)

    return json_str


# ---------------------------------------------------------------------------
# Main entry point (standalone / Katib sidecar)
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the metrics collector as a standalone script.

    Reads structured JSON logs from stdin or a log file, extracts the final
    validation_nll, and outputs it in Katib-compatible format.

    Usage:
        # From stdin (Katib sidecar mode):
        cat training.log | python metrics_collector.py

        # From a log file:
        python metrics_collector.py --log-file /var/log/training.log

        # With trial recording:
        python metrics_collector.py --log-file /var/log/training.log \\
            --trial-name trial-001 \\
            --hyperparameters '{"learning_rate": 0.001, "num_layers": 3}'

        # Output best config from multiple trial results:
        python metrics_collector.py --best-config \\
            --trials-file /tmp/trials.json \\
            --output-path /tmp/best_config.json
    """
    parser = argparse.ArgumentParser(
        description="Katib metrics collector for Probabilistic Forecaster"
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

    # Mode: Output best configuration from trials
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
            TrialResult(
                trial_name=t["trial_name"],
                hyperparameters=t["hyperparameters"],
                validation_nll=t["validation_nll"],
                epoch=t.get("epoch", 0),
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

    # Mode: Collect metrics from training logs
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

    # Get the final validation NLL (last epoch)
    final = get_final_validation_nll(metrics)
    if final is None:
        print(
            "Error: no validation_nll found in logs",
            file=sys.stderr,
        )
        sys.exit(1)

    epoch, nll = final

    # Output in Katib-compatible format
    print(format_katib_metric("validation_nll", nll))

    # Record in Model Registry if trial info provided
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

        trial_result = TrialResult(
            trial_name=args.trial_name,
            hyperparameters=hyperparams,
            validation_nll=nll,
            epoch=epoch,
        )

        record_trial_in_registry(trial_result, registry_url=args.registry_url)


if __name__ == "__main__":
    main()
