"""Unit tests for the DQN Katib metrics collector.

Tests cover log parsing, metric extraction, best configuration determination,
and structured artifact output.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pytest

from deepqnetwork.kubeflow.katib.dqn_metrics_collector import (
    DQNBestConfiguration,
    DQNTrialResult,
    collect_metrics_from_stream,
    determine_best_configuration,
    extract_avg_episode_reward,
    extract_episode,
    format_katib_metric,
    get_final_avg_episode_reward,
    output_best_configuration,
    parse_log_line,
)


# ---------------------------------------------------------------------------
# parse_log_line
# ---------------------------------------------------------------------------


class TestParseLogLine:
    def test_valid_json(self):
        line = '{"message": "hello", "level": "INFO"}'
        result = parse_log_line(line)
        assert result == {"message": "hello", "level": "INFO"}

    def test_empty_line(self):
        assert parse_log_line("") is None
        assert parse_log_line("   ") is None

    def test_non_json(self):
        assert parse_log_line("not json at all") is None

    def test_whitespace_stripped(self):
        line = '  {"key": "value"}  \n'
        result = parse_log_line(line)
        assert result == {"key": "value"}


# ---------------------------------------------------------------------------
# extract_avg_episode_reward
# ---------------------------------------------------------------------------


class TestExtractAvgEpisodeReward:
    def test_direct_field(self):
        entry = {"avg_episode_reward": 42.5, "message": "training"}
        assert extract_avg_episode_reward(entry) == 42.5

    def test_direct_field_string_value(self):
        entry = {"avg_episode_reward": "3.14", "message": "training"}
        assert extract_avg_episode_reward(entry) == 3.14

    def test_nested_in_metrics(self):
        entry = {"metrics": {"avg_episode_reward": -1.5}, "message": "log"}
        assert extract_avg_episode_reward(entry) == -1.5

    def test_nested_in_extras(self):
        entry = {"extras": {"avg_episode_reward": 100.0}, "message": "log"}
        assert extract_avg_episode_reward(entry) == 100.0

    def test_in_message_equals_pattern(self):
        entry = {"message": "step done avg_episode_reward=7.25 done"}
        assert extract_avg_episode_reward(entry) == 7.25

    def test_in_message_colon_pattern(self):
        entry = {"message": "result avg_episode_reward: 12.3 end"}
        assert extract_avg_episode_reward(entry) == 12.3

    def test_not_present(self):
        entry = {"message": "no reward here", "level": "INFO"}
        assert extract_avg_episode_reward(entry) is None

    def test_invalid_value(self):
        entry = {"avg_episode_reward": "not_a_number"}
        assert extract_avg_episode_reward(entry) is None


# ---------------------------------------------------------------------------
# extract_episode
# ---------------------------------------------------------------------------


class TestExtractEpisode:
    def test_direct_field(self):
        entry = {"episode": 10, "message": "training"}
        assert extract_episode(entry) == 10

    def test_nested_in_metrics(self):
        entry = {"metrics": {"episode": 50}, "message": "log"}
        assert extract_episode(entry) == 50

    def test_in_message_equals_pattern(self):
        entry = {"message": "episode=25 reward=3.0"}
        assert extract_episode(entry) == 25

    def test_in_message_colon_pattern(self):
        entry = {"message": "Episode: 100 completed"}
        assert extract_episode(entry) == 100

    def test_not_present(self):
        entry = {"message": "no episode info", "level": "INFO"}
        assert extract_episode(entry) is None


# ---------------------------------------------------------------------------
# collect_metrics_from_stream
# ---------------------------------------------------------------------------


class TestCollectMetricsFromStream:
    def test_basic_collection(self):
        lines = [
            '{"episode": 1, "avg_episode_reward": 10.0}\n',
            '{"episode": 2, "avg_episode_reward": 20.0}\n',
            '{"episode": 3, "avg_episode_reward": 30.0}\n',
        ]
        stream = io.StringIO("".join(lines))
        result = collect_metrics_from_stream(stream)
        assert result == [(1, 10.0), (2, 20.0), (3, 30.0)]

    def test_skips_malformed_lines(self):
        lines = [
            '{"episode": 1, "avg_episode_reward": 5.0}\n',
            "not json\n",
            '{"episode": 2, "avg_episode_reward": 15.0}\n',
        ]
        stream = io.StringIO("".join(lines))
        result = collect_metrics_from_stream(stream)
        assert result == [(1, 5.0), (2, 15.0)]

    def test_episode_tracking(self):
        """Episode number carries forward when not explicitly in reward line."""
        lines = [
            '{"episode": 5, "message": "starting episode"}\n',
            '{"avg_episode_reward": 7.5}\n',
        ]
        stream = io.StringIO("".join(lines))
        result = collect_metrics_from_stream(stream)
        assert result == [(5, 7.5)]

    def test_empty_stream(self):
        stream = io.StringIO("")
        result = collect_metrics_from_stream(stream)
        assert result == []

    def test_no_reward_lines(self):
        lines = [
            '{"episode": 1, "message": "no reward"}\n',
            '{"level": "INFO", "message": "still no reward"}\n',
        ]
        stream = io.StringIO("".join(lines))
        result = collect_metrics_from_stream(stream)
        assert result == []


# ---------------------------------------------------------------------------
# get_final_avg_episode_reward
# ---------------------------------------------------------------------------


class TestGetFinalAvgEpisodeReward:
    def test_returns_last(self):
        metrics = [(1, 10.0), (2, 20.0), (3, 30.0)]
        assert get_final_avg_episode_reward(metrics) == (3, 30.0)

    def test_empty_returns_none(self):
        assert get_final_avg_episode_reward([]) is None

    def test_single_entry(self):
        assert get_final_avg_episode_reward([(0, 5.5)]) == (0, 5.5)


# ---------------------------------------------------------------------------
# format_katib_metric
# ---------------------------------------------------------------------------


class TestFormatKatibMetric:
    def test_format(self):
        assert format_katib_metric("avg_episode_reward", 42.5) == "avg_episode_reward=42.5"

    def test_negative_value(self):
        assert format_katib_metric("avg_episode_reward", -3.14) == "avg_episode_reward=-3.14"


# ---------------------------------------------------------------------------
# determine_best_configuration
# ---------------------------------------------------------------------------


class TestDetermineBestConfiguration:
    def test_selects_highest_reward(self):
        trials = [
            DQNTrialResult("t1", {"lr": 0.001}, avg_episode_reward=10.0),
            DQNTrialResult("t2", {"lr": 0.01}, avg_episode_reward=50.0),
            DQNTrialResult("t3", {"lr": 0.0001}, avg_episode_reward=30.0),
        ]
        best = determine_best_configuration(trials)
        assert best is not None
        assert best.trial_name == "t2"
        assert best.avg_episode_reward == 50.0
        assert best.hyperparameters == {"lr": 0.01}

    def test_empty_returns_none(self):
        assert determine_best_configuration([]) is None

    def test_single_trial(self):
        trials = [DQNTrialResult("t1", {"gamma": 0.99}, avg_episode_reward=7.0)]
        best = determine_best_configuration(trials)
        assert best is not None
        assert best.trial_name == "t1"

    def test_negative_rewards(self):
        """DQN can have negative rewards; best is the least negative."""
        trials = [
            DQNTrialResult("t1", {}, avg_episode_reward=-10.0),
            DQNTrialResult("t2", {}, avg_episode_reward=-2.0),
            DQNTrialResult("t3", {}, avg_episode_reward=-5.0),
        ]
        best = determine_best_configuration(trials)
        assert best is not None
        assert best.trial_name == "t2"
        assert best.avg_episode_reward == -2.0


# ---------------------------------------------------------------------------
# output_best_configuration
# ---------------------------------------------------------------------------


class TestOutputBestConfiguration:
    def test_json_structure(self):
        config = DQNBestConfiguration(
            hyperparameters={"learning_rate": 0.001, "gamma": 0.99},
            avg_episode_reward=42.5,
            trial_name="trial-007",
        )
        json_str = output_best_configuration(config)
        data = json.loads(json_str)

        assert data["best_trial"] == "trial-007"
        assert data["objective_metric"]["name"] == "avg_episode_reward"
        assert data["objective_metric"]["value"] == 42.5
        assert data["hyperparameters"]["learning_rate"] == 0.001
        assert data["hyperparameters"]["gamma"] == 0.99

    def test_writes_to_file(self):
        config = DQNBestConfiguration(
            hyperparameters={"batch_size": 64},
            avg_episode_reward=100.0,
            trial_name="trial-best",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "subdir" / "best_config.json")
            json_str = output_best_configuration(config, output_path=output_path)

            # File should exist and contain valid JSON
            written = Path(output_path).read_text()
            assert json.loads(written) == json.loads(json_str)

    def test_no_file_when_path_none(self):
        config = DQNBestConfiguration(
            hyperparameters={},
            avg_episode_reward=0.0,
            trial_name="t",
        )
        result = output_best_configuration(config, output_path=None)
        assert isinstance(result, str)
        assert json.loads(result)  # Valid JSON


# ---------------------------------------------------------------------------
# record_trial_in_registry (import failure path)
# ---------------------------------------------------------------------------


class TestRecordTrialInRegistry:
    def test_handles_import_error_gracefully(self):
        """When the DQN registry client is not available, returns None without crashing."""
        from deepqnetwork.kubeflow.katib.dqn_metrics_collector import (
            record_trial_in_registry,
        )

        trial = DQNTrialResult(
            trial_name="test-trial",
            hyperparameters={"lr": 0.001},
            avg_episode_reward=25.0,
        )
        # This should not raise; it handles ImportError gracefully
        result = record_trial_in_registry(trial)
        assert result is None
