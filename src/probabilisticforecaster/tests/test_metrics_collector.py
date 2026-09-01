"""Unit tests for the Katib metrics collector.

Tests cover:
- JSON log line parsing (valid and malformed)
- Validation NLL extraction from various log formats
- Epoch extraction
- Stream-based metric collection
- Final validation NLL selection (last epoch)
- Katib-compatible metric output formatting
- Best configuration determination
- Best configuration artifact output
"""

import json
import io
from pathlib import Path

import pytest

from probabilisticforecaster.kubeflow.katib.metrics_collector import (
    BestConfiguration,
    TrialResult,
    collect_metrics_from_stream,
    determine_best_configuration,
    extract_epoch,
    extract_validation_nll,
    format_katib_metric,
    get_final_validation_nll,
    output_best_configuration,
    parse_log_line,
)


class TestParseLogLine:
    """Tests for parse_log_line."""

    def test_valid_json_line(self):
        line = '{"timestamp": "2024-01-01T00:00:00", "level": "INFO", "message": "hello"}'
        result = parse_log_line(line)
        assert result is not None
        assert result["level"] == "INFO"
        assert result["message"] == "hello"

    def test_empty_line(self):
        assert parse_log_line("") is None
        assert parse_log_line("   ") is None
        assert parse_log_line("\n") is None

    def test_non_json_line(self):
        assert parse_log_line("This is not JSON") is None
        assert parse_log_line("INFO: some log message") is None

    def test_partial_json(self):
        assert parse_log_line('{"incomplete": ') is None

    def test_strips_whitespace(self):
        line = '  {"key": "value"}  \n'
        result = parse_log_line(line)
        assert result == {"key": "value"}


class TestExtractValidationNll:
    """Tests for extract_validation_nll."""

    def test_direct_field(self):
        entry = {"validation_nll": 1.234, "message": "training complete"}
        assert extract_validation_nll(entry) == 1.234

    def test_direct_field_string_value(self):
        entry = {"validation_nll": "2.567", "message": "done"}
        assert extract_validation_nll(entry) == 2.567

    def test_extras_dict(self):
        entry = {"message": "epoch done", "extras": {"validation_nll": 0.89}}
        assert extract_validation_nll(entry) == 0.89

    def test_metrics_dict(self):
        entry = {"message": "epoch done", "metrics": {"validation_nll": 1.5}}
        assert extract_validation_nll(entry) == 1.5

    def test_message_equals_pattern(self):
        entry = {"message": "epoch 5 validation_nll=0.456 da=0.55"}
        assert extract_validation_nll(entry) == 0.456

    def test_message_colon_pattern(self):
        entry = {"message": "validation_nll: 0.789"}
        assert extract_validation_nll(entry) == 0.789

    def test_no_validation_nll(self):
        entry = {"message": "training started", "level": "INFO"}
        assert extract_validation_nll(entry) is None

    def test_invalid_value(self):
        entry = {"validation_nll": "not_a_number"}
        assert extract_validation_nll(entry) is None


class TestExtractEpoch:
    """Tests for extract_epoch."""

    def test_direct_field(self):
        entry = {"epoch": 3, "message": "training"}
        assert extract_epoch(entry) == 3

    def test_extras_field(self):
        entry = {"message": "done", "extras": {"epoch": 5}}
        assert extract_epoch(entry) == 5

    def test_message_equals_pattern(self):
        entry = {"message": "epoch=2 loss=0.5"}
        assert extract_epoch(entry) == 2

    def test_message_colon_pattern(self):
        entry = {"message": "Epoch: 4 completed"}
        assert extract_epoch(entry) == 4

    def test_no_epoch(self):
        entry = {"message": "starting training"}
        assert extract_epoch(entry) is None


class TestCollectMetricsFromStream:
    """Tests for collect_metrics_from_stream."""

    def test_multiple_epochs(self):
        lines = [
            '{"epoch": 1, "validation_nll": 2.5, "message": "epoch done"}\n',
            '{"epoch": 2, "validation_nll": 2.1, "message": "epoch done"}\n',
            '{"epoch": 3, "validation_nll": 1.8, "message": "epoch done"}\n',
        ]
        stream = io.StringIO("".join(lines))
        metrics = collect_metrics_from_stream(stream)
        assert metrics == [(1, 2.5), (2, 2.1), (3, 1.8)]

    def test_skips_malformed_lines(self):
        lines = [
            "This is not JSON\n",
            '{"epoch": 1, "validation_nll": 2.5, "message": "done"}\n',
            "Another bad line\n",
            '{"epoch": 2, "validation_nll": 1.9, "message": "done"}\n',
        ]
        stream = io.StringIO("".join(lines))
        metrics = collect_metrics_from_stream(stream)
        assert metrics == [(1, 2.5), (2, 1.9)]

    def test_empty_stream(self):
        stream = io.StringIO("")
        metrics = collect_metrics_from_stream(stream)
        assert metrics == []

    def test_no_nll_in_logs(self):
        lines = [
            '{"epoch": 1, "message": "training started"}\n',
            '{"epoch": 2, "message": "training in progress"}\n',
        ]
        stream = io.StringIO("".join(lines))
        metrics = collect_metrics_from_stream(stream)
        assert metrics == []

    def test_epoch_tracked_across_lines(self):
        lines = [
            '{"epoch": 1, "message": "starting epoch"}\n',
            '{"message": "validation_nll=2.3"}\n',
            '{"epoch": 2, "message": "starting epoch"}\n',
            '{"message": "validation_nll=1.9"}\n',
        ]
        stream = io.StringIO("".join(lines))
        metrics = collect_metrics_from_stream(stream)
        assert metrics == [(1, 2.3), (2, 1.9)]


class TestGetFinalValidationNll:
    """Tests for get_final_validation_nll."""

    def test_returns_last_entry(self):
        metrics = [(1, 2.5), (2, 2.1), (3, 1.8)]
        assert get_final_validation_nll(metrics) == (3, 1.8)

    def test_single_entry(self):
        metrics = [(0, 3.0)]
        assert get_final_validation_nll(metrics) == (0, 3.0)

    def test_empty_list(self):
        assert get_final_validation_nll([]) is None


class TestFormatKatibMetric:
    """Tests for format_katib_metric."""

    def test_basic_format(self):
        assert format_katib_metric("validation_nll", 1.234) == "validation_nll=1.234"

    def test_integer_value(self):
        assert format_katib_metric("epoch", 5) == "epoch=5"

    def test_scientific_notation(self):
        result = format_katib_metric("validation_nll", 1e-5)
        assert "validation_nll=" in result
        assert "1e-05" in result


class TestDetermineBestConfiguration:
    """Tests for determine_best_configuration."""

    def test_selects_lowest_nll(self):
        trials = [
            TrialResult("trial-1", {"lr": 0.001}, 2.5, epoch=5),
            TrialResult("trial-2", {"lr": 0.01}, 1.8, epoch=5),
            TrialResult("trial-3", {"lr": 0.0001}, 2.1, epoch=5),
        ]
        best = determine_best_configuration(trials)
        assert best is not None
        assert best.trial_name == "trial-2"
        assert best.validation_nll == 1.8
        assert best.hyperparameters == {"lr": 0.01}

    def test_single_trial(self):
        trials = [TrialResult("trial-1", {"lr": 0.001}, 2.5)]
        best = determine_best_configuration(trials)
        assert best is not None
        assert best.trial_name == "trial-1"

    def test_empty_list(self):
        assert determine_best_configuration([]) is None


class TestOutputBestConfiguration:
    """Tests for output_best_configuration."""

    def test_returns_valid_json(self):
        best = BestConfiguration(
            hyperparameters={"learning_rate": 0.001, "num_layers": 3},
            validation_nll=1.5,
            trial_name="trial-best",
        )
        result = output_best_configuration(best)
        parsed = json.loads(result)
        assert parsed["best_trial"] == "trial-best"
        assert parsed["objective_metric"]["name"] == "validation_nll"
        assert parsed["objective_metric"]["value"] == 1.5
        assert parsed["hyperparameters"]["learning_rate"] == 0.001
        assert parsed["hyperparameters"]["num_layers"] == 3

    def test_writes_to_file(self, tmp_path):
        best = BestConfiguration(
            hyperparameters={"lr": 0.01},
            validation_nll=1.2,
            trial_name="trial-42",
        )
        output_file = tmp_path / "best_config.json"
        result = output_best_configuration(best, str(output_file))

        assert output_file.exists()
        file_content = json.loads(output_file.read_text())
        assert file_content["best_trial"] == "trial-42"
        assert file_content == json.loads(result)

    def test_creates_parent_directories(self, tmp_path):
        best = BestConfiguration(
            hyperparameters={"lr": 0.01},
            validation_nll=1.0,
            trial_name="trial-1",
        )
        output_file = tmp_path / "nested" / "dir" / "config.json"
        output_best_configuration(best, str(output_file))
        assert output_file.exists()
