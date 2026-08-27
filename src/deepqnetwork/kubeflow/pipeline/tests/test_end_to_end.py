"""Tests for the DQN end-to-end pipeline wiring.

Tests the degradation gate logic, Katib parameter injection,
and client-side config builder.

Requirements: DQN-R11, DQN-R12, DQN-R19, DQN-R20
"""

import json

import pytest

from deepqnetwork.kubeflow.pipeline.end_to_end import (
    MODEL_REGISTRY_URL,
    PNL_ABSOLUTE_THRESHOLD,
    SHARPE_ABSOLUTE_THRESHOLD,
    SHARPE_DEGRADATION_THRESHOLD,
    build_dqn_pipeline_e2e_config,
    evaluate_degradation_gate,
)


# ---------------------------------------------------------------------------
# Tests for evaluate_degradation_gate
# ---------------------------------------------------------------------------


class TestDegradationGate:
    """Tests for the degradation gate evaluation logic."""

    def test_gate_passes_when_all_checks_pass(self):
        """Gate passes when candidate beats all thresholds."""
        passed, reason = evaluate_degradation_gate(
            candidate_sharpe=1.5,
            candidate_pnl=100.0,
            production_sharpe=1.4,
            production_pnl=90.0,
        )
        assert passed is True
        assert "passed" in reason.lower()

    def test_gate_blocks_sharpe_below_absolute_threshold(self):
        """Gate blocks when Sharpe < 1.0 (absolute threshold)."""
        passed, reason = evaluate_degradation_gate(
            candidate_sharpe=0.8,
            candidate_pnl=50.0,
            production_sharpe=1.2,
            production_pnl=40.0,
        )
        assert passed is False
        assert "absolute threshold" in reason.lower()
        assert "sharpe" in reason.lower()

    def test_gate_blocks_pnl_at_zero(self):
        """Gate blocks when P&L <= 0 (absolute threshold)."""
        passed, reason = evaluate_degradation_gate(
            candidate_sharpe=1.5,
            candidate_pnl=0.0,
            production_sharpe=1.4,
            production_pnl=50.0,
        )
        assert passed is False
        assert "absolute threshold" in reason.lower()
        assert "p&l" in reason.lower()

    def test_gate_blocks_pnl_negative(self):
        """Gate blocks when P&L is negative (absolute threshold)."""
        passed, reason = evaluate_degradation_gate(
            candidate_sharpe=1.5,
            candidate_pnl=-10.0,
            production_sharpe=1.4,
            production_pnl=50.0,
        )
        assert passed is False
        assert "absolute threshold" in reason.lower()

    def test_gate_blocks_sharpe_degradation_vs_production(self):
        """Gate blocks when Sharpe degrades > 0.1 vs production."""
        passed, reason = evaluate_degradation_gate(
            candidate_sharpe=1.2,
            candidate_pnl=50.0,
            production_sharpe=1.5,  # degradation = 0.3 > 0.1
            production_pnl=40.0,
        )
        assert passed is False
        assert "degraded" in reason.lower()

    def test_gate_allows_small_sharpe_degradation(self):
        """Gate allows Sharpe degradation <= 0.1."""
        passed, reason = evaluate_degradation_gate(
            candidate_sharpe=1.35,
            candidate_pnl=50.0,
            production_sharpe=1.4,  # degradation = 0.05 <= 0.1
            production_pnl=40.0,
        )
        assert passed is True

    def test_gate_blocks_pnl_flip_negative(self):
        """Gate blocks when P&L flips negative while production is positive."""
        passed, reason = evaluate_degradation_gate(
            candidate_sharpe=1.5,
            candidate_pnl=-5.0,
            production_sharpe=1.4,
            production_pnl=50.0,
        )
        # Note: this is caught by the absolute P&L check first
        assert passed is False

    def test_gate_allows_both_negative_pnl(self):
        """Gate allows when both candidate and production have negative P&L.

        The P&L flip check only triggers when production P&L > 0.
        But the absolute P&L check still blocks.
        """
        passed, reason = evaluate_degradation_gate(
            candidate_sharpe=1.5,
            candidate_pnl=-5.0,
            production_sharpe=1.4,
            production_pnl=-10.0,
        )
        # Blocked by absolute P&L threshold
        assert passed is False

    def test_bootstrap_no_production_model(self):
        """Gate passes when no production model exists (bootstrap)."""
        passed, reason = evaluate_degradation_gate(
            candidate_sharpe=1.5,
            candidate_pnl=50.0,
            production_sharpe=None,
            production_pnl=None,
        )
        assert passed is True

    def test_bootstrap_still_checks_absolute_thresholds(self):
        """Even without production model, absolute thresholds apply."""
        passed, reason = evaluate_degradation_gate(
            candidate_sharpe=0.5,
            candidate_pnl=50.0,
            production_sharpe=None,
            production_pnl=None,
        )
        assert passed is False
        assert "absolute threshold" in reason.lower()

    def test_sharpe_exactly_at_threshold(self):
        """Sharpe exactly at 1.0 passes (>= check)."""
        passed, reason = evaluate_degradation_gate(
            candidate_sharpe=1.0,
            candidate_pnl=10.0,
            production_sharpe=None,
            production_pnl=None,
        )
        # Sharpe < 1.0 blocks, so exactly 1.0 should pass
        assert passed is True

    def test_sharpe_degradation_exactly_at_threshold(self):
        """Sharpe degradation exactly at 0.1 passes (> check, not >=)."""
        passed, reason = evaluate_degradation_gate(
            candidate_sharpe=1.3,
            candidate_pnl=50.0,
            production_sharpe=1.4,  # degradation = 0.1, not > 0.1
            production_pnl=40.0,
        )
        assert passed is True


# ---------------------------------------------------------------------------
# Tests for build_dqn_pipeline_e2e_config
# ---------------------------------------------------------------------------


class TestBuildPipelineConfig:
    """Tests for the client-side pipeline config builder."""

    def test_basic_config_build(self):
        """Builds valid config with default parameters."""
        params = build_dqn_pipeline_e2e_config(
            symbol="USDJPY",
            date_start="2024-01-01",
            date_end="2024-02-01",
        )
        assert params["symbol"] == "USDJPY"
        assert params["date_start"] == "2024-01-01"
        assert params["date_end"] == "2024-02-01"
        assert params["training_mode"] == "scratch"
        assert params["model_registry_url"] == MODEL_REGISTRY_URL

    def test_finetune_mode_overrides_lr_and_episodes(self):
        """Finetune mode uses reduced LR and fewer episodes."""
        params = build_dqn_pipeline_e2e_config(
            symbol="USDJPY",
            date_start="2024-01-01",
            date_end="2024-02-01",
            learning_rate=1e-4,
            num_episodes_per_range=3000,
            training_mode="finetune",
            checkpoint="s3://bucket/checkpoint.pt",
        )
        # Finetune uses finetune_learning_rate (1e-5) and finetune_num_episodes_per_range (500)
        assert params["learning_rate"] == 1e-5
        assert params["num_episodes_per_range"] == 500

    def test_finetune_without_checkpoint_raises(self):
        """Finetune mode without checkpoint raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint"):
            build_dqn_pipeline_e2e_config(
                symbol="USDJPY",
                date_start="2024-01-01",
                date_end="2024-02-01",
                training_mode="finetune",
                checkpoint="",
            )

    def test_invalid_symbol_raises(self):
        """Invalid symbol raises ValueError."""
        with pytest.raises(ValueError, match="Invalid symbol"):
            build_dqn_pipeline_e2e_config(
                symbol="EURUSD",
                date_start="2024-01-01",
                date_end="2024-02-01",
            )

    def test_invalid_learning_rate_raises(self):
        """Out-of-range learning rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate"):
            build_dqn_pipeline_e2e_config(
                symbol="USDJPY",
                date_start="2024-01-01",
                date_end="2024-02-01",
                learning_rate=0.1,  # Too high
            )

# ---------------------------------------------------------------------------
# Tests for constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_sharpe_absolute_threshold(self):
        """Sharpe absolute threshold is 1.0 (buy & hold baseline)."""
        assert SHARPE_ABSOLUTE_THRESHOLD == 1.0

    def test_pnl_absolute_threshold(self):
        """P&L absolute threshold is 0.0 (must be profitable)."""
        assert PNL_ABSOLUTE_THRESHOLD == 0.0

    def test_sharpe_degradation_threshold(self):
        """Sharpe degradation threshold is 0.1."""
        assert SHARPE_DEGRADATION_THRESHOLD == 0.1

    def test_model_registry_url(self):
        """Model Registry URL is the in-cluster service address."""
        assert MODEL_REGISTRY_URL == (
            "http://model-registry-service.kubeflow.svc.cluster.local:8080"
        )
