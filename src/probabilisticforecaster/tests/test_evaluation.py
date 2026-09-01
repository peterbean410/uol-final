"""Unit tests for the evaluation module.

Tests hourly metric grouping and metric computations on trivial known-answer cases.
Validates: Requirements 8.5
"""

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.evaluation import (
    EvaluationMetrics,
    _compute_covered_ratio_95,
    _compute_directional_accuracy,
    _compute_metrics,
    _compute_nll,
    _compute_rmse,
    evaluate_by_hour,
)
from probabilisticforecaster.model import ProbabilisticTransformer


class TestEvaluateByHour:
    """Tests for evaluate_by_hour producing 24 rows, Requirement 8.5."""

    def test_evaluate_by_hour_produces_24_rows(self):
        """evaluate_by_hour should produce a DataFrame with exactly 24 rows (hours 0-23)."""
        config = ForecasterConfig()
        model = ProbabilisticTransformer(config)
        model.eval()

        num_samples = 24 * 3

        features = np.random.randn(num_samples, 36, 16).astype(np.float32)
        labels = np.random.randn(num_samples, 1).astype(np.float32)

        class MockDataset(torch.utils.data.Dataset):
            def __len__(self):
                return num_samples

            def __getitem__(self, idx):
                return (
                    torch.from_numpy(features[idx]),
                    torch.from_numpy(labels[idx]),
                )

        mock_dataset = MockDataset()

        base_date = pd.Timestamp("2023-06-01", tz="UTC")
        timestamps = pd.Series(
            [base_date + pd.Timedelta(hours=i // 3, minutes=(i % 3) * 5) for i in range(num_samples)]
        )

        result = evaluate_by_hour(model, mock_dataset, timestamps, config)

        assert len(result) == 24
        assert list(result.index) == list(range(24))

    def test_evaluate_by_hour_has_expected_columns(self):
        """evaluate_by_hour should produce columns: nll, directional_accuracy, covered_ratio_95, rmse, count."""
        config = ForecasterConfig()
        model = ProbabilisticTransformer(config)
        model.eval()

        num_samples = 24
        features = np.random.randn(num_samples, 36, 16).astype(np.float32)
        labels = np.random.randn(num_samples, 1).astype(np.float32)

        class MockDataset(torch.utils.data.Dataset):
            def __len__(self):
                return num_samples

            def __getitem__(self, idx):
                return (
                    torch.from_numpy(features[idx]),
                    torch.from_numpy(labels[idx]),
                )

        mock_dataset = MockDataset()

        base_date = pd.Timestamp("2023-06-01", tz="UTC")
        timestamps = pd.Series(
            [base_date + pd.Timedelta(hours=i) for i in range(num_samples)]
        )

        result = evaluate_by_hour(model, mock_dataset, timestamps, config)

        expected_columns = {"nll", "directional_accuracy", "covered_ratio_95", "rmse", "count"}
        assert set(result.columns) == expected_columns


class TestDirectionalAccuracy:
    """Tests for directional accuracy on trivial known-answer cases."""

    def test_da_all_same_sign_returns_1(self):
        """If all mu and actual have the same sign, DA should be 1.0."""
        mu = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
        actual = np.array([0.2, 0.1, 0.4, 0.6, 0.9])
        assert _compute_directional_accuracy(mu, actual) == 1.0

    def test_da_all_negative_same_sign_returns_1(self):
        """If all mu and actual are negative, DA should be 1.0."""
        mu = np.array([-0.1, -0.5, -0.3])
        actual = np.array([-0.2, -0.1, -0.4])
        assert _compute_directional_accuracy(mu, actual) == 1.0

    def test_da_all_opposite_sign_returns_0(self):
        """If all mu and actual have opposite signs, DA should be 0.0."""
        mu = np.array([0.1, 0.5, 0.3, 0.8])
        actual = np.array([-0.2, -0.1, -0.4, -0.6])
        assert _compute_directional_accuracy(mu, actual) == 0.0


class TestCoveredRatio95:
    """Tests for 95% covered ratio on trivial known-answer cases."""

    def test_cr95_actuals_equal_mu_returns_1(self):
        """If all actuals are exactly equal to mu, CR95 should be 1.0."""
        mu = np.array([0.1, -0.2, 0.3, 0.0, -0.5])
        sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        actual = mu.copy()
        assert _compute_covered_ratio_95(mu, sigma, actual) == 1.0

    def test_cr95_actuals_far_from_mu_returns_0(self):
        """If all actuals are far outside 2*sigma, CR95 should be 0.0."""
        mu = np.array([0.0, 0.0, 0.0])
        sigma = np.array([0.1, 0.1, 0.1])
        actual = np.array([10.0, -10.0, 10.0])
        assert _compute_covered_ratio_95(mu, sigma, actual) == 0.0


class TestRMSE:
    """Tests for RMSE on trivial known-answer cases."""

    def test_rmse_mu_equals_actual_returns_0(self):
        """If mu == actual for all samples, RMSE should be 0.0."""
        mu = np.array([0.1, -0.2, 0.3, 0.5, -0.1])
        actual = mu.copy()
        assert _compute_rmse(mu, actual) == 0.0

    def test_rmse_known_value(self):
        """RMSE for a known case: mu=[1,2,3], actual=[1,2,4] -> sqrt(1/3)."""
        mu = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.0, 2.0, 4.0])
        expected = math.sqrt(1.0 / 3.0)
        assert abs(_compute_rmse(mu, actual) - expected) < 1e-10


class TestNLL:
    """Tests for NLL on a manually computed known case."""

    def test_nll_known_case(self):
        """Verify NLL against a manually computed known case.

        For mu=0, sigma=1, actual=0:
        NLL = 0.5 * (log(1) + (0/1)^2 + log(2*pi))
            = 0.5 * (0 + 0 + log(2*pi))
            = 0.5 * log(2*pi)
            ≈ 0.9189
        """
        mu = np.array([0.0])
        sigma = np.array([1.0])
        actual = np.array([0.0])

        expected_nll = 0.5 * math.log(2 * math.pi)
        computed_nll = _compute_nll(mu, sigma, actual)
        assert abs(computed_nll - expected_nll) < 1e-10

    def test_nll_known_case_nonzero(self):
        """Verify NLL for mu=1, sigma=2, actual=3.

        NLL = 0.5 * (log(4) + ((3-1)/2)^2 + log(2*pi))
            = 0.5 * (log(4) + 1 + log(2*pi))
            = 0.5 * (1.3863 + 1 + 1.8379)
            ≈ 2.1121
        """
        mu = np.array([1.0])
        sigma = np.array([2.0])
        actual = np.array([3.0])

        expected_nll = 0.5 * (math.log(4.0) + 1.0 + math.log(2 * math.pi))
        computed_nll = _compute_nll(mu, sigma, actual)
        assert abs(computed_nll - expected_nll) < 1e-10
