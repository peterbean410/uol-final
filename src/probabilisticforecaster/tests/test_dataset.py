"""Unit tests for the ForexDataset (dataset.py).

Tests cover:
- Train/test date split (train: 2012-2022, test: 2023-2026)
- Sample shape is (36, 16) for features and (1,) for labels
- Label computation correctness for different horizons

Requirements: 4.1, 4.2, 4.3
"""

import numpy as np
import pandas as pd
import pytest
import torch

from probabilisticforecaster.dataset import ForexDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_data(
    start: str = "2023-01-01",
    n_bars: int = 100,
    freq: str = "5min",
    base_price: float = 150.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic feature DataFrame and close prices for testing.

    Returns:
        Tuple of (features_df with 16 columns and DatetimeIndex, close_prices Series).
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=n_bars, freq=freq)

    # 16 feature columns
    feature_data = rng.uniform(-3.0, 3.0, size=(n_bars, 16)).astype(np.float32)
    features_df = pd.DataFrame(
        feature_data,
        index=timestamps,
        columns=[f"feat_{i}" for i in range(16)],
    )

    # Close prices (positive, random walk)
    close_values = base_price + np.cumsum(rng.normal(0, 0.01, n_bars))
    close_prices = pd.Series(close_values, index=timestamps)

    return features_df, close_prices


# ---------------------------------------------------------------------------
# Test: Train/Test Date Split (Requirement 4.3)
# ---------------------------------------------------------------------------


class TestTrainTestDateSplit:
    """Test that data can be correctly split into train (2012-2022) and test (2023-2026).

    Requirement 4.3: Data split into training set (2012-01-01 to 2022-12-31)
    and test set (2023-01-01 to 2026-04-30).
    """

    def test_train_split_contains_only_pre_2023_data(self):
        """Training dataset should only contain data from 2012-2022."""
        # Create data spanning 2021-2024 to test the boundary
        train_data, train_close = _make_synthetic_data(
            start="2022-12-30", n_bars=200, freq="5min"
        )

        # Filter for training period: up to 2022-12-31
        train_mask = train_data.index < pd.Timestamp("2023-01-01")
        train_features = train_data[train_mask]
        train_prices = train_close[train_mask]

        dataset = ForexDataset(
            features_df=train_features,
            close_prices=train_prices,
            lookback=36,
            horizon=1,
        )

        # All samples should have timestamps before 2023
        for idx in range(len(dataset)):
            start_bar = dataset.valid_indices[idx]
            end_bar = start_bar + dataset.lookback - 1 + dataset.horizon
            assert train_features.index[end_bar] < pd.Timestamp("2023-01-01")

    def test_test_split_contains_only_post_2023_data(self):
        """Test dataset should only contain data from 2023 onwards."""
        # Create data starting in 2023 (test period)
        test_data, test_close = _make_synthetic_data(
            start="2023-01-01", n_bars=100, freq="5min"
        )

        dataset = ForexDataset(
            features_df=test_data,
            close_prices=test_close,
            lookback=36,
            horizon=1,
        )

        # All samples should have timestamps from 2023 onwards
        for idx in range(len(dataset)):
            start_bar = dataset.valid_indices[idx]
            assert test_data.index[start_bar] >= pd.Timestamp("2023-01-01")

    def test_train_test_split_boundary_no_overlap(self):
        """Train and test datasets should not overlap at the boundary."""
        # Create continuous data across the boundary
        all_data, all_close = _make_synthetic_data(
            start="2022-12-31 23:00", n_bars=100, freq="5min"
        )

        # Split at 2023-01-01
        train_mask = all_data.index < pd.Timestamp("2023-01-01")
        test_mask = all_data.index >= pd.Timestamp("2023-01-01")

        train_features = all_data[train_mask]
        train_close = all_close[train_mask]
        test_features = all_data[test_mask]
        test_close = all_close[test_mask]

        # Train timestamps and test timestamps should not overlap
        train_timestamps = set(train_features.index)
        test_timestamps = set(test_features.index)
        assert train_timestamps.isdisjoint(test_timestamps)


# ---------------------------------------------------------------------------
# Test: Sample Shape (Requirement 4.1)
# ---------------------------------------------------------------------------


class TestSampleShape:
    """Test that __getitem__ returns correct tensor shapes.

    Requirement 4.1: Each training sample is a sequence of 36 consecutive
    5-minute feature vectors.
    """

    @pytest.fixture
    def dataset(self):
        """Create a dataset with enough bars for multiple samples."""
        features_df, close_prices = _make_synthetic_data(n_bars=100)
        return ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=36,
            horizon=1,
        )

    def test_features_shape_is_36_by_16(self, dataset):
        """Feature tensor should be (36, 16), lookback window × num_features."""
        features, label = dataset[0]
        assert features.shape == (36, 16)

    def test_label_shape_is_1(self, dataset):
        """Label tensor should be (1,); a single scalar forward return."""
        features, label = dataset[0]
        assert label.shape == (1,)

    def test_features_dtype_is_float32(self, dataset):
        """Feature tensor should be float32."""
        features, label = dataset[0]
        assert features.dtype == torch.float32

    def test_label_dtype_is_float32(self, dataset):
        """Label tensor should be float32."""
        features, label = dataset[0]
        assert label.dtype == torch.float32

    def test_all_samples_have_consistent_shape(self, dataset):
        """Every sample in the dataset should have the same shape."""
        for idx in range(min(len(dataset), 10)):
            features, label = dataset[idx]
            assert features.shape == (36, 16)
            assert label.shape == (1,)

    def test_custom_lookback_shape(self):
        """Dataset with custom lookback should produce matching feature shape."""
        features_df, close_prices = _make_synthetic_data(n_bars=100)
        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=20,
            horizon=1,
        )
        features, label = dataset[0]
        assert features.shape == (20, 16)
        assert label.shape == (1,)


# ---------------------------------------------------------------------------
# Test: Label Computation (Requirement 4.2)
# ---------------------------------------------------------------------------


class TestLabelComputation:
    """Test forward return label computation correctness.

    Requirement 4.2: Labels are forward return over specified forecasting period:
    label = (close_{t+h} - close_t) / close_t
    """

    def test_label_horizon_1_correctness(self):
        """Label with horizon=1 should be (close_{t+1} - close_t) / close_t."""
        # Use deterministic close prices for exact verification
        n_bars = 50
        timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="5min")
        features_df = pd.DataFrame(
            np.ones((n_bars, 16), dtype=np.float32),
            index=timestamps,
            columns=[f"feat_{i}" for i in range(16)],
        )
        # Set specific close prices
        close_values = np.linspace(100.0, 110.0, n_bars)
        close_prices = pd.Series(close_values, index=timestamps)

        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=36,
            horizon=1,
        )

        # For the first sample: t = 35 (last bar in lookback), t+h = 36
        _, label = dataset[0]
        expected = (close_values[36] - close_values[35]) / close_values[35]
        assert pytest.approx(label.item(), rel=1e-5) == expected

    def test_label_horizon_3_correctness(self):
        """Label with horizon=3 should be (close_{t+3} - close_t) / close_t."""
        n_bars = 50
        timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="5min")
        features_df = pd.DataFrame(
            np.ones((n_bars, 16), dtype=np.float32),
            index=timestamps,
            columns=[f"feat_{i}" for i in range(16)],
        )
        close_values = np.linspace(100.0, 110.0, n_bars)
        close_prices = pd.Series(close_values, index=timestamps)

        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=36,
            horizon=3,
        )

        # For the first sample: t = 35, t+h = 38
        _, label = dataset[0]
        expected = (close_values[38] - close_values[35]) / close_values[35]
        assert pytest.approx(label.item(), rel=1e-5) == expected

    def test_label_horizon_6_correctness(self):
        """Label with horizon=6 should be (close_{t+6} - close_t) / close_t."""
        n_bars = 60
        timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="5min")
        features_df = pd.DataFrame(
            np.ones((n_bars, 16), dtype=np.float32),
            index=timestamps,
            columns=[f"feat_{i}" for i in range(16)],
        )
        close_values = np.linspace(100.0, 120.0, n_bars)
        close_prices = pd.Series(close_values, index=timestamps)

        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=36,
            horizon=6,
        )

        # For the first sample: t = 35, t+h = 41
        _, label = dataset[0]
        expected = (close_values[41] - close_values[35]) / close_values[35]
        assert pytest.approx(label.item(), rel=1e-5) == expected

    def test_label_horizon_12_correctness(self):
        """Label with horizon=12 should be (close_{t+12} - close_t) / close_t."""
        n_bars = 60
        timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="5min")
        features_df = pd.DataFrame(
            np.ones((n_bars, 16), dtype=np.float32),
            index=timestamps,
            columns=[f"feat_{i}" for i in range(16)],
        )
        close_values = np.linspace(100.0, 120.0, n_bars)
        close_prices = pd.Series(close_values, index=timestamps)

        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=36,
            horizon=12,
        )

        # For the first sample: t = 35, t+h = 47
        _, label = dataset[0]
        expected = (close_values[47] - close_values[35]) / close_values[35]
        assert pytest.approx(label.item(), rel=1e-5) == expected

    def test_label_positive_return(self):
        """Label should be positive when future price is higher."""
        n_bars = 50
        timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="5min")
        features_df = pd.DataFrame(
            np.ones((n_bars, 16), dtype=np.float32),
            index=timestamps,
            columns=[f"feat_{i}" for i in range(16)],
        )
        # Price goes up monotonically
        close_values = np.linspace(100.0, 200.0, n_bars)
        close_prices = pd.Series(close_values, index=timestamps)

        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=36,
            horizon=1,
        )

        _, label = dataset[0]
        assert label.item() > 0

    def test_label_negative_return(self):
        """Label should be negative when future price is lower."""
        n_bars = 50
        timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="5min")
        features_df = pd.DataFrame(
            np.ones((n_bars, 16), dtype=np.float32),
            index=timestamps,
            columns=[f"feat_{i}" for i in range(16)],
        )
        # Price goes down monotonically
        close_values = np.linspace(200.0, 100.0, n_bars)
        close_prices = pd.Series(close_values, index=timestamps)

        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=36,
            horizon=1,
        )

        _, label = dataset[0]
        assert label.item() < 0

    def test_label_zero_return_when_price_unchanged(self):
        """Label should be zero when future price equals current price."""
        n_bars = 50
        timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="5min")
        features_df = pd.DataFrame(
            np.ones((n_bars, 16), dtype=np.float32),
            index=timestamps,
            columns=[f"feat_{i}" for i in range(16)],
        )
        # Constant price
        close_values = np.full(n_bars, 150.0)
        close_prices = pd.Series(close_values, index=timestamps)

        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=36,
            horizon=1,
        )

        _, label = dataset[0]
        assert label.item() == 0.0


# ---------------------------------------------------------------------------
# Test: Dataset Length and Stride
# ---------------------------------------------------------------------------


class TestDatasetLength:
    """Test dataset length computation with different parameters."""

    def test_length_with_no_gaps(self):
        """Dataset length should be (n - lookback - horizon + 1) with stride=1 and no gaps."""
        n_bars = 50
        features_df, close_prices = _make_synthetic_data(n_bars=n_bars)
        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=36,
            horizon=1,
            stride=1,
        )
        expected_len = n_bars - 36 - 1 + 1  # 14
        assert len(dataset) == expected_len

    def test_length_with_stride(self):
        """Dataset length should account for stride."""
        n_bars = 100
        features_df, close_prices = _make_synthetic_data(n_bars=n_bars)
        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=36,
            horizon=1,
            stride=2,
        )
        # With stride=2, we take every other valid starting index
        # Total valid range: 0 to n - lookback - horizon = 63
        # Indices: 0, 2, 4, ..., 62 → 32 samples
        expected_len = len(range(0, n_bars - 36 - 1 + 1, 2))
        assert len(dataset) == expected_len


# ---------------------------------------------------------------------------
# Test: Validation Errors
# ---------------------------------------------------------------------------


class TestDatasetValidation:
    """Test that ForexDataset raises appropriate errors for invalid inputs."""

    def test_wrong_feature_dimension_raises_error(self):
        """Should raise ValueError if features don't have 16 columns."""
        n_bars = 50
        timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="5min")
        # Only 10 features instead of 16
        features_df = pd.DataFrame(
            np.ones((n_bars, 10), dtype=np.float32),
            index=timestamps,
            columns=[f"feat_{i}" for i in range(10)],
        )
        close_prices = pd.Series(np.full(n_bars, 150.0), index=timestamps)

        with pytest.raises(ValueError, match="Expected 16 features, got 10"):
            ForexDataset(
                features_df=features_df,
                close_prices=close_prices,
                lookback=36,
                horizon=1,
            )

    def test_empty_dataset_after_gap_exclusion_raises_error(self):
        """Should raise ValueError if all samples are excluded by gaps."""
        n_bars = 40
        timestamps = []
        current = pd.Timestamp("2023-01-01")
        for i in range(n_bars):
            timestamps.append(current)
            # Every bar has a gap after it (>10 min)
            current += pd.Timedelta(minutes=15)

        index = pd.DatetimeIndex(timestamps)
        features_df = pd.DataFrame(
            np.ones((n_bars, 16), dtype=np.float32),
            index=index,
            columns=[f"feat_{i}" for i in range(16)],
        )
        close_prices = pd.Series(np.full(n_bars, 150.0), index=index)

        with pytest.raises(ValueError, match="No valid samples after gap exclusion"):
            ForexDataset(
                features_df=features_df,
                close_prices=close_prices,
                lookback=36,
                horizon=1,
            )
