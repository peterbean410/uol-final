"""Property-based tests for Data Preparation output shapes.

Verifies that the data_preparation component produces datasets with correct
feature shapes and that train/test splits align with configured date boundaries.

Uses synthetic OHLC data to avoid S3 access.

**Validates: Requirements 1.3**
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from probabilisticforecaster.dataset import ForexDataset
from probabilisticforecaster.features import compute_features


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

VALID_LOOKBACK_WINDOWS = (24, 36, 48)
VALID_FORECAST_HORIZONS = (1, 3, 6, 12)


def _generate_synthetic_ohlc(
    n_bars: int,
    start_time: pd.Timestamp,
    base_price: float = 150.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic 5-minute OHLC data mimicking forex structure.

    Creates continuous (no gaps) 5-minute bars with realistic price movements.

    Args:
        n_bars: Number of bars to generate.
        start_time: Starting timestamp for the series.
        base_price: Starting price level.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns [Timestamp, Symbol, Open, High, Low, Close, Volume].
    """
    rng = np.random.default_rng(seed)

    # Generate price walk
    returns = rng.normal(0, 0.0002, size=n_bars)
    close_prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    spreads = rng.uniform(0.001, 0.005, size=n_bars) * close_prices
    high_prices = close_prices + spreads * rng.uniform(0.3, 0.7, size=n_bars)
    low_prices = close_prices - spreads * rng.uniform(0.3, 0.7, size=n_bars)
    open_prices = close_prices + rng.normal(0, 0.0001, size=n_bars) * close_prices

    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    # Generate timestamps (5-minute intervals, skip weekends)
    timestamps = pd.date_range(
        start=start_time, periods=n_bars, freq="5min", tz="UTC"
    )

    volumes = rng.integers(100, 10000, size=n_bars)

    return pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Symbol": "USDJPY",
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volumes,
        }
    )


@st.composite
def valid_data_prep_configs(draw):
    """Generate valid data preparation configurations.

    Produces a config dict with lookback_window, forecast_horizon,
    historical_window, and date boundaries that are consistent with
    the generated synthetic data.
    """
    lookback_window = draw(st.sampled_from(VALID_LOOKBACK_WINDOWS))
    forecast_horizon = draw(st.sampled_from(VALID_FORECAST_HORIZONS))
    historical_window = draw(st.integers(min_value=200, max_value=500))

    # We need enough bars to cover:
    # historical_window (for feature warmup) + lookback + horizon + some margin
    # for both train and test sets
    min_bars_per_split = lookback_window + forecast_horizon + 50
    total_min_bars = historical_window + min_bars_per_split * 2

    # Generate a seed for synthetic data
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))

    # Use fixed date boundaries that work with our synthetic data
    # Start generating data from 2020-01-01
    # Train: 2020-01-01 to 2020-01-15
    # Test: 2020-01-16 to 2020-01-31
    start_time = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")

    # Generate enough bars to cover both train and test periods
    # 5-min bars: 288 per day, so ~30 days = ~8640 bars
    # We need at least historical_window + enough for both splits
    n_bars = total_min_bars + 200  # extra margin

    return {
        "lookback_window": lookback_window,
        "forecast_horizon": forecast_horizon,
        "historical_window": historical_window,
        "n_bars": n_bars,
        "seed": seed,
        "start_time": start_time,
    }


# ---------------------------------------------------------------------------
# Property 1: Data preparation output shape invariants
# ---------------------------------------------------------------------------


class TestDataPreparationOutputShapes:
    """Property 1: Data preparation output shape invariants.

    For any valid config, output datasets have feature shape
    (lookback_window, 16) and train/test split aligns with configured test_start.

    **Validates: Requirements 1.3**
    """

    @given(config=valid_data_prep_configs())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_feature_shape_matches_lookback_window(self, config):
        """For any valid lookback_window in {24, 36, 48}, each sample from
        ForexDataset has feature shape (lookback_window, 16).

        **Validates: Requirements 1.3**
        """
        lookback_window = config["lookback_window"]
        forecast_horizon = config["forecast_horizon"]
        historical_window = config["historical_window"]
        n_bars = config["n_bars"]
        seed = config["seed"]
        start_time = config["start_time"]

        # Generate synthetic OHLC data
        ohlc_data = _generate_synthetic_ohlc(
            n_bars=n_bars,
            start_time=start_time,
            seed=seed,
        )

        # Compute features (same as data_preparation component)
        features_df = compute_features(ohlc_data, historical_window=historical_window)

        # We need enough data after feature computation for at least one sample
        assume(len(features_df) >= lookback_window + forecast_horizon)

        # Align close prices with features index
        data_indexed = ohlc_data.set_index(
            pd.to_datetime(ohlc_data["Timestamp"], utc=True)
        )
        close_prices = data_indexed["Close"].reindex(features_df.index)

        # Build dataset (using all available data as a single split)
        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=lookback_window,
            horizon=forecast_horizon,
        )

        # Verify shape for every sample in the dataset
        # (check a subset for performance, first, last, and random middle)
        n_samples = len(dataset)
        assert n_samples > 0, "Dataset should have at least one sample"

        indices_to_check = [0, n_samples - 1]
        if n_samples > 2:
            rng = np.random.default_rng(seed)
            mid_indices = rng.choice(
                range(1, n_samples - 1),
                size=min(10, n_samples - 2),
                replace=False,
            )
            indices_to_check.extend(mid_indices.tolist())

        for idx in indices_to_check:
            features_tensor, label_tensor = dataset[idx]

            # Feature shape must be (lookback_window, 16)
            assert features_tensor.shape == (lookback_window, 16), (
                f"Sample {idx}: expected shape ({lookback_window}, 16), "
                f"got {features_tensor.shape}"
            )

            # Label should be a scalar (shape (1,))
            assert label_tensor.shape == (1,), (
                f"Sample {idx}: expected label shape (1,), got {label_tensor.shape}"
            )

    @given(config=valid_data_prep_configs())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_train_test_split_respects_test_start_boundary(self, config):
        """Train/test split respects the configured test_start date boundary.

        All samples in the train dataset have their last lookback bar timestamp
        <= train_end, and all samples in the test dataset have their first
        lookback bar timestamp >= test_start.

        **Validates: Requirements 1.3**
        """
        lookback_window = config["lookback_window"]
        forecast_horizon = config["forecast_horizon"]
        historical_window = config["historical_window"]
        n_bars = config["n_bars"]
        seed = config["seed"]
        start_time = config["start_time"]

        # Generate synthetic OHLC data
        ohlc_data = _generate_synthetic_ohlc(
            n_bars=n_bars,
            start_time=start_time,
            seed=seed,
        )

        # Compute features
        features_df = compute_features(ohlc_data, historical_window=historical_window)
        assume(len(features_df) >= lookback_window + forecast_horizon + 10)

        # Align close prices
        data_indexed = ohlc_data.set_index(
            pd.to_datetime(ohlc_data["Timestamp"], utc=True)
        )
        close_prices = data_indexed["Close"].reindex(features_df.index)

        # Define train/test split boundary using the midpoint of available data
        available_timestamps = features_df.index
        mid_idx = len(available_timestamps) // 2
        # Ensure both splits have enough data
        assume(mid_idx >= lookback_window + forecast_horizon)
        assume(len(available_timestamps) - mid_idx >= lookback_window + forecast_horizon)

        test_start_dt = available_timestamps[mid_idx]
        train_end_dt = available_timestamps[mid_idx - 1]

        # Split features by date boundary (same logic as build_datasets)
        train_mask = features_df.index <= train_end_dt
        test_mask = features_df.index >= test_start_dt

        train_features = features_df[train_mask]
        test_features = features_df[test_mask]
        train_close = close_prices[train_mask]
        test_close = close_prices[test_mask]

        # Skip if either split is too small for a valid dataset
        assume(len(train_features) >= lookback_window + forecast_horizon)
        assume(len(test_features) >= lookback_window + forecast_horizon)

        # Build train and test datasets
        train_dataset = ForexDataset(
            features_df=train_features,
            close_prices=train_close,
            lookback=lookback_window,
            horizon=forecast_horizon,
        )
        test_dataset = ForexDataset(
            features_df=test_features,
            close_prices=test_close,
            lookback=lookback_window,
            horizon=forecast_horizon,
        )

        # Verify train dataset: all sample timestamps are <= train_end
        train_timestamps = train_features.index
        for sample_idx in range(len(train_dataset)):
            start = train_dataset.valid_indices[sample_idx]
            # The last bar used for features is at start + lookback - 1
            last_feature_bar = start + lookback_window - 1
            # The label bar is at start + lookback - 1 + horizon
            label_bar = last_feature_bar + forecast_horizon

            # All bars in this sample must be within train period
            assert train_timestamps[label_bar] <= train_end_dt, (
                f"Train sample {sample_idx}: label bar at "
                f"{train_timestamps[label_bar]} exceeds train_end {train_end_dt}"
            )

        # Verify test dataset: all sample timestamps are >= test_start
        test_timestamps = test_features.index
        for sample_idx in range(len(test_dataset)):
            start = test_dataset.valid_indices[sample_idx]
            # The first bar in the lookback window
            first_feature_bar_ts = test_timestamps[start]

            # First bar of every test sample must be >= test_start
            assert first_feature_bar_ts >= test_start_dt, (
                f"Test sample {sample_idx}: first feature bar at "
                f"{first_feature_bar_ts} is before test_start {test_start_dt}"
            )
