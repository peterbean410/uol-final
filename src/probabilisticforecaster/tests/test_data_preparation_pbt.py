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

    returns = rng.normal(0, 0.0002, size=n_bars)
    close_prices = base_price * np.exp(np.cumsum(returns))

    spreads = rng.uniform(0.001, 0.005, size=n_bars) * close_prices
    high_prices = close_prices + spreads * rng.uniform(0.3, 0.7, size=n_bars)
    low_prices = close_prices - spreads * rng.uniform(0.3, 0.7, size=n_bars)
    open_prices = close_prices + rng.normal(0, 0.0001, size=n_bars) * close_prices

    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

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

    min_bars_per_split = lookback_window + forecast_horizon + 50
    total_min_bars = historical_window + min_bars_per_split * 2

    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))

    start_time = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")

    n_bars = total_min_bars + 200

    return {
        "lookback_window": lookback_window,
        "forecast_horizon": forecast_horizon,
        "historical_window": historical_window,
        "n_bars": n_bars,
        "seed": seed,
        "start_time": start_time,
    }


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

        ohlc_data = _generate_synthetic_ohlc(
            n_bars=n_bars,
            start_time=start_time,
            seed=seed,
        )

        features_df = compute_features(ohlc_data, historical_window=historical_window)

        assume(len(features_df) >= lookback_window + forecast_horizon)

        data_indexed = ohlc_data.set_index(
            pd.to_datetime(ohlc_data["Timestamp"], utc=True)
        )
        close_prices = data_indexed["Close"].reindex(features_df.index)

        dataset = ForexDataset(
            features_df=features_df,
            close_prices=close_prices,
            lookback=lookback_window,
            horizon=forecast_horizon,
        )

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

            assert features_tensor.shape == (lookback_window, 16), (
                f"Sample {idx}: expected shape ({lookback_window}, 16), "
                f"got {features_tensor.shape}"
            )

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

        ohlc_data = _generate_synthetic_ohlc(
            n_bars=n_bars,
            start_time=start_time,
            seed=seed,
        )

        features_df = compute_features(ohlc_data, historical_window=historical_window)
        assume(len(features_df) >= lookback_window + forecast_horizon + 10)

        data_indexed = ohlc_data.set_index(
            pd.to_datetime(ohlc_data["Timestamp"], utc=True)
        )
        close_prices = data_indexed["Close"].reindex(features_df.index)

        available_timestamps = features_df.index
        mid_idx = len(available_timestamps) // 2
        assume(mid_idx >= lookback_window + forecast_horizon)
        assume(len(available_timestamps) - mid_idx >= lookback_window + forecast_horizon)

        test_start_dt = available_timestamps[mid_idx]
        train_end_dt = available_timestamps[mid_idx - 1]

        train_mask = features_df.index <= train_end_dt
        test_mask = features_df.index >= test_start_dt

        train_features = features_df[train_mask]
        test_features = features_df[test_mask]
        train_close = close_prices[train_mask]
        test_close = close_prices[test_mask]

        assume(len(train_features) >= lookback_window + forecast_horizon)
        assume(len(test_features) >= lookback_window + forecast_horizon)

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

        train_timestamps = train_features.index
        for sample_idx in range(len(train_dataset)):
            start = train_dataset.valid_indices[sample_idx]
            last_feature_bar = start + lookback_window - 1
            label_bar = last_feature_bar + forecast_horizon

            assert train_timestamps[label_bar] <= train_end_dt, (
                f"Train sample {sample_idx}: label bar at "
                f"{train_timestamps[label_bar]} exceeds train_end {train_end_dt}"
            )

        test_timestamps = test_features.index
        for sample_idx in range(len(test_dataset)):
            start = test_dataset.valid_indices[sample_idx]
            first_feature_bar_ts = test_timestamps[start]

            assert first_feature_bar_ts >= test_start_dt, (
                f"Test sample {sample_idx}: first feature bar at "
                f"{first_feature_bar_ts} is before test_start {test_start_dt}"
            )
