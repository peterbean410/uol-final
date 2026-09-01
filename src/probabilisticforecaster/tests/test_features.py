"""Unit tests for the Feature Engine (features.py)."""

import numpy as np
import pandas as pd
import pytest

from probabilisticforecaster.features import compute_features


def _make_ohlc_df(n: int, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic OHLC DataFrame with n bars."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="5min")
    close = start_price + np.cumsum(rng.normal(0, 0.01, n))
    high = close + rng.uniform(0.001, 0.05, n)
    low = close - rng.uniform(0.001, 0.05, n)
    open_ = close + rng.normal(0, 0.01, n)
    volume = rng.integers(100, 10000, n)

    return pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


class TestComputeFeaturesValidation:
    """Tests for input validation."""

    def test_missing_columns_raises_value_error(self):
        df = pd.DataFrame({"Timestamp": [1], "Open": [1], "High": [1]})
        with pytest.raises(ValueError, match="Missing columns"):
            compute_features(df)

    def test_insufficient_history_raises_value_error(self):
        df = _make_ohlc_df(100)
        with pytest.raises(ValueError, match="Insufficient history: 100 bars"):
            compute_features(df)

    def test_zero_previous_price_raises_value_error(self):
        df = _make_ohlc_df(1500)
        df.loc[500, "High"] = 0.0
        with pytest.raises(ValueError, match="Zero price at index"):
            compute_features(df)


class TestComputeFeaturesOutput:
    """Tests for correct output shape and content."""

    @pytest.fixture
    def features_df(self):
        """Compute features on a standard synthetic dataset."""
        df = _make_ohlc_df(2000)
        return compute_features(df)

    def test_output_has_16_columns(self, features_df):
        assert features_df.shape[1] == 16

    def test_output_column_names(self, features_df):
        expected = [
            "z_high", "z_low", "z_close", "z_hl_spread",
            "z_ema5", "z_ema20", "z_ema30", "z_ema60",
            "ret_high", "ret_low", "ret_close",
            "vol_high", "vol_low", "vol_close",
            "time_sin", "time_cos",
        ]
        assert list(features_df.columns) == expected

    def test_output_drops_first_1440_rows(self, features_df):
        assert len(features_df) == 560

    def test_no_nan_values_in_output(self, features_df):
        assert not features_df.isna().any().any()

    def test_time_features_bounded(self, features_df):
        assert features_df["time_sin"].between(-1, 1).all()
        assert features_df["time_cos"].between(-1, 1).all()

    def test_volatility_features_non_negative(self, features_df):
        assert (features_df["vol_high"] >= 0).all()
        assert (features_df["vol_low"] >= 0).all()
        assert (features_df["vol_close"] >= 0).all()


class TestComputeFeaturesZeroStd:
    """Tests for zero standard deviation edge case."""

    def test_constant_price_gives_zero_zscore(self):
        """When all prices are constant, z-scores should be 0."""
        n = 1500
        timestamps = pd.date_range("2023-01-01", periods=n, freq="5min")
        df = pd.DataFrame(
            {
                "Timestamp": timestamps,
                "Open": np.full(n, 100.0),
                "High": np.full(n, 101.0),
                "Low": np.full(n, 99.0),
                "Close": np.full(n, 100.0),
                "Volume": np.full(n, 1000),
            }
        )
        features = compute_features(df)
        zscore_cols = [
            "z_high", "z_low", "z_close", "z_hl_spread",
            "z_ema5", "z_ema20", "z_ema30", "z_ema60",
        ]
        for col in zscore_cols:
            assert (features[col] == 0.0).all(), f"{col} should be 0 for constant prices"


class TestComputeFeaturesTimeFeatures:
    """Tests for time feature correctness."""

    def test_midnight_time_features(self):
        """At midnight (00:00), sin should be 0 and cos should be 1."""
        n = 1500
        timestamps = pd.date_range("2023-01-01 00:00", periods=n, freq="5min")
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
        df = pd.DataFrame(
            {
                "Timestamp": timestamps,
                "Open": close + rng.normal(0, 0.005, n),
                "High": close + 0.02,
                "Low": close - 0.02,
                "Close": close,
                "Volume": np.full(n, 1000),
            }
        )
        features = compute_features(df)
        midnight_mask = features.index.hour == 0
        midnight_mask &= features.index.minute == 0
        if midnight_mask.any():
            assert np.allclose(features.loc[midnight_mask, "time_sin"], 0.0, atol=1e-10)
            assert np.allclose(features.loc[midnight_mask, "time_cos"], 1.0, atol=1e-10)

    def test_custom_historical_window(self):
        """Test with a smaller historical window."""
        df = _make_ohlc_df(200)
        features = compute_features(df, historical_window=100)
        assert len(features) == 100
        assert features.shape[1] == 16
