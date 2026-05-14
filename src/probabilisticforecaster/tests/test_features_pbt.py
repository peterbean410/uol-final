"""Property-based tests for the Feature Engine (features.py).

Uses Hypothesis to verify correctness properties across randomly generated inputs.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from probabilisticforecaster.features import compute_features

TWO_PI = 2 * np.pi


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Use a smaller historical window for faster PBT execution
HIST_WINDOW = 50


@st.composite
def ohlc_dataframes(draw, min_bars=HIST_WINDOW + 10, max_bars=HIST_WINDOW + 50):
    """Generate valid OHLC DataFrames with realistic price data.

    Ensures:
    - All prices are strictly positive (no zero prices)
    - High >= Close >= Low (valid OHLC relationship)
    - At least min_bars rows
    - Sufficient price variation to avoid near-zero std edge cases
      (the zero-std case is tested separately)
    """
    n = draw(st.integers(min_value=min_bars, max_value=max_bars))

    # Generate a base close price series via random walk with guaranteed variation
    start_price = draw(st.floats(min_value=80.0, max_value=150.0))
    # Use increments from two disjoint ranges to guarantee non-zero variation
    # without filtering
    increments = draw(
        st.lists(
            st.one_of(
                st.floats(min_value=-0.5, max_value=-0.02),
                st.floats(min_value=0.02, max_value=0.5),
            ),
            min_size=n - 1,
            max_size=n - 1,
        )
    )
    close = np.empty(n)
    close[0] = start_price
    for i, inc in enumerate(increments):
        close[i + 1] = close[i] + inc

    # Ensure all close prices are strictly positive
    min_close = close.min()
    if min_close <= 1.0:
        close = close - min_close + 10.0

    # Generate high and low with valid OHLC relationship
    spreads = draw(
        st.lists(
            st.floats(min_value=0.05, max_value=1.5),
            min_size=n,
            max_size=n,
        )
    )
    spreads = np.array(spreads)

    high = close + spreads
    low = close - spreads

    # Ensure low is strictly positive
    low = np.maximum(low, 0.01)

    # Open is between low and high
    open_ = (high + low) / 2.0

    timestamps = pd.date_range("2023-01-01", periods=n, freq="5min")
    volume = np.full(n, 1000)

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


@st.composite
def constant_price_dataframes(draw, min_bars=HIST_WINDOW + 10, max_bars=HIST_WINDOW + 50):
    """Generate OHLC DataFrames where prices are constant (std = 0).

    This tests the zero-std edge case for z-scores.
    """
    n = draw(st.integers(min_value=min_bars, max_value=max_bars))
    price = draw(st.floats(min_value=50.0, max_value=200.0))
    spread = draw(st.floats(min_value=0.01, max_value=1.0))

    timestamps = pd.date_range("2023-01-01", periods=n, freq="5min")

    return pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Open": np.full(n, price),
            "High": np.full(n, price + spread),
            "Low": np.full(n, price - spread),
            "Close": np.full(n, price),
            "Volume": np.full(n, 1000),
        }
    )


# ---------------------------------------------------------------------------
# Property 1: Z-Score Computation Correctness
# ---------------------------------------------------------------------------


class TestZScoreComputationCorrectness:
    """Property 1: Z-Score Computation Correctness.

    For any valid OHLC DataFrame with at least historical_window bars and any
    position t >= historical_window, the computed z-score feature for each of
    the 8 z-score columns SHALL equal
    (x_t - mean(x[t-window+1:t+1])) / std(x[t-window+1:t+1]) with ddof=0,
    and when std equals zero the result SHALL be zero.

    **Validates: Requirements 2.1, 2.2, 2.3**
    """

    @given(df=ohlc_dataframes())
    @settings(max_examples=50, deadline=None)
    def test_zscore_equals_manual_computation(self, df: pd.DataFrame):
        """Z-score features match manual rolling z-score computation.

        **Validates: Requirements 2.1, 2.2, 2.3**
        """
        features = compute_features(df, historical_window=HIST_WINDOW)

        # Reconstruct the raw series used for z-score computation
        data = df.reset_index(drop=True)
        high = data["High"]
        low = data["Low"]
        close = data["Close"]
        hl_spread = high - low
        ema5 = close.ewm(span=5, adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema30 = close.ewm(span=30, adjust=False).mean()
        ema60 = close.ewm(span=60, adjust=False).mean()

        raw_series = {
            "z_high": high,
            "z_low": low,
            "z_close": close,
            "z_hl_spread": hl_spread,
            "z_ema5": ema5,
            "z_ema20": ema20,
            "z_ema30": ema30,
            "z_ema60": ema60,
        }

        # Check a sample of positions in the output
        # Output starts at index HIST_WINDOW in the original data
        output_len = len(features)
        # Check up to 5 random positions to keep test fast
        positions_to_check = min(5, output_len)
        rng = np.random.default_rng(42)
        check_indices = rng.choice(output_len, size=positions_to_check, replace=False)

        for feat_name, series in raw_series.items():
            for out_idx in check_indices:
                # Map output index to original data index
                t = out_idx + HIST_WINDOW

                # Manual z-score: window is [t - window + 1, t + 1)
                # i.e. indices t-49 to t inclusive (50 elements)
                window_data = series.iloc[t - HIST_WINDOW + 1: t + 1].values
                mean_val = np.mean(window_data)
                std_val = np.std(window_data, ddof=0)

                # Use same tolerance as implementation: std is considered zero
                # if it's negligible relative to the mean or in absolute terms
                abs_mean = abs(mean_val)
                is_zero_std = (std_val < 1e-14) or (abs_mean > 0 and std_val < abs_mean * 1e-10)

                if is_zero_std:
                    # When std is effectively zero, implementation returns 0
                    expected = 0.0
                else:
                    expected = (series.iloc[t] - mean_val) / std_val

                actual = features.iloc[out_idx][feat_name]

                assert np.isclose(actual, expected, rtol=1e-5, atol=1e-9), (
                    f"{feat_name} at output index {out_idx} (data index {t}): "
                    f"expected {expected}, got {actual}"
                )

    @given(df=constant_price_dataframes())
    @settings(max_examples=30, deadline=None)
    def test_zero_std_produces_zero_zscore(self, df: pd.DataFrame):
        """When std equals zero, z-score result SHALL be zero.

        **Validates: Requirements 2.1, 2.2, 2.3**
        """
        features = compute_features(df, historical_window=HIST_WINDOW)

        zscore_cols = [
            "z_high", "z_low", "z_close", "z_hl_spread",
            "z_ema5", "z_ema20", "z_ema30", "z_ema60",
        ]

        for col in zscore_cols:
            assert (features[col] == 0.0).all(), (
                f"{col} should be 0 when prices are constant (zero std), "
                f"but got non-zero values: {features[col][features[col] != 0.0].values[:5]}"
            )


# ---------------------------------------------------------------------------
# Property 2: Return Computation Correctness
# ---------------------------------------------------------------------------


class TestReturnComputationCorrectness:
    """Property 2: Return Computation Correctness.

    For any valid OHLC DataFrame with positive prices and any consecutive bars
    at positions t-1 and t, the computed return feature SHALL equal
    (x_t - x_{t-1}) / x_{t-1} for each of high, low, and close.

    **Validates: Requirements 3.1**
    """

    @given(df=ohlc_dataframes())
    @settings(max_examples=50, deadline=None)
    def test_return_equals_manual_computation(self, df: pd.DataFrame):
        """Return features match manual (x_t - x_{t-1}) / x_{t-1} computation.

        **Validates: Requirements 3.1**
        """
        features = compute_features(df, historical_window=HIST_WINDOW)

        data = df.reset_index(drop=True)
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        output_len = len(features)

        # Check all positions in the output
        for out_idx in range(output_len):
            # Map output index to original data index
            t = out_idx + HIST_WINDOW

            # Manual return: (x_t - x_{t-1}) / x_{t-1}
            expected_ret_high = (high.iloc[t] - high.iloc[t - 1]) / high.iloc[t - 1]
            expected_ret_low = (low.iloc[t] - low.iloc[t - 1]) / low.iloc[t - 1]
            expected_ret_close = (close.iloc[t] - close.iloc[t - 1]) / close.iloc[t - 1]

            actual_ret_high = features.iloc[out_idx]["ret_high"]
            actual_ret_low = features.iloc[out_idx]["ret_low"]
            actual_ret_close = features.iloc[out_idx]["ret_close"]

            assert np.isclose(actual_ret_high, expected_ret_high, rtol=1e-10, atol=1e-14), (
                f"ret_high at output index {out_idx} (data index {t}): "
                f"expected {expected_ret_high}, got {actual_ret_high}"
            )
            assert np.isclose(actual_ret_low, expected_ret_low, rtol=1e-10, atol=1e-14), (
                f"ret_low at output index {out_idx} (data index {t}): "
                f"expected {expected_ret_low}, got {actual_ret_low}"
            )
            assert np.isclose(actual_ret_close, expected_ret_close, rtol=1e-10, atol=1e-14), (
                f"ret_close at output index {out_idx} (data index {t}): "
                f"expected {expected_ret_close}, got {actual_ret_close}"
            )


# ---------------------------------------------------------------------------
# Strategy: OHLC DataFrames with varied starting timestamps
# ---------------------------------------------------------------------------


@st.composite
def ohlc_dataframes_varied_timestamps(draw, min_bars=HIST_WINDOW + 10, max_bars=HIST_WINDOW + 50):
    """Generate valid OHLC DataFrames with varied starting timestamps.

    Unlike ohlc_dataframes which always starts at "2023-01-01", this strategy
    generates DataFrames starting at random dates/times to ensure time features
    are correct for any time of day.
    """
    n = draw(st.integers(min_value=min_bars, max_value=max_bars))

    # Generate a random starting timestamp (any hour/minute combination)
    start_year = draw(st.integers(min_value=2020, max_value=2024))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))
    start_hour = draw(st.integers(min_value=0, max_value=23))
    start_minute = draw(st.sampled_from([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]))

    start_ts = pd.Timestamp(
        year=start_year, month=start_month, day=start_day,
        hour=start_hour, minute=start_minute
    )

    # Generate a base close price series via random walk
    start_price = draw(st.floats(min_value=80.0, max_value=150.0))
    increments = draw(
        st.lists(
            st.one_of(
                st.floats(min_value=-0.5, max_value=-0.02),
                st.floats(min_value=0.02, max_value=0.5),
            ),
            min_size=n - 1,
            max_size=n - 1,
        )
    )
    close = np.empty(n)
    close[0] = start_price
    for i, inc in enumerate(increments):
        close[i + 1] = close[i] + inc

    # Ensure all close prices are strictly positive
    min_close = close.min()
    if min_close <= 1.0:
        close = close - min_close + 10.0

    # Generate high and low with valid OHLC relationship
    spreads = draw(
        st.lists(
            st.floats(min_value=0.05, max_value=1.5),
            min_size=n,
            max_size=n,
        )
    )
    spreads = np.array(spreads)

    high = close + spreads
    low = close - spreads
    low = np.maximum(low, 0.01)
    open_ = (high + low) / 2.0

    timestamps = pd.date_range(start_ts, periods=n, freq="5min")
    volume = np.full(n, 1000)

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


# ---------------------------------------------------------------------------
# Property 3: Time Feature Trigonometric Identity
# ---------------------------------------------------------------------------


class TestTimeFeatureTrigonometricIdentity:
    """Property 3: Time Feature Trigonometric Identity.

    For any valid OHLC DataFrame, the computed time features SHALL satisfy
    time_sin² + time_cos² = 1 for every row, and the values SHALL equal
    sin((h*60+m)/1440 * 2π) and cos((h*60+m)/1440 * 2π) respectively.

    **Validates: Requirements 3.3**
    """

    @given(df=ohlc_dataframes())
    @settings(max_examples=50, deadline=None)
    def test_trig_identity_holds(self, df: pd.DataFrame):
        """time_sin² + time_cos² = 1 for all rows (fixed start timestamp).

        **Validates: Requirements 3.3**
        """
        features = compute_features(df, historical_window=HIST_WINDOW)

        sin_vals = features["time_sin"].values
        cos_vals = features["time_cos"].values

        identity = sin_vals**2 + cos_vals**2

        assert np.allclose(identity, 1.0, rtol=1e-12, atol=1e-12), (
            f"Trigonometric identity violated: "
            f"max deviation = {np.max(np.abs(identity - 1.0))}"
        )

    @given(df=ohlc_dataframes())
    @settings(max_examples=50, deadline=None)
    def test_time_features_match_formula(self, df: pd.DataFrame):
        """time_sin and time_cos match sin/cos((h*60+m)/1440 * 2π).

        **Validates: Requirements 3.3**
        """
        features = compute_features(df, historical_window=HIST_WINDOW)

        # Reconstruct expected values from timestamps
        timestamps = features.index
        hours = timestamps.hour
        minutes = timestamps.minute
        time_fraction = (hours * 60 + minutes) / 1440.0

        expected_sin = np.sin(time_fraction * TWO_PI)
        expected_cos = np.cos(time_fraction * TWO_PI)

        actual_sin = features["time_sin"].values
        actual_cos = features["time_cos"].values

        assert np.allclose(actual_sin, expected_sin, rtol=1e-12, atol=1e-12), (
            f"time_sin mismatch: max diff = "
            f"{np.max(np.abs(actual_sin - expected_sin))}"
        )
        assert np.allclose(actual_cos, expected_cos, rtol=1e-12, atol=1e-12), (
            f"time_cos mismatch: max diff = "
            f"{np.max(np.abs(actual_cos - expected_cos))}"
        )

    @given(df=ohlc_dataframes_varied_timestamps())
    @settings(max_examples=50, deadline=None)
    def test_trig_identity_varied_timestamps(self, df: pd.DataFrame):
        """time_sin² + time_cos² = 1 for varied starting timestamps.

        Ensures the trigonometric identity holds for any time of day,
        not just timestamps starting from midnight.

        **Validates: Requirements 3.3**
        """
        features = compute_features(df, historical_window=HIST_WINDOW)

        sin_vals = features["time_sin"].values
        cos_vals = features["time_cos"].values

        identity = sin_vals**2 + cos_vals**2

        assert np.allclose(identity, 1.0, rtol=1e-12, atol=1e-12), (
            f"Trigonometric identity violated with varied timestamps: "
            f"max deviation = {np.max(np.abs(identity - 1.0))}"
        )

    @given(df=ohlc_dataframes_varied_timestamps())
    @settings(max_examples=50, deadline=None)
    def test_time_features_match_formula_varied_timestamps(self, df: pd.DataFrame):
        """time_sin and time_cos match formula for varied starting timestamps.

        **Validates: Requirements 3.3**
        """
        features = compute_features(df, historical_window=HIST_WINDOW)

        timestamps = features.index
        hours = timestamps.hour
        minutes = timestamps.minute
        time_fraction = (hours * 60 + minutes) / 1440.0

        expected_sin = np.sin(time_fraction * TWO_PI)
        expected_cos = np.cos(time_fraction * TWO_PI)

        actual_sin = features["time_sin"].values
        actual_cos = features["time_cos"].values

        assert np.allclose(actual_sin, expected_sin, rtol=1e-12, atol=1e-12), (
            f"time_sin mismatch with varied timestamps: max diff = "
            f"{np.max(np.abs(actual_sin - expected_sin))}"
        )
        assert np.allclose(actual_cos, expected_cos, rtol=1e-12, atol=1e-12), (
            f"time_cos mismatch with varied timestamps: max diff = "
            f"{np.max(np.abs(actual_cos - expected_cos))}"
        )
