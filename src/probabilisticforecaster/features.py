"""Feature Engine for the Transformer Probabilistic Forex Forecaster.

Computes 16 engineered features from 5-minute OHLC price bars:
- 8 z-score features (high, low, close, hl_spread, ema5, ema20, ema30, ema60)
- 3 return features (high, low, close)
- 3 volatility features (high, low, close)
- 2 time features (sin, cos of intraday cycle)
"""

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {"Timestamp", "Open", "High", "Low", "Close", "Volume"}
_EMA_SPANS = [5, 20, 30, 60]


def _validate_input(df: pd.DataFrame, historical_window: int) -> None:
    """Validate input DataFrame has required columns and sufficient rows.

    Args:
        df: Input DataFrame to validate.
        historical_window: Minimum number of bars required.

    Raises:
        ValueError: If columns are missing or insufficient history.
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    n = len(df)
    if n < historical_window:
        raise ValueError(
            f"Insufficient history: {n} bars provided, {historical_window} required"
        )


def _compute_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score, returning 0 when std is zero or near-zero.

    Args:
        series: Input price or indicator series.
        window: Rolling window size.

    Returns:
        Z-score series with 0 where std is effectively zero.
    """
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=0)

    abs_mean = rolling_mean.abs()
    is_zero_std = (rolling_std < 1e-14) | (rolling_std < abs_mean * 1e-10)

    zscore = (series - rolling_mean) / rolling_std
    zscore = zscore.where(~is_zero_std, 0.0)
    return zscore


def _check_zero_prices(series: pd.Series, name: str) -> None:
    """Check for zero prices that would cause division by zero in returns.

    Args:
        series: Price series to check (shifted by 1 for previous prices).
        name: Name of the series for error messaging.

    Raises:
        ValueError: If any previous price is zero.
    """
    prev = series.shift(1)
    zero_mask = prev == 0
    if zero_mask.any():
        first_zero_idx = zero_mask.idxmax()
        raise ValueError(
            f"Zero price at index {first_zero_idx}, cannot compute return"
        )


def compute_features(
    df: pd.DataFrame, historical_window: int = 1440
) -> pd.DataFrame:
    """Compute 16 features from 5-min OHLC DataFrame.

    Args:
        df: DataFrame with columns [Timestamp, Open, High, Low, Close, Volume]
        historical_window: Rolling window size for z-score and volatility
            (default 1440)

    Returns:
        DataFrame with 16 feature columns, indexed by Timestamp.
        Rows with insufficient history are dropped.

    Raises:
        ValueError: If fewer than historical_window bars are available.
        ValueError: If required columns are missing.
        ValueError: If a previous bar price is zero (cannot compute return).
    """
    _validate_input(df, historical_window)

    data = df.copy().reset_index(drop=True)

    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    timestamp = pd.to_datetime(data["Timestamp"])

    _check_zero_prices(high, "High")
    _check_zero_prices(low, "Low")
    _check_zero_prices(close, "Close")

    z_high = _compute_zscore(high, historical_window)
    z_low = _compute_zscore(low, historical_window)
    z_close = _compute_zscore(close, historical_window)

    hl_spread = high - low
    z_hl_spread = _compute_zscore(hl_spread, historical_window)

    ema5 = close.ewm(span=5, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema30 = close.ewm(span=30, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()

    z_ema5 = _compute_zscore(ema5, historical_window)
    z_ema20 = _compute_zscore(ema20, historical_window)
    z_ema30 = _compute_zscore(ema30, historical_window)
    z_ema60 = _compute_zscore(ema60, historical_window)

    ret_high = high.pct_change()
    ret_low = low.pct_change()
    ret_close = close.pct_change()

    vol_high = ret_high.rolling(
        window=historical_window, min_periods=historical_window
    ).std(ddof=0)
    vol_low = ret_low.rolling(
        window=historical_window, min_periods=historical_window
    ).std(ddof=0)
    vol_close = ret_close.rolling(
        window=historical_window, min_periods=historical_window
    ).std(ddof=0)

    hours = timestamp.dt.hour
    minutes = timestamp.dt.minute
    time_fraction = (hours * 60 + minutes) / 1440.0
    time_sin = np.sin(time_fraction * 2 * np.pi)
    time_cos = np.cos(time_fraction * 2 * np.pi)

    features = pd.DataFrame(
        {
            "z_high": z_high,
            "z_low": z_low,
            "z_close": z_close,
            "z_hl_spread": z_hl_spread,
            "z_ema5": z_ema5,
            "z_ema20": z_ema20,
            "z_ema30": z_ema30,
            "z_ema60": z_ema60,
            "ret_high": ret_high,
            "ret_low": ret_low,
            "ret_close": ret_close,
            "vol_high": vol_high,
            "vol_low": vol_low,
            "vol_close": vol_close,
            "time_sin": time_sin,
            "time_cos": time_cos,
        }
    )

    features.index = timestamp

    features = features.iloc[historical_window:]

    return features
