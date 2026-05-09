"""
Fibonacci Retracements indicator.

Pass a DataFrame with High, Low columns to compute() and receive
a DataFrame with Fibonacci retracement level columns appended.
"""

import pandas as pd

_LABELS = ["000", "236", "382", "500", "618", "786", "1000"]
_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]


def compute(
    df: pd.DataFrame,
    window: int = 50,
) -> pd.DataFrame:
    """Compute Fibonacci retracement levels and return *df* with level columns appended.

    For each row, finds the highest high and lowest low over the trailing
    *window* bars, then computes the seven standard retracement levels
    between those extremes and adds them as columns.

    Parameters
    ----------
    df : DataFrame with columns High, Low.
    window : rolling lookback window (default 50).

    Returns
    -------
    A copy of *df* with ``FR_000`` through ``FR_1000`` columns added.
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    roll_high = high.rolling(window=window, min_periods=1).max()
    roll_low = low.rolling(window=window, min_periods=1).min()
    delta = roll_high - roll_low

    result = df.copy()
    for label, ratio in zip(_LABELS, _RATIOS):
        result[f"FR_{label}"] = roll_low + delta * ratio
    return result
