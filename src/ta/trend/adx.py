"""
Average Directional Index (ADX) indicator powered by TA-Lib.

Pass a DataFrame with High, Low, Close columns to compute() and receive
a DataFrame with ADX appended.
"""

import talib
import pandas as pd


def compute(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """Compute ADX and return *df* with an ``ADX_{period}`` column appended.

    Parameters
    ----------
    df : DataFrame with columns High, Low, Close.
    period : lookback period (default 14).

    Returns
    -------
    A copy of *df* with an ``ADX_14`` (or similar) column added.
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    result = df.copy()
    result[f"ADX_{period}"] = talib.ADX(high, low, close, timeperiod=period)
    return result
