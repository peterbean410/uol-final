"""
Commodity Channel Index (CCI) indicator powered by TA-Lib.

Pass a DataFrame with High, Low, Close columns to compute() and receive
a DataFrame with a CCI column appended.
"""

import talib
import pandas as pd


def compute(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """Compute CCI and return *df* with a ``CCI_{period}`` column appended.

    Parameters
    ----------
    df : DataFrame with columns High, Low, Close.
    period : lookback period (default 14).

    Returns
    -------
    A copy of *df* with a ``CCI_14`` (or similar) column added.
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    result = df.copy()
    result[f"CCI_{period}"] = talib.CCI(high, low, close, timeperiod=period)
    return result
