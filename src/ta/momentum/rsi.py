"""
Relative Strength Index (RSI) indicator powered by TA-Lib.

Pass a DataFrame with a price column (e.g. 'Close') to compute() and
receive a DataFrame with an RSI column appended.
"""

import talib
import pandas as pd


def compute(
    df: pd.DataFrame,
    price_col: str = "Close",
    period: int = 14,
) -> pd.DataFrame:
    """Compute RSI and return *df* with an ``RSI_{period}`` column appended.

    Parameters
    ----------
    df : DataFrame with at least a numeric price column.
    price_col : name of the column to compute RSI over.
    period : lookback period (default 14).

    Returns
    -------
    A copy of *df* with an ``RSI_14`` (or similar) column added.
    """
    prices = df[price_col].astype(float)
    result = df.copy()
    result[f"RSI_{period}"] = talib.RSI(prices, timeperiod=period)
    return result
