"""
Bollinger Bands indicator powered by TA-Lib.

Pass a DataFrame with a price column (e.g. 'Close') to compute() and
receive a DataFrame with upper, middle, and lower band columns appended.
"""

import talib
import pandas as pd


def compute(
    df: pd.DataFrame,
    price_col: str = "Close",
    period: int = 20,
    nbdev: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Bands and return *df* with band columns appended.

    Parameters
    ----------
    df : DataFrame with at least a numeric price column.
    price_col : name of the column to compute bands over.
    period : lookback period (default 20).
    nbdev : number of standard deviations for the bands (default 2.0).

    Returns
    -------
    A copy of *df* with ``BB_upper``, ``BB_middle``, ``BB_lower`` columns
    added.
    """
    prices = df[price_col].astype(float)
    upper, middle, lower = talib.BBANDS(prices, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev)
    result = df.copy()
    result["BB_upper"] = upper
    result["BB_middle"] = middle
    result["BB_lower"] = lower
    return result
