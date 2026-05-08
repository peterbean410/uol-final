"""
MACD indicator powered by TA-Lib.

Pass a DataFrame with a price column (e.g. 'Close') to compute() and
receive a DataFrame with MACD, MACD signal, and MACD histogram columns
appended.
"""

import talib
import pandas as pd


def compute(
    df: pd.DataFrame,
    price_col: str = "Close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute MACD and return *df* with MACD columns appended.

    Parameters
    ----------
    df : DataFrame with at least a numeric price column.
    price_col : name of the column to compute MACD over.
    fast : fast EMA period (default 12).
    slow : slow EMA period (default 26).
    signal : signal line EMA period (default 9).

    Returns
    -------
    A copy of *df* with ``MACD``, ``MACD_signal``, and ``MACD_hist``
    columns added.
    """
    prices = df[price_col].astype(float)
    macd, macd_signal, macd_hist = talib.MACD(
        prices, fastperiod=fast, slowperiod=slow, signalperiod=signal,
    )
    result = df.copy()
    result["MACD"] = macd
    result["MACD_signal"] = macd_signal
    result["MACD_hist"] = macd_hist
    return result
