"""
Moving Average indicators powered by TA-Lib.

Pass a DataFrame with a price column (e.g. 'Close') to compute() and
receive a DataFrame with moving-average columns appended.
"""

import talib
import pandas as pd


_MA_FUNCS = {
    "SMA":   talib.SMA,
    "EMA":   talib.EMA,
    "WMA":   talib.WMA,
    "DEMA":  talib.DEMA,
    "TEMA":  talib.TEMA,
    "KAMA":  talib.KAMA,
    "TRIMA": talib.TRIMA,
}


def compute(
    df: pd.DataFrame,
    price_col: str = "Close",
    periods: tuple[int, ...] = (10, 20, 50),
    kinds: tuple[str, ...] = ("SMA", "EMA"),
) -> pd.DataFrame:
    """Compute moving averages and return *df* with new columns appended.

    Parameters
    ----------
    df : DataFrame with at least a numeric price column.
    price_col : name of the column to compute MAs over.
    periods : time periods for each MA (e.g. 10, 20, 50).
    kinds : which moving-average types to compute (SMA, EMA, WMA, DEMA,
            TEMA, KAMA, TRIMA).

    Returns
    -------
    A copy of *df* with columns like ``SMA_10``, ``EMA_20``, etc. added.
    """
    prices = df[price_col].astype(float)
    result = df.copy()

    for kind in kinds:
        func = _MA_FUNCS[kind]
        for period in periods:
            col = f"{kind}_{period}"
            result[col] = func(prices, timeperiod=period)

    return result
