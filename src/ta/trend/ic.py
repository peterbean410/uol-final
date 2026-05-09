"""
Ichimoku Cloud (Ichimoku Kinko Hyo) indicator.

Pass a DataFrame with High, Low, Close columns to compute() and receive
a DataFrame with all five Ichimoku lines appended.
"""

import pandas as pd


def compute(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> pd.DataFrame:
    """Compute Ichimoku Cloud lines and return *df* with them appended.

    Parameters
    ----------
    df : DataFrame with columns High, Low, Close.
    tenkan : Tenkan-sen (conversion line) lookback (default 9).
    kijun : Kijun-sen (base line) lookback (default 26).
    senkou_b : Senkou Span B lookback (default 52).

    Returns
    -------
    A copy of *df* with ``Tenkan``, ``Kijun``, ``SenkouA``, ``SenkouB``,
    and ``Chikou`` columns added.
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    tenkan_val = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_val = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_val + kijun_val) / 2).shift(kijun)
    senkou_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    chikou = close.shift(-kijun)

    result = df.copy()
    result["Tenkan"] = tenkan_val
    result["Kijun"] = kijun_val
    result["SenkouA"] = senkou_a
    result["SenkouB"] = senkou_b
    result["Chikou"] = chikou
    return result
