"""
Double Bottom Pattern Detection

A double bottom is a bullish reversal pattern defined by:

    1.  A decline to a local low  (first bottom)
    2.  A rally to a resistance level  (the neckline)
    3.  A second decline to approximately the same low  (second bottom)
    4.  A rally above the neckline  (confirmation)

Pass a DataFrame with columns Low, High, Close, Timestamp to
detect_double_bottoms() and receive a DataFrame of detected patterns.
"""

from dataclasses import dataclass, asdict

import pandas as pd


# ──────────────────────────────────────────────
# Pattern detection
# ──────────────────────────────────────────────
@dataclass
class DoubleBottom:
    """A detected double-bottom pattern."""
    idx1: int               # DataFrame index of first bottom
    idx2: int               # DataFrame index of second bottom
    ts1: str                # Timestamp of first bottom
    ts2: str                # Timestamp of second bottom
    low1: float             # Price at first bottom
    low2: float             # Price at second bottom
    neckline: float         # Highest high between the two bottoms
    neckline_idx: int       # DataFrame index of the neckline bar
    depth_pct: float        # (neckline - avg_bottom) / neckline * 100
    width_bars: int         # Number of bars between the two bottoms
    confirmed: bool         # True if price closed above the neckline after bottom 2

    # Local extrema before the pattern
    min_before_val: float | None = None
    min_before_ts: str | None = None
    max_before_val: float | None = None
    max_before_ts: str | None = None

    # Local extrema after the pattern
    min_after_val: float | None = None
    min_after_ts: str | None = None
    max_after_val: float | None = None
    max_after_ts: str | None = None


def find_local_minima(lows: pd.Series, window: int) -> list[int]:
    """Return indices where *lows* is the minimum in a centred rolling window."""
    half = window // 2
    minima = []
    for i in range(half, len(lows) - half):
        segment = lows.iloc[i - half : i + half + 1]
        if lows.iloc[i] == segment.min() and segment.min() != segment.max():
            if not minima or i - minima[-1] >= half:
                minima.append(i)
    return minima


def find_local_maxima(highs: pd.Series, window: int) -> list[int]:
    """Return indices where *highs* is the maximum in a centred rolling window."""
    half = window // 2
    maxima = []
    for i in range(half, len(highs) - half):
        segment = highs.iloc[i - half : i + half + 1]
        if highs.iloc[i] == segment.max() and segment.min() != segment.max():
            if not maxima or i - maxima[-1] >= half:
                maxima.append(i)
    return maxima


def _extrema_before(
    minima: list[int], maxima: list[int], idx: int,
    lows: pd.Series, highs: pd.Series, timestamps: pd.Series,
) -> tuple[float | None, str | None, float | None, str | None]:
    """Return (min_val, min_ts, max_val, max_ts) of the last extremum before *idx*."""
    before_min = [m for m in minima if m < idx]
    min_val = float(lows.iloc[before_min[-1]]) if before_min else None
    min_ts = str(timestamps.iloc[before_min[-1]]) if before_min else None

    before_max = [m for m in maxima if m < idx]
    max_val = float(highs.iloc[before_max[-1]]) if before_max else None
    max_ts = str(timestamps.iloc[before_max[-1]]) if before_max else None

    return min_val, min_ts, max_val, max_ts


def _extrema_after(
    minima: list[int], maxima: list[int], idx: int,
    lows: pd.Series, highs: pd.Series, timestamps: pd.Series,
) -> tuple[float | None, str | None, float | None, str | None]:
    """Return (min_val, min_ts, max_val, max_ts) of the first extremum after *idx*."""
    after_min = [m for m in minima if m > idx]
    min_val = float(lows.iloc[after_min[0]]) if after_min else None
    min_ts = str(timestamps.iloc[after_min[0]]) if after_min else None

    after_max = [m for m in maxima if m > idx]
    max_val = float(highs.iloc[after_max[0]]) if after_max else None
    max_ts = str(timestamps.iloc[after_max[0]]) if after_max else None

    return min_val, min_ts, max_val, max_ts


def detect_double_bottoms(
    df: pd.DataFrame,
    window: int = 5,
    tolerance_pct: float = 0.3,
    min_width: int = 5,
) -> tuple[pd.DataFrame, float | None, float | None]:
    """Scan *df* for double-bottom patterns.

    Parameters
    ----------
    df : DataFrame with columns Low, High, Close, Timestamp
    window : rolling window size for local-minima detection
    tolerance_pct : max % difference between the two bottoms
    min_width : minimum bars between the two bottoms

    Returns
    -------
    (patterns, latest_min, latest_max); a DataFrame of detected patterns
    (one row per pattern), the latest local minimum, and the latest local
    maximum.
    """
    lows = df["Low"].astype(float)
    highs = df["High"].astype(float)
    closes = df["Close"].astype(float)
    timestamps = df["Timestamp"]

    minima = find_local_minima(lows, window)
    maxima = find_local_maxima(highs, window)
    patterns: list[DoubleBottom] = []

    for a, b in zip(minima, minima[1:]):
        low1, low2 = lows.iloc[a], lows.iloc[b]
        avg_low = (low1 + low2) / 2

        # ── Tolerance check ──
        diff_pct = abs(low1 - low2) / avg_low * 100
        if diff_pct > tolerance_pct:
            continue

        # ── Minimum width ──
        width = b - a
        if width < min_width:
            continue

        # ── Neckline: highest high between the two bottoms ──
        between = highs.iloc[a + 1 : b]
        if between.empty:
            continue
        neckline_idx = int(between.idxmax())
        neckline = float(highs.iloc[neckline_idx])

        # ── Depth: the neckline must be meaningfully above the bottoms ──
        depth_pct = (neckline - avg_low) / neckline * 100
        if depth_pct < 0.1:
            continue

        # ── Confirmation: did price close above the neckline after bottom 2? ──
        remaining = closes.iloc[b + 1 :]
        confirmed = bool((remaining > neckline).any()) if not remaining.empty else False

        ts1 = str(timestamps.iloc[a])
        ts2 = str(timestamps.iloc[b])

        min_before_val, min_before_ts, max_before_val, max_before_ts = _extrema_before(
            minima, maxima, a, lows, highs, timestamps,
        )
        min_after_val, min_after_ts, max_after_val, max_after_ts = _extrema_after(
            minima, maxima, b, lows, highs, timestamps,
        )

        patterns.append(DoubleBottom(
            idx1=a, idx2=b,
            ts1=ts1, ts2=ts2,
            low1=round(low1, 5), low2=round(low2, 5),
            neckline=round(neckline, 5),
            neckline_idx=neckline_idx,
            depth_pct=round(depth_pct, 3),
            width_bars=width,
            confirmed=confirmed,
            min_before_val=min_before_val, min_before_ts=min_before_ts,
            max_before_val=max_before_val, max_before_ts=max_before_ts,
            min_after_val=min_after_val, min_after_ts=min_after_ts,
            max_after_val=max_after_val, max_after_ts=max_after_ts,
        ))

    # ── Check for potential forming double bottom at the right edge ──
    if minima:
        a = minima[-1]
        if a + 1 < len(df):
            b = int(lows.iloc[a + 1 :].idxmin())
            if b not in minima:
                low1, low2 = float(lows.iloc[a]), float(lows.iloc[b])
                avg_low = (low1 + low2) / 2
                diff_pct = abs(low1 - low2) / avg_low * 100

                if diff_pct <= tolerance_pct:
                    width = b - a
                    if width >= min_width:
                        between = highs.iloc[a + 1 : b]
                        if not between.empty:
                            neckline_idx = int(between.idxmax())
                            neckline = float(highs.iloc[neckline_idx])
                            depth_pct = (neckline - avg_low) / neckline * 100

                            if depth_pct >= 0.1:
                                remaining = closes.iloc[b + 1 :]
                                confirmed = bool((remaining > neckline).any()) if not remaining.empty else False
                                ts1 = str(timestamps.iloc[a])
                                ts2 = str(timestamps.iloc[b])

                                min_before_val, min_before_ts, max_before_val, max_before_ts = _extrema_before(
                                    minima, maxima, a, lows, highs, timestamps,
                                )

                                patterns.append(DoubleBottom(
                                    idx1=a, idx2=b,
                                    ts1=ts1, ts2=ts2,
                                    low1=round(low1, 5), low2=round(low2, 5),
                                    neckline=round(neckline, 5),
                                    neckline_idx=neckline_idx,
                                    depth_pct=round(depth_pct, 3),
                                    width_bars=width,
                                    confirmed=confirmed,
                                    min_before_val=min_before_val, min_before_ts=min_before_ts,
                                    max_before_val=max_before_val, max_before_ts=max_before_ts,
                                    min_after_val=None, min_after_ts=None,
                                    max_after_val=None, max_after_ts=None,
                                ))

    latest_min = float(lows.iloc[minima[-1]]) if minima else None
    latest_max = float(highs.iloc[maxima[-1]]) if maxima else None

    return patterns_to_frame(patterns), latest_min, latest_max


def patterns_to_frame(patterns: list[DoubleBottom]) -> pd.DataFrame:
    """Convert a list of DoubleBottom instances to a DataFrame."""
    if not patterns:
        return pd.DataFrame(columns=[
            "idx1", "idx2", "ts1", "ts2", "low1", "low2",
            "neckline", "neckline_idx", "depth_pct", "width_bars", "confirmed",
            "min_before_val", "min_before_ts", "max_before_val", "max_before_ts",
            "min_after_val", "min_after_ts", "max_after_val", "max_after_ts",
        ])
    return pd.DataFrame([asdict(p) for p in patterns])
