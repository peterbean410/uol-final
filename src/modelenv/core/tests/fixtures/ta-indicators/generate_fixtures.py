#!/usr/bin/env python3
"""
Generate JSON fixtures for TA indicator parity tests.

This script generates synthetic OHLCV data and computes each indicator using
the Python ta/ package, then saves the input, parameters, output values, and
NaN indices to JSON files for use in Rust parity tests.

Usage:
    python generate_fixtures.py

Output:
    Creates JSON fixture files in the same directory as this script.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from ta.momentum import rsi as rsi_mod
from ta.momentum import cci as cci_mod
from ta.trend import adx as adx_mod
from ta.trend import macd as macd_mod
from ta.trend import movingavg as ma_mod
from ta.trend import ic as ichimoku_mod
from ta.volatility import bb as bb_mod
from ta.support import fr as fr_mod
from ta.patterns import doublebottom as db_mod
from ta.patterns import doubletop as dt_mod


def generate_synthetic_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing.
    
    Creates a realistic price series with:
    - Random walk for close prices
    - High/Low derived from close with random spread
    - Open derived from previous close with gap
    - Volume as random positive integers
    """
    np.random.seed(seed)
    
    returns = np.random.normal(0.0001, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    
    spread = np.abs(np.random.normal(0, 0.005, n)) * close
    high = close + spread * np.random.uniform(0.5, 1.5, n)
    low = close - spread * np.random.uniform(0.5, 1.5, n)
    
    high = np.maximum(high, close)
    low = np.minimum(low, close)
    
    open_prices = np.roll(close, 1)
    open_prices[0] = close[0]
    open_prices = open_prices * (1 + np.random.normal(0, 0.001, n))
    
    open_prices = np.clip(open_prices, low, high)
    
    volume = np.random.randint(1000, 100000, n).astype(float)
    
    base_ts = 1704067200000000000
    interval_ns = 3600000000000
    timestamp_ns = [base_ts + i * interval_ns for i in range(n)]
    
    return pd.DataFrame({
        "Timestamp": timestamp_ns,
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


def get_nan_indices(series: pd.Series) -> list[int]:
    """Return list of indices where the series has NaN values."""
    return [int(i) for i in range(len(series)) if pd.isna(series.iloc[i]) or math.isnan(series.iloc[i])]


def to_json_safe(values: list) -> list:
    """Convert values to JSON-safe format, replacing NaN with null."""
    result = []
    for v in values:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or pd.isna(v):
                result.append(None)
            elif math.isinf(v):
                result.append(None)
            else:
                result.append(float(v))
        elif isinstance(v, (int, np.integer)):
            result.append(int(v))
        else:
            result.append(v)
    return result


def save_fixture(filename: str, data: dict, output_dir: Path):
    """Save fixture data to a JSON file."""
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Generated: {filename}")


def create_input_dict(df: pd.DataFrame) -> dict:
    """Create the input dictionary for a fixture."""
    return {
        "timestamp_ns": [int(ts) for ts in df["Timestamp"].tolist()],
        "open": to_json_safe(df["Open"].tolist()),
        "high": to_json_safe(df["High"].tolist()),
        "low": to_json_safe(df["Low"].tolist()),
        "close": to_json_safe(df["Close"].tolist()),
        "volume": to_json_safe(df["Volume"].tolist()),
    }


def generate_rsi_fixture(df: pd.DataFrame, period: int, output_dir: Path):
    """Generate RSI fixture."""
    result = rsi_mod.compute(df, price_col="Close", period=period)
    col_name = f"RSI_{period}"
    values = result[col_name]
    
    fixture = {
        "input": create_input_dict(df),
        "params": {"period": period},
        "output": {
            "values": to_json_safe(values.tolist()),
            "nan_indices": get_nan_indices(values),
        }
    }
    save_fixture(f"rsi_{period}.json", fixture, output_dir)


def generate_cci_fixture(df: pd.DataFrame, period: int, output_dir: Path):
    """Generate CCI fixture."""
    result = cci_mod.compute(df, period=period)
    col_name = f"CCI_{period}"
    values = result[col_name]
    
    fixture = {
        "input": create_input_dict(df),
        "params": {"period": period},
        "output": {
            "values": to_json_safe(values.tolist()),
            "nan_indices": get_nan_indices(values),
        }
    }
    save_fixture(f"cci_{period}.json", fixture, output_dir)


def generate_adx_fixture(df: pd.DataFrame, period: int, output_dir: Path):
    """Generate ADX fixture."""
    result = adx_mod.compute(df, period=period)
    col_name = f"ADX_{period}"
    values = result[col_name]
    
    fixture = {
        "input": create_input_dict(df),
        "params": {"period": period},
        "output": {
            "values": to_json_safe(values.tolist()),
            "nan_indices": get_nan_indices(values),
        }
    }
    save_fixture(f"adx_{period}.json", fixture, output_dir)


def generate_macd_fixture(df: pd.DataFrame, fast: int, slow: int, signal: int, output_dir: Path):
    """Generate MACD fixture."""
    result = macd_mod.compute(df, price_col="Close", fast=fast, slow=slow, signal=signal)
    
    macd_values = result["MACD"]
    signal_values = result["MACD_signal"]
    hist_values = result["MACD_hist"]
    
    fixture = {
        "input": create_input_dict(df),
        "params": {
            "fast": fast,
            "slow": slow,
            "signal": signal,
        },
        "output": {
            "macd": to_json_safe(macd_values.tolist()),
            "macd_nan_indices": get_nan_indices(macd_values),
            "signal": to_json_safe(signal_values.tolist()),
            "signal_nan_indices": get_nan_indices(signal_values),
            "hist": to_json_safe(hist_values.tolist()),
            "hist_nan_indices": get_nan_indices(hist_values),
        }
    }
    save_fixture(f"macd_{fast}_{slow}_{signal}.json", fixture, output_dir)


def generate_ma_fixture(df: pd.DataFrame, kind: str, period: int, output_dir: Path):
    """Generate moving average fixture."""
    result = ma_mod.compute(df, price_col="Close", periods=(period,), kinds=(kind,))
    col_name = f"{kind}_{period}"
    values = result[col_name]
    
    fixture = {
        "input": create_input_dict(df),
        "params": {
            "kind": kind,
            "period": period,
        },
        "output": {
            "values": to_json_safe(values.tolist()),
            "nan_indices": get_nan_indices(values),
        }
    }
    save_fixture(f"{kind.lower()}_{period}.json", fixture, output_dir)


def generate_ichimoku_fixture(df: pd.DataFrame, tenkan: int, kijun: int, senkou_b: int, output_dir: Path):
    """Generate Ichimoku Cloud fixture."""
    result = ichimoku_mod.compute(df, tenkan=tenkan, kijun=kijun, senkou_b=senkou_b)
    
    fixture = {
        "input": create_input_dict(df),
        "params": {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_b_period": senkou_b,
        },
        "output": {
            "tenkan": to_json_safe(result["Tenkan"].tolist()),
            "tenkan_nan_indices": get_nan_indices(result["Tenkan"]),
            "kijun": to_json_safe(result["Kijun"].tolist()),
            "kijun_nan_indices": get_nan_indices(result["Kijun"]),
            "senkou_a": to_json_safe(result["SenkouA"].tolist()),
            "senkou_a_nan_indices": get_nan_indices(result["SenkouA"]),
            "senkou_b": to_json_safe(result["SenkouB"].tolist()),
            "senkou_b_nan_indices": get_nan_indices(result["SenkouB"]),
            "chikou": to_json_safe(result["Chikou"].tolist()),
            "chikou_nan_indices": get_nan_indices(result["Chikou"]),
        }
    }
    save_fixture(f"ichimoku_{tenkan}_{kijun}_{senkou_b}.json", fixture, output_dir)


def generate_bollinger_fixture(df: pd.DataFrame, period: int, nbdev: float, output_dir: Path):
    """Generate Bollinger Bands fixture."""
    result = bb_mod.compute(df, price_col="Close", period=period, nbdev=nbdev)
    
    fixture = {
        "input": create_input_dict(df),
        "params": {
            "period": period,
            "nbdev": nbdev,
        },
        "output": {
            "upper": to_json_safe(result["BB_upper"].tolist()),
            "upper_nan_indices": get_nan_indices(result["BB_upper"]),
            "middle": to_json_safe(result["BB_middle"].tolist()),
            "middle_nan_indices": get_nan_indices(result["BB_middle"]),
            "lower": to_json_safe(result["BB_lower"].tolist()),
            "lower_nan_indices": get_nan_indices(result["BB_lower"]),
        }
    }
    save_fixture(f"bollinger_{period}_{int(nbdev)}.json", fixture, output_dir)


def generate_fibonacci_fixture(df: pd.DataFrame, window: int, output_dir: Path):
    """Generate Fibonacci retracements fixture."""
    result = fr_mod.compute(df, window=window)
    
    fixture = {
        "input": create_input_dict(df),
        "params": {
            "window": window,
        },
        "output": {
            "fr_000": to_json_safe(result["FR_000"].tolist()),
            "fr_000_nan_indices": get_nan_indices(result["FR_000"]),
            "fr_236": to_json_safe(result["FR_236"].tolist()),
            "fr_236_nan_indices": get_nan_indices(result["FR_236"]),
            "fr_382": to_json_safe(result["FR_382"].tolist()),
            "fr_382_nan_indices": get_nan_indices(result["FR_382"]),
            "fr_500": to_json_safe(result["FR_500"].tolist()),
            "fr_500_nan_indices": get_nan_indices(result["FR_500"]),
            "fr_618": to_json_safe(result["FR_618"].tolist()),
            "fr_618_nan_indices": get_nan_indices(result["FR_618"]),
            "fr_786": to_json_safe(result["FR_786"].tolist()),
            "fr_786_nan_indices": get_nan_indices(result["FR_786"]),
            "fr_1000": to_json_safe(result["FR_1000"].tolist()),
            "fr_1000_nan_indices": get_nan_indices(result["FR_1000"]),
        }
    }
    save_fixture(f"fibonacci_{window}.json", fixture, output_dir)


def generate_double_bottom_fixture(df: pd.DataFrame, output_dir: Path):
    """Generate double bottom pattern fixture."""
    patterns_df, latest_min, latest_max = db_mod.detect_double_bottoms(
        df, window=5, tolerance_pct=0.3, min_width=5
    )
    
    patterns = []
    for _, row in patterns_df.iterrows():
        pattern = {
            "idx1": int(row["idx1"]),
            "idx2": int(row["idx2"]),
            "ts1": row["ts1"],
            "ts2": row["ts2"],
            "low1": float(row["low1"]) if pd.notna(row["low1"]) else None,
            "low2": float(row["low2"]) if pd.notna(row["low2"]) else None,
            "neckline": float(row["neckline"]) if pd.notna(row["neckline"]) else None,
            "neckline_idx": int(row["neckline_idx"]),
            "depth_pct": float(row["depth_pct"]) if pd.notna(row["depth_pct"]) else None,
            "width_bars": int(row["width_bars"]),
            "confirmed": bool(row["confirmed"]),
            "min_before_val": float(row["min_before_val"]) if pd.notna(row["min_before_val"]) else None,
            "min_before_ts": row["min_before_ts"] if pd.notna(row["min_before_ts"]) else None,
            "max_before_val": float(row["max_before_val"]) if pd.notna(row["max_before_val"]) else None,
            "max_before_ts": row["max_before_ts"] if pd.notna(row["max_before_ts"]) else None,
            "min_after_val": float(row["min_after_val"]) if pd.notna(row["min_after_val"]) else None,
            "min_after_ts": row["min_after_ts"] if pd.notna(row["min_after_ts"]) else None,
            "max_after_val": float(row["max_after_val"]) if pd.notna(row["max_after_val"]) else None,
            "max_after_ts": row["max_after_ts"] if pd.notna(row["max_after_ts"]) else None,
        }
        patterns.append(pattern)
    
    fixture = {
        "input": create_input_dict(df),
        "params": {
            "window": 5,
            "tolerance_pct": 0.3,
            "min_width": 5,
        },
        "output": {
            "patterns": patterns,
            "latest_min": float(latest_min) if latest_min is not None else None,
            "latest_max": float(latest_max) if latest_max is not None else None,
        }
    }
    save_fixture("double_bottom.json", fixture, output_dir)


def generate_double_top_fixture(df: pd.DataFrame, output_dir: Path):
    """Generate double top pattern fixture."""
    patterns_df, latest_min, latest_max = dt_mod.detect_double_tops(
        df, window=5, tolerance_pct=0.3, min_width=5
    )
    
    patterns = []
    for _, row in patterns_df.iterrows():
        pattern = {
            "idx1": int(row["idx1"]),
            "idx2": int(row["idx2"]),
            "ts1": row["ts1"],
            "ts2": row["ts2"],
            "high1": float(row["high1"]) if pd.notna(row["high1"]) else None,
            "high2": float(row["high2"]) if pd.notna(row["high2"]) else None,
            "neckline": float(row["neckline"]) if pd.notna(row["neckline"]) else None,
            "neckline_idx": int(row["neckline_idx"]),
            "depth_pct": float(row["depth_pct"]) if pd.notna(row["depth_pct"]) else None,
            "width_bars": int(row["width_bars"]),
            "confirmed": bool(row["confirmed"]),
            "min_before_val": float(row["min_before_val"]) if pd.notna(row["min_before_val"]) else None,
            "min_before_ts": row["min_before_ts"] if pd.notna(row["min_before_ts"]) else None,
            "max_before_val": float(row["max_before_val"]) if pd.notna(row["max_before_val"]) else None,
            "max_before_ts": row["max_before_ts"] if pd.notna(row["max_before_ts"]) else None,
            "min_after_val": float(row["min_after_val"]) if pd.notna(row["min_after_val"]) else None,
            "min_after_ts": row["min_after_ts"] if pd.notna(row["min_after_ts"]) else None,
            "max_after_val": float(row["max_after_val"]) if pd.notna(row["max_after_val"]) else None,
            "max_after_ts": row["max_after_ts"] if pd.notna(row["max_after_ts"]) else None,
        }
        patterns.append(pattern)
    
    fixture = {
        "input": create_input_dict(df),
        "params": {
            "window": 5,
            "tolerance_pct": 0.3,
            "min_width": 5,
        },
        "output": {
            "patterns": patterns,
            "latest_min": float(latest_min) if latest_min is not None else None,
            "latest_max": float(latest_max) if latest_max is not None else None,
        }
    }
    save_fixture("double_top.json", fixture, output_dir)


def main():
    """Generate all fixtures."""
    output_dir = Path(__file__).parent
    
    print("Generating TA indicator fixtures...")
    print(f"Output directory: {output_dir}")
    print()
    
    print("Generating synthetic OHLCV data (200 bars)...")
    df = generate_synthetic_ohlcv(n=200, seed=42)
    print(f"  Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
    print()
    
    print("Generating momentum indicator fixtures...")
    generate_rsi_fixture(df, period=14, output_dir=output_dir)
    generate_cci_fixture(df, period=14, output_dir=output_dir)
    print()
    
    print("Generating trend indicator fixtures...")
    generate_adx_fixture(df, period=14, output_dir=output_dir)
    generate_macd_fixture(df, fast=12, slow=26, signal=9, output_dir=output_dir)
    print()
    
    print("Generating moving average fixtures...")
    generate_ma_fixture(df, kind="SMA", period=10, output_dir=output_dir)
    generate_ma_fixture(df, kind="EMA", period=20, output_dir=output_dir)
    generate_ma_fixture(df, kind="WMA", period=50, output_dir=output_dir)
    generate_ma_fixture(df, kind="DEMA", period=10, output_dir=output_dir)
    generate_ma_fixture(df, kind="TEMA", period=20, output_dir=output_dir)
    generate_ma_fixture(df, kind="KAMA", period=10, output_dir=output_dir)
    generate_ma_fixture(df, kind="TRIMA", period=20, output_dir=output_dir)
    print()
    
    print("Generating Ichimoku Cloud fixture...")
    generate_ichimoku_fixture(df, tenkan=9, kijun=26, senkou_b=52, output_dir=output_dir)
    print()
    
    print("Generating volatility indicator fixtures...")
    generate_bollinger_fixture(df, period=20, nbdev=2.0, output_dir=output_dir)
    print()
    
    print("Generating support indicator fixtures...")
    generate_fibonacci_fixture(df, window=50, output_dir=output_dir)
    print()
    
    print("Generating pattern detection fixtures...")
    generate_double_bottom_fixture(df, output_dir=output_dir)
    generate_double_top_fixture(df, output_dir=output_dir)
    print()
    
    print("All fixtures generated successfully!")


if __name__ == "__main__":
    main()
