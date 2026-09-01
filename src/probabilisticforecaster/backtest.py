"""Backtesting engine for the Transformer Probabilistic Forex Forecaster.

Simulates trading over a test period using model predictions and a trading strategy,
computing daily PnL, annualised return, Sharpe ratio, maximum drawdown, and hourly
PnL contribution analysis.

Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from probabilisticforecaster.config import ForecasterConfig
from probabilisticforecaster.strategy import TradingStrategy


@dataclass
class BacktestResult:
    """Results from a backtest run.

    Attributes:
        daily_pnl: Series indexed by date with daily PnL values (sum of intraday PnLs).
        annualised_return: Total return divided by number of years in the test period.
        sharpe_ratio: mean(daily_returns) / std(daily_returns) × sqrt(252).
        max_drawdown: Largest peak-to-trough decline in cumulative PnL.
        hourly_pnl: DataFrame with PnL contribution analysis grouped by hour of day.
    """

    daily_pnl: pd.Series
    annualised_return: float
    sharpe_ratio: float
    max_drawdown: float
    hourly_pnl: pd.DataFrame


def run_backtest(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    strategy: TradingStrategy,
    config: ForecasterConfig,
) -> BacktestResult:
    """Run a backtest over the test period using model predictions and a trading strategy.

    The backtest loop for each 5-minute bar:
    1. Model prediction (μ̂, σ̂) is used to compute position size via the strategy.
    2. PnL for the bar is: position_t × (close_{t+1} - close_t) / close_t
    3. Intraday PnLs are aggregated to daily PnL.

    Args:
        predictions: DataFrame with columns ['timestamp', 'mu', 'sigma'].
            Each row is a model prediction at a given timestamp.
        prices: DataFrame with columns ['timestamp', 'close'].
            Price data aligned with predictions (must have one extra row for final close).
        strategy: A TradingStrategy instance that computes position sizes.
        config: ForecasterConfig with position_size and risk_aversion parameters.

    Returns:
        BacktestResult with daily PnL, annualised return, Sharpe ratio,
        max drawdown, and hourly PnL analysis.

    Raises:
        ValueError: If predictions is empty or timestamps do not align.
    """
    if predictions.empty:
        raise ValueError("No predictions to backtest")

    pred_timestamps = pd.to_datetime(predictions["timestamp"])
    price_timestamps = pd.to_datetime(prices["timestamp"])

    pred_set = set(pred_timestamps)
    price_set = set(price_timestamps)

    if not pred_set.issubset(price_set):
        raise ValueError("Prediction and price timestamps do not align")

    price_lookup = pd.Series(
        prices["close"].values, index=price_timestamps
    )

    predictions_sorted = predictions.copy()
    predictions_sorted["timestamp"] = pred_timestamps
    predictions_sorted = predictions_sorted.sort_values("timestamp").reset_index(drop=True)

    prices_sorted = prices.copy()
    prices_sorted["timestamp"] = price_timestamps
    prices_sorted = prices_sorted.sort_values("timestamp").reset_index(drop=True)

    price_ts_list = prices_sorted["timestamp"].tolist()
    price_ts_to_idx = {ts: idx for idx, ts in enumerate(price_ts_list)}

    pnl_records = []

    for _, row in predictions_sorted.iterrows():
        ts = row["timestamp"]
        mu = float(row["mu"])
        sigma = float(row["sigma"])

        idx = price_ts_to_idx.get(ts)
        if idx is None:
            continue

        next_idx = idx + 1
        if next_idx >= len(prices_sorted):
            continue

        close_t = float(prices_sorted.iloc[idx]["close"])
        close_t1 = float(prices_sorted.iloc[next_idx]["close"])

        if close_t == 0.0:
            continue

        position = strategy.compute_position(mu, sigma, config)

        actual_return = (close_t1 - close_t) / close_t
        pnl = position * actual_return

        pnl_records.append(
            {
                "timestamp": ts,
                "pnl": pnl,
                "position": position,
                "actual_return": actual_return,
            }
        )

    if not pnl_records:
        return BacktestResult(
            daily_pnl=pd.Series(dtype=float),
            annualised_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            hourly_pnl=pd.DataFrame(columns=["hour", "total_pnl", "mean_pnl", "count"]),
        )

    pnl_df = pd.DataFrame(pnl_records)
    pnl_df["timestamp"] = pd.to_datetime(pnl_df["timestamp"])
    pnl_df["date"] = pnl_df["timestamp"].dt.date
    pnl_df["hour"] = pnl_df["timestamp"].dt.hour

    daily_pnl = pnl_df.groupby("date")["pnl"].sum()
    daily_pnl.index = pd.to_datetime(daily_pnl.index)
    daily_pnl.index.name = "date"

    total_return = daily_pnl.sum()
    num_days = (daily_pnl.index.max() - daily_pnl.index.min()).days
    num_years = max(num_days / 365.25, 1.0 / 365.25)
    annualised_return = total_return / num_years

    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe_ratio = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    max_drawdown = _compute_max_drawdown(daily_pnl)

    hourly_pnl = (
        pnl_df.groupby("hour")["pnl"]
        .agg(total_pnl="sum", mean_pnl="mean", count="count")
        .reset_index()
    )

    return BacktestResult(
        daily_pnl=daily_pnl,
        annualised_return=annualised_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        hourly_pnl=hourly_pnl,
    )


def _compute_max_drawdown(daily_pnl: pd.Series) -> float:
    """Compute maximum drawdown from a daily PnL series.

    Maximum drawdown is the largest peak-to-trough decline in the cumulative
    sum of daily PnL values.

    Args:
        daily_pnl: Series of daily PnL values.

    Returns:
        Maximum drawdown as a positive float (magnitude of largest decline).
        Returns 0.0 if the series is empty or monotonically increasing.
    """
    if daily_pnl.empty:
        return 0.0

    cumulative = daily_pnl.cumsum()
    running_max = cumulative.cummax()
    drawdowns = running_max - cumulative

    max_dd = drawdowns.max()
    return float(max_dd) if max_dd > 0 else 0.0
