//! Trend indicators: ADX, MACD, Moving Averages, Ichimoku Cloud
//!
//! These indicators identify the direction and strength of price trends.

use crate::indicators::IndicatorError;
use std::collections::HashMap;
use talib_rs::momentum::adx as talib_adx;
use talib_rs::momentum::macd as talib_macd;
use talib_rs::overlap::{dema, ema, kama, sma, tema, trima, wma};

/// Output structure for Ichimoku Cloud indicator.
///
/// Contains five vectors aligned index-for-index with the input:
/// - `tenkan`: Tenkan-sen (Conversion Line) - (9-period high + 9-period low) / 2
/// - `kijun`: Kijun-sen (Base Line) - (26-period high + 26-period low) / 2
/// - `senkou_a`: Senkou Span A - (Tenkan + Kijun) / 2, shifted forward by kijun periods
/// - `senkou_b`: Senkou Span B - (52-period high + 52-period low) / 2, shifted forward by kijun periods
/// - `chikou`: Chikou Span (Lagging Span) - Close price shifted backward by kijun periods
#[derive(Debug, Clone, PartialEq)]
pub struct IchimokuOutput {
    /// Tenkan-sen (Conversion Line): (period-high + period-low) / 2
    pub tenkan: Vec<f64>,
    /// Kijun-sen (Base Line): (period-high + period-low) / 2
    pub kijun: Vec<f64>,
    /// Senkou Span A: (Tenkan + Kijun) / 2, shifted forward by kijun periods
    pub senkou_a: Vec<f64>,
    /// Senkou Span B: (period-high + period-low) / 2, shifted forward by kijun periods
    pub senkou_b: Vec<f64>,
    /// Chikou Span (Lagging Span): Close price shifted backward by kijun periods
    pub chikou: Vec<f64>,
}

/// Compute Ichimoku Cloud (Ichimoku Kinko Hyo) indicator.
///
/// Returns an `IchimokuOutput` struct containing five vectors of length `n`:
/// - `tenkan`: Tenkan-sen (Conversion Line)
/// - `kijun`: Kijun-sen (Base Line)
/// - `senkou_a`: Senkou Span A (shifted forward)
/// - `senkou_b`: Senkou Span B (shifted forward)
/// - `chikou`: Chikou Span (Lagging Span)
///
/// # Arguments
///
/// * `high` - Slice of high prices, ordered oldest to newest
/// * `low` - Slice of low prices, ordered oldest to newest
/// * `close` - Slice of closing prices, ordered oldest to newest
/// * `tenkan_period` - The Tenkan-sen lookback period (typically 9)
/// * `kijun_period` - The Kijun-sen lookback period (typically 26)
/// * `senkou_b_period` - The Senkou Span B lookback period (typically 52)
///
/// # Returns
///
/// A `Result<IchimokuOutput, IndicatorError>` where each vector is aligned
/// index-for-index with the input slices.
///
/// # Errors
///
/// - `IndicatorError::LengthMismatch` if `high`, `low`, and `close` have different lengths
/// - `IndicatorError::InvalidPeriod` if any period is 0
///
/// # Computation Details
///
/// - `tenkan[i] = (max(high[i-tenkan+1..=i]) + min(low[i-tenkan+1..=i])) / 2`
///   for `i >= tenkan - 1`, NaN otherwise
/// - `kijun[i]` computed analogously with `kijun_period` window
/// - `senkou_a[i] = (tenkan[i-kijun] + kijun[i-kijun]) / 2` for `i >= kijun`
///   where both values are finite, NaN otherwise
/// - `senkou_b[i]` computed with `senkou_b_period` window shifted by `kijun_period`
/// - `chikou[i] = close[i + kijun]` when `i + kijun < n`, NaN otherwise
///
/// # Edge Cases
///
/// - If any period exceeds `n`, the affected line is all-NaN (no error returned)
/// - If all inputs are empty, returns `Ok` with empty vectors
///
/// # Example
///
/// ```
/// use modelenv_core::indicators::trend::ichimoku;
///
/// let high = vec![45.0, 45.5, 46.0, 45.5, 46.0, 45.5, 45.0, 44.5, 45.0, 45.5,
///                 46.0, 45.5, 46.0, 45.5, 45.0, 44.5, 45.0, 45.5, 46.0, 45.5,
///                 46.0, 45.5, 46.0, 45.5, 45.0, 44.5, 45.0, 45.5, 46.0, 45.5];
/// let low = vec![44.0, 44.5, 45.0, 44.5, 45.0, 44.5, 44.0, 43.5, 44.0, 44.5,
///                45.0, 44.5, 45.0, 44.5, 44.0, 43.5, 44.0, 44.5, 45.0, 44.5,
///                45.0, 44.5, 45.0, 44.5, 44.0, 43.5, 44.0, 44.5, 45.0, 44.5];
/// let close = vec![44.5, 45.0, 45.5, 45.0, 45.5, 45.0, 44.5, 44.0, 44.5, 45.0,
///                  45.5, 45.0, 45.5, 45.0, 44.5, 44.0, 44.5, 45.0, 45.5, 45.0,
///                  45.5, 45.0, 45.5, 45.0, 44.5, 44.0, 44.5, 45.0, 45.5, 45.0];
/// let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();
/// assert_eq!(result.tenkan.len(), close.len());
/// assert_eq!(result.kijun.len(), close.len());
/// assert_eq!(result.senkou_a.len(), close.len());
/// assert_eq!(result.senkou_b.len(), close.len());
/// assert_eq!(result.chikou.len(), close.len());
/// ```
pub fn ichimoku(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    tenkan_period: usize,
    kijun_period: usize,
    senkou_b_period: usize,
) -> Result<IchimokuOutput, IndicatorError> {
    let n = close.len();

        if high.len() != n {
        return Err(IndicatorError::LengthMismatch {
            expected: n,
            actual: high.len(),
            param_name: "high",
        });
    }
    if low.len() != n {
        return Err(IndicatorError::LengthMismatch {
            expected: n,
            actual: low.len(),
            param_name: "low",
        });
    }

        if tenkan_period == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "tenkan",
            value: 0,
            reason: "tenkan period must be greater than 0",
        });
    }
    if kijun_period == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "kijun",
            value: 0,
            reason: "kijun period must be greater than 0",
        });
    }
    if senkou_b_period == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "senkou_b_period",
            value: 0,
            reason: "senkou_b_period must be greater than 0",
        });
    }

        if n == 0 {
        return Ok(IchimokuOutput {
            tenkan: Vec::new(),
            kijun: Vec::new(),
            senkou_a: Vec::new(),
            senkou_b: Vec::new(),
            chikou: Vec::new(),
        });
    }

            let tenkan_line = compute_midpoint_line(high, low, tenkan_period);

            let kijun_line = compute_midpoint_line(high, low, kijun_period);

                let mut senkou_a = vec![f64::NAN; n];
    for i in kijun_period..n {
        let src_idx = i - kijun_period;
        let tenkan_val = tenkan_line[src_idx];
        let kijun_val = kijun_line[src_idx];
        if tenkan_val.is_finite() && kijun_val.is_finite() {
            senkou_a[i] = (tenkan_val + kijun_val) / 2.0;
        }
    }

                    let senkou_b_unshifted = compute_midpoint_line(high, low, senkou_b_period);
    let mut senkou_b = vec![f64::NAN; n];
    for i in kijun_period..n {
        let src_idx = i - kijun_period;
        senkou_b[i] = senkou_b_unshifted[src_idx];
    }

                    let mut chikou = vec![f64::NAN; n];
    for i in 0..n {
        if i + kijun_period < n {
            chikou[i] = close[i + kijun_period];
        }
    }

    Ok(IchimokuOutput {
        tenkan: tenkan_line,
        kijun: kijun_line,
        senkou_a,
        senkou_b,
        chikou,
    })
}

/// Helper function to compute the midpoint line (rolling max + rolling min) / 2.
///
/// This is used for Tenkan-sen, Kijun-sen, and the unshifted Senkou Span B.
///
/// Returns a vector of length `n` where:
/// - Indices `[0, period - 1)` are NaN (insufficient data)
/// - Indices `[period - 1, n)` contain (rolling_max + rolling_min) / 2
fn compute_midpoint_line(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];

    if period == 0 || n == 0 {
        return result;
    }

        for i in (period - 1)..n {
        let start = i + 1 - period;
        let end = i + 1;

        let mut max_high = f64::NEG_INFINITY;
        let mut min_low = f64::INFINITY;

        for j in start..end {
            let h = high[j];
            let l = low[j];

            if h.is_nan() || l.is_nan() {
                                max_high = f64::NAN;
                min_low = f64::NAN;
                break;
            }

            if h > max_high {
                max_high = h;
            }
            if l < min_low {
                min_low = l;
            }
        }

        if max_high.is_finite() && min_low.is_finite() {
            result[i] = (max_high + min_low) / 2.0;
        }
    }

    result
}

/// Enumeration of supported moving average types.
///
/// Each variant corresponds to a different moving average algorithm:
/// - `Sma`: Simple Moving Average
/// - `Ema`: Exponential Moving Average
/// - `Wma`: Weighted Moving Average
/// - `Dema`: Double Exponential Moving Average
/// - `Tema`: Triple Exponential Moving Average
/// - `Kama`: Kaufman Adaptive Moving Average
/// - `Trima`: Triangular Moving Average
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MovingAverageKind {
    /// Simple Moving Average - arithmetic mean of the last `period` values
    Sma,
    /// Exponential Moving Average - weighted average with exponentially decreasing weights
    Ema,
    /// Weighted Moving Average - linear weighted average giving more weight to recent values
    Wma,
    /// Double Exponential Moving Average - reduces lag by applying EMA twice
    Dema,
    /// Triple Exponential Moving Average - further reduces lag by applying EMA three times
    Tema,
    /// Kaufman Adaptive Moving Average - adapts to market volatility
    Kama,
    /// Triangular Moving Average - double-smoothed SMA
    Trima,
}

impl std::fmt::Display for MovingAverageKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MovingAverageKind::Sma => write!(f, "SMA"),
            MovingAverageKind::Ema => write!(f, "EMA"),
            MovingAverageKind::Wma => write!(f, "WMA"),
            MovingAverageKind::Dema => write!(f, "DEMA"),
            MovingAverageKind::Tema => write!(f, "TEMA"),
            MovingAverageKind::Kama => write!(f, "KAMA"),
            MovingAverageKind::Trima => write!(f, "TRIMA"),
        }
    }
}

/// Default moving average kinds used by the Python `ta/trend/movingavg.py` module.
///
/// Contains `[Sma, Ema]` as the default kinds.
pub const DEFAULT_MA_KINDS: &[MovingAverageKind] = &[MovingAverageKind::Sma, MovingAverageKind::Ema];

/// Default moving average periods used by the Python `ta/trend/movingavg.py` module.
///
/// Contains `[10, 20, 50]` as the default periods.
pub const DEFAULT_MA_PERIODS: &[usize] = &[10, 20, 50];

/// Output structure for MACD indicator.
///
/// Contains three vectors aligned index-for-index with the input:
/// - `macd`: The MACD line (fast EMA - slow EMA)
/// - `signal`: The signal line (EMA of MACD line)
/// - `hist`: The histogram (MACD line - signal line)
#[derive(Debug, Clone, PartialEq)]
pub struct MacdOutput {
    /// MACD line: fast EMA - slow EMA
    pub macd: Vec<f64>,
    /// Signal line: EMA of MACD line
    pub signal: Vec<f64>,
    /// Histogram: MACD line - signal line
    pub hist: Vec<f64>,
}

/// Compute Moving Average Convergence/Divergence (MACD).
///
/// Returns a `MacdOutput` struct containing three vectors of length `close.len()`:
/// - `macd`: The MACD line (fast EMA - slow EMA)
/// - `signal`: The signal line (EMA of MACD line)
/// - `hist`: The histogram (MACD line - signal line)
///
/// Leading elements up to index `slow + signal - 2` are NaN.
///
/// # Arguments
///
/// * `close` - Slice of closing prices, ordered oldest to newest
/// * `fast` - The fast EMA period (typically 12)
/// * `slow` - The slow EMA period (typically 26)
/// * `signal` - The signal line EMA period (typically 9)
///
/// # Returns
///
/// A `Result<MacdOutput, IndicatorError>` where each vector is aligned index-for-index
/// with `close`. Leading `slow + signal - 2` elements are `f64::NAN`.
///
/// # Errors
///
/// - `IndicatorError::InvalidPeriod` if `fast == 0`, `slow == 0`, `signal == 0`, or `fast >= slow`
///
/// # Edge Cases
///
/// - If `close.len() < slow + signal - 1`: returns `Ok` with all-NaN vectors
/// - If `close` is empty: returns `Ok` with empty vectors
///
/// # Invariants
///
/// For all finite indices `i`: `|hist[i] - (macd[i] - signal[i])| <= 1e-9`
///
/// # Example
///
/// ```
/// use modelenv_core::indicators::trend::macd;
///
/// let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
/// let result = macd(&close, 12, 26, 9).unwrap();
/// assert_eq!(result.macd.len(), close.len());
/// assert_eq!(result.signal.len(), close.len());
/// assert_eq!(result.hist.len(), close.len());
/// // First slow + signal - 2 = 33 elements are NaN
/// assert!(result.macd[32].is_nan());
/// assert!(!result.macd[33].is_nan());
/// ```
pub fn macd(
    close: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
) -> Result<MacdOutput, IndicatorError> {
    let n = close.len();

        if fast == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "fast",
            value: 0,
            reason: "fast period must be greater than 0",
        });
    }
    if slow == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "slow",
            value: 0,
            reason: "slow period must be greater than 0",
        });
    }
    if signal == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "signal",
            value: 0,
            reason: "signal period must be greater than 0",
        });
    }
    if fast >= slow {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "fast",
            value: fast,
            reason: "fast period must be less than slow period",
        });
    }

        if n == 0 {
        return Ok(MacdOutput {
            macd: Vec::new(),
            signal: Vec::new(),
            hist: Vec::new(),
        });
    }

            let min_required = slow + signal - 1;
    if n < min_required {
        return Ok(MacdOutput {
            macd: vec![f64::NAN; n],
            signal: vec![f64::NAN; n],
            hist: vec![f64::NAN; n],
        });
    }

            if fast < 2 || slow < 2 {
                        return Ok(MacdOutput {
            macd: vec![f64::NAN; n],
            signal: vec![f64::NAN; n],
            hist: vec![f64::NAN; n],
        });
    }

        match talib_macd(close, fast, slow, signal) {
        Ok((macd_line, signal_line, histogram)) => {
                                    if macd_line.len() == n && signal_line.len() == n && histogram.len() == n {
                Ok(MacdOutput {
                    macd: macd_line,
                    signal: signal_line,
                    hist: histogram,
                })
            } else {
                                                let mut macd_out = vec![f64::NAN; n];
                let mut signal_out = vec![f64::NAN; n];
                let mut hist_out = vec![f64::NAN; n];

                let start_idx = n.saturating_sub(macd_line.len());
                for (i, &val) in macd_line.iter().enumerate() {
                    if start_idx + i < n {
                        macd_out[start_idx + i] = val;
                    }
                }

                let start_idx = n.saturating_sub(signal_line.len());
                for (i, &val) in signal_line.iter().enumerate() {
                    if start_idx + i < n {
                        signal_out[start_idx + i] = val;
                    }
                }

                let start_idx = n.saturating_sub(histogram.len());
                for (i, &val) in histogram.iter().enumerate() {
                    if start_idx + i < n {
                        hist_out[start_idx + i] = val;
                    }
                }

                Ok(MacdOutput {
                    macd: macd_out,
                    signal: signal_out,
                    hist: hist_out,
                })
            }
        }
        Err(talib_rs::error::TaError::InsufficientData { .. }) => {
                        Ok(MacdOutput {
                macd: vec![f64::NAN; n],
                signal: vec![f64::NAN; n],
                hist: vec![f64::NAN; n],
            })
        }
        Err(talib_rs::error::TaError::InvalidParameter { name, reason, .. }) => {
                        Err(IndicatorError::InvalidPeriod {
                param_name: name,
                value: fast,
                reason,
            })
        }
        Err(_) => {
                        Ok(MacdOutput {
                macd: vec![f64::NAN; n],
                signal: vec![f64::NAN; n],
                hist: vec![f64::NAN; n],
            })
        }
    }
}

/// Compute Average Directional Index (ADX).
///
/// Returns a vector of length `close.len()` where indices `[0, 2*period - 1)` are NaN
/// and indices `[2*period - 1, n)` contain ADX values in `[0.0, 100.0]`.
///
/// # Arguments
///
/// * `high` - Slice of high prices, ordered oldest to newest
/// * `low` - Slice of low prices, ordered oldest to newest
/// * `close` - Slice of closing prices, ordered oldest to newest
/// * `period` - The lookback period for ADX calculation
///
/// # Returns
///
/// A `Result<Vec<f64>, IndicatorError>` aligned index-for-index with the input slices.
/// Leading `2*period - 1` elements are `f64::NAN`, and subsequent elements are ADX values
/// in `[0.0, 100.0]`.
///
/// # Errors
///
/// - `IndicatorError::LengthMismatch` if `high`, `low`, and `close` have different lengths
/// - `IndicatorError::InvalidPeriod` if `period == 0`
///
/// # Edge Cases
///
/// - If `close.len() < 2*period - 1`: returns `Ok` with all-NaN vector
/// - If all inputs are empty: returns `Ok` with empty vector
///
/// # Example
///
/// ```
/// use modelenv_core::indicators::trend::adx;
///
/// let high = vec![45.0, 45.5, 46.0, 45.5, 46.0, 45.5, 45.0, 44.5, 45.0, 45.5,
///                 46.0, 45.5, 46.0, 45.5, 45.0, 44.5, 45.0, 45.5, 46.0, 45.5];
/// let low = vec![44.0, 44.5, 45.0, 44.5, 45.0, 44.5, 44.0, 43.5, 44.0, 44.5,
///                45.0, 44.5, 45.0, 44.5, 44.0, 43.5, 44.0, 44.5, 45.0, 44.5];
/// let close = vec![44.5, 45.0, 45.5, 45.0, 45.5, 45.0, 44.5, 44.0, 44.5, 45.0,
///                  45.5, 45.0, 45.5, 45.0, 44.5, 44.0, 44.5, 45.0, 45.5, 45.0];
/// let result = adx(&high, &low, &close, 5).unwrap();
/// assert_eq!(result.len(), close.len());
/// // First 2*period - 1 = 9 elements are NaN
/// assert!(result[8].is_nan());
/// // ADX computed from index 9 onwards
/// assert!(!result[9].is_nan());
/// ```
pub fn adx(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Result<Vec<f64>, IndicatorError> {
    let n = close.len();

        if high.len() != n {
        return Err(IndicatorError::LengthMismatch {
            expected: n,
            actual: high.len(),
            param_name: "high",
        });
    }
    if low.len() != n {
        return Err(IndicatorError::LengthMismatch {
            expected: n,
            actual: low.len(),
            param_name: "low",
        });
    }

        if period == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "period",
            value: 0,
            reason: "period must be greater than 0",
        });
    }

        if n == 0 {
        return Ok(Vec::new());
    }

        let lookback = 2 * period - 1;

            if n < lookback {
        return Ok(vec![f64::NAN; n]);
    }

        match talib_adx(high, low, close, period) {
        Ok(result) => {
                                    if result.len() == n {
                Ok(result)
            } else {
                                                let mut output = vec![f64::NAN; n];
                let start_idx = n.saturating_sub(result.len());
                for (i, &val) in result.iter().enumerate() {
                    if start_idx + i < n {
                        output[start_idx + i] = val;
                    }
                }
                Ok(output)
            }
        }
        Err(talib_rs::error::TaError::InsufficientData { .. }) => {
                        Ok(vec![f64::NAN; n])
        }
        Err(talib_rs::error::TaError::LengthMismatch { expected, got }) => {
                        Err(IndicatorError::LengthMismatch {
                expected,
                actual: got,
                param_name: "input slices",
            })
        }
        Err(talib_rs::error::TaError::InvalidParameter { name, reason, .. }) => {
                        Err(IndicatorError::InvalidPeriod {
                param_name: name,
                value: period,
                reason,
            })
        }
        Err(_) => {
                        Ok(vec![f64::NAN; n])
        }
    }
}

/// Compute a moving average of the specified kind.
///
/// Returns a vector of length `close.len()` where leading elements are NaN
/// (the warm-up period varies by moving average type) and subsequent elements
/// contain the computed moving average values.
///
/// # Arguments
///
/// * `close` - Slice of closing prices, ordered oldest to newest
/// * `kind` - The type of moving average to compute
/// * `period` - The lookback period for the moving average
///
/// # Returns
///
/// A `Result<Vec<f64>, IndicatorError>` aligned index-for-index with `close`.
/// Leading elements are `f64::NAN` during the warm-up period.
///
/// # Errors
///
/// - `IndicatorError::InvalidPeriod` if `period == 0`
/// - `IndicatorError::EmptyInput` if `close` is empty
///
/// # Warm-up Periods by Type
///
/// - SMA: `period - 1` leading NaNs
/// - EMA: `period - 1` leading NaNs
/// - WMA: `period - 1` leading NaNs
/// - DEMA: `2 * period - 2` leading NaNs (approximately)
/// - TEMA: `3 * period - 3` leading NaNs (approximately)
/// - KAMA: `period - 1` leading NaNs
/// - TRIMA: `period - 1` leading NaNs
///
/// # NaN Propagation
///
/// If `close` contains NaN values, those NaN values will propagate into every
/// output index whose computation window includes a NaN input.
///
/// # Example
///
/// ```
/// use modelenv_core::indicators::trend::{moving_average, MovingAverageKind};
///
/// let close: Vec<f64> = (1..=20).map(|x| x as f64).collect();
/// let result = moving_average(&close, MovingAverageKind::Sma, 5).unwrap();
/// assert_eq!(result.len(), close.len());
/// // First period - 1 = 4 elements are NaN for SMA
/// assert!(result[3].is_nan());
/// assert!(!result[4].is_nan());
/// ```
pub fn moving_average(
    close: &[f64],
    kind: MovingAverageKind,
    period: usize,
) -> Result<Vec<f64>, IndicatorError> {
    let n = close.len();

        if period == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "period",
            value: 0,
            reason: "period must be greater than 0",
        });
    }

        if n == 0 {
        return Err(IndicatorError::EmptyInput);
    }

        let has_nan = close.iter().any(|&x| x.is_nan());

            if has_nan {
        return compute_ma_with_nan_propagation(close, kind, period);
    }

        let result = match kind {
        MovingAverageKind::Sma => sma(close, period),
        MovingAverageKind::Ema => ema(close, period),
        MovingAverageKind::Wma => wma(close, period),
        MovingAverageKind::Dema => dema(close, period),
        MovingAverageKind::Tema => tema(close, period),
        MovingAverageKind::Kama => kama(close, period),
        MovingAverageKind::Trima => trima(close, period),
    };

    match result {
        Ok(output) => {
                                    if output.len() == n {
                Ok(output)
            } else {
                                let mut padded = vec![f64::NAN; n];
                let start_idx = n.saturating_sub(output.len());
                for (i, &val) in output.iter().enumerate() {
                    if start_idx + i < n {
                        padded[start_idx + i] = val;
                    }
                }
                Ok(padded)
            }
        }
        Err(talib_rs::error::TaError::InsufficientData { .. }) => {
                        Ok(vec![f64::NAN; n])
        }
        Err(talib_rs::error::TaError::InvalidParameter { name, reason, .. }) => {
                        Err(IndicatorError::InvalidPeriod {
                param_name: name,
                value: period,
                reason,
            })
        }
        Err(_) => {
                        Ok(vec![f64::NAN; n])
        }
    }
}

/// Helper function to compute moving average with NaN propagation.
///
/// When the input contains NaN values, we need to propagate NaN into every
/// output index whose computation window includes a NaN input.
fn compute_ma_with_nan_propagation(
    close: &[f64],
    kind: MovingAverageKind,
    period: usize,
) -> Result<Vec<f64>, IndicatorError> {
    let n = close.len();

        
            let close_cleaned: Vec<f64> = close.iter().map(|&x| if x.is_nan() { 0.0 } else { x }).collect();

        let result = match kind {
        MovingAverageKind::Sma => sma(&close_cleaned, period),
        MovingAverageKind::Ema => ema(&close_cleaned, period),
        MovingAverageKind::Wma => wma(&close_cleaned, period),
        MovingAverageKind::Dema => dema(&close_cleaned, period),
        MovingAverageKind::Tema => tema(&close_cleaned, period),
        MovingAverageKind::Kama => kama(&close_cleaned, period),
        MovingAverageKind::Trima => trima(&close_cleaned, period),
    };

    let mut output = match result {
        Ok(output) => {
            if output.len() == n {
                output
            } else {
                let mut padded = vec![f64::NAN; n];
                let start_idx = n.saturating_sub(output.len());
                for (i, &val) in output.iter().enumerate() {
                    if start_idx + i < n {
                        padded[start_idx + i] = val;
                    }
                }
                padded
            }
        }
        Err(talib_rs::error::TaError::InsufficientData { .. }) => {
            vec![f64::NAN; n]
        }
        Err(talib_rs::error::TaError::InvalidParameter { name, reason, .. }) => {
            return Err(IndicatorError::InvalidPeriod {
                param_name: name,
                value: period,
                reason,
            });
        }
        Err(_) => {
            vec![f64::NAN; n]
        }
    };

                        
    let window_size = match kind {
        MovingAverageKind::Sma | MovingAverageKind::Wma | MovingAverageKind::Trima => period,
        MovingAverageKind::Ema | MovingAverageKind::Kama => period,
        MovingAverageKind::Dema => 2 * period,
        MovingAverageKind::Tema => 3 * period,
    };

        for i in 0..n {
        if output[i].is_nan() {
            continue;
        }

                let start = i.saturating_sub(window_size - 1);
        for j in start..=i {
            if close[j].is_nan() {
                output[i] = f64::NAN;
                break;
            }
        }
    }

    Ok(output)
}

/// Compute a matrix of moving averages for all (kind, period) combinations.
///
/// Returns a `HashMap` containing exactly one entry per `(kind, period)` pair
/// in the cartesian product of the two input slices. Keys are formatted as
/// `"{KIND}_{period}"` in uppercase (e.g., `"SMA_10"`, `"EMA_20"`).
///
/// # Arguments
///
/// * `close` - Slice of closing prices, ordered oldest to newest
/// * `kinds` - Slice of moving average types to compute
/// * `periods` - Slice of lookback periods to use
///
/// # Returns
///
/// A `Result<HashMap<String, Vec<f64>>, IndicatorError>` where each entry
/// contains a moving average vector aligned index-for-index with `close`.
///
/// # Errors
///
/// - `IndicatorError::InvalidPeriod` if any period is 0
/// - `IndicatorError::EmptyInput` if `close` is empty
///
/// # Example
///
/// ```
/// use modelenv_core::indicators::trend::{moving_average_matrix, MovingAverageKind};
///
/// let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
/// let kinds = &[MovingAverageKind::Sma, MovingAverageKind::Ema];
/// let periods = &[10, 20];
/// let result = moving_average_matrix(&close, kinds, periods).unwrap();
///
/// // Should have 4 entries: SMA_10, SMA_20, EMA_10, EMA_20
/// assert_eq!(result.len(), 4);
/// assert!(result.contains_key("SMA_10"));
/// assert!(result.contains_key("SMA_20"));
/// assert!(result.contains_key("EMA_10"));
/// assert!(result.contains_key("EMA_20"));
/// ```
pub fn moving_average_matrix(
    close: &[f64],
    kinds: &[MovingAverageKind],
    periods: &[usize],
) -> Result<HashMap<String, Vec<f64>>, IndicatorError> {
    let mut result = HashMap::new();

        for kind in kinds {
        for &period in periods {
                        let ma_values = moving_average(close, *kind, period)?;

                        let key = format!("{}_{}", kind, period);

            result.insert(key, ma_values);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    
    #[test]
    fn macd_returns_correct_length() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = macd(&close, 12, 26, 9).unwrap();
        assert_eq!(result.macd.len(), close.len());
        assert_eq!(result.signal.len(), close.len());
        assert_eq!(result.hist.len(), close.len());
    }

    #[test]
    fn macd_leading_nans() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = macd(&close, 12, 26, 9).unwrap();
                let lookback = 26 + 9 - 2;
        for i in 0..lookback {
            assert!(result.macd[i].is_nan(), "Expected NaN macd at index {}", i);
            assert!(
                result.signal[i].is_nan(),
                "Expected NaN signal at index {}",
                i
            );
            assert!(result.hist[i].is_nan(), "Expected NaN hist at index {}", i);
        }
                assert!(
            !result.macd[lookback].is_nan(),
            "Expected non-NaN macd at index {}",
            lookback
        );
        assert!(
            !result.signal[lookback].is_nan(),
            "Expected non-NaN signal at index {}",
            lookback
        );
        assert!(
            !result.hist[lookback].is_nan(),
            "Expected non-NaN hist at index {}",
            lookback
        );
    }

    #[test]
    fn macd_histogram_identity() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = macd(&close, 12, 26, 9).unwrap();
                for i in 0..result.macd.len() {
            if !result.macd[i].is_nan() && !result.signal[i].is_nan() && !result.hist[i].is_nan() {
                let expected_hist = result.macd[i] - result.signal[i];
                let diff = (result.hist[i] - expected_hist).abs();
                assert!(
                    diff <= 1e-9,
                    "Histogram identity violated at index {}: hist={}, macd-signal={}, diff={}",
                    i,
                    result.hist[i],
                    expected_hist,
                    diff
                );
            }
        }
    }

    #[test]
    fn macd_empty_input() {
        let close: Vec<f64> = vec![];
        let result = macd(&close, 12, 26, 9).unwrap();
        assert!(result.macd.is_empty());
        assert!(result.signal.is_empty());
        assert!(result.hist.is_empty());
    }

    #[test]
    fn macd_insufficient_data() {
                        let close: Vec<f64> = (1..=33).map(|x| x as f64).collect();
        let result = macd(&close, 12, 26, 9).unwrap();
        assert_eq!(result.macd.len(), 33);
        assert_eq!(result.signal.len(), 33);
        assert_eq!(result.hist.len(), 33);
        for i in 0..33 {
            assert!(result.macd[i].is_nan());
            assert!(result.signal[i].is_nan());
            assert!(result.hist[i].is_nan());
        }
    }

    #[test]
    fn macd_fast_zero_returns_error() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = macd(&close, 0, 26, 9);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod {
                param_name: "fast",
                value: 0,
                ..
            })
        ));
    }

    #[test]
    fn macd_slow_zero_returns_error() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = macd(&close, 12, 0, 9);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod {
                param_name: "slow",
                value: 0,
                ..
            })
        ));
    }

    #[test]
    fn macd_signal_zero_returns_error() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = macd(&close, 12, 26, 0);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod {
                param_name: "signal",
                value: 0,
                ..
            })
        ));
    }

    #[test]
    fn macd_fast_equals_slow_returns_error() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = macd(&close, 12, 12, 9);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod {
                param_name: "fast",
                ..
            })
        ));
    }

    #[test]
    fn macd_fast_greater_than_slow_returns_error() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = macd(&close, 26, 12, 9);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod {
                param_name: "fast",
                ..
            })
        ));
    }

    #[test]
    fn macd_custom_periods() {
                let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = macd(&close, 5, 10, 3).unwrap();
        assert_eq!(result.macd.len(), 100);
                let lookback = 10 + 3 - 2;
        assert!(result.macd[lookback - 1].is_nan());
        assert!(!result.macd[lookback].is_nan());
    }

    
    #[test]
    fn adx_returns_correct_length() {
        let high = vec![
            45.0, 45.5, 46.0, 45.5, 46.0, 45.5, 45.0, 44.5, 45.0, 45.5, 46.0, 45.5, 46.0, 45.5,
            45.0, 44.5, 45.0, 45.5, 46.0, 45.5,
        ];
        let low = vec![
            44.0, 44.5, 45.0, 44.5, 45.0, 44.5, 44.0, 43.5, 44.0, 44.5, 45.0, 44.5, 45.0, 44.5,
            44.0, 43.5, 44.0, 44.5, 45.0, 44.5,
        ];
        let close = vec![
            44.5, 45.0, 45.5, 45.0, 45.5, 45.0, 44.5, 44.0, 44.5, 45.0, 45.5, 45.0, 45.5, 45.0,
            44.5, 44.0, 44.5, 45.0, 45.5, 45.0,
        ];
        let result = adx(&high, &low, &close, 5).unwrap();
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn adx_leading_nans() {
        let high = vec![
            45.0, 45.5, 46.0, 45.5, 46.0, 45.5, 45.0, 44.5, 45.0, 45.5, 46.0, 45.5, 46.0, 45.5,
            45.0, 44.5, 45.0, 45.5, 46.0, 45.5,
        ];
        let low = vec![
            44.0, 44.5, 45.0, 44.5, 45.0, 44.5, 44.0, 43.5, 44.0, 44.5, 45.0, 44.5, 45.0, 44.5,
            44.0, 43.5, 44.0, 44.5, 45.0, 44.5,
        ];
        let close = vec![
            44.5, 45.0, 45.5, 45.0, 45.5, 45.0, 44.5, 44.0, 44.5, 45.0, 45.5, 45.0, 45.5, 45.0,
            44.5, 44.0, 44.5, 45.0, 45.5, 45.0,
        ];
        let result = adx(&high, &low, &close, 5).unwrap();
                for i in 0..9 {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }
                assert!(
            !result[9].is_nan(),
            "Expected non-NaN ADX at index 9"
        );
    }

    #[test]
    fn adx_values_in_range() {
        let high = vec![
            45.0, 45.5, 46.0, 45.5, 46.0, 45.5, 45.0, 44.5, 45.0, 45.5, 46.0, 45.5, 46.0, 45.5,
            45.0, 44.5, 45.0, 45.5, 46.0, 45.5,
        ];
        let low = vec![
            44.0, 44.5, 45.0, 44.5, 45.0, 44.5, 44.0, 43.5, 44.0, 44.5, 45.0, 44.5, 45.0, 44.5,
            44.0, 43.5, 44.0, 44.5, 45.0, 44.5,
        ];
        let close = vec![
            44.5, 45.0, 45.5, 45.0, 45.5, 45.0, 44.5, 44.0, 44.5, 45.0, 45.5, 45.0, 45.5, 45.0,
            44.5, 44.0, 44.5, 45.0, 45.5, 45.0,
        ];
        let result = adx(&high, &low, &close, 5).unwrap();
                for i in 9..result.len() {
            assert!(
                !result[i].is_nan(),
                "Expected non-NaN ADX at index {}",
                i
            );
            assert!(
                result[i] >= 0.0 && result[i] <= 100.0,
                "ADX at index {} is {} which is out of range [0, 100]",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn adx_empty_input() {
        let high: Vec<f64> = vec![];
        let low: Vec<f64> = vec![];
        let close: Vec<f64> = vec![];
        let result = adx(&high, &low, &close, 14).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn adx_insufficient_data() {
                let high = vec![45.0, 45.5, 46.0];
        let low = vec![44.0, 44.5, 45.0];
        let close = vec![44.5, 45.0, 45.5];
        let result = adx(&high, &low, &close, 5).unwrap();
        assert_eq!(result.len(), 3);
        for val in &result {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn adx_period_zero_returns_error() {
        let high = vec![45.0, 45.5, 46.0];
        let low = vec![44.0, 44.5, 45.0];
        let close = vec![44.5, 45.0, 45.5];
        let result = adx(&high, &low, &close, 0);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { value: 0, .. })
        ));
    }

    #[test]
    fn adx_length_mismatch_high() {
        let high = vec![45.0, 45.5];
        let low = vec![44.0, 44.5, 45.0];
        let close = vec![44.5, 45.0, 45.5];
        let result = adx(&high, &low, &close, 2);
        assert!(matches!(
            result,
            Err(IndicatorError::LengthMismatch {
                param_name: "high",
                ..
            })
        ));
    }

    #[test]
    fn adx_length_mismatch_low() {
        let high = vec![45.0, 45.5, 46.0];
        let low = vec![44.0, 44.5];
        let close = vec![44.5, 45.0, 45.5];
        let result = adx(&high, &low, &close, 2);
        assert!(matches!(
            result,
            Err(IndicatorError::LengthMismatch {
                param_name: "low",
                ..
            })
        ));
    }

    #[test]
    fn adx_period_one_returns_error() {
                let high = vec![
            45.0, 45.5, 46.0, 45.5, 46.0, 45.5, 45.0, 44.5, 45.0, 45.5,
        ];
        let low = vec![
            44.0, 44.5, 45.0, 44.5, 45.0, 44.5, 44.0, 43.5, 44.0, 44.5,
        ];
        let close = vec![
            44.5, 45.0, 45.5, 45.0, 45.5, 45.0, 44.5, 44.0, 44.5, 45.0,
        ];
        let result = adx(&high, &low, &close, 1);
                assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { .. })
        ));
    }

    
    #[test]
    fn moving_average_sma_returns_correct_length() {
        let close: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = moving_average(&close, MovingAverageKind::Sma, 5).unwrap();
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn moving_average_sma_leading_nans() {
        let close: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = moving_average(&close, MovingAverageKind::Sma, 5).unwrap();
                for i in 0..4 {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }
                assert!(!result[4].is_nan(), "Expected non-NaN SMA at index 4");
    }

    #[test]
    fn moving_average_ema_returns_correct_length() {
        let close: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = moving_average(&close, MovingAverageKind::Ema, 5).unwrap();
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn moving_average_wma_returns_correct_length() {
        let close: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = moving_average(&close, MovingAverageKind::Wma, 5).unwrap();
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn moving_average_dema_returns_correct_length() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = moving_average(&close, MovingAverageKind::Dema, 5).unwrap();
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn moving_average_tema_returns_correct_length() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = moving_average(&close, MovingAverageKind::Tema, 5).unwrap();
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn moving_average_kama_returns_correct_length() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = moving_average(&close, MovingAverageKind::Kama, 5).unwrap();
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn moving_average_trima_returns_correct_length() {
        let close: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = moving_average(&close, MovingAverageKind::Trima, 5).unwrap();
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn moving_average_period_zero_returns_error() {
        let close: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = moving_average(&close, MovingAverageKind::Sma, 0);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { value: 0, .. })
        ));
    }

    #[test]
    fn moving_average_empty_input_returns_error() {
        let close: Vec<f64> = vec![];
        let result = moving_average(&close, MovingAverageKind::Sma, 5);
        assert!(matches!(result, Err(IndicatorError::EmptyInput)));
    }

    #[test]
    fn moving_average_nan_propagation() {
                let mut close: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        close[10] = f64::NAN;

        let result = moving_average(&close, MovingAverageKind::Sma, 5).unwrap();
        assert_eq!(result.len(), close.len());

                                for i in 10..=14 {
            assert!(
                result[i].is_nan(),
                "Expected NaN at index {} due to NaN propagation",
                i
            );
        }

                if close.len() > 15 {
            assert!(
                !result[15].is_nan(),
                "Expected non-NaN at index 15 (window doesn't include NaN)"
            );
        }
    }

    #[test]
    fn moving_average_all_kinds_work() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let kinds = [
            MovingAverageKind::Sma,
            MovingAverageKind::Ema,
            MovingAverageKind::Wma,
            MovingAverageKind::Dema,
            MovingAverageKind::Tema,
            MovingAverageKind::Kama,
            MovingAverageKind::Trima,
        ];

        for kind in &kinds {
            let result = moving_average(&close, *kind, 10);
            assert!(
                result.is_ok(),
                "moving_average failed for kind {:?}: {:?}",
                kind,
                result
            );
            let output = result.unwrap();
            assert_eq!(
                output.len(),
                close.len(),
                "Output length mismatch for kind {:?}",
                kind
            );
        }
    }

    #[test]
    fn moving_average_kind_display() {
        assert_eq!(format!("{}", MovingAverageKind::Sma), "SMA");
        assert_eq!(format!("{}", MovingAverageKind::Ema), "EMA");
        assert_eq!(format!("{}", MovingAverageKind::Wma), "WMA");
        assert_eq!(format!("{}", MovingAverageKind::Dema), "DEMA");
        assert_eq!(format!("{}", MovingAverageKind::Tema), "TEMA");
        assert_eq!(format!("{}", MovingAverageKind::Kama), "KAMA");
        assert_eq!(format!("{}", MovingAverageKind::Trima), "TRIMA");
    }

    #[test]
    fn default_ma_kinds_and_periods() {
                assert_eq!(DEFAULT_MA_KINDS.len(), 2);
        assert_eq!(DEFAULT_MA_KINDS[0], MovingAverageKind::Sma);
        assert_eq!(DEFAULT_MA_KINDS[1], MovingAverageKind::Ema);

        assert_eq!(DEFAULT_MA_PERIODS.len(), 3);
        assert_eq!(DEFAULT_MA_PERIODS[0], 10);
        assert_eq!(DEFAULT_MA_PERIODS[1], 20);
        assert_eq!(DEFAULT_MA_PERIODS[2], 50);
    }

    #[test]
    fn moving_average_insufficient_data() {
                let close: Vec<f64> = vec![1.0, 2.0, 3.0];
        let result = moving_average(&close, MovingAverageKind::Sma, 10).unwrap();
        assert_eq!(result.len(), 3);
        for val in &result {
            assert!(val.is_nan());
        }
    }

    
    #[test]
    fn moving_average_matrix_returns_correct_entries() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let kinds = &[MovingAverageKind::Sma, MovingAverageKind::Ema];
        let periods = &[10, 20];
        let result = moving_average_matrix(&close, kinds, periods).unwrap();

                assert_eq!(result.len(), 4);
        assert!(result.contains_key("SMA_10"));
        assert!(result.contains_key("SMA_20"));
        assert!(result.contains_key("EMA_10"));
        assert!(result.contains_key("EMA_20"));
    }

    #[test]
    fn moving_average_matrix_correct_key_format() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let kinds = &[
            MovingAverageKind::Sma,
            MovingAverageKind::Ema,
            MovingAverageKind::Wma,
            MovingAverageKind::Dema,
            MovingAverageKind::Tema,
            MovingAverageKind::Kama,
            MovingAverageKind::Trima,
        ];
        let periods = &[5];
        let result = moving_average_matrix(&close, kinds, periods).unwrap();

                assert!(result.contains_key("SMA_5"));
        assert!(result.contains_key("EMA_5"));
        assert!(result.contains_key("WMA_5"));
        assert!(result.contains_key("DEMA_5"));
        assert!(result.contains_key("TEMA_5"));
        assert!(result.contains_key("KAMA_5"));
        assert!(result.contains_key("TRIMA_5"));
    }

    #[test]
    fn moving_average_matrix_output_lengths() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let kinds = &[MovingAverageKind::Sma, MovingAverageKind::Ema];
        let periods = &[10, 20];
        let result = moving_average_matrix(&close, kinds, periods).unwrap();

                for (key, values) in &result {
            assert_eq!(
                values.len(),
                close.len(),
                "Output length mismatch for key {}",
                key
            );
        }
    }

    #[test]
    fn moving_average_matrix_empty_kinds() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let kinds: &[MovingAverageKind] = &[];
        let periods = &[10, 20];
        let result = moving_average_matrix(&close, kinds, periods).unwrap();

                assert!(result.is_empty());
    }

    #[test]
    fn moving_average_matrix_empty_periods() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let kinds = &[MovingAverageKind::Sma, MovingAverageKind::Ema];
        let periods: &[usize] = &[];
        let result = moving_average_matrix(&close, kinds, periods).unwrap();

                assert!(result.is_empty());
    }

    #[test]
    fn moving_average_matrix_period_zero_returns_error() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let kinds = &[MovingAverageKind::Sma];
        let periods = &[0];
        let result = moving_average_matrix(&close, kinds, periods);

        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { value: 0, .. })
        ));
    }

    #[test]
    fn moving_average_matrix_empty_input_returns_error() {
        let close: Vec<f64> = vec![];
        let kinds = &[MovingAverageKind::Sma];
        let periods = &[10];
        let result = moving_average_matrix(&close, kinds, periods);

        assert!(matches!(result, Err(IndicatorError::EmptyInput)));
    }

    #[test]
    fn moving_average_matrix_with_defaults() {
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = moving_average_matrix(&close, DEFAULT_MA_KINDS, DEFAULT_MA_PERIODS).unwrap();

                        assert_eq!(result.len(), 6);
        assert!(result.contains_key("SMA_10"));
        assert!(result.contains_key("SMA_20"));
        assert!(result.contains_key("SMA_50"));
        assert!(result.contains_key("EMA_10"));
        assert!(result.contains_key("EMA_20"));
        assert!(result.contains_key("EMA_50"));
    }

    #[test]
    fn moving_average_matrix_values_match_individual_calls() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let kinds = &[MovingAverageKind::Sma, MovingAverageKind::Ema];
        let periods = &[10, 20];
        let matrix_result = moving_average_matrix(&close, kinds, periods).unwrap();

                for kind in kinds {
            for &period in periods {
                let key = format!("{}_{}", kind, period);
                let individual_result = moving_average(&close, *kind, period).unwrap();
                let matrix_values = matrix_result.get(&key).unwrap();

                for (i, (&matrix_val, &individual_val)) in
                    matrix_values.iter().zip(individual_result.iter()).enumerate()
                {
                    if matrix_val.is_nan() && individual_val.is_nan() {
                        continue;
                    }
                    assert!(
                        (matrix_val - individual_val).abs() < 1e-12,
                        "Value mismatch at index {} for key {}: matrix={}, individual={}",
                        i,
                        key,
                        matrix_val,
                        individual_val
                    );
                }
            }
        }
    }

    
    #[test]
    fn ichimoku_returns_correct_length() {
        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();
        assert_eq!(result.tenkan.len(), close.len());
        assert_eq!(result.kijun.len(), close.len());
        assert_eq!(result.senkou_a.len(), close.len());
        assert_eq!(result.senkou_b.len(), close.len());
        assert_eq!(result.chikou.len(), close.len());
    }

    #[test]
    fn ichimoku_tenkan_leading_nans() {
        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();

                for i in 0..8 {
            assert!(result.tenkan[i].is_nan(), "Expected NaN tenkan at index {}", i);
        }
                assert!(!result.tenkan[8].is_nan(), "Expected non-NaN tenkan at index 8");
    }

    #[test]
    fn ichimoku_kijun_leading_nans() {
        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();

                for i in 0..25 {
            assert!(result.kijun[i].is_nan(), "Expected NaN kijun at index {}", i);
        }
                assert!(!result.kijun[25].is_nan(), "Expected non-NaN kijun at index 25");
    }

    #[test]
    fn ichimoku_senkou_a_leading_nans() {
        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();

                                                        for i in 0..26 {
            assert!(result.senkou_a[i].is_nan(), "Expected NaN senkou_a at index {}", i);
        }
                assert!(!result.senkou_a[51].is_nan(), "Expected non-NaN senkou_a at index 51");
    }

    #[test]
    fn ichimoku_senkou_b_leading_nans() {
        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();

                        for i in 0..77 {
            assert!(result.senkou_b[i].is_nan(), "Expected NaN senkou_b at index {}", i);
        }
                assert!(!result.senkou_b[77].is_nan(), "Expected non-NaN senkou_b at index 77");
    }

    #[test]
    fn ichimoku_chikou_trailing_nans() {
        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();

                        for i in 74..100 {
            assert!(result.chikou[i].is_nan(), "Expected NaN chikou at index {}", i);
        }
                assert!(!result.chikou[73].is_nan(), "Expected non-NaN chikou at index 73");
    }

    #[test]
    fn ichimoku_chikou_values() {
        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();

                for i in 0..74 {
            let expected = close[i + 26];
            assert!(
                (result.chikou[i] - expected).abs() < 1e-12,
                "Chikou mismatch at index {}: expected {}, got {}",
                i, expected, result.chikou[i]
            );
        }
    }

    #[test]
    fn ichimoku_tenkan_computation() {
                let high = vec![10.0, 12.0, 11.0, 13.0, 12.0];
        let low = vec![8.0, 9.0, 9.0, 10.0, 10.0];
        let close = vec![9.0, 11.0, 10.0, 12.0, 11.0];
        let result = ichimoku(&high, &low, &close, 3, 3, 3).unwrap();

                        assert!(
            (result.tenkan[2] - 10.0).abs() < 1e-12,
            "Tenkan at index 2: expected 10.0, got {}",
            result.tenkan[2]
        );

                        assert!(
            (result.tenkan[3] - 11.0).abs() < 1e-12,
            "Tenkan at index 3: expected 11.0, got {}",
            result.tenkan[3]
        );
    }

    #[test]
    fn ichimoku_empty_input() {
        let high: Vec<f64> = vec![];
        let low: Vec<f64> = vec![];
        let close: Vec<f64> = vec![];
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();
        assert!(result.tenkan.is_empty());
        assert!(result.kijun.is_empty());
        assert!(result.senkou_a.is_empty());
        assert!(result.senkou_b.is_empty());
        assert!(result.chikou.is_empty());
    }

    #[test]
    fn ichimoku_period_exceeds_length() {
                let high = vec![10.0, 12.0, 11.0];
        let low = vec![8.0, 9.0, 9.0];
        let close = vec![9.0, 11.0, 10.0];
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();

                for val in &result.tenkan {
            assert!(val.is_nan());
        }
        for val in &result.kijun {
            assert!(val.is_nan());
        }
        for val in &result.senkou_a {
            assert!(val.is_nan());
        }
        for val in &result.senkou_b {
            assert!(val.is_nan());
        }
        for val in &result.chikou {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn ichimoku_tenkan_zero_returns_error() {
        let high = vec![10.0, 12.0, 11.0];
        let low = vec![8.0, 9.0, 9.0];
        let close = vec![9.0, 11.0, 10.0];
        let result = ichimoku(&high, &low, &close, 0, 26, 52);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod {
                param_name: "tenkan",
                value: 0,
                ..
            })
        ));
    }

    #[test]
    fn ichimoku_kijun_zero_returns_error() {
        let high = vec![10.0, 12.0, 11.0];
        let low = vec![8.0, 9.0, 9.0];
        let close = vec![9.0, 11.0, 10.0];
        let result = ichimoku(&high, &low, &close, 9, 0, 52);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod {
                param_name: "kijun",
                value: 0,
                ..
            })
        ));
    }

    #[test]
    fn ichimoku_senkou_b_period_zero_returns_error() {
        let high = vec![10.0, 12.0, 11.0];
        let low = vec![8.0, 9.0, 9.0];
        let close = vec![9.0, 11.0, 10.0];
        let result = ichimoku(&high, &low, &close, 9, 26, 0);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod {
                param_name: "senkou_b_period",
                value: 0,
                ..
            })
        ));
    }

    #[test]
    fn ichimoku_length_mismatch_high() {
        let high = vec![10.0, 12.0];
        let low = vec![8.0, 9.0, 9.0];
        let close = vec![9.0, 11.0, 10.0];
        let result = ichimoku(&high, &low, &close, 2, 2, 2);
        assert!(matches!(
            result,
            Err(IndicatorError::LengthMismatch {
                param_name: "high",
                ..
            })
        ));
    }

    #[test]
    fn ichimoku_length_mismatch_low() {
        let high = vec![10.0, 12.0, 11.0];
        let low = vec![8.0, 9.0];
        let close = vec![9.0, 11.0, 10.0];
        let result = ichimoku(&high, &low, &close, 2, 2, 2);
        assert!(matches!(
            result,
            Err(IndicatorError::LengthMismatch {
                param_name: "low",
                ..
            })
        ));
    }

    #[test]
    fn ichimoku_senkou_a_computation() {
                let high = vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0];
        let low = vec![8.0, 9.0, 9.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0];
        let close = vec![9.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0];
        let result = ichimoku(&high, &low, &close, 2, 3, 2).unwrap();

                                                                                                assert!(
            (result.senkou_a[5] - 10.25).abs() < 1e-12,
            "Senkou A at index 5: expected 10.25, got {}",
            result.senkou_a[5]
        );
    }

    #[test]
    fn ichimoku_with_nan_in_input() {
                let mut high: Vec<f64> = (1..=20).map(|x| x as f64 + 1.0).collect();
        let mut low: Vec<f64> = (1..=20).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=20).map(|x| x as f64).collect();

                high[5] = f64::NAN;
        low[5] = f64::NAN;

        let result = ichimoku(&high, &low, &close, 3, 5, 3).unwrap();

                        for i in 5..8 {
            assert!(
                result.tenkan[i].is_nan(),
                "Expected NaN tenkan at index {} due to NaN input",
                i
            );
        }
    }

            
    #[test]
    fn ichimoku_nan_placement_tenkan_matches_python_reference() {
                        let high: Vec<f64> = (1..=50).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=50).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        
        let tenkan_period = 9;
        let result = ichimoku(&high, &low, &close, tenkan_period, 26, 52).unwrap();
        
                for i in 0..(tenkan_period - 1) {
            assert!(
                result.tenkan[i].is_nan(),
                "tenkan[{}] should be NaN (< tenkan_period - 1 = {})",
                i, tenkan_period - 1
            );
        }
        
                assert!(
            !result.tenkan[tenkan_period - 1].is_nan(),
            "tenkan[{}] should be finite (first valid index)",
            tenkan_period - 1
        );
    }

    #[test]
    fn ichimoku_nan_placement_kijun_matches_python_reference() {
                        let high: Vec<f64> = (1..=50).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=50).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        
        let kijun_period = 26;
        let result = ichimoku(&high, &low, &close, 9, kijun_period, 52).unwrap();
        
                for i in 0..(kijun_period - 1) {
            assert!(
                result.kijun[i].is_nan(),
                "kijun[{}] should be NaN (< kijun_period - 1 = {})",
                i, kijun_period - 1
            );
        }
        
                assert!(
            !result.kijun[kijun_period - 1].is_nan(),
            "kijun[{}] should be finite (first valid index)",
            kijun_period - 1
        );
    }

    #[test]
    fn ichimoku_nan_placement_senkou_a_matches_python_reference() {
                                        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        
        let tenkan_period = 9;
        let kijun_period = 26;
        let result = ichimoku(&high, &low, &close, tenkan_period, kijun_period, 52).unwrap();
        
                for i in 0..kijun_period {
            assert!(
                result.senkou_a[i].is_nan(),
                "senkou_a[{}] should be NaN (< kijun_period = {})",
                i, kijun_period
            );
        }
        
                                let first_valid_senkou_a = 2 * kijun_period - 1;
        assert!(
            !result.senkou_a[first_valid_senkou_a].is_nan(),
            "senkou_a[{}] should be finite (first valid index)",
            first_valid_senkou_a
        );
    }

    #[test]
    fn ichimoku_nan_placement_senkou_b_matches_python_reference() {
                        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        
        let kijun_period = 26;
        let senkou_b_period = 52;
        let result = ichimoku(&high, &low, &close, 9, kijun_period, senkou_b_period).unwrap();
        
                let first_valid_senkou_b = kijun_period + senkou_b_period - 1;
        for i in 0..first_valid_senkou_b {
            assert!(
                result.senkou_b[i].is_nan(),
                "senkou_b[{}] should be NaN (< kijun + senkou_b_period - 1 = {})",
                i, first_valid_senkou_b
            );
        }
        
                assert!(
            !result.senkou_b[first_valid_senkou_b].is_nan(),
            "senkou_b[{}] should be finite (first valid index)",
            first_valid_senkou_b
        );
    }

    #[test]
    fn ichimoku_nan_placement_chikou_matches_python_reference() {
                        let n = 100;
        let high: Vec<f64> = (1..=n).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=n).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=n).map(|x| x as f64).collect();
        
        let kijun_period = 26;
        let result = ichimoku(&high, &low, &close, 9, kijun_period, 52).unwrap();
        
                let first_nan_chikou = n as usize - kijun_period;
        
                for i in 0..first_nan_chikou {
            assert!(
                !result.chikou[i].is_nan(),
                "chikou[{}] should be finite (i + kijun < n)",
                i
            );
        }
        
                for i in first_nan_chikou..n as usize {
            assert!(
                result.chikou[i].is_nan(),
                "chikou[{}] should be NaN (i + kijun >= n)",
                i
            );
        }
    }

    #[test]
    fn ichimoku_all_nan_when_all_periods_exceed_length() {
                let high = vec![10.0, 12.0, 11.0, 13.0, 12.0];
        let low = vec![8.0, 9.0, 9.0, 10.0, 10.0];
        let close = vec![9.0, 11.0, 10.0, 12.0, 11.0];
        
                let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();
        
                assert_eq!(result.tenkan.len(), 5);
        assert_eq!(result.kijun.len(), 5);
        assert_eq!(result.senkou_a.len(), 5);
        assert_eq!(result.senkou_b.len(), 5);
        assert_eq!(result.chikou.len(), 5);
        
        for i in 0..5 {
            assert!(result.tenkan[i].is_nan(), "tenkan[{}] should be NaN", i);
            assert!(result.kijun[i].is_nan(), "kijun[{}] should be NaN", i);
            assert!(result.senkou_a[i].is_nan(), "senkou_a[{}] should be NaN", i);
            assert!(result.senkou_b[i].is_nan(), "senkou_b[{}] should be NaN", i);
            assert!(result.chikou[i].is_nan(), "chikou[{}] should be NaN", i);
        }
    }

    #[test]
    fn ichimoku_partial_nan_when_some_periods_exceed_length() {
                let high: Vec<f64> = (1..=15).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=15).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=15).map(|x| x as f64).collect();
        
                let result = ichimoku(&high, &low, &close, 3, 5, 20).unwrap();
        
                assert!(result.tenkan[1].is_nan());
        assert!(!result.tenkan[2].is_nan());
        
                assert!(result.kijun[3].is_nan());
        assert!(!result.kijun[4].is_nan());
        
                assert!(result.senkou_a[8].is_nan());
        assert!(!result.senkou_a[9].is_nan());
        
                for i in 0..15 {
            assert!(result.senkou_b[i].is_nan(), "senkou_b[{}] should be NaN", i);
        }
        
                assert!(!result.chikou[9].is_nan());
        assert!(result.chikou[10].is_nan());
    }

    #[test]
    fn ichimoku_output_vector_lengths_always_match_input() {
                        let test_cases = vec![
            (1, 9, 26, 52),
            (10, 9, 26, 52),
            (50, 9, 26, 52),
            (100, 9, 26, 52),
            (200, 9, 26, 52),
            (50, 3, 5, 10),
            (50, 1, 1, 1),
        ];
        
        for (n, tenkan, kijun, senkou_b) in test_cases {
            let high: Vec<f64> = (1..=n).map(|x| x as f64 + 1.0).collect();
            let low: Vec<f64> = (1..=n).map(|x| x as f64 - 1.0).collect();
            let close: Vec<f64> = (1..=n).map(|x| x as f64).collect();
            
            let result = ichimoku(&high, &low, &close, tenkan, kijun, senkou_b).unwrap();
            
            assert_eq!(
                result.tenkan.len(), n as usize,
                "tenkan length mismatch for n={}, periods=({},{},{})",
                n, tenkan, kijun, senkou_b
            );
            assert_eq!(
                result.kijun.len(), n as usize,
                "kijun length mismatch for n={}, periods=({},{},{})",
                n, tenkan, kijun, senkou_b
            );
            assert_eq!(
                result.senkou_a.len(), n as usize,
                "senkou_a length mismatch for n={}, periods=({},{},{})",
                n, tenkan, kijun, senkou_b
            );
            assert_eq!(
                result.senkou_b.len(), n as usize,
                "senkou_b length mismatch for n={}, periods=({},{},{})",
                n, tenkan, kijun, senkou_b
            );
            assert_eq!(
                result.chikou.len(), n as usize,
                "chikou length mismatch for n={}, periods=({},{},{})",
                n, tenkan, kijun, senkou_b
            );
        }
    }

    #[test]
    fn ichimoku_computation_correctness_tenkan_kijun() {
                        let high = vec![10.0, 15.0, 12.0, 18.0, 14.0, 20.0, 16.0, 22.0, 18.0, 24.0];
        let low = vec![5.0, 8.0, 7.0, 10.0, 9.0, 12.0, 11.0, 14.0, 13.0, 16.0];
        let close = vec![7.0, 12.0, 10.0, 15.0, 12.0, 17.0, 14.0, 19.0, 16.0, 21.0];
        
        let result = ichimoku(&high, &low, &close, 3, 5, 3).unwrap();
        
                        assert!(
            (result.tenkan[2] - 10.0).abs() < 1e-9,
            "tenkan[2] expected 10.0, got {}",
            result.tenkan[2]
        );
        
                        assert!(
            (result.tenkan[5] - 14.5).abs() < 1e-9,
            "tenkan[5] expected 14.5, got {}",
            result.tenkan[5]
        );
        
                        assert!(
            (result.kijun[4] - 11.5).abs() < 1e-9,
            "kijun[4] expected 11.5, got {}",
            result.kijun[4]
        );
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

                            proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn adx_bounded_range(
                                    close in prop::collection::vec(1.0f64..10000.0f64, 1..500),
                        period in 2usize..50
        ) {
            let n = close.len();

                                    let high: Vec<f64> = close.iter().map(|&c| c * 1.01).collect();
            let low: Vec<f64> = close.iter().map(|&c| c * 0.99).collect();

            let result = adx(&high, &low, &close, period);

                        prop_assert!(result.is_ok(), "ADX returned error: {:?}", result);

            let result = result.unwrap();

                        prop_assert_eq!(result.len(), n);

                        for (i, &val) in result.iter().enumerate() {
                if !val.is_nan() {
                    prop_assert!(
                        val >= 0.0 && val <= 100.0,
                        "ADX at index {} is {} which is outside [0.0, 100.0]. \
                         Input length: {}, period: {}",
                        i, val, n, period
                    );
                }
            }
        }
    }

                                proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn macd_histogram_identity(
                                    close in prop::collection::vec(1.0f64..10000.0f64, 1..500),
                        fast in 2usize..20,
                        slow_offset in 1usize..30,
                        signal in 1usize..20
        ) {
                        let slow = fast + slow_offset;

            let result = macd(&close, fast, slow, signal);

                        prop_assert!(result.is_ok(), "MACD returned error: {:?}", result);

            let output = result.unwrap();

                        let n = close.len();
            prop_assert_eq!(output.macd.len(), n);
            prop_assert_eq!(output.signal.len(), n);
            prop_assert_eq!(output.hist.len(), n);

                                    for i in 0..n {
                let macd_val = output.macd[i];
                let signal_val = output.signal[i];
                let hist_val = output.hist[i];

                                if macd_val.is_finite() && signal_val.is_finite() && hist_val.is_finite() {
                    let expected_hist = macd_val - signal_val;
                    let diff = (hist_val - expected_hist).abs();

                    prop_assert!(
                        diff <= 1e-9,
                        "MACD histogram identity violated at index {}. \
                         hist[{}] = {}, macd[{}] - signal[{}] = {} - {} = {}, \
                         diff = {}. Input length: {}, fast: {}, slow: {}, signal: {}",
                        i, i, hist_val, i, i, macd_val, signal_val, expected_hist,
                        diff, n, fast, slow, signal
                    );
                }
            }
        }
    }
}
