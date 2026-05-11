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

    // Check for length mismatch first
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

    // Check for invalid periods
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

    // Handle empty input
    if n == 0 {
        return Ok(IchimokuOutput {
            tenkan: Vec::new(),
            kijun: Vec::new(),
            senkou_a: Vec::new(),
            senkou_b: Vec::new(),
            chikou: Vec::new(),
        });
    }

    // Compute Tenkan-sen (Conversion Line)
    // tenkan[i] = (max(high[i-tenkan+1..=i]) + min(low[i-tenkan+1..=i])) / 2
    let tenkan_line = compute_midpoint_line(high, low, tenkan_period);

    // Compute Kijun-sen (Base Line)
    // kijun[i] = (max(high[i-kijun+1..=i]) + min(low[i-kijun+1..=i])) / 2
    let kijun_line = compute_midpoint_line(high, low, kijun_period);

    // Compute Senkou Span A
    // senkou_a[i] = (tenkan[i-kijun] + kijun[i-kijun]) / 2 for i >= kijun
    // This is the average of tenkan and kijun, shifted forward by kijun periods
    let mut senkou_a = vec![f64::NAN; n];
    for i in kijun_period..n {
        let src_idx = i - kijun_period;
        let tenkan_val = tenkan_line[src_idx];
        let kijun_val = kijun_line[src_idx];
        if tenkan_val.is_finite() && kijun_val.is_finite() {
            senkou_a[i] = (tenkan_val + kijun_val) / 2.0;
        }
    }

    // Compute Senkou Span B
    // senkou_b[i] = (max(high[i-kijun-senkou_b+1..=i-kijun]) + min(low[i-kijun-senkou_b+1..=i-kijun])) / 2
    // This is the midpoint of senkou_b_period window, shifted forward by kijun periods
    // Per Python: senkou_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    let senkou_b_unshifted = compute_midpoint_line(high, low, senkou_b_period);
    let mut senkou_b = vec![f64::NAN; n];
    for i in kijun_period..n {
        let src_idx = i - kijun_period;
        senkou_b[i] = senkou_b_unshifted[src_idx];
    }

    // Compute Chikou Span (Lagging Span)
    // chikou[i] = close[i + kijun] when i + kijun < n, NaN otherwise
    // Per Python: chikou = close.shift(-kijun)
    // shift(-kijun) means chikou[i] = close[i + kijun]
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

    // For each index i >= period - 1, compute the rolling max/min over [i - period + 1, i]
    for i in (period - 1)..n {
        let start = i + 1 - period;
        let end = i + 1; // exclusive

        let mut max_high = f64::NEG_INFINITY;
        let mut min_low = f64::INFINITY;

        for j in start..end {
            let h = high[j];
            let l = low[j];

            if h.is_nan() || l.is_nan() {
                // If any value in the window is NaN, the result is NaN
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

    // Check for invalid parameters
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

    // Handle empty input
    if n == 0 {
        return Ok(MacdOutput {
            macd: Vec::new(),
            signal: Vec::new(),
            hist: Vec::new(),
        });
    }

    // MACD lookback is slow + signal - 2 (first valid index is slow + signal - 2)
    // Per requirements: return all-NaN if close.len() < slow + signal - 1
    let min_required = slow + signal - 1;
    if n < min_required {
        return Ok(MacdOutput {
            macd: vec![f64::NAN; n],
            signal: vec![f64::NAN; n],
            hist: vec![f64::NAN; n],
        });
    }

    // talib-rs requires fast >= 2 and slow >= 2
    // If fast == 1, we need to handle it specially
    if fast < 2 || slow < 2 {
        // For periods < 2, talib-rs will return InvalidParameter
        // Return all-NaN as a fallback
        return Ok(MacdOutput {
            macd: vec![f64::NAN; n],
            signal: vec![f64::NAN; n],
            hist: vec![f64::NAN; n],
        });
    }

    // Call talib-rs MACD function
    match talib_macd(close, fast, slow, signal) {
        Ok((macd_line, signal_line, histogram)) => {
            // talib-rs should return vectors of the same length as input
            // with leading NaNs for the warm-up period
            if macd_line.len() == n && signal_line.len() == n && histogram.len() == n {
                Ok(MacdOutput {
                    macd: macd_line,
                    signal: signal_line,
                    hist: histogram,
                })
            } else {
                // If talib-rs returns different lengths, pad with NaNs
                // This shouldn't happen with talib-rs 0.1.2, but handle it defensively
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
            // If talib-rs says insufficient data, return all-NaN
            Ok(MacdOutput {
                macd: vec![f64::NAN; n],
                signal: vec![f64::NAN; n],
                hist: vec![f64::NAN; n],
            })
        }
        Err(talib_rs::error::TaError::InvalidParameter { name, reason, .. }) => {
            // Map talib-rs invalid parameter to our InvalidPeriod
            Err(IndicatorError::InvalidPeriod {
                param_name: name,
                value: fast, // Use fast as a representative value
                reason,
            })
        }
        Err(_) => {
            // On any other error from talib-rs, return all-NaN
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

    // Check for length mismatch first (before period validation)
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

    // Check for invalid period
    if period == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "period",
            value: 0,
            reason: "period must be greater than 0",
        });
    }

    // Handle empty input
    if n == 0 {
        return Ok(Vec::new());
    }

    // ADX lookback is 2*period - 1
    let lookback = 2 * period - 1;

    // Handle insufficient data: close.len() < lookback
    // Return all-NaN vector per requirements
    if n < lookback {
        return Ok(vec![f64::NAN; n]);
    }

    // Call talib-rs ADX function
    match talib_adx(high, low, close, period) {
        Ok(result) => {
            // talib-rs should return a vector of the same length as input
            // with leading NaNs for the warm-up period
            if result.len() == n {
                Ok(result)
            } else {
                // If talib-rs returns a different length, pad with NaNs
                // This shouldn't happen with talib-rs 0.1.2, but handle it defensively
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
            // If talib-rs says insufficient data, return all-NaN
            Ok(vec![f64::NAN; n])
        }
        Err(talib_rs::error::TaError::LengthMismatch { expected, got }) => {
            // This shouldn't happen since we check lengths above, but handle it
            Err(IndicatorError::LengthMismatch {
                expected,
                actual: got,
                param_name: "input slices",
            })
        }
        Err(talib_rs::error::TaError::InvalidParameter { name, reason, .. }) => {
            // Map talib-rs invalid parameter to our InvalidPeriod
            Err(IndicatorError::InvalidPeriod {
                param_name: name,
                value: period,
                reason,
            })
        }
        Err(_) => {
            // On any other error from talib-rs, return all-NaN
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

    // Check for invalid period
    if period == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "period",
            value: 0,
            reason: "period must be greater than 0",
        });
    }

    // Check for empty input
    if n == 0 {
        return Err(IndicatorError::EmptyInput);
    }

    // Check if input contains NaN values - we need to handle NaN propagation
    let has_nan = close.iter().any(|&x| x.is_nan());

    // If input has NaN values, we need to handle NaN propagation manually
    // since talib-rs may not handle NaN inputs correctly
    if has_nan {
        return compute_ma_with_nan_propagation(close, kind, period);
    }

    // Call the appropriate talib-rs function based on the kind
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
            // talib-rs should return a vector of the same length as input
            // with leading NaNs for the warm-up period
            if output.len() == n {
                Ok(output)
            } else {
                // If talib-rs returns a different length, pad with NaNs
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
            // If talib-rs says insufficient data, return all-NaN
            Ok(vec![f64::NAN; n])
        }
        Err(talib_rs::error::TaError::InvalidParameter { name, reason, .. }) => {
            // Map talib-rs invalid parameter to our InvalidPeriod
            Err(IndicatorError::InvalidPeriod {
                param_name: name,
                value: period,
                reason,
            })
        }
        Err(_) => {
            // On any other error from talib-rs, return all-NaN
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

    // First, compute the MA on the original data (talib-rs may produce incorrect results with NaN)
    // We'll compute on a cleaned version and then apply NaN propagation

    // Create a version of close with NaN replaced by 0.0 for computation
    // (we'll overwrite the affected indices with NaN afterward)
    let close_cleaned: Vec<f64> = close.iter().map(|&x| if x.is_nan() { 0.0 } else { x }).collect();

    // Compute the MA on cleaned data
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

    // Now propagate NaN: for each output index, check if any input in its window is NaN
    // The window size depends on the MA type, but we use a conservative approach:
    // - For SMA/WMA/TRIMA: window is exactly `period` elements
    // - For EMA/KAMA: technically uses all previous data, but we approximate with `period`
    // - For DEMA: uses 2*period elements approximately
    // - For TEMA: uses 3*period elements approximately

    let window_size = match kind {
        MovingAverageKind::Sma | MovingAverageKind::Wma | MovingAverageKind::Trima => period,
        MovingAverageKind::Ema | MovingAverageKind::Kama => period,
        MovingAverageKind::Dema => 2 * period,
        MovingAverageKind::Tema => 3 * period,
    };

    // For each output index, check if any input in the window contains NaN
    for i in 0..n {
        if output[i].is_nan() {
            continue; // Already NaN, skip
        }

        // Check the window [i - window_size + 1, i] for NaN values
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

    // Iterate over the cartesian product of kinds and periods
    for kind in kinds {
        for &period in periods {
            // Compute the moving average for this (kind, period) pair
            let ma_values = moving_average(close, *kind, period)?;

            // Format the key as "{KIND}_{period}" in uppercase
            let key = format!("{}_{}", kind, period);

            result.insert(key, ma_values);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== MACD Tests ====================

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
        // First slow + signal - 2 = 26 + 9 - 2 = 33 elements should be NaN
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
        // From index lookback onwards, should have values
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
        // Verify hist[i] == macd[i] - signal[i] within tolerance for all finite indices
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
        // close.len() < slow + signal - 1 should return all-NaN
        // slow + signal - 1 = 26 + 9 - 1 = 34
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
        // Test with different periods
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = macd(&close, 5, 10, 3).unwrap();
        assert_eq!(result.macd.len(), 100);
        // lookback = slow + signal - 2 = 10 + 3 - 2 = 11
        let lookback = 10 + 3 - 2;
        assert!(result.macd[lookback - 1].is_nan());
        assert!(!result.macd[lookback].is_nan());
    }

    // ==================== ADX Tests ====================

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
        // First 2*period - 1 = 9 elements should be NaN
        for i in 0..9 {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }
        // From index 9 onwards, should have values
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
        // ADX values should be in [0.0, 100.0]
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
        // close.len() < 2*period - 1 should return all-NaN
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
        let high = vec![45.0, 45.5]; // Different length
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
        let low = vec![44.0, 44.5]; // Different length
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
        // talib-rs ADX requires period >= 2
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
        // talib-rs requires period >= 2, so period == 1 returns InvalidPeriod
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { .. })
        ));
    }

    // ==================== Moving Average Tests ====================

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
        // First period - 1 = 4 elements should be NaN for SMA
        for i in 0..4 {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }
        // From index 4 onwards, should have values
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
        // Test that NaN values in input propagate to output
        let mut close: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        close[10] = f64::NAN; // Insert NaN at index 10

        let result = moving_average(&close, MovingAverageKind::Sma, 5).unwrap();
        assert_eq!(result.len(), close.len());

        // The NaN at index 10 should propagate to indices 10-14 (window includes index 10)
        // For SMA with period 5, the window for index i is [i-4, i]
        // So indices 10, 11, 12, 13, 14 should be NaN
        for i in 10..=14 {
            assert!(
                result[i].is_nan(),
                "Expected NaN at index {} due to NaN propagation",
                i
            );
        }

        // Index 15 should not be NaN (window is [11, 15], doesn't include index 10)
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
        // Verify the default constants match the Python ta/trend/movingavg.py defaults
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
        // When close.len() < period, should return all-NaN
        let close: Vec<f64> = vec![1.0, 2.0, 3.0];
        let result = moving_average(&close, MovingAverageKind::Sma, 10).unwrap();
        assert_eq!(result.len(), 3);
        for val in &result {
            assert!(val.is_nan());
        }
    }

    // ==================== Moving Average Matrix Tests ====================

    #[test]
    fn moving_average_matrix_returns_correct_entries() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let kinds = &[MovingAverageKind::Sma, MovingAverageKind::Ema];
        let periods = &[10, 20];
        let result = moving_average_matrix(&close, kinds, periods).unwrap();

        // Should have 4 entries: SMA_10, SMA_20, EMA_10, EMA_20
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

        // Verify all keys are in uppercase format
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

        // All output vectors should have the same length as input
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

        // Should return empty HashMap
        assert!(result.is_empty());
    }

    #[test]
    fn moving_average_matrix_empty_periods() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let kinds = &[MovingAverageKind::Sma, MovingAverageKind::Ema];
        let periods: &[usize] = &[];
        let result = moving_average_matrix(&close, kinds, periods).unwrap();

        // Should return empty HashMap
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

        // DEFAULT_MA_KINDS = [Sma, Ema], DEFAULT_MA_PERIODS = [10, 20, 50]
        // Should have 6 entries
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

        // Verify that matrix values match individual moving_average calls
        for kind in kinds {
            for &period in periods {
                let key = format!("{}_{}", kind, period);
                let individual_result = moving_average(&close, *kind, period).unwrap();
                let matrix_values = matrix_result.get(&key).unwrap();

                for (i, (&matrix_val, &individual_val)) in
                    matrix_values.iter().zip(individual_result.iter()).enumerate()
                {
                    if matrix_val.is_nan() && individual_val.is_nan() {
                        continue; // Both NaN, considered equal
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

    // ==================== Ichimoku Cloud Tests ====================

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

        // Tenkan: first tenkan_period - 1 = 8 elements should be NaN
        for i in 0..8 {
            assert!(result.tenkan[i].is_nan(), "Expected NaN tenkan at index {}", i);
        }
        // From index 8 onwards, should have values
        assert!(!result.tenkan[8].is_nan(), "Expected non-NaN tenkan at index 8");
    }

    #[test]
    fn ichimoku_kijun_leading_nans() {
        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();

        // Kijun: first kijun_period - 1 = 25 elements should be NaN
        for i in 0..25 {
            assert!(result.kijun[i].is_nan(), "Expected NaN kijun at index {}", i);
        }
        // From index 25 onwards, should have values
        assert!(!result.kijun[25].is_nan(), "Expected non-NaN kijun at index 25");
    }

    #[test]
    fn ichimoku_senkou_a_leading_nans() {
        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();

        // Senkou A: shifted by kijun (26), so first 26 elements should be NaN
        // Plus the source needs kijun-1 elements to be valid
        // So first kijun + kijun - 1 = 51 elements should be NaN
        // Actually: senkou_a[i] = (tenkan[i-kijun] + kijun[i-kijun]) / 2
        // For i = 26, we need tenkan[0] and kijun[0], but kijun[0] is NaN
        // For i = 51, we need tenkan[25] and kijun[25], kijun[25] is valid
        for i in 0..26 {
            assert!(result.senkou_a[i].is_nan(), "Expected NaN senkou_a at index {}", i);
        }
        // At index 51, both tenkan[25] and kijun[25] should be valid
        assert!(!result.senkou_a[51].is_nan(), "Expected non-NaN senkou_a at index 51");
    }

    #[test]
    fn ichimoku_senkou_b_leading_nans() {
        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();

        // Senkou B: shifted by kijun (26), and needs senkou_b_period (52) elements
        // So first kijun + senkou_b_period - 1 = 26 + 52 - 1 = 77 elements should be NaN
        for i in 0..77 {
            assert!(result.senkou_b[i].is_nan(), "Expected NaN senkou_b at index {}", i);
        }
        // From index 77 onwards, should have values
        assert!(!result.senkou_b[77].is_nan(), "Expected non-NaN senkou_b at index 77");
    }

    #[test]
    fn ichimoku_chikou_trailing_nans() {
        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();

        // Chikou: chikou[i] = close[i + kijun], so last kijun elements should be NaN
        // For n=100, kijun=26: indices 74..100 should be NaN
        for i in 74..100 {
            assert!(result.chikou[i].is_nan(), "Expected NaN chikou at index {}", i);
        }
        // Index 73 should have a value (close[73 + 26] = close[99])
        assert!(!result.chikou[73].is_nan(), "Expected non-NaN chikou at index 73");
    }

    #[test]
    fn ichimoku_chikou_values() {
        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();

        // Verify chikou[i] = close[i + kijun]
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
        // Test with simple data where we can verify the computation
        let high = vec![10.0, 12.0, 11.0, 13.0, 12.0];
        let low = vec![8.0, 9.0, 9.0, 10.0, 10.0];
        let close = vec![9.0, 11.0, 10.0, 12.0, 11.0];
        let result = ichimoku(&high, &low, &close, 3, 3, 3).unwrap();

        // tenkan[2] = (max(high[0..=2]) + min(low[0..=2])) / 2
        // = (max(10, 12, 11) + min(8, 9, 9)) / 2 = (12 + 8) / 2 = 10.0
        assert!(
            (result.tenkan[2] - 10.0).abs() < 1e-12,
            "Tenkan at index 2: expected 10.0, got {}",
            result.tenkan[2]
        );

        // tenkan[3] = (max(high[1..=3]) + min(low[1..=3])) / 2
        // = (max(12, 11, 13) + min(9, 9, 10)) / 2 = (13 + 9) / 2 = 11.0
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
        // When period exceeds n, the affected line should be all-NaN (no error)
        let high = vec![10.0, 12.0, 11.0];
        let low = vec![8.0, 9.0, 9.0];
        let close = vec![9.0, 11.0, 10.0];
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();

        // All lines should be NaN since periods exceed length
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
        let high = vec![10.0, 12.0]; // Different length
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
        let low = vec![8.0, 9.0]; // Different length
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
        // Test senkou_a computation with simple data
        let high = vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0];
        let low = vec![8.0, 9.0, 9.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0];
        let close = vec![9.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0];
        let result = ichimoku(&high, &low, &close, 2, 3, 2).unwrap();

        // senkou_a[i] = (tenkan[i-kijun] + kijun[i-kijun]) / 2
        // For kijun=3, senkou_a[3] = (tenkan[0] + kijun[0]) / 2
        // But tenkan[0] is NaN (needs period-1=1 elements), so senkou_a[3] is NaN
        // senkou_a[4] = (tenkan[1] + kijun[1]) / 2
        // tenkan[1] = (max(high[0..=1]) + min(low[0..=1])) / 2 = (12 + 8) / 2 = 10.0
        // kijun[1] is NaN (needs period-1=2 elements)
        // So senkou_a[4] is NaN
        // senkou_a[5] = (tenkan[2] + kijun[2]) / 2
        // tenkan[2] = (max(high[1..=2]) + min(low[1..=2])) / 2 = (12 + 9) / 2 = 10.5
        // kijun[2] = (max(high[0..=2]) + min(low[0..=2])) / 2 = (12 + 8) / 2 = 10.0
        // senkou_a[5] = (10.5 + 10.0) / 2 = 10.25
        assert!(
            (result.senkou_a[5] - 10.25).abs() < 1e-12,
            "Senkou A at index 5: expected 10.25, got {}",
            result.senkou_a[5]
        );
    }

    #[test]
    fn ichimoku_with_nan_in_input() {
        // Test that NaN in input propagates correctly
        let mut high: Vec<f64> = (1..=20).map(|x| x as f64 + 1.0).collect();
        let mut low: Vec<f64> = (1..=20).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=20).map(|x| x as f64).collect();

        // Insert NaN at index 5
        high[5] = f64::NAN;
        low[5] = f64::NAN;

        let result = ichimoku(&high, &low, &close, 3, 5, 3).unwrap();

        // The NaN at index 5 should affect tenkan and kijun for indices 5, 6, 7
        // (window includes index 5)
        for i in 5..8 {
            assert!(
                result.tenkan[i].is_nan(),
                "Expected NaN tenkan at index {} due to NaN input",
                i
            );
        }
    }

    // ==================== Ichimoku Cloud NaN Placement Tests (Python Reference Parity) ====================
    // These tests verify NaN placement matches the Python ta/trend/ic.py reference
    // Requirements: 7.8

    #[test]
    fn ichimoku_nan_placement_tenkan_matches_python_reference() {
        // Validates: Requirements 7.3, 7.8
        // tenkan[i] should be NaN for indices < tenkan_period - 1
        let high: Vec<f64> = (1..=50).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=50).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        
        let tenkan_period = 9;
        let result = ichimoku(&high, &low, &close, tenkan_period, 26, 52).unwrap();
        
        // NaN for indices < tenkan_period - 1 (i.e., indices 0..8)
        for i in 0..(tenkan_period - 1) {
            assert!(
                result.tenkan[i].is_nan(),
                "tenkan[{}] should be NaN (< tenkan_period - 1 = {})",
                i, tenkan_period - 1
            );
        }
        
        // First valid value at index tenkan_period - 1
        assert!(
            !result.tenkan[tenkan_period - 1].is_nan(),
            "tenkan[{}] should be finite (first valid index)",
            tenkan_period - 1
        );
    }

    #[test]
    fn ichimoku_nan_placement_kijun_matches_python_reference() {
        // Validates: Requirements 7.4, 7.8
        // kijun[i] should be NaN for indices < kijun_period - 1
        let high: Vec<f64> = (1..=50).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=50).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        
        let kijun_period = 26;
        let result = ichimoku(&high, &low, &close, 9, kijun_period, 52).unwrap();
        
        // NaN for indices < kijun_period - 1 (i.e., indices 0..25)
        for i in 0..(kijun_period - 1) {
            assert!(
                result.kijun[i].is_nan(),
                "kijun[{}] should be NaN (< kijun_period - 1 = {})",
                i, kijun_period - 1
            );
        }
        
        // First valid value at index kijun_period - 1
        assert!(
            !result.kijun[kijun_period - 1].is_nan(),
            "kijun[{}] should be finite (first valid index)",
            kijun_period - 1
        );
    }

    #[test]
    fn ichimoku_nan_placement_senkou_a_matches_python_reference() {
        // Validates: Requirements 7.5, 7.8
        // senkou_a[i] = (tenkan[i-kijun] + kijun[i-kijun]) / 2
        // senkou_a[i] is NaN when i < kijun OR when tenkan[i-kijun] or kijun[i-kijun] is NaN
        // First valid senkou_a is at index kijun + max(tenkan_period, kijun_period) - 1
        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        
        let tenkan_period = 9;
        let kijun_period = 26;
        let result = ichimoku(&high, &low, &close, tenkan_period, kijun_period, 52).unwrap();
        
        // senkou_a[i] is NaN for i < kijun (due to shift)
        for i in 0..kijun_period {
            assert!(
                result.senkou_a[i].is_nan(),
                "senkou_a[{}] should be NaN (< kijun_period = {})",
                i, kijun_period
            );
        }
        
        // First valid senkou_a is at index kijun + kijun - 1 = 2*kijun - 1
        // because we need kijun[i-kijun] to be valid, which requires i-kijun >= kijun-1
        // i.e., i >= 2*kijun - 1
        let first_valid_senkou_a = 2 * kijun_period - 1;
        assert!(
            !result.senkou_a[first_valid_senkou_a].is_nan(),
            "senkou_a[{}] should be finite (first valid index)",
            first_valid_senkou_a
        );
    }

    #[test]
    fn ichimoku_nan_placement_senkou_b_matches_python_reference() {
        // Validates: Requirements 7.6, 7.8
        // senkou_b[i] is NaN for i < kijun + senkou_b_period - 1
        let high: Vec<f64> = (1..=100).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=100).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        
        let kijun_period = 26;
        let senkou_b_period = 52;
        let result = ichimoku(&high, &low, &close, 9, kijun_period, senkou_b_period).unwrap();
        
        // senkou_b[i] is NaN for i < kijun + senkou_b_period - 1
        let first_valid_senkou_b = kijun_period + senkou_b_period - 1;
        for i in 0..first_valid_senkou_b {
            assert!(
                result.senkou_b[i].is_nan(),
                "senkou_b[{}] should be NaN (< kijun + senkou_b_period - 1 = {})",
                i, first_valid_senkou_b
            );
        }
        
        // First valid value at index kijun + senkou_b_period - 1
        assert!(
            !result.senkou_b[first_valid_senkou_b].is_nan(),
            "senkou_b[{}] should be finite (first valid index)",
            first_valid_senkou_b
        );
    }

    #[test]
    fn ichimoku_nan_placement_chikou_matches_python_reference() {
        // Validates: Requirements 7.7, 7.8
        // chikou[i] = close[i + kijun] when i + kijun < n, NaN otherwise
        let n = 100;
        let high: Vec<f64> = (1..=n).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=n).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=n).map(|x| x as f64).collect();
        
        let kijun_period = 26;
        let result = ichimoku(&high, &low, &close, 9, kijun_period, 52).unwrap();
        
        // chikou[i] is NaN for i + kijun >= n, i.e., i >= n - kijun
        let first_nan_chikou = n as usize - kijun_period;
        
        // Valid values for i < n - kijun
        for i in 0..first_nan_chikou {
            assert!(
                !result.chikou[i].is_nan(),
                "chikou[{}] should be finite (i + kijun < n)",
                i
            );
        }
        
        // NaN for i >= n - kijun
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
        // Edge case: all periods exceed input length
        let high = vec![10.0, 12.0, 11.0, 13.0, 12.0];
        let low = vec![8.0, 9.0, 9.0, 10.0, 10.0];
        let close = vec![9.0, 11.0, 10.0, 12.0, 11.0];
        
        // All periods (9, 26, 52) exceed length (5)
        let result = ichimoku(&high, &low, &close, 9, 26, 52).unwrap();
        
        // All outputs should be NaN
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
        // Edge case: some periods exceed input length
        let high: Vec<f64> = (1..=15).map(|x| x as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=15).map(|x| x as f64 - 1.0).collect();
        let close: Vec<f64> = (1..=15).map(|x| x as f64).collect();
        
        // tenkan=3 (valid), kijun=5 (valid), senkou_b=20 (exceeds length 15)
        let result = ichimoku(&high, &low, &close, 3, 5, 20).unwrap();
        
        // tenkan should have valid values from index 2 onwards
        assert!(result.tenkan[1].is_nan());
        assert!(!result.tenkan[2].is_nan());
        
        // kijun should have valid values from index 4 onwards
        assert!(result.kijun[3].is_nan());
        assert!(!result.kijun[4].is_nan());
        
        // senkou_a should have valid values from index 2*kijun - 1 = 9 onwards
        assert!(result.senkou_a[8].is_nan());
        assert!(!result.senkou_a[9].is_nan());
        
        // senkou_b should be all NaN (kijun + senkou_b - 1 = 5 + 20 - 1 = 24 > 15)
        for i in 0..15 {
            assert!(result.senkou_b[i].is_nan(), "senkou_b[{}] should be NaN", i);
        }
        
        // chikou should have valid values for i < n - kijun = 15 - 5 = 10
        assert!(!result.chikou[9].is_nan());
        assert!(result.chikou[10].is_nan());
    }

    #[test]
    fn ichimoku_output_vector_lengths_always_match_input() {
        // Validates: Requirements 7.2
        // Output vectors should always have the same length as input
        let test_cases = vec![
            (1, 9, 26, 52),   // Very short input
            (10, 9, 26, 52),  // Short input
            (50, 9, 26, 52),  // Medium input
            (100, 9, 26, 52), // Standard input
            (200, 9, 26, 52), // Long input
            (50, 3, 5, 10),   // Custom periods
            (50, 1, 1, 1),    // Minimum periods
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
        // Validates: Requirements 7.3, 7.4, 7.8
        // Verify the computation formulas are correct
        let high = vec![10.0, 15.0, 12.0, 18.0, 14.0, 20.0, 16.0, 22.0, 18.0, 24.0];
        let low = vec![5.0, 8.0, 7.0, 10.0, 9.0, 12.0, 11.0, 14.0, 13.0, 16.0];
        let close = vec![7.0, 12.0, 10.0, 15.0, 12.0, 17.0, 14.0, 19.0, 16.0, 21.0];
        
        let result = ichimoku(&high, &low, &close, 3, 5, 3).unwrap();
        
        // tenkan[2] = (max(high[0..=2]) + min(low[0..=2])) / 2
        // = (max(10, 15, 12) + min(5, 8, 7)) / 2 = (15 + 5) / 2 = 10.0
        assert!(
            (result.tenkan[2] - 10.0).abs() < 1e-9,
            "tenkan[2] expected 10.0, got {}",
            result.tenkan[2]
        );
        
        // tenkan[5] = (max(high[3..=5]) + min(low[3..=5])) / 2
        // = (max(18, 14, 20) + min(10, 9, 12)) / 2 = (20 + 9) / 2 = 14.5
        assert!(
            (result.tenkan[5] - 14.5).abs() < 1e-9,
            "tenkan[5] expected 14.5, got {}",
            result.tenkan[5]
        );
        
        // kijun[4] = (max(high[0..=4]) + min(low[0..=4])) / 2
        // = (max(10, 15, 12, 18, 14) + min(5, 8, 7, 10, 9)) / 2 = (18 + 5) / 2 = 11.5
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

    // **Property 3: ADX Bounded Range**
    //
    // For any OHLC input with no NaN values and `period >= 1`,
    // all non-NaN ADX output values SHALL be in the closed range `[0.0, 100.0]`.
    //
    // **Validates: Requirements 4.3**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn adx_bounded_range(
            // Generate a vector of positive close prices (no NaN values)
            // Using positive values to simulate realistic price data
            close in prop::collection::vec(1.0f64..10000.0f64, 1..500),
            // Period must be >= 2 for talib-rs ADX (period 1 returns InvalidPeriod)
            period in 2usize..50
        ) {
            let n = close.len();

            // Generate high prices that are >= close prices
            // and low prices that are <= close prices to maintain OHLC validity
            let high: Vec<f64> = close.iter().map(|&c| c * 1.01).collect();
            let low: Vec<f64> = close.iter().map(|&c| c * 0.99).collect();

            let result = adx(&high, &low, &close, period);

            // ADX should return Ok for valid inputs
            prop_assert!(result.is_ok(), "ADX returned error: {:?}", result);

            let result = result.unwrap();

            // Verify output length matches input length
            prop_assert_eq!(result.len(), n);

            // Verify all non-NaN ADX values are in [0.0, 100.0]
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

    // **Property 4: MACD Histogram Identity**
    //
    // For any close price slice and valid MACD parameters, at every index `i`
    // where `macd[i]`, `signal[i]`, and `hist[i]` are all finite, the identity
    // `|hist[i] - (macd[i] - signal[i])| <= 1e-9` SHALL hold.
    //
    // **Validates: Requirements 5.4**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn macd_histogram_identity(
            // Generate a vector of positive close prices (no NaN values)
            // Using positive values to simulate realistic price data
            close in prop::collection::vec(1.0f64..10000.0f64, 1..500),
            // Fast period must be >= 2 for talib-rs
            fast in 2usize..20,
            // Slow period must be > fast, so we generate an offset and add to fast
            slow_offset in 1usize..30,
            // Signal period must be >= 1
            signal in 1usize..20
        ) {
            // Ensure slow > fast
            let slow = fast + slow_offset;

            let result = macd(&close, fast, slow, signal);

            // MACD should return Ok for valid inputs
            prop_assert!(result.is_ok(), "MACD returned error: {:?}", result);

            let output = result.unwrap();

            // Verify output lengths match input length
            let n = close.len();
            prop_assert_eq!(output.macd.len(), n);
            prop_assert_eq!(output.signal.len(), n);
            prop_assert_eq!(output.hist.len(), n);

            // Verify histogram identity: hist[i] == macd[i] - signal[i] within 1e-9
            // for all indices where all three values are finite
            for i in 0..n {
                let macd_val = output.macd[i];
                let signal_val = output.signal[i];
                let hist_val = output.hist[i];

                // Only check when all three values are finite (not NaN)
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
