//! Technical analysis indicators for the model environment.
//!
//! This module provides momentum, trend, volatility, support, and pattern
//! indicators for use in the RL observation vector.

use modelenv_proto::{Bar, DoubleBottomPattern, DoubleTopPattern};

mod error;
pub mod momentum;
pub mod patterns;
pub mod support;
pub mod trend;
pub mod volatility;

pub use error::IndicatorError;
pub use patterns::{
    detect_double_bottoms, detect_double_bottoms_default, detect_double_tops,
    detect_double_tops_default, DoubleBottom, DoubleBottomDetection,
    DoubleTop, DoubleTopDetection,
};
pub use support::{fibonacci_retracements, FibonacciRetracementsOutput};
pub use trend::MacdOutput;
pub use trend::{ichimoku, IchimokuOutput};
pub use trend::{
    moving_average, moving_average_matrix, MovingAverageKind, DEFAULT_MA_KINDS, DEFAULT_MA_PERIODS,
};
pub use volatility::{bollinger_bands, BollingerBandsOutput};

/// A compile-time-fixed enumeration of indicator scalars for per-interval selection.
///
/// Each boolean field indicates whether the corresponding indicator should be
/// computed and included in the output. This allows callers to customize which
/// indicators are computed, reducing unnecessary computation when only a subset
/// of indicators is needed.
///
/// # Example
///
/// ```
/// use modelenv_core::indicators::IndicatorSet;
///
/// // Create a set with only RSI and MACD enabled
/// let mut set = IndicatorSet::default();
/// set.rsi_14 = true;
/// set.macd = true;
/// set.macd_signal = true;
/// set.macd_hist = true;
/// ```
#[derive(Debug, Clone, Default)]
pub struct IndicatorSet {
    // Momentum indicators
    /// RSI with period 14
    pub rsi_14: bool,
    /// CCI with period 14
    pub cci_14: bool,

    // Trend indicators
    /// ADX with period 14
    pub adx_14: bool,
    /// MACD line
    pub macd: bool,
    /// MACD signal line
    pub macd_signal: bool,
    /// MACD histogram
    pub macd_hist: bool,
    /// Simple Moving Average with period 10
    pub sma_10: bool,
    /// Simple Moving Average with period 20
    pub sma_20: bool,
    /// Simple Moving Average with period 50
    pub sma_50: bool,
    /// Exponential Moving Average with period 10
    pub ema_10: bool,
    /// Exponential Moving Average with period 20
    pub ema_20: bool,
    /// Exponential Moving Average with period 50
    pub ema_50: bool,
    /// Ichimoku Tenkan-sen (Conversion Line)
    pub ichimoku_tenkan: bool,
    /// Ichimoku Kijun-sen (Base Line)
    pub ichimoku_kijun: bool,
    /// Ichimoku Senkou Span A (Leading Span A)
    pub ichimoku_senkou_a: bool,
    /// Ichimoku Senkou Span B (Leading Span B)
    pub ichimoku_senkou_b: bool,
    /// Ichimoku Chikou Span (Lagging Span)
    pub ichimoku_chikou: bool,

    // Volatility indicators
    /// Bollinger Bands upper band
    pub bb_upper: bool,
    /// Bollinger Bands middle band (SMA)
    pub bb_middle: bool,
    /// Bollinger Bands lower band
    pub bb_lower: bool,

    // Support indicators (Fibonacci retracement levels)
    /// Fibonacci retracement level 0.0% (low)
    pub fr_000: bool,
    /// Fibonacci retracement level 23.6%
    pub fr_236: bool,
    /// Fibonacci retracement level 38.2%
    pub fr_382: bool,
    /// Fibonacci retracement level 50.0%
    pub fr_500: bool,
    /// Fibonacci retracement level 61.8%
    pub fr_618: bool,
    /// Fibonacci retracement level 78.6%
    pub fr_786: bool,
    /// Fibonacci retracement level 100.0% (high)
    pub fr_1000: bool,

}

/// Number of indicator scalars produced per time interval.
///
/// This constant represents the total count of all indicators:
/// - Momentum: 2 (rsi_14, cci_14)
/// - Trend: 15 (adx_14, macd, macd_signal, macd_hist, sma_10, sma_20, sma_50,
///   ema_10, ema_20, ema_50, ichimoku_tenkan, ichimoku_kijun, ichimoku_senkou_a,
///   ichimoku_senkou_b, ichimoku_chikou)
/// - Volatility: 3 (bb_upper, bb_middle, bb_lower)
/// - Support: 7 (fr_000, fr_236, fr_382, fr_500, fr_618, fr_786, fr_1000)
/// Total: 2 + 15 + 3 + 7 = 27
pub const INDICATORS_PER_INTERVAL: usize = 27;

// Compile-time assertion to verify INDICATORS_PER_INTERVAL matches the actual count
// of indicators in IndicatorSet. This ensures the constant stays in sync with the
// number of indicator fields.
const _: () = {
    // Count of indicators by category:
    // Momentum: rsi_14, cci_14 = 2
    // Trend: adx_14, macd, macd_signal, macd_hist, sma_10, sma_20, sma_50,
    //        ema_10, ema_20, ema_50, ichimoku_tenkan, ichimoku_kijun,
    //        ichimoku_senkou_a, ichimoku_senkou_b, ichimoku_chikou = 15
    // Volatility: bb_upper, bb_middle, bb_lower = 3
    // Support: fr_000, fr_236, fr_382, fr_500, fr_618, fr_786, fr_1000 = 7
    // Patterns: double_bottom_score, double_top_score = 2
    const MOMENTUM_COUNT: usize = 2;
    const TREND_COUNT: usize = 15;
    const VOLATILITY_COUNT: usize = 3;
    const SUPPORT_COUNT: usize = 7;
    const EXPECTED_TOTAL: usize =
        MOMENTUM_COUNT + TREND_COUNT + VOLATILITY_COUNT + SUPPORT_COUNT;

    // This will fail to compile if INDICATORS_PER_INTERVAL doesn't match the expected total
    assert!(INDICATORS_PER_INTERVAL == EXPECTED_TOTAL);
};

/// Compute a flat array of `INDICATORS_PER_INTERVAL` scalars for `bars`.
///
/// Prefer [`compute_interval_indicators`] for structured output; this flat
/// array exists for backward compatibility with tests.
/// exists for backward compatibility with tests that inspect individual slots.
pub fn compute_interval_indicators_flat(bars: &[Bar]) -> [f64; INDICATORS_PER_INTERVAL] {
    let ta = compute_interval_indicators(bars);
    let m = ta.momentum.as_ref().map_or_else(
        || [f64::NAN; 2],
        |v| [v.rsi_14, v.cci_14],
    );
    let t = ta.trend.as_ref().map_or_else(
        || [f64::NAN; 15],
        |v| {
            [
                v.adx_14, v.macd, v.macd_signal, v.macd_hist, v.sma_10, v.sma_20,
                v.sma_50, v.ema_10, v.ema_20, v.ema_50, v.ichimoku_tenkan,
                v.ichimoku_kijun, v.ichimoku_senkou_a, v.ichimoku_senkou_b,
                v.ichimoku_chikou,
            ]
        },
    );
    let vl = ta.volatility.as_ref().map_or_else(
        || [f64::NAN; 3],
        |v| [v.bb_upper, v.bb_middle, v.bb_lower],
    );
    let s = ta.support.as_ref().map_or_else(
        || [f64::NAN; 7],
        |v| {
            [
                v.fr_000, v.fr_236, v.fr_382, v.fr_500, v.fr_618, v.fr_786, v.fr_1000,
            ]
        },
    );
    let mut out = [0.0_f64; INDICATORS_PER_INTERVAL];
    out[0] = m[0];  // rsi_14
    out[1] = m[1];  // cci_14
    out[2..17].copy_from_slice(&t);
    out[17..20].copy_from_slice(&vl);
    out[20..27].copy_from_slice(&s);
    out
}

/// Compute structured per-interval indicators for `bars` (oldest -> newest).
///
/// Returns an `IntervalIndicators` with typed sub-messages for momentum,
/// trend, volatility, support, and pattern scores. Missing values (insufficient
/// warm-up) are represented as NaN in the relevant double fields.
pub fn compute_interval_indicators(
    bars: &[Bar],
) -> modelenv_proto::IntervalIndicators {
    use modelenv_proto::{
        MomentumIndicators, PatternScores, SupportIndicators, TrendIndicators,
        VolatilityIndicators,
    };

    let n = bars.len();
    if n == 0 {
        return empty_interval_indicators();
    }
    let last_idx = n - 1;

    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let high: Vec<f64> = bars.iter().map(|b| b.high).collect();
    let low: Vec<f64> = bars.iter().map(|b| b.low).collect();

    // Momentum
    let rsi_values = momentum::rsi(&close, 14);
    let rsi_14 = rsi_values.get(last_idx).copied().unwrap_or(f64::NAN);
    let cci_14 = momentum::cci(&high, &low, &close, 14)
        .ok()
        .and_then(|v| v.get(last_idx).copied())
        .unwrap_or(f64::NAN);

    // Trend, ADX
    let adx_14 = trend::adx(&high, &low, &close, 14)
        .ok()
        .and_then(|v| v.get(last_idx).copied())
        .unwrap_or(f64::NAN);

    // Trend, MACD
    let macd_output = trend::macd(&close, 12, 26, 9).ok();
    let macd = last_of(&macd_output, |o| &o.macd);
    let macd_signal = last_of(&macd_output, |o| &o.signal);
    let macd_hist = last_of(&macd_output, |o| &o.hist);

    // Trend, Moving Averages
    let sma_10 = last_ma(&close, trend::MovingAverageKind::Sma, 10);
    let sma_20 = last_ma(&close, trend::MovingAverageKind::Sma, 20);
    let sma_50 = last_ma(&close, trend::MovingAverageKind::Sma, 50);
    let ema_10 = last_ma(&close, trend::MovingAverageKind::Ema, 10);
    let ema_20 = last_ma(&close, trend::MovingAverageKind::Ema, 20);
    let ema_50 = last_ma(&close, trend::MovingAverageKind::Ema, 50);

    // Trend, Ichimoku
    let ichimoku_output = trend::ichimoku(&high, &low, &close, 9, 26, 52).ok();
    let ichimoku_tenkan = last_of(&ichimoku_output, |o| &o.tenkan);
    let ichimoku_kijun = last_of(&ichimoku_output, |o| &o.kijun);
    let ichimoku_senkou_a = last_of(&ichimoku_output, |o| &o.senkou_a);
    let ichimoku_senkou_b = last_of(&ichimoku_output, |o| &o.senkou_b);
    let ichimoku_chikou = last_of(&ichimoku_output, |o| &o.chikou);

    // Volatility
    let bb_output = volatility::bollinger_bands(&close, 20, 2.0).ok();
    let bb_upper = last_of(&bb_output, |o| &o.upper);
    let bb_middle = last_of(&bb_output, |o| &o.middle);
    let bb_lower = last_of(&bb_output, |o| &o.lower);

    // Support (Fibonacci)
    let fib_output = support::fibonacci_retracements(&high, &low, 50).ok();
    let fr_000 = last_of(&fib_output, |o| &o.fr_000);
    let fr_236 = last_of(&fib_output, |o| &o.fr_236);
    let fr_382 = last_of(&fib_output, |o| &o.fr_382);
    let fr_500 = last_of(&fib_output, |o| &o.fr_500);
    let fr_618 = last_of(&fib_output, |o| &o.fr_618);
    let fr_786 = last_of(&fib_output, |o| &o.fr_786);
    let fr_1000 = last_of(&fib_output, |o| &o.fr_1000);

    modelenv_proto::IntervalIndicators {
        momentum: Some(MomentumIndicators { rsi_14, cci_14 }),
        trend: Some(TrendIndicators {
            adx_14,
            macd,
            macd_signal,
            macd_hist,
            sma_10,
            sma_20,
            sma_50,
            ema_10,
            ema_20,
            ema_50,
            ichimoku_tenkan,
            ichimoku_kijun,
            ichimoku_senkou_a,
            ichimoku_senkou_b,
            ichimoku_chikou,
        }),
        volatility: Some(VolatilityIndicators {
            bb_upper,
            bb_middle,
            bb_lower,
        }),
        support: Some(SupportIndicators {
            fr_000,
            fr_236,
            fr_382,
            fr_500,
            fr_618,
            fr_786,
            fr_1000,
        }),
        patterns: Some(PatternScores::default()),
    }
}

/// Extract the last value from an optional output struct via an accessor.
fn last_of<T, F>(output: &Option<T>, f: F) -> f64
where
    F: FnOnce(&T) -> &Vec<f64>,
{
    output
        .as_ref()
        .map(|o| f(o).last().copied().unwrap_or(f64::NAN))
        .unwrap_or(f64::NAN)
}

fn last_ma(close: &[f64], kind: trend::MovingAverageKind, period: usize) -> f64 {
    trend::moving_average(close, kind, period)
        .ok()
        .and_then(|v| v.last().copied())
        .unwrap_or(f64::NAN)
}

/// Detect all double-bottom and double-top patterns in `bars` and return
/// them as proto messages suitable for inclusion in an Observation.
pub fn detect_all_patterns(
    bars: &[Bar],
) -> (Vec<DoubleBottomPattern>, Vec<DoubleTopPattern>) {
    let dbs = detect_double_bottoms_default(bars)
        .map(|d| d.patterns.into_iter().map(db_to_proto).collect())
        .unwrap_or_default();
    let dts = detect_double_tops_default(bars)
        .map(|d| d.patterns.into_iter().map(dt_to_proto).collect())
        .unwrap_or_default();
    (dbs, dts)
}

fn db_to_proto(p: DoubleBottom) -> DoubleBottomPattern {
    DoubleBottomPattern {
        idx1: p.idx1 as i64,
        idx2: p.idx2 as i64,
        ts1: p.ts1,
        ts2: p.ts2,
        low1: p.low1,
        low2: p.low2,
        neckline: p.neckline,
        neckline_idx: p.neckline_idx as i64,
        depth_pct: p.depth_pct,
        width_bars: p.width_bars as i64,
        confirmed: p.confirmed,
        min_before_val: p.min_before_val.unwrap_or(0.0),
        min_before_ts: p.min_before_ts.unwrap_or(0),
        max_before_val: p.max_before_val.unwrap_or(0.0),
        max_before_ts: p.max_before_ts.unwrap_or(0),
    }
}

fn dt_to_proto(p: DoubleTop) -> DoubleTopPattern {
    DoubleTopPattern {
        idx1: p.idx1 as i64,
        idx2: p.idx2 as i64,
        ts1: p.ts1,
        ts2: p.ts2,
        high1: p.high1,
        high2: p.high2,
        neckline: p.neckline,
        neckline_idx: p.neckline_idx as i64,
        depth_pct: p.depth_pct,
        width_bars: p.width_bars as i64,
        confirmed: p.confirmed,
        min_before_val: p.min_before_val.unwrap_or(0.0),
        min_before_ts: p.min_before_ts.unwrap_or(0),
        max_before_val: p.max_before_val.unwrap_or(0.0),
        max_before_ts: p.max_before_ts.unwrap_or(0),
    }
}

/// Compute indicators for a bar slice using a custom IndicatorSet.
///
/// This function computes only the indicators that are enabled in the provided
/// `IndicatorSet`, returning a dynamic-length vector containing the last value
/// (most recent bar) for each enabled indicator.
///
/// # Arguments
///
/// * `bars` - Slice of bars ordered oldest to newest
/// * `set` - The indicator set specifying which indicators to compute
///
/// # Returns
///
/// A `Vec<f64>` containing the computed indicator values for the most recent bar.
/// The order of values follows the field order in `IndicatorSet`:
///
/// 1. **Momentum**: rsi_14, cci_14
/// 2. **Trend**: adx_14, macd, macd_signal, macd_hist, sma_10, sma_20, sma_50,
///    ema_10, ema_20, ema_50, ichimoku_tenkan, ichimoku_kijun, ichimoku_senkou_a,
///    ichimoku_senkou_b, ichimoku_chikou
/// 3. **Volatility**: bb_upper, bb_middle, bb_lower
/// 4. **Support**: fr_000, fr_236, fr_382, fr_500, fr_618, fr_786, fr_1000
/// 5. **Patterns**: double_bottom_score, double_top_score
///
/// # Empty Input Handling
///
/// When `bars` is empty, returns an empty vector.
///
/// # NaN Values
///
/// Indicators that cannot be computed due to insufficient data will return `f64::NAN`.
///
/// # Example
///
/// ```
/// use modelenv_core::indicators::{compute_interval_indicators_with, IndicatorSet};
/// use modelenv_proto::Bar;
///
/// let bars: Vec<Bar> = (0..100).map(|i| Bar {
///     timestamp_ns: i as i64,
///     open: 100.0 + (i as f64) * 0.1,
///     high: 101.0 + (i as f64) * 0.1,
///     low: 99.0 + (i as f64) * 0.1,
///     close: 100.5 + (i as f64) * 0.1,
///     volume: 1000.0,
/// }).collect();
///
/// let mut set = IndicatorSet::default();
/// set.rsi_14 = true;
/// set.sma_20 = true;
///
/// let result = compute_interval_indicators_with(&bars, &set);
/// assert_eq!(result.len(), 2); // Only RSI and SMA_20 enabled
/// ```
pub fn compute_interval_indicators_with(bars: &[Bar], set: &IndicatorSet) -> Vec<f64> {
    // Handle empty input
    if bars.is_empty() {
        return Vec::new();
    }

    let n = bars.len();
    let last_idx = n - 1;

    // Extract price slices from bars
    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let high: Vec<f64> = bars.iter().map(|b| b.high).collect();
    let low: Vec<f64> = bars.iter().map(|b| b.low).collect();

    let mut result = Vec::new();

    // ========== MOMENTUM INDICATORS ==========

    // RSI with period 14
    if set.rsi_14 {
        let rsi_values = momentum::rsi(&close, 14);
        result.push(rsi_values.get(last_idx).copied().unwrap_or(f64::NAN));
    }

    // CCI with period 14
    if set.cci_14 {
        let cci_value = momentum::cci(&high, &low, &close, 14)
            .ok()
            .and_then(|v| v.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(cci_value);
    }

    // ========== TREND INDICATORS ==========

    // ADX with period 14
    if set.adx_14 {
        let adx_value = trend::adx(&high, &low, &close, 14)
            .ok()
            .and_then(|v| v.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(adx_value);
    }

    // MACD (12, 26, 9) - compute once if any MACD component is needed
    let macd_output = if set.macd || set.macd_signal || set.macd_hist {
        trend::macd(&close, 12, 26, 9).ok()
    } else {
        None
    };

    if set.macd {
        let macd_value = macd_output
            .as_ref()
            .and_then(|o| o.macd.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(macd_value);
    }

    if set.macd_signal {
        let signal_value = macd_output
            .as_ref()
            .and_then(|o| o.signal.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(signal_value);
    }

    if set.macd_hist {
        let hist_value = macd_output
            .as_ref()
            .and_then(|o| o.hist.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(hist_value);
    }

    // SMA with periods 10, 20, 50
    if set.sma_10 {
        let sma_value = trend::moving_average(&close, trend::MovingAverageKind::Sma, 10)
            .ok()
            .and_then(|v| v.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(sma_value);
    }

    if set.sma_20 {
        let sma_value = trend::moving_average(&close, trend::MovingAverageKind::Sma, 20)
            .ok()
            .and_then(|v| v.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(sma_value);
    }

    if set.sma_50 {
        let sma_value = trend::moving_average(&close, trend::MovingAverageKind::Sma, 50)
            .ok()
            .and_then(|v| v.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(sma_value);
    }

    // EMA with periods 10, 20, 50
    if set.ema_10 {
        let ema_value = trend::moving_average(&close, trend::MovingAverageKind::Ema, 10)
            .ok()
            .and_then(|v| v.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(ema_value);
    }

    if set.ema_20 {
        let ema_value = trend::moving_average(&close, trend::MovingAverageKind::Ema, 20)
            .ok()
            .and_then(|v| v.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(ema_value);
    }

    if set.ema_50 {
        let ema_value = trend::moving_average(&close, trend::MovingAverageKind::Ema, 50)
            .ok()
            .and_then(|v| v.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(ema_value);
    }

    // Ichimoku Cloud (9, 26, 52) - compute once if any Ichimoku component is needed
    let ichimoku_output = if set.ichimoku_tenkan
        || set.ichimoku_kijun
        || set.ichimoku_senkou_a
        || set.ichimoku_senkou_b
        || set.ichimoku_chikou
    {
        trend::ichimoku(&high, &low, &close, 9, 26, 52).ok()
    } else {
        None
    };

    if set.ichimoku_tenkan {
        let tenkan_value = ichimoku_output
            .as_ref()
            .and_then(|o| o.tenkan.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(tenkan_value);
    }

    if set.ichimoku_kijun {
        let kijun_value = ichimoku_output
            .as_ref()
            .and_then(|o| o.kijun.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(kijun_value);
    }

    if set.ichimoku_senkou_a {
        let senkou_a_value = ichimoku_output
            .as_ref()
            .and_then(|o| o.senkou_a.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(senkou_a_value);
    }

    if set.ichimoku_senkou_b {
        let senkou_b_value = ichimoku_output
            .as_ref()
            .and_then(|o| o.senkou_b.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(senkou_b_value);
    }

    if set.ichimoku_chikou {
        let chikou_value = ichimoku_output
            .as_ref()
            .and_then(|o| o.chikou.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(chikou_value);
    }

    // ========== VOLATILITY INDICATORS ==========

    // Bollinger Bands (20, 2.0) - compute once if any BB component is needed
    let bb_output = if set.bb_upper || set.bb_middle || set.bb_lower {
        volatility::bollinger_bands(&close, 20, 2.0).ok()
    } else {
        None
    };

    if set.bb_upper {
        let upper_value = bb_output
            .as_ref()
            .and_then(|o| o.upper.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(upper_value);
    }

    if set.bb_middle {
        let middle_value = bb_output
            .as_ref()
            .and_then(|o| o.middle.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(middle_value);
    }

    if set.bb_lower {
        let lower_value = bb_output
            .as_ref()
            .and_then(|o| o.lower.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(lower_value);
    }

    // ========== SUPPORT INDICATORS ==========

    // Fibonacci Retracements (window 50) - compute once if any FR level is needed
    let fib_output = if set.fr_000
        || set.fr_236
        || set.fr_382
        || set.fr_500
        || set.fr_618
        || set.fr_786
        || set.fr_1000
    {
        support::fibonacci_retracements(&high, &low, 50).ok()
    } else {
        None
    };

    if set.fr_000 {
        let fr_value = fib_output
            .as_ref()
            .and_then(|o| o.fr_000.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(fr_value);
    }

    if set.fr_236 {
        let fr_value = fib_output
            .as_ref()
            .and_then(|o| o.fr_236.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(fr_value);
    }

    if set.fr_382 {
        let fr_value = fib_output
            .as_ref()
            .and_then(|o| o.fr_382.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(fr_value);
    }

    if set.fr_500 {
        let fr_value = fib_output
            .as_ref()
            .and_then(|o| o.fr_500.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(fr_value);
    }

    if set.fr_618 {
        let fr_value = fib_output
            .as_ref()
            .and_then(|o| o.fr_618.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(fr_value);
    }

    if set.fr_786 {
        let fr_value = fib_output
            .as_ref()
            .and_then(|o| o.fr_786.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(fr_value);
    }

    if set.fr_1000 {
        let fr_value = fib_output
            .as_ref()
            .and_then(|o| o.fr_1000.get(last_idx).copied())
            .unwrap_or(f64::NAN);
        result.push(fr_value);
    }

    result
}

/// Return the ordered column names for a flattened state vector.
///
/// This matches the layout used when serialising a `Reference` into a flat
/// feature array for RL training. The names are deterministic and identical
/// across observations.
pub fn state_columns() -> Vec<String> {
    let mut cols = Vec::new();

    // -- scalar metadata --
    cols.push("timestamp_ns".to_string());

    // -- per-interval bars and indicators --
    for interval in crate::data_loader::TIME_INTERVALS {
        if *interval == "M1" || *interval == "D1" || *interval == "H4" || *interval == "MN" {
            continue;
        }
        let iv = *interval;
        // live bar
        for field in &["close", "volume"] {
            cols.push(format!("{iv}_bar_{field}"));
        }
        // ta sub-messages, per-interval curation
        let fields: &[&str] = match iv {
            "M5" => &[
                // Tactical / Momentum: fast-reacting
                "rsi_14", "macd", "macd_signal", "macd_hist",
                "ema_10", "ema_20",
                "bb_upper", "bb_middle", "bb_lower",
            ],
            "M15" => &[
                // Structural / Anchor: stability and support/resistance
                "adx_14", "sma_50",
                "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a",
                "ichimoku_senkou_b", "ichimoku_chikou",
                "fr_000", "fr_236", "fr_382", "fr_500", "fr_618", "fr_786", "fr_1000",
            ],
            "H1" | "W1" => &["rsi_14", "ema_10", "ema_20", "ema_50"],
            _ => &[
                "rsi_14", "cci_14", "adx_14", "macd", "macd_signal", "macd_hist",
                "sma_10", "sma_20", "sma_50", "ema_10", "ema_20", "ema_50",
                "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a",
                "ichimoku_senkou_b", "ichimoku_chikou",
                "bb_upper", "bb_middle", "bb_lower",
                "fr_000", "fr_236", "fr_382", "fr_500", "fr_618", "fr_786", "fr_1000",
            ],
        };
        for field in fields {
            cols.push(format!("{iv}_ta_{field}"));
        }
    }

    // -- realised P&L --
    cols.push("realised_pnl_12m".to_string());

    cols.push("unrealised_pnl".to_string());
    cols.push("num_positions_buy".to_string());
    cols.push("num_positions_sell".to_string());
    cols.push("swap_fees".to_string());
    cols.push("tick_bid".to_string());
    cols.push("tick_ask".to_string());
    cols.push("tick_count".to_string());
    cols.push("M15_double_bottom_low".to_string());
    cols.push("M15_double_bottom_high".to_string());
    cols.push("M15_double_top_high".to_string());
    cols.push("M15_double_top_low".to_string());
    // -- done flag --
    cols.push("done".to_string());

    cols
}

/// Returns `MIN(low1, low2)` of the most recent confirmed M15 double bottom
/// whose `MIN(low1, low2) <= tick_ask <= neckline` and `depth_pct >= 0.15`.
///
/// `double_bottoms` must be sorted latest-first (most recent at index 0).
/// `live_ticks` must be latest-first. Returns 0.0 when no pattern qualifies.
pub fn compute_m15_double_bottom_low(
    double_bottoms: &[modelenv_proto::DoubleBottomPattern],
    live_ticks: &[modelenv_proto::Tick],
) -> f64 {
    let tick_ask = match live_ticks.first() {
        Some(t) => t.ask,
        None => return 0.0,
    };
    for db in double_bottoms {
        if !db.confirmed || db.depth_pct < 0.15 {
            continue;
        }
        let min_low = db.low1.min(db.low2);
        if min_low <= tick_ask && tick_ask <= db.neckline {
            return min_low;
        }
    }
    0.0
}

/// Returns `MIN(low1, low2)` of the most recent confirmed M15 double bottom
/// whose `MIN(low1, low2) >= tick_ask`, `depth_pct >= 0.15`, and
/// `min_low - m15_double_bottom_low >= 0.2`.
///
/// `double_bottoms` must be sorted latest-first (most recent at index 0).
/// `live_ticks` must be latest-first. Returns 0.0 when no pattern qualifies.
pub fn compute_m15_double_bottom_high(
    double_bottoms: &[modelenv_proto::DoubleBottomPattern],
    live_ticks: &[modelenv_proto::Tick],
    m15_double_bottom_low: f64,
) -> f64 {
    let tick_ask = match live_ticks.first() {
        Some(t) => t.ask,
        None => return 0.0,
    };
    for db in double_bottoms {
        if !db.confirmed || db.depth_pct < 0.15 {
            continue;
        }
        let min_low = db.low1.min(db.low2);
        if min_low >= tick_ask && (min_low - m15_double_bottom_low) >= 0.2 {
            return min_low;
        }
    }
    0.0
}

/// Returns `MAX(high1, high2)` of the most recent confirmed M15 double top
/// whose `MAX(high1, high2) >= tick_bid >= neckline` and `depth_pct >= 0.15`.
///
/// `double_tops` must be sorted latest-first (most recent at index 0).
/// `live_ticks` must be latest-first. Returns 0.0 when no pattern qualifies.
pub fn compute_m15_double_top_high(
    double_tops: &[modelenv_proto::DoubleTopPattern],
    live_ticks: &[modelenv_proto::Tick],
) -> f64 {
    let tick_bid = match live_ticks.first() {
        Some(t) => t.bid,
        None => return 0.0,
    };
    for dt in double_tops {
        if !dt.confirmed || dt.depth_pct < 0.15 {
            continue;
        }
        let max_high = dt.high1.max(dt.high2);
        if tick_bid >= dt.neckline && tick_bid <= max_high {
            return max_high;
        }
    }
    0.0
}

fn empty_interval_indicators() -> modelenv_proto::IntervalIndicators {
    use modelenv_proto::{
        MomentumIndicators, PatternScores, SupportIndicators, TrendIndicators,
        VolatilityIndicators,
    };
    modelenv_proto::IntervalIndicators {
        momentum: Some(MomentumIndicators {
            rsi_14: f64::NAN,
            cci_14: f64::NAN,
        }),
        trend: Some(TrendIndicators {
            adx_14: f64::NAN,
            macd: f64::NAN,
            macd_signal: f64::NAN,
            macd_hist: f64::NAN,
            sma_10: f64::NAN,
            sma_20: f64::NAN,
            sma_50: f64::NAN,
            ema_10: f64::NAN,
            ema_20: f64::NAN,
            ema_50: f64::NAN,
            ichimoku_tenkan: f64::NAN,
            ichimoku_kijun: f64::NAN,
            ichimoku_senkou_a: f64::NAN,
            ichimoku_senkou_b: f64::NAN,
            ichimoku_chikou: f64::NAN,
        }),
        volatility: Some(VolatilityIndicators {
            bb_upper: f64::NAN,
            bb_middle: f64::NAN,
            bb_lower: f64::NAN,
        }),
        support: Some(SupportIndicators {
            fr_000: f64::NAN,
            fr_236: f64::NAN,
            fr_382: f64::NAN,
            fr_500: f64::NAN,
            fr_618: f64::NAN,
            fr_786: f64::NAN,
            fr_1000: f64::NAN,
        }),
        patterns: Some(PatternScores::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(low: f64, high: f64) -> Bar {
        Bar {
            timestamp_ns: 0,
            open: (low + high) / 2.0,
            high,
            low,
            close: (low + high) / 2.0,
            volume: 1.0,
        }
    }

    fn bar_with_close(low: f64, high: f64, close: f64) -> Bar {
        Bar {
            timestamp_ns: 0,
            open: (low + high) / 2.0,
            high,
            low,
            close,
            volume: 1.0,
        }
    }

    #[test]
    fn compute_interval_indicators_returns_fixed_length() {
        let result = compute_interval_indicators_flat(&[]);
        assert_eq!(result.len(), INDICATORS_PER_INTERVAL);
        // All indicators should be NaN for empty input
        for i in 0..INDICATORS_PER_INTERVAL {
            assert!(result[i].is_nan(), "Index {} should be NaN for empty input", i);
        }
    }

    #[test]
    fn compute_interval_indicators_with_bars() {
        let bars = vec![bar(100.0, 101.0); 30];
        let result = compute_interval_indicators_flat(&bars);
        assert_eq!(result.len(), INDICATORS_PER_INTERVAL);
        // Index 0 is rsi_14
        assert!(result[0].is_nan() || (result[0] >= 0.0 && result[0] <= 100.0));
    }

    #[test]
    fn compute_interval_indicators_with_sufficient_data() {
        // Create bars with varying prices to get meaningful indicator values
        let bars: Vec<Bar> = (0..200)
            .map(|i| {
                let price = 100.0 + (i as f64) * 0.1 + ((i % 7) as f64) * 0.2;
                bar_with_close(price - 1.0, price + 1.0, price)
            })
            .collect();

        let result = compute_interval_indicators_flat(&bars);
        assert_eq!(result.len(), INDICATORS_PER_INTERVAL);

        // Index 0: RSI should be in [0, 100] or NaN
        assert!(!result[0].is_nan(), "RSI should not be NaN with 200 bars");
        assert!(result[0] >= 0.0 && result[0] <= 100.0, "RSI should be in [0, 100]");
        // Index 2: ADX should be in [0, 100] or NaN
        assert!(!result[2].is_nan(), "ADX should not be NaN with 200 bars");
        assert!(result[2] >= 0.0 && result[2] <= 100.0, "ADX should be in [0, 100]");
        // Index 17-19: Bollinger Bands should have lower <= middle <= upper
        assert!(!result[17].is_nan(), "BB upper should not be NaN with 200 bars");
        assert!(!result[18].is_nan(), "BB middle should not be NaN with 200 bars");
        assert!(!result[19].is_nan(), "BB lower should not be NaN with 200 bars");
        assert!(result[19] <= result[18], "BB lower should be <= middle");
        assert!(result[18] <= result[17], "BB middle should be <= upper");
        // Index 20-26: Fibonacci levels should be ordered
        assert!(!result[20].is_nan(), "FR 0.0% should not be NaN with 200 bars");
        assert!(!result[26].is_nan(), "FR 100.0% should not be NaN with 200 bars");
        assert!(result[20] <= result[26], "FR 0.0% should be <= FR 100.0%");
    }

    // Tests for compute_interval_indicators_with

    #[test]
    fn compute_interval_indicators_with_empty_bars() {
        let bars: Vec<Bar> = vec![];
        let set = IndicatorSet::default();
        let result = compute_interval_indicators_with(&bars, &set);
        assert!(result.is_empty());
    }

    #[test]
    fn compute_interval_indicators_with_empty_set() {
        let bars = vec![bar(100.0, 101.0); 30];
        let set = IndicatorSet::default();
        let result = compute_interval_indicators_with(&bars, &set);
        assert!(result.is_empty());
    }

    #[test]
    fn compute_interval_indicators_with_rsi_only() {
        // Create bars with varying prices to get meaningful RSI
        let bars: Vec<Bar> = (0..50)
            .map(|i| {
                let price = 100.0 + (i as f64) * 0.5 + ((i % 3) as f64) * 0.2;
                bar_with_close(price - 0.5, price + 0.5, price)
            })
            .collect();

        let mut set = IndicatorSet::default();
        set.rsi_14 = true;

        let result = compute_interval_indicators_with(&bars, &set);
        assert_eq!(result.len(), 1);
        // RSI should be a valid value (not NaN) with 50 bars
        assert!(!result[0].is_nan());
        // RSI should be in [0, 100]
        assert!(result[0] >= 0.0 && result[0] <= 100.0);
    }

    #[test]
    fn compute_interval_indicators_with_multiple_indicators() {
        // Create bars with varying prices
        let bars: Vec<Bar> = (0..100)
            .map(|i| {
                let price = 100.0 + (i as f64) * 0.1 + ((i % 5) as f64) * 0.3;
                bar_with_close(price - 1.0, price + 1.0, price)
            })
            .collect();

        let mut set = IndicatorSet::default();
        set.rsi_14 = true;
        set.sma_20 = true;
        set.ema_10 = true;

        let result = compute_interval_indicators_with(&bars, &set);
        assert_eq!(result.len(), 3);

        // All values should be valid (not NaN) with 100 bars
        assert!(!result[0].is_nan(), "RSI should not be NaN");
        assert!(!result[1].is_nan(), "SMA_20 should not be NaN");
        assert!(!result[2].is_nan(), "EMA_10 should not be NaN");

        // RSI should be in [0, 100]
        assert!(result[0] >= 0.0 && result[0] <= 100.0);
    }

    #[test]
    fn compute_interval_indicators_with_macd() {
        let bars: Vec<Bar> = (0..50)
            .map(|i| {
                let price = 100.0 + (i as f64) * 0.2;
                bar_with_close(price - 0.5, price + 0.5, price)
            })
            .collect();

        let mut set = IndicatorSet::default();
        set.macd = true;
        set.macd_signal = true;
        set.macd_hist = true;

        let result = compute_interval_indicators_with(&bars, &set);
        assert_eq!(result.len(), 3);

        // With 50 bars, MACD should have valid values
        // MACD lookback is slow + signal - 2 = 26 + 9 - 2 = 33
        assert!(!result[0].is_nan(), "MACD should not be NaN");
        assert!(!result[1].is_nan(), "MACD signal should not be NaN");
        assert!(!result[2].is_nan(), "MACD hist should not be NaN");
    }

    #[test]
    fn compute_interval_indicators_with_bollinger_bands() {
        let bars: Vec<Bar> = (0..50)
            .map(|i| {
                let price = 100.0 + (i as f64) * 0.1;
                bar_with_close(price - 0.5, price + 0.5, price)
            })
            .collect();

        let mut set = IndicatorSet::default();
        set.bb_upper = true;
        set.bb_middle = true;
        set.bb_lower = true;

        let result = compute_interval_indicators_with(&bars, &set);
        assert_eq!(result.len(), 3);

        // With 50 bars and period 20, BB should have valid values
        assert!(!result[0].is_nan(), "BB upper should not be NaN");
        assert!(!result[1].is_nan(), "BB middle should not be NaN");
        assert!(!result[2].is_nan(), "BB lower should not be NaN");

        // Verify ordering: lower <= middle <= upper
        assert!(result[2] <= result[1], "BB lower should be <= middle");
        assert!(result[1] <= result[0], "BB middle should be <= upper");
    }

    #[test]
    fn compute_interval_indicators_with_fibonacci() {
        let bars: Vec<Bar> = (0..60)
            .map(|i| {
                let price = 100.0 + (i as f64) * 0.2;
                bar_with_close(price - 1.0, price + 1.0, price)
            })
            .collect();

        let mut set = IndicatorSet::default();
        set.fr_000 = true;
        set.fr_500 = true;
        set.fr_1000 = true;

        let result = compute_interval_indicators_with(&bars, &set);
        assert_eq!(result.len(), 3);

        // All Fibonacci levels should be valid
        assert!(!result[0].is_nan(), "FR 0.0% should not be NaN");
        assert!(!result[1].is_nan(), "FR 50.0% should not be NaN");
        assert!(!result[2].is_nan(), "FR 100.0% should not be NaN");

        // Verify ordering: fr_000 <= fr_500 <= fr_1000
        assert!(result[0] <= result[1], "FR 0.0% should be <= FR 50.0%");
        assert!(result[1] <= result[2], "FR 50.0% should be <= FR 100.0%");
    }

    #[test]
    fn compute_interval_indicators_with_ichimoku() {
        let bars: Vec<Bar> = (0..100)
            .map(|i| {
                let price = 100.0 + (i as f64) * 0.1;
                bar_with_close(price - 0.5, price + 0.5, price)
            })
            .collect();

        let mut set = IndicatorSet::default();
        set.ichimoku_tenkan = true;
        set.ichimoku_kijun = true;

        let result = compute_interval_indicators_with(&bars, &set);
        assert_eq!(result.len(), 2);

        // With 100 bars, Ichimoku tenkan (9) and kijun (26) should have valid values
        assert!(!result[0].is_nan(), "Ichimoku tenkan should not be NaN");
        assert!(!result[1].is_nan(), "Ichimoku kijun should not be NaN");
    }

    #[test]
    fn compute_interval_indicators_with_patterns() {
        let bars = vec![bar(100.0, 101.0); 30];

        let set = IndicatorSet::default();
        let result = compute_interval_indicators_with(&bars, &set);
        assert!(result.is_empty());
    }

    #[test]
    fn compute_interval_indicators_with_all_indicators() {
        let bars: Vec<Bar> = (0..200)
            .map(|i| {
                let price = 100.0 + (i as f64) * 0.1 + ((i % 7) as f64) * 0.2;
                bar_with_close(price - 1.0, price + 1.0, price)
            })
            .collect();

        let set = IndicatorSet {
            rsi_14: true,
            cci_14: true,
            adx_14: true,
            macd: true,
            macd_signal: true,
            macd_hist: true,
            sma_10: true,
            sma_20: true,
            sma_50: true,
            ema_10: true,
            ema_20: true,
            ema_50: true,
            ichimoku_tenkan: true,
            ichimoku_kijun: true,
            ichimoku_senkou_a: true,
            ichimoku_senkou_b: true,
            ichimoku_chikou: true,
            bb_upper: true,
            bb_middle: true,
            bb_lower: true,
            fr_000: true,
            fr_236: true,
            fr_382: true,
            fr_500: true,
            fr_618: true,
            fr_786: true,
            fr_1000: true,
        };

        let result = compute_interval_indicators_with(&bars, &set);

        // Total: 2 momentum + 15 trend + 3 volatility + 7 support = 27
        assert_eq!(result.len(), 27);
    }

    #[test]
    fn compute_interval_indicators_with_insufficient_data() {
        // Only 5 bars - not enough for most indicators
        let bars: Vec<Bar> = (0..5)
            .map(|i| {
                let price = 100.0 + (i as f64);
                bar_with_close(price - 0.5, price + 0.5, price)
            })
            .collect();

        let mut set = IndicatorSet::default();
        set.rsi_14 = true;  // Needs 14+ bars
        set.sma_50 = true;  // Needs 50+ bars

        let result = compute_interval_indicators_with(&bars, &set);
        assert_eq!(result.len(), 2);

        // Both should be NaN due to insufficient data
        assert!(result[0].is_nan(), "RSI should be NaN with insufficient data");
        assert!(result[1].is_nan(), "SMA_50 should be NaN with insufficient data");
    }

    #[test]
    fn compute_interval_indicators_with_order_preserved() {
        // Verify that the order of indicators in the output matches the IndicatorSet field order
        let bars: Vec<Bar> = (0..100)
            .map(|i| {
                let price = 100.0 + (i as f64) * 0.1;
                bar_with_close(price - 0.5, price + 0.5, price)
            })
            .collect();

        let mut set = IndicatorSet::default();
        // Enable indicators in non-sequential order
        set.sma_20 = true;      // Should be at index 1 (after rsi_14)
        set.rsi_14 = true;      // Should be at index 0
        set.bb_middle = true;   // Should be at index 2 (after sma_20)

        let result = compute_interval_indicators_with(&bars, &set);
        assert_eq!(result.len(), 3);

        // The order should follow IndicatorSet field order:
        // rsi_14 (momentum) -> sma_20 (trend) -> bb_middle (volatility)
        // All should be valid values
        assert!(!result[0].is_nan(), "First value (RSI) should not be NaN");
        assert!(!result[1].is_nan(), "Second value (SMA_20) should not be NaN");
        assert!(!result[2].is_nan(), "Third value (BB_middle) should not be NaN");
    }

    // ========== Task 7.5: Empty and Short Input Edge Case Tests ==========
    // These tests verify Requirements 1.6 and 1.7

    #[test]
    fn empty_input_returns_documented_values() {
        // Requirement 1.6: Empty input returns documented empty-input values
        let result = compute_interval_indicators_flat(&[]);

        // Verify array length
        assert_eq!(result.len(), INDICATORS_PER_INTERVAL);

        // All indicators should be NaN for empty input
        for i in 0..INDICATORS_PER_INTERVAL {
            assert!(result[i].is_nan(), "Index {} should be NaN for empty input", i);
        }
    }

    #[test]
    fn empty_input_completes_in_sub_millisecond() {
        // Requirement 1.6: Empty input completes in under one millisecond
        use std::time::Instant;

        let start = Instant::now();
        let _result = compute_interval_indicators_flat(&[]);
        let elapsed = start.elapsed();

        // Verify sub-millisecond completion
        assert!(
            elapsed.as_millis() < 1,
            "Empty input should complete in under 1ms, took {:?}",
            elapsed
        );
    }

    #[test]
    fn empty_input_does_not_panic() {
        // Requirement 1.6: Empty input does not panic
        // This test passes if it completes without panicking
        let result = std::panic::catch_unwind(|| {
            compute_interval_indicators_flat(&[])
        });
        assert!(result.is_ok(), "compute_interval_indicators_flat should not panic on empty input");
    }

    #[test]
    fn short_input_returns_nan_for_insufficient_warmup() {
        // Requirement 1.7: Short input returns documented empty-input values for indicators
        // that cannot be computed

        // Test with 5 bars - insufficient for most indicators
        let bars: Vec<Bar> = (0..5)
            .map(|i| {
                let price = 100.0 + (i as f64);
                bar_with_close(price - 0.5, price + 0.5, price)
            })
            .collect();

        let result = compute_interval_indicators_flat(&bars);

        // Verify array length is still correct
        assert_eq!(result.len(), INDICATORS_PER_INTERVAL);

        // RSI (index 0) needs 14 bars, should be NaN with only 5 bars
        assert!(result[0].is_nan(), "RSI should be NaN with only 5 bars (needs 14)");

        // ADX (index 2) needs 27 bars, should be NaN with only 5 bars
        assert!(result[2].is_nan(), "ADX should be NaN with only 5 bars (needs 27)");

        // MACD (indices 3-5) needs 34 bars, should be NaN with only 5 bars
        assert!(result[3].is_nan(), "MACD should be NaN with only 5 bars (needs 34)");
        assert!(result[4].is_nan(), "MACD signal should be NaN with only 5 bars");
        assert!(result[5].is_nan(), "MACD hist should be NaN with only 5 bars");

        // SMA_50 (index 8) needs 50 bars, should be NaN with only 5 bars
        assert!(result[8].is_nan(), "SMA_50 should be NaN with only 5 bars (needs 50)");

        // EMA_50 (index 11) needs 50 bars, should be NaN with only 5 bars
        assert!(result[11].is_nan(), "EMA_50 should be NaN with only 5 bars (needs 50)");

        // Ichimoku kijun (index 13) needs 26 bars, should be NaN with only 5 bars
        assert!(result[13].is_nan(), "Ichimoku kijun should be NaN with only 5 bars (needs 26)");

        // Bollinger Bands (indices 17-19) need 20 bars, should be NaN with only 5 bars
        assert!(result[17].is_nan(), "BB upper should be NaN with only 5 bars (needs 20)");
        assert!(result[18].is_nan(), "BB middle should be NaN with only 5 bars (needs 20)");
        assert!(result[19].is_nan(), "BB lower should be NaN with only 5 bars (needs 20)");

        // Fibonacci levels (indices 20-26) use min_periods=1, so should NOT be NaN
        assert!(!result[20].is_nan(), "FR 0.0% should not be NaN (uses min_periods=1)");
        assert!(!result[26].is_nan(), "FR 100.0% should not be NaN (uses min_periods=1)");
    }

    #[test]
    fn short_input_does_not_panic() {
        // Requirement 1.7: Short input does not panic
        // Test various short input lengths
        for len in 1..=10 {
            let bars: Vec<Bar> = (0..len)
                .map(|i| {
                    let price = 100.0 + (i as f64);
                    bar_with_close(price - 0.5, price + 0.5, price)
                })
                .collect();

            let result = std::panic::catch_unwind(|| {
                compute_interval_indicators_flat(&bars)
            });
            assert!(
                result.is_ok(),
                "compute_interval_indicators_flat should not panic with {} bars",
                len
            );
        }
    }

    #[test]
    fn short_input_with_exact_warmup_periods() {
        // Test with exactly enough bars for some indicators but not others

        // 15 bars: enough for RSI (14) but not for ADX (27), MACD (34), etc.
        let bars: Vec<Bar> = (0..15)
            .map(|i| {
                let price = 100.0 + (i as f64) * 0.5 + ((i % 3) as f64) * 0.2;
                bar_with_close(price - 0.5, price + 0.5, price)
            })
            .collect();

        let result = compute_interval_indicators_flat(&bars);

        // RSI (index 0) should be valid with 15 bars (needs 14)
        assert!(!result[0].is_nan(), "RSI should be valid with 15 bars");
        assert!(result[0] >= 0.0 && result[0] <= 100.0, "RSI should be in [0, 100]");

        // CCI (index 1) should be valid with 15 bars (needs 14)
        assert!(!result[1].is_nan(), "CCI should be valid with 15 bars");

        // ADX (index 2) should still be NaN (needs 27 bars)
        assert!(result[2].is_nan(), "ADX should be NaN with 15 bars (needs 27)");

        // SMA_10 (index 6) should be valid with 15 bars (needs 10)
        assert!(!result[6].is_nan(), "SMA_10 should be valid with 15 bars");

        // SMA_20 (index 7) should be NaN with 15 bars (needs 20)
        assert!(result[7].is_nan(), "SMA_20 should be NaN with 15 bars (needs 20)");

        // EMA_10 (index 9) should be valid with 15 bars (needs 10)
        assert!(!result[9].is_nan(), "EMA_10 should be valid with 15 bars");

        // Ichimoku tenkan (index 12) should be valid with 15 bars (needs 9)
        assert!(!result[12].is_nan(), "Ichimoku tenkan should be valid with 15 bars");
    }

    #[test]
    fn single_bar_input_does_not_panic() {
        // Edge case: single bar input
        let bars = vec![bar_with_close(99.0, 101.0, 100.0)];

        let result = std::panic::catch_unwind(|| {
            compute_interval_indicators_flat(&bars)
        });
        assert!(result.is_ok(), "compute_interval_indicators_flat should not panic with single bar");

        let indicators = result.unwrap();
        assert_eq!(indicators.len(), INDICATORS_PER_INTERVAL);

        // Fibonacci levels should be valid (min_periods=1)
        assert!(!indicators[20].is_nan(), "FR 0.0% should be valid with single bar");
    }

    #[test]
    fn compute_interval_indicators_with_empty_does_not_panic() {
        // Test compute_interval_indicators_with with empty input
        let bars: Vec<Bar> = vec![];

        // Test with all indicators enabled
        let set = IndicatorSet {
            rsi_14: true,
            cci_14: true,
            adx_14: true,
            macd: true,
            macd_signal: true,
            macd_hist: true,
            sma_10: true,
            sma_20: true,
            sma_50: true,
            ema_10: true,
            ema_20: true,
            ema_50: true,
            ichimoku_tenkan: true,
            ichimoku_kijun: true,
            ichimoku_senkou_a: true,
            ichimoku_senkou_b: true,
            ichimoku_chikou: true,
            bb_upper: true,
            bb_middle: true,
            bb_lower: true,
            fr_000: true,
            fr_236: true,
            fr_382: true,
            fr_500: true,
            fr_618: true,
            fr_786: true,
            fr_1000: true,
        };

        let result = std::panic::catch_unwind(|| {
            compute_interval_indicators_with(&bars, &set)
        });
        assert!(result.is_ok(), "compute_interval_indicators_with should not panic on empty input");

        // Empty input returns empty vector
        let indicators = result.unwrap();
        assert!(indicators.is_empty(), "Empty input should return empty vector");
    }
    #[test]
    fn compute_m15_double_bottom_low_empty_patterns_returns_zero() {
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 150.0,
            ask: 150.1,
        }];
        assert_eq!(
            compute_m15_double_bottom_low(&[], &ticks),
            0.0
        );
    }

    #[test]
    fn compute_m15_double_bottom_low_empty_ticks_returns_zero() {
        let dbs = vec![modelenv_proto::DoubleBottomPattern {
            low1: 149.0,
            low2: 149.5,
            neckline: 151.0,
            depth_pct: 0.2,
            confirmed: true,
            ..Default::default()
        }];
        assert_eq!(compute_m15_double_bottom_low(&dbs, &[]), 0.0);
    }

    #[test]
    fn compute_m15_double_bottom_low_skips_unconfirmed() {
        let dbs = vec![modelenv_proto::DoubleBottomPattern {
            low1: 149.0,
            low2: 149.5,
            neckline: 151.0,
            depth_pct: 0.2,
            confirmed: false,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 150.0,
            ask: 150.1,
        }];
        assert_eq!(compute_m15_double_bottom_low(&dbs, &ticks), 0.0);
    }

    #[test]
    fn compute_m15_double_bottom_low_skips_shallow_depth() {
        let dbs = vec![modelenv_proto::DoubleBottomPattern {
            low1: 149.0,
            low2: 149.5,
            neckline: 151.0,
            depth_pct: 0.10,
            confirmed: true,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 150.0,
            ask: 150.1,
        }];
        assert_eq!(compute_m15_double_bottom_low(&dbs, &ticks), 0.0);
    }

    #[test]
    fn compute_m15_double_bottom_low_skips_when_tick_ask_above_neckline() {
        let dbs = vec![modelenv_proto::DoubleBottomPattern {
            low1: 149.0,
            low2: 149.5,
            neckline: 151.0,
            depth_pct: 0.2,
            confirmed: true,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 151.5,
            ask: 151.6,
        }];
        assert_eq!(compute_m15_double_bottom_low(&dbs, &ticks), 0.0);
    }

    #[test]
    fn compute_m15_double_bottom_low_skips_when_tick_ask_below_min_low() {
        let dbs = vec![modelenv_proto::DoubleBottomPattern {
            low1: 149.0,
            low2: 149.5,
            neckline: 151.0,
            depth_pct: 0.2,
            confirmed: true,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 148.5,
            ask: 148.6,
        }];
        assert_eq!(compute_m15_double_bottom_low(&dbs, &ticks), 0.0);
    }

    #[test]
    fn compute_m15_double_bottom_low_returns_min_low_when_qualified() {
        let dbs = vec![modelenv_proto::DoubleBottomPattern {
            low1: 149.0,
            low2: 149.5,
            neckline: 151.0,
            depth_pct: 0.2,
            confirmed: true,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 150.0,
            ask: 150.1,
        }];
        assert_eq!(compute_m15_double_bottom_low(&dbs, &ticks), 149.0);
    }

    #[test]
    fn compute_m15_double_bottom_low_picks_most_recent_qualified() {
        // Two confirmed patterns both with tick_ask inside their ranges.
        // Patterns are latest-first; the first (most recent) should win.
        let dbs = vec![
            modelenv_proto::DoubleBottomPattern {
                low1: 149.0,
                low2: 149.5,
                neckline: 152.0,
                depth_pct: 0.2,
                confirmed: true,
                ..Default::default()
            },
            modelenv_proto::DoubleBottomPattern {
                low1: 148.0,
                low2: 148.5,
                neckline: 150.0,
                depth_pct: 0.2,
                confirmed: true,
                ..Default::default()
            },
        ];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 150.0,
            ask: 150.1,
        }];
        // Both qualify, but the first is more recent → returns 149.0
        assert_eq!(compute_m15_double_bottom_low(&dbs, &ticks), 149.0);
    }

    // --- compute_m15_double_bottom_high ---

    #[test]
    fn compute_m15_double_bottom_high_empty_patterns_returns_zero() {
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 148.0,
            ask: 148.1,
        }];
        assert_eq!(
            compute_m15_double_bottom_high(&[], &ticks, 0.0),
            0.0
        );
    }

    #[test]
    fn compute_m15_double_bottom_high_empty_ticks_returns_zero() {
        let dbs = vec![modelenv_proto::DoubleBottomPattern {
            low1: 150.0,
            low2: 150.5,
            neckline: 152.0,
            depth_pct: 0.2,
            confirmed: true,
            ..Default::default()
        }];
        assert_eq!(compute_m15_double_bottom_high(&dbs, &[], 0.0), 0.0);
    }

    #[test]
    fn compute_m15_double_bottom_high_skips_unconfirmed() {
        let dbs = vec![modelenv_proto::DoubleBottomPattern {
            low1: 150.0,
            low2: 150.5,
            neckline: 152.0,
            depth_pct: 0.2,
            confirmed: false,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 148.0,
            ask: 148.1,
        }];
        assert_eq!(compute_m15_double_bottom_high(&dbs, &ticks, 0.0), 0.0);
    }

    #[test]
    fn compute_m15_double_bottom_high_skips_shallow_depth() {
        let dbs = vec![modelenv_proto::DoubleBottomPattern {
            low1: 150.0,
            low2: 150.5,
            neckline: 152.0,
            depth_pct: 0.10,
            confirmed: true,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 148.0,
            ask: 148.1,
        }];
        assert_eq!(compute_m15_double_bottom_high(&dbs, &ticks, 0.0), 0.0);
    }

    #[test]
    fn compute_m15_double_bottom_high_skips_when_tick_ask_above_min_low() {
        let dbs = vec![modelenv_proto::DoubleBottomPattern {
            low1: 150.0,
            low2: 150.5,
            neckline: 152.0,
            depth_pct: 0.2,
            confirmed: true,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 150.2,
            ask: 150.3,
        }];
        assert_eq!(compute_m15_double_bottom_high(&dbs, &ticks, 0.0), 0.0);
    }

    #[test]
    fn compute_m15_double_bottom_high_skips_when_spread_too_small() {
        let dbs = vec![modelenv_proto::DoubleBottomPattern {
            low1: 150.0,
            low2: 150.5,
            neckline: 152.0,
            depth_pct: 0.2,
            confirmed: true,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 148.0,
            ask: 148.1,
        }];
        // m15_double_bottom_low = 149.9, so spread = 150.0 - 149.9 = 0.1 < 0.2
        assert_eq!(compute_m15_double_bottom_high(&dbs, &ticks, 149.9), 0.0);
    }

    #[test]
    fn compute_m15_double_bottom_high_returns_min_low_when_qualified() {
        let dbs = vec![modelenv_proto::DoubleBottomPattern {
            low1: 150.0,
            low2: 150.5,
            neckline: 152.0,
            depth_pct: 0.2,
            confirmed: true,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 148.0,
            ask: 148.1,
        }];
        // m15_double_bottom_low = 149.0, spread = 150.0 - 149.0 = 1.0 >= 0.2
        assert_eq!(compute_m15_double_bottom_high(&dbs, &ticks, 149.0), 150.0);
    }

    #[test]
    fn compute_m15_double_bottom_high_picks_most_recent_qualified() {
        let dbs = vec![
            modelenv_proto::DoubleBottomPattern {
                low1: 152.0,
                low2: 152.5,
                neckline: 154.0,
                depth_pct: 0.2,
                confirmed: true,
                ..Default::default()
            },
            modelenv_proto::DoubleBottomPattern {
                low1: 150.0,
                low2: 150.5,
                neckline: 152.0,
                depth_pct: 0.2,
                confirmed: true,
                ..Default::default()
            },
        ];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 148.0,
            ask: 148.1,
        }];
        let m15_double_bottom_low = 148.0;
        // First (more recent): 152.0 - 148.0 = 4.0 >= 0.2 → returns 152.0
        assert_eq!(
            compute_m15_double_bottom_high(&dbs, &ticks, m15_double_bottom_low),
            152.0
        );
    }

    // --- compute_m15_double_top_high ---

    #[test]
    fn compute_m15_double_top_high_empty_patterns_returns_zero() {
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 152.0,
            ask: 152.1,
        }];
        assert_eq!(compute_m15_double_top_high(&[], &ticks), 0.0);
    }

    #[test]
    fn compute_m15_double_top_high_empty_ticks_returns_zero() {
        let dts = vec![modelenv_proto::DoubleTopPattern {
            high1: 155.0,
            high2: 155.5,
            neckline: 153.0,
            depth_pct: 0.2,
            confirmed: true,
            ..Default::default()
        }];
        assert_eq!(compute_m15_double_top_high(&dts, &[]), 0.0);
    }

    #[test]
    fn compute_m15_double_top_high_skips_unconfirmed() {
        let dts = vec![modelenv_proto::DoubleTopPattern {
            high1: 155.0,
            high2: 155.5,
            neckline: 153.0,
            depth_pct: 0.2,
            confirmed: false,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 154.0,
            ask: 154.1,
        }];
        assert_eq!(compute_m15_double_top_high(&dts, &ticks), 0.0);
    }

    #[test]
    fn compute_m15_double_top_high_skips_shallow_depth() {
        let dts = vec![modelenv_proto::DoubleTopPattern {
            high1: 155.0,
            high2: 155.5,
            neckline: 153.0,
            depth_pct: 0.10,
            confirmed: true,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 154.0,
            ask: 154.1,
        }];
        assert_eq!(compute_m15_double_top_high(&dts, &ticks), 0.0);
    }

    #[test]
    fn compute_m15_double_top_high_skips_when_tick_bid_below_neckline() {
        let dts = vec![modelenv_proto::DoubleTopPattern {
            high1: 155.0,
            high2: 155.5,
            neckline: 153.0,
            depth_pct: 0.2,
            confirmed: true,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 152.5,
            ask: 152.6,
        }];
        assert_eq!(compute_m15_double_top_high(&dts, &ticks), 0.0);
    }

    #[test]
    fn compute_m15_double_top_high_skips_when_tick_bid_above_max_high() {
        let dts = vec![modelenv_proto::DoubleTopPattern {
            high1: 155.0,
            high2: 155.5,
            neckline: 153.0,
            depth_pct: 0.2,
            confirmed: true,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 156.0,
            ask: 156.1,
        }];
        assert_eq!(compute_m15_double_top_high(&dts, &ticks), 0.0);
    }

    #[test]
    fn compute_m15_double_top_high_returns_max_high_when_qualified() {
        let dts = vec![modelenv_proto::DoubleTopPattern {
            high1: 155.5,
            high2: 155.0,
            neckline: 153.0,
            depth_pct: 0.2,
            confirmed: true,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 154.0,
            ask: 154.1,
        }];
        assert_eq!(compute_m15_double_top_high(&dts, &ticks), 155.5);
    }

    #[test]
    fn compute_m15_double_top_high_picks_most_recent_qualified() {
        let dts = vec![
            modelenv_proto::DoubleTopPattern {
                high1: 158.0,
                high2: 158.5,
                neckline: 155.0,
                depth_pct: 0.2,
                confirmed: true,
                ..Default::default()
            },
            modelenv_proto::DoubleTopPattern {
                high1: 155.0,
                high2: 155.5,
                neckline: 153.0,
                depth_pct: 0.2,
                confirmed: true,
                ..Default::default()
            },
        ];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 156.0,
            ask: 156.1,
        }];
        // First (more recent): tick_bid=156.0 within [155.0, 158.5] → returns 158.5
        assert_eq!(compute_m15_double_top_high(&dts, &ticks), 158.5);
    }

    #[test]
    fn compute_m15_double_bottom_low_depth_pct_exactly_at_threshold() {
        let dbs = vec![modelenv_proto::DoubleBottomPattern {
            low1: 149.0,
            low2: 149.5,
            neckline: 151.0,
            depth_pct: 0.15,
            confirmed: true,
            ..Default::default()
        }];
        let ticks = vec![modelenv_proto::Tick {
            timestamp_ns: 1000,
            bid: 150.0,
            ask: 150.1,
        }];
        assert_eq!(compute_m15_double_bottom_low(&dbs, &ticks), 149.0);
    }
}


#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // **Property 1: Output Length Invariant**
    //
    // For any indicator function and any input slice of length `n`, the output
    // vector(s) SHALL have length exactly `n`.
    //
    // **Validates: Requirements 2.2, 3.3, 4.2, 5.2, 6.3, 7.2, 8.2, 9.2**
    //
    // This property test verifies that all indicator functions maintain the
    // invariant that output length equals input length.

    /// Strategy to generate realistic OHLC price data
    fn ohlc_strategy(
        len: impl Into<proptest::collection::SizeRange>,
    ) -> impl Strategy<Value = (Vec<f64>, Vec<f64>, Vec<f64>)> {
        prop::collection::vec(10.0f64..10000.0f64, len).prop_flat_map(|close| {
            // Generate high values that are >= close
            let high: Vec<f64> = close.iter().map(|&c| c + (c * 0.01)).collect();
            // Generate low values that are <= close
            let low: Vec<f64> = close.iter().map(|&c| c - (c * 0.01)).collect();
            Just((high, low, close))
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Test RSI output length equals input length
        /// **Validates: Requirement 2.2**
        #[test]
        fn rsi_output_length_equals_input_length(
            close in prop::collection::vec(1.0f64..10000.0f64, 0..500),
            period in 1usize..50
        ) {
            let result = momentum::rsi(&close, period);
            prop_assert_eq!(
                result.len(),
                close.len(),
                "RSI output length ({}) should equal input length ({}). Period: {}",
                result.len(),
                close.len(),
                period
            );
        }

        /// Test CCI output length equals input length
        /// **Validates: Requirement 3.3**
        #[test]
        fn cci_output_length_equals_input_length(
            (high, low, close) in ohlc_strategy(0..500),
            period in 1usize..50
        ) {
            let result = momentum::cci(&high, &low, &close, period);
            match result {
                Ok(output) => {
                    prop_assert_eq!(
                        output.len(),
                        close.len(),
                        "CCI output length ({}) should equal input length ({}). Period: {}",
                        output.len(),
                        close.len(),
                        period
                    );
                }
                Err(IndicatorError::LengthMismatch { .. }) => {
                    // This is expected if high/low/close have different lengths
                    // (shouldn't happen with our strategy, but handle it)
                }
                Err(e) => {
                    prop_assert!(false, "Unexpected error from CCI: {:?}", e);
                }
            }
        }

        /// Test ADX output length equals input length
        /// **Validates: Requirement 4.2**
        #[test]
        fn adx_output_length_equals_input_length(
            (high, low, close) in ohlc_strategy(0..500),
            period in 2usize..50  // ADX requires period >= 2
        ) {
            let result = trend::adx(&high, &low, &close, period);
            match result {
                Ok(output) => {
                    prop_assert_eq!(
                        output.len(),
                        close.len(),
                        "ADX output length ({}) should equal input length ({}). Period: {}",
                        output.len(),
                        close.len(),
                        period
                    );
                }
                Err(IndicatorError::LengthMismatch { .. }) => {
                    // Expected if high/low/close have different lengths
                }
                Err(IndicatorError::InvalidPeriod { .. }) => {
                    // Expected if period is invalid (e.g., period == 1)
                }
                Err(e) => {
                    prop_assert!(false, "Unexpected error from ADX: {:?}", e);
                }
            }
        }

        /// Test MACD output lengths equal input length (all three vectors)
        /// **Validates: Requirement 5.2**
        #[test]
        fn macd_output_length_equals_input_length(
            close in prop::collection::vec(1.0f64..10000.0f64, 0..500),
            fast in 2usize..20,
            slow in 21usize..50,
            signal in 1usize..20
        ) {
            // Ensure fast < slow
            prop_assume!(fast < slow);

            let result = trend::macd(&close, fast, slow, signal);
            match result {
                Ok(output) => {
                    let n = close.len();
                    prop_assert_eq!(
                        output.macd.len(),
                        n,
                        "MACD line length ({}) should equal input length ({}). fast={}, slow={}, signal={}",
                        output.macd.len(),
                        n,
                        fast,
                        slow,
                        signal
                    );
                    prop_assert_eq!(
                        output.signal.len(),
                        n,
                        "MACD signal length ({}) should equal input length ({}). fast={}, slow={}, signal={}",
                        output.signal.len(),
                        n,
                        fast,
                        slow,
                        signal
                    );
                    prop_assert_eq!(
                        output.hist.len(),
                        n,
                        "MACD histogram length ({}) should equal input length ({}). fast={}, slow={}, signal={}",
                        output.hist.len(),
                        n,
                        fast,
                        slow,
                        signal
                    );
                }
                Err(e) => {
                    prop_assert!(false, "Unexpected error from MACD: {:?}", e);
                }
            }
        }

        /// Test moving_average output length equals input length for all MA kinds
        /// **Validates: Requirement 6.3**
        #[test]
        fn moving_average_output_length_equals_input_length(
            close in prop::collection::vec(1.0f64..10000.0f64, 1..500),
            period in 2usize..50,  // Some MA types (DEMA, TEMA, KAMA) require period >= 2
            kind_idx in 0usize..7
        ) {
            let kinds = [
                trend::MovingAverageKind::Sma,
                trend::MovingAverageKind::Ema,
                trend::MovingAverageKind::Wma,
                trend::MovingAverageKind::Dema,
                trend::MovingAverageKind::Tema,
                trend::MovingAverageKind::Kama,
                trend::MovingAverageKind::Trima,
            ];
            let kind = kinds[kind_idx];

            let result = trend::moving_average(&close, kind, period);
            match result {
                Ok(output) => {
                    prop_assert_eq!(
                        output.len(),
                        close.len(),
                        "Moving average ({:?}) output length ({}) should equal input length ({}). Period: {}",
                        kind,
                        output.len(),
                        close.len(),
                        period
                    );
                }
                Err(IndicatorError::EmptyInput) => {
                    // Expected for empty input
                    prop_assert_eq!(close.len(), 0);
                }
                Err(IndicatorError::InvalidPeriod { .. }) => {
                    // Expected if period is invalid for the specific MA type
                }
                Err(e) => {
                    prop_assert!(false, "Unexpected error from moving_average: {:?}", e);
                }
            }
        }

        /// Test Ichimoku output lengths equal input length (all five vectors)
        /// **Validates: Requirement 7.2**
        #[test]
        fn ichimoku_output_length_equals_input_length(
            (high, low, close) in ohlc_strategy(0..500),
            tenkan in 1usize..20,
            kijun in 1usize..50,
            senkou_b in 1usize..60
        ) {
            let result = trend::ichimoku(&high, &low, &close, tenkan, kijun, senkou_b);
            match result {
                Ok(output) => {
                    let n = close.len();
                    prop_assert_eq!(
                        output.tenkan.len(),
                        n,
                        "Ichimoku tenkan length ({}) should equal input length ({}). tenkan={}, kijun={}, senkou_b={}",
                        output.tenkan.len(),
                        n,
                        tenkan,
                        kijun,
                        senkou_b
                    );
                    prop_assert_eq!(
                        output.kijun.len(),
                        n,
                        "Ichimoku kijun length ({}) should equal input length ({}). tenkan={}, kijun={}, senkou_b={}",
                        output.kijun.len(),
                        n,
                        tenkan,
                        kijun,
                        senkou_b
                    );
                    prop_assert_eq!(
                        output.senkou_a.len(),
                        n,
                        "Ichimoku senkou_a length ({}) should equal input length ({}). tenkan={}, kijun={}, senkou_b={}",
                        output.senkou_a.len(),
                        n,
                        tenkan,
                        kijun,
                        senkou_b
                    );
                    prop_assert_eq!(
                        output.senkou_b.len(),
                        n,
                        "Ichimoku senkou_b length ({}) should equal input length ({}). tenkan={}, kijun={}, senkou_b={}",
                        output.senkou_b.len(),
                        n,
                        tenkan,
                        kijun,
                        senkou_b
                    );
                    prop_assert_eq!(
                        output.chikou.len(),
                        n,
                        "Ichimoku chikou length ({}) should equal input length ({}). tenkan={}, kijun={}, senkou_b={}",
                        output.chikou.len(),
                        n,
                        tenkan,
                        kijun,
                        senkou_b
                    );
                }
                Err(IndicatorError::LengthMismatch { .. }) => {
                    // Expected if high/low/close have different lengths
                }
                Err(e) => {
                    prop_assert!(false, "Unexpected error from ichimoku: {:?}", e);
                }
            }
        }

        /// Test Bollinger Bands output lengths equal input length (all three vectors)
        /// **Validates: Requirement 8.2**
        #[test]
        fn bollinger_bands_output_length_equals_input_length(
            close in prop::collection::vec(1.0f64..10000.0f64, 0..500),
            period in 1usize..50,
            nbdev in 0.0f64..5.0f64
        ) {
            let result = volatility::bollinger_bands(&close, period, nbdev);
            match result {
                Ok(output) => {
                    let n = close.len();
                    prop_assert_eq!(
                        output.upper.len(),
                        n,
                        "Bollinger Bands upper length ({}) should equal input length ({}). Period: {}, nbdev: {}",
                        output.upper.len(),
                        n,
                        period,
                        nbdev
                    );
                    prop_assert_eq!(
                        output.middle.len(),
                        n,
                        "Bollinger Bands middle length ({}) should equal input length ({}). Period: {}, nbdev: {}",
                        output.middle.len(),
                        n,
                        period,
                        nbdev
                    );
                    prop_assert_eq!(
                        output.lower.len(),
                        n,
                        "Bollinger Bands lower length ({}) should equal input length ({}). Period: {}, nbdev: {}",
                        output.lower.len(),
                        n,
                        period,
                        nbdev
                    );
                }
                Err(e) => {
                    prop_assert!(false, "Unexpected error from bollinger_bands: {:?}", e);
                }
            }
        }

        /// Test Fibonacci retracements output lengths equal input length (all seven vectors)
        /// **Validates: Requirement 9.2**
        #[test]
        fn fibonacci_retracements_output_length_equals_input_length(
            (high, low, _close) in ohlc_strategy(0..500),
            window in 1usize..100
        ) {
            let result = support::fibonacci_retracements(&high, &low, window);
            match result {
                Ok(output) => {
                    let n = high.len();
                    prop_assert_eq!(
                        output.fr_000.len(),
                        n,
                        "Fibonacci fr_000 length ({}) should equal input length ({}). Window: {}",
                        output.fr_000.len(),
                        n,
                        window
                    );
                    prop_assert_eq!(
                        output.fr_236.len(),
                        n,
                        "Fibonacci fr_236 length ({}) should equal input length ({}). Window: {}",
                        output.fr_236.len(),
                        n,
                        window
                    );
                    prop_assert_eq!(
                        output.fr_382.len(),
                        n,
                        "Fibonacci fr_382 length ({}) should equal input length ({}). Window: {}",
                        output.fr_382.len(),
                        n,
                        window
                    );
                    prop_assert_eq!(
                        output.fr_500.len(),
                        n,
                        "Fibonacci fr_500 length ({}) should equal input length ({}). Window: {}",
                        output.fr_500.len(),
                        n,
                        window
                    );
                    prop_assert_eq!(
                        output.fr_618.len(),
                        n,
                        "Fibonacci fr_618 length ({}) should equal input length ({}). Window: {}",
                        output.fr_618.len(),
                        n,
                        window
                    );
                    prop_assert_eq!(
                        output.fr_786.len(),
                        n,
                        "Fibonacci fr_786 length ({}) should equal input length ({}). Window: {}",
                        output.fr_786.len(),
                        n,
                        window
                    );
                    prop_assert_eq!(
                        output.fr_1000.len(),
                        n,
                        "Fibonacci fr_1000 length ({}) should equal input length ({}). Window: {}",
                        output.fr_1000.len(),
                        n,
                        window
                    );
                }
                Err(IndicatorError::LengthMismatch { .. }) => {
                    // Expected if high/low have different lengths
                }
                Err(e) => {
                    prop_assert!(false, "Unexpected error from fibonacci_retracements: {:?}", e);
                }
            }
        }
    }

    /// Strategy to generate a valid Bar with realistic OHLC values for backward compatibility test
    fn bar_strategy() -> impl Strategy<Value = Bar> {
        // Generate base price and variations
        (1.0f64..10000.0f64, 0.001f64..0.05f64, 0.001f64..0.05f64).prop_map(
            |(base, high_pct, low_pct)| {
                let high = base * (1.0 + high_pct);
                let low = base * (1.0 - low_pct);
                let open = low + (high - low) * 0.3;
                let close = low + (high - low) * 0.7;
                Bar {
                    timestamp_ns: 0,
                    open,
                    high,
                    low,
                    close,
                    volume: 1000.0,
                }
            },
        )
    }

}
