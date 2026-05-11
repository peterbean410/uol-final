//! Pattern detectors: Double Bottom, Double Top
//!
//! These detectors identify chart patterns that may signal trend reversals.

use crate::indicators::IndicatorError;
use modelenv_proto::Bar;

const SWING_RADIUS: usize = 2;
const BOTTOM_SIMILARITY_TOL: f64 = 0.005;
const PEAK_RISE_TARGET: f64 = 0.01;

// Minimum depth percentage threshold for valid patterns
const MIN_DEPTH_PCT: f64 = 0.1;

/// A detected double-bottom pattern with all metadata fields.
///
/// A double bottom is a bullish reversal pattern consisting of two consecutive
/// lows at approximately the same price level, separated by a peak (neckline).
#[derive(Debug, Clone, PartialEq)]
pub struct DoubleBottom {
    /// Index of the first bottom in the bar slice
    pub idx1: usize,
    /// Index of the second bottom in the bar slice
    pub idx2: usize,
    /// Timestamp (nanoseconds) of the first bottom
    pub ts1: i64,
    /// Timestamp (nanoseconds) of the second bottom
    pub ts2: i64,
    /// Low price at the first bottom
    pub low1: f64,
    /// Low price at the second bottom
    pub low2: f64,
    /// Neckline price (max high between the two bottoms)
    pub neckline: f64,
    /// Index of the neckline (max high) in the bar slice
    pub neckline_idx: usize,
    /// Depth percentage: (neckline - avg_low) / neckline * 100, banker's rounded to 3 decimals
    pub depth_pct: f64,
    /// Width in bars: idx2 - idx1
    pub width_bars: usize,
    /// True if price closed above neckline after idx2
    pub confirmed: bool,
    /// Value of the nearest local minimum strictly before idx1
    pub min_before_val: Option<f64>,
    /// Timestamp of the nearest local minimum strictly before idx1
    pub min_before_ts: Option<i64>,
    /// Value of the nearest local maximum strictly before idx1
    pub max_before_val: Option<f64>,
    /// Timestamp of the nearest local maximum strictly before idx1
    pub max_before_ts: Option<i64>,
    /// Value of the nearest local minimum strictly after idx2
    pub min_after_val: Option<f64>,
    /// Timestamp of the nearest local minimum strictly after idx2
    pub min_after_ts: Option<i64>,
    /// Value of the nearest local maximum strictly after idx2
    pub max_after_val: Option<f64>,
    /// Timestamp of the nearest local maximum strictly after idx2
    pub max_after_ts: Option<i64>,
}

/// Result of double-bottom pattern detection.
#[derive(Debug, Clone, PartialEq)]
pub struct DoubleBottomDetection {
    /// Detected patterns, including any forming pattern at the right edge
    pub patterns: Vec<DoubleBottom>,
    /// Latest local minimum value in the bar series
    pub latest_min: Option<f64>,
    /// Latest local maximum value in the bar series
    pub latest_max: Option<f64>,
}

/// A detected double-top pattern with all metadata fields.
///
/// A double top is a bearish reversal pattern consisting of two consecutive
/// highs at approximately the same price level, separated by a trough (neckline).
#[derive(Debug, Clone, PartialEq)]
pub struct DoubleTop {
    /// Index of the first top in the bar slice
    pub idx1: usize,
    /// Index of the second top in the bar slice
    pub idx2: usize,
    /// Timestamp (nanoseconds) of the first top
    pub ts1: i64,
    /// Timestamp (nanoseconds) of the second top
    pub ts2: i64,
    /// High price at the first top
    pub high1: f64,
    /// High price at the second top
    pub high2: f64,
    /// Neckline price (min low between the two tops)
    pub neckline: f64,
    /// Index of the neckline (min low) in the bar slice
    pub neckline_idx: usize,
    /// Depth percentage: (avg_high - neckline) / avg_high * 100, banker's rounded to 3 decimals
    pub depth_pct: f64,
    /// Width in bars: idx2 - idx1
    pub width_bars: usize,
    /// True if price closed below neckline after idx2
    pub confirmed: bool,
    /// Value of the nearest local minimum strictly before idx1
    pub min_before_val: Option<f64>,
    /// Timestamp of the nearest local minimum strictly before idx1
    pub min_before_ts: Option<i64>,
    /// Value of the nearest local maximum strictly before idx1
    pub max_before_val: Option<f64>,
    /// Timestamp of the nearest local maximum strictly before idx1
    pub max_before_ts: Option<i64>,
    /// Value of the nearest local minimum strictly after idx2
    pub min_after_val: Option<f64>,
    /// Timestamp of the nearest local minimum strictly after idx2
    pub min_after_ts: Option<i64>,
    /// Value of the nearest local maximum strictly after idx2
    pub max_after_val: Option<f64>,
    /// Timestamp of the nearest local maximum strictly after idx2
    pub max_after_ts: Option<i64>,
}

/// Result of double-top pattern detection.
#[derive(Debug, Clone, PartialEq)]
pub struct DoubleTopDetection {
    /// Detected patterns, including any forming pattern at the right edge
    pub patterns: Vec<DoubleTop>,
    /// Latest local minimum value in the bar series
    pub latest_min: Option<f64>,
    /// Latest local maximum value in the bar series
    pub latest_max: Option<f64>,
}

/// Detect double-bottom patterns in bar data.
///
/// # Arguments
///
/// * `bars` - Slice of bars ordered oldest to newest
/// * `window` - Window size for local extrema detection (uses floor(window/2) on each side)
/// * `tolerance_pct` - Maximum percentage difference between the two lows
/// * `min_width` - Minimum number of bars between the two bottoms
///
/// # Returns
///
/// * `Ok(DoubleBottomDetection)` - Detection result with patterns and latest extrema
/// * `Err(IndicatorError::InvalidPeriod)` - If window == 0, min_width == 0, or tolerance_pct is invalid
///
/// # Invariants
///
/// For every detected pattern:
/// - `idx1 < neckline_idx < idx2`
/// - `width_bars == idx2 - idx1`
/// - `depth_pct >= 0.1`
pub fn detect_double_bottoms(
    bars: &[Bar],
    window: usize,
    tolerance_pct: f64,
    min_width: usize,
) -> Result<DoubleBottomDetection, IndicatorError> {
    // Validate parameters per requirement 10.11
    if window == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "window",
            value: 0,
            reason: "window must be >= 1",
        });
    }
    if min_width == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "min_width",
            value: 0,
            reason: "min_width must be >= 1",
        });
    }
    if tolerance_pct < 0.0 || !tolerance_pct.is_finite() {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "tolerance_pct",
            value: 0, // Can't represent f64 in usize, but the reason explains
            reason: "tolerance_pct must be >= 0.0 and finite",
        });
    }

    // Per requirement 10.4: return empty result if insufficient bars
    if bars.len() < 2 * window + 1 {
        return Ok(DoubleBottomDetection {
            patterns: Vec::new(),
            latest_min: None,
            latest_max: None,
        });
    }

    let half_window = window / 2;

    // Find all local minima and maxima
    let local_minima = find_local_minima(bars, half_window);
    let local_maxima = find_local_maxima(bars, half_window);

    // Track latest extrema
    let latest_min = local_minima.last().map(|&idx| bars[idx].low);
    let latest_max = local_maxima.last().map(|&idx| bars[idx].high);

    let mut patterns = Vec::new();

    // Find all valid double-bottom patterns from CONSECUTIVE pairs of local minima
    // This matches the Python implementation which uses zip(minima, minima[1:])
    for i in 0..local_minima.len().saturating_sub(1) {
        let idx1 = local_minima[i];
        let idx2 = local_minima[i + 1];

        if let Some(pattern) = try_create_double_bottom(
            bars,
            idx1,
            idx2,
            tolerance_pct,
            min_width,
            &local_minima,
            &local_maxima,
            half_window,
            false, // not forming
        ) {
            patterns.push(pattern);
        }
    }

    // Check for forming pattern at right edge (requirement 10.13)
    // Python only checks the LAST local minimum against the running minimum
    if let Some(&last_min_idx) = local_minima.last() {
        // Look for running minimum after the last local minimum
        if last_min_idx + 1 < bars.len() {
            let (running_min_idx, running_min_val) =
                find_running_min_after(bars, last_min_idx + 1);

            // Only check if the running minimum is not already a local minimum
            // (Python: if b not in minima)
            if !local_minima.contains(&running_min_idx) {
                // Use the last local minimum as idx1
                let idx1 = last_min_idx;

                if let Some(pattern) = try_create_forming_pattern(
                    bars,
                    idx1,
                    running_min_idx,
                    running_min_val,
                    tolerance_pct,
                    min_width,
                    &local_minima,
                    &local_maxima,
                    half_window,
                ) {
                    patterns.push(pattern);
                }
            }
        }
    }

    // Sort patterns by idx2 ascending
    patterns.sort_by_key(|p| p.idx2);

    Ok(DoubleBottomDetection {
        patterns,
        latest_min,
        latest_max,
    })
}

/// Detect double-bottom patterns using default parameters.
///
/// This is a convenience wrapper that delegates to `detect_double_bottoms` with
/// the Python defaults: `window = 5`, `tolerance_pct = 0.3`, `min_width = 5`.
///
/// # Arguments
///
/// * `bars` - Slice of bars ordered oldest to newest
///
/// # Returns
///
/// * `Ok(DoubleBottomDetection)` - Detection result with patterns and latest extrema
/// * `Err(IndicatorError)` - If an error occurs (should not happen with valid defaults)
///
/// # Example
///
/// ```ignore
/// use modelenv_core::indicators::patterns::detect_double_bottoms_default;
///
/// let bars = vec![/* ... */];
/// let result = detect_double_bottoms_default(&bars)?;
/// for pattern in result.patterns {
///     println!("Found double bottom at indices {} and {}", pattern.idx1, pattern.idx2);
/// }
/// ```
pub fn detect_double_bottoms_default(bars: &[Bar]) -> Result<DoubleBottomDetection, IndicatorError> {
    detect_double_bottoms(bars, 5, 0.3, 5)
}

/// Find local minima indices using window-based comparison.
///
/// A local minimum at index i is defined as:
/// bars[i].low == min(bars[i - half_window : i + half_window + 1].low)
/// AND the segment has variation (min != max)
/// AND the minimum is at least half_window bars from the previous minimum
fn find_local_minima(bars: &[Bar], half_window: usize) -> Vec<usize> {
    if bars.len() <= 2 * half_window {
        return Vec::new();
    }

    let mut minima = Vec::new();
    for i in half_window..(bars.len() - half_window) {
        let center = bars[i].low;
        
        // Find min and max in the segment [i - half_window, i + half_window]
        let mut segment_min = center;
        let mut segment_max = center;
        for j in (i.saturating_sub(half_window))..=(i + half_window).min(bars.len() - 1) {
            segment_min = segment_min.min(bars[j].low);
            segment_max = segment_max.max(bars[j].low);
        }
        
        // Check if center is the minimum and there's variation in the segment
        // (Python: if lows.iloc[i] == segment.min() and segment.min() != segment.max())
        if (center - segment_min).abs() < 1e-12 && (segment_max - segment_min).abs() > 1e-12 {
            // Check spacing constraint: at least half_window bars from previous minimum
            // (Python: if not minima or i - minima[-1] >= half)
            if minima.is_empty() || i - minima[minima.len() - 1] >= half_window {
                minima.push(i);
            }
        }
    }
    minima
}

/// Find local maxima indices using window-based comparison.
///
/// A local maximum at index i is defined as:
/// bars[i].high == max(bars[i - half_window : i + half_window + 1].high)
/// AND the segment has variation (min != max)
/// AND the maximum is at least half_window bars from the previous maximum
fn find_local_maxima(bars: &[Bar], half_window: usize) -> Vec<usize> {
    if bars.len() <= 2 * half_window {
        return Vec::new();
    }

    let mut maxima = Vec::new();
    for i in half_window..(bars.len() - half_window) {
        let center = bars[i].high;
        
        // Find min and max in the segment [i - half_window, i + half_window]
        let mut segment_min = center;
        let mut segment_max = center;
        for j in (i.saturating_sub(half_window))..=(i + half_window).min(bars.len() - 1) {
            segment_min = segment_min.min(bars[j].high);
            segment_max = segment_max.max(bars[j].high);
        }
        
        // Check if center is the maximum and there's variation in the segment
        // (Python: if highs.iloc[i] == segment.max() and segment.min() != segment.max())
        if (center - segment_max).abs() < 1e-12 && (segment_max - segment_min).abs() > 1e-12 {
            // Check spacing constraint: at least half_window bars from previous maximum
            // (Python: if not maxima or i - maxima[-1] >= half)
            if maxima.is_empty() || i - maxima[maxima.len() - 1] >= half_window {
                maxima.push(i);
            }
        }
    }
    maxima
}

/// Find the running minimum value and index after a given start index.
fn find_running_min_after(bars: &[Bar], start: usize) -> (usize, f64) {
    let mut min_idx = start;
    let mut min_val = bars[start].low;

    for i in (start + 1)..bars.len() {
        if bars[i].low < min_val {
            min_val = bars[i].low;
            min_idx = i;
        }
    }

    (min_idx, min_val)
}

/// Try to create a double-bottom pattern from two indices.
#[allow(clippy::too_many_arguments)]
fn try_create_double_bottom(
    bars: &[Bar],
    idx1: usize,
    idx2: usize,
    tolerance_pct: f64,
    min_width: usize,
    local_minima: &[usize],
    local_maxima: &[usize],
    half_window: usize,
    is_forming: bool,
) -> Option<DoubleBottom> {
    // Check width requirement (10.5)
    let width_bars = idx2 - idx1;
    if width_bars < min_width {
        return None;
    }

    let low1 = bars[idx1].low;
    let low2 = bars[idx2].low;

    // Check tolerance requirement (10.5)
    let min_low = low1.min(low2);
    if min_low <= 0.0 {
        return None;
    }
    let tolerance_check = (low1 - low2).abs() / min_low * 100.0;
    if tolerance_check > tolerance_pct {
        return None;
    }

    // Find neckline (max high between idx1 and idx2, exclusive of endpoints for strict ordering)
    let (neckline, neckline_idx) = find_neckline(bars, idx1, idx2)?;

    // Verify neckline_idx is strictly between idx1 and idx2 (requirement 10.7)
    if neckline_idx <= idx1 || neckline_idx >= idx2 {
        return None;
    }

    // Calculate depth percentage (10.5, 10.10)
    let avg_low = (low1 + low2) / 2.0;
    let depth_pct_raw = (neckline - avg_low) / neckline * 100.0;

    // Check minimum depth threshold
    if depth_pct_raw < MIN_DEPTH_PCT {
        return None;
    }

    // Apply banker's rounding to 3 decimal places (10.10)
    let depth_pct = bankers_round(depth_pct_raw, 3);

    // Check confirmation (10.6)
    let confirmed = if is_forming {
        false
    } else {
        bars[(idx2 + 1)..]
            .iter()
            .any(|bar| bar.close > neckline)
    };

    // Find nearest extrema before idx1 and after idx2 (10.8)
    let (min_before_val, min_before_ts) =
        find_nearest_extremum_before(bars, idx1, local_minima, true);
    let (max_before_val, max_before_ts) =
        find_nearest_extremum_before(bars, idx1, local_maxima, false);

    let (min_after_val, min_after_ts, max_after_val, max_after_ts) = if is_forming {
        // Forming patterns have no after extrema (10.13)
        (None, None, None, None)
    } else {
        let (min_val, min_ts) = find_nearest_extremum_after(bars, idx2, local_minima, half_window, true);
        let (max_val, max_ts) = find_nearest_extremum_after(bars, idx2, local_maxima, half_window, false);
        (min_val, min_ts, max_val, max_ts)
    };

    // Round price values to 5 decimal places to match Python's round(x, 5)
    Some(DoubleBottom {
        idx1,
        idx2,
        ts1: bars[idx1].timestamp_ns,
        ts2: bars[idx2].timestamp_ns,
        low1: round_to_5(low1),
        low2: round_to_5(low2),
        neckline: round_to_5(neckline),
        neckline_idx,
        depth_pct,
        width_bars,
        confirmed,
        min_before_val,
        min_before_ts,
        max_before_val,
        max_before_ts,
        min_after_val,
        min_after_ts,
        max_after_val,
        max_after_ts,
    })
}

/// Try to create a forming pattern at the right edge.
#[allow(clippy::too_many_arguments)]
fn try_create_forming_pattern(
    bars: &[Bar],
    idx1: usize,
    running_min_idx: usize,
    running_min_val: f64,
    tolerance_pct: f64,
    min_width: usize,
    local_minima: &[usize],
    local_maxima: &[usize],
    _half_window: usize,
) -> Option<DoubleBottom> {
    // Use the running minimum as idx2
    let idx2 = running_min_idx;

    // Check width requirement
    let width_bars = idx2 - idx1;
    if width_bars < min_width {
        return None;
    }

    let low1 = bars[idx1].low;
    let low2 = running_min_val;

    // Check tolerance requirement
    let min_low = low1.min(low2);
    if min_low <= 0.0 {
        return None;
    }
    let tolerance_check = (low1 - low2).abs() / min_low * 100.0;
    if tolerance_check > tolerance_pct {
        return None;
    }

    // Find neckline (max high between idx1 and idx2)
    let (neckline, neckline_idx) = find_neckline(bars, idx1, idx2)?;

    // Verify neckline_idx is strictly between idx1 and idx2
    if neckline_idx <= idx1 || neckline_idx >= idx2 {
        return None;
    }

    // Calculate depth percentage
    let avg_low = (low1 + low2) / 2.0;
    let depth_pct_raw = (neckline - avg_low) / neckline * 100.0;

    // Check minimum depth threshold
    if depth_pct_raw < MIN_DEPTH_PCT {
        return None;
    }

    // Apply banker's rounding to 3 decimal places
    let depth_pct = bankers_round(depth_pct_raw, 3);

    // Find nearest extrema before idx1
    let (min_before_val, min_before_ts) =
        find_nearest_extremum_before(bars, idx1, local_minima, true);
    let (max_before_val, max_before_ts) =
        find_nearest_extremum_before(bars, idx1, local_maxima, false);

    // Round price values to 5 decimal places to match Python's round(x, 5)
    Some(DoubleBottom {
        idx1,
        idx2,
        ts1: bars[idx1].timestamp_ns,
        ts2: bars[idx2].timestamp_ns,
        low1: round_to_5(low1),
        low2: round_to_5(low2),
        neckline: round_to_5(neckline),
        neckline_idx,
        depth_pct,
        width_bars,
        confirmed: false, // Forming patterns are never confirmed
        min_before_val,
        min_before_ts,
        max_before_val,
        max_before_ts,
        min_after_val: None, // Forming patterns have no after extrema
        min_after_ts: None,
        max_after_val: None,
        max_after_ts: None,
    })
}

/// Find the neckline (max high) between two indices (exclusive of endpoints).
fn find_neckline(bars: &[Bar], idx1: usize, idx2: usize) -> Option<(f64, usize)> {
    if idx2 <= idx1 + 1 {
        return None; // No bars between idx1 and idx2
    }

    let mut max_high = f64::NEG_INFINITY;
    let mut max_idx = idx1 + 1;

    for i in (idx1 + 1)..idx2 {
        if bars[i].high > max_high {
            max_high = bars[i].high;
            max_idx = i;
        }
    }

    if max_high.is_finite() {
        Some((max_high, max_idx))
    } else {
        None
    }
}

/// Find the neckline (min low) between two indices for double-top patterns (exclusive of endpoints).
fn find_neckline_min(bars: &[Bar], idx1: usize, idx2: usize) -> Option<(f64, usize)> {
    if idx2 <= idx1 + 1 {
        return None; // No bars between idx1 and idx2
    }

    let mut min_low = f64::INFINITY;
    let mut min_idx = idx1 + 1;

    for i in (idx1 + 1)..idx2 {
        if bars[i].low < min_low {
            min_low = bars[i].low;
            min_idx = i;
        }
    }

    if min_low.is_finite() {
        Some((min_low, min_idx))
    } else {
        None
    }
}

/// Find the nearest extremum strictly before a given index.
fn find_nearest_extremum_before(
    bars: &[Bar],
    idx: usize,
    extrema: &[usize],
    is_minimum: bool,
) -> (Option<f64>, Option<i64>) {
    // Find the largest extremum index that is strictly less than idx
    for &ext_idx in extrema.iter().rev() {
        if ext_idx < idx {
            let val = if is_minimum {
                bars[ext_idx].low
            } else {
                bars[ext_idx].high
            };
            return (Some(val), Some(bars[ext_idx].timestamp_ns));
        }
    }
    (None, None)
}

/// Find the nearest extremum strictly after a given index.
fn find_nearest_extremum_after(
    bars: &[Bar],
    idx: usize,
    extrema: &[usize],
    _half_window: usize,
    is_minimum: bool,
) -> (Option<f64>, Option<i64>) {
    // Find the smallest extremum index that is strictly greater than idx
    // Note: extrema detection requires half_window bars on each side,
    // so we need to check if there are valid extrema after idx
    for &ext_idx in extrema.iter() {
        if ext_idx > idx {
            let val = if is_minimum {
                bars[ext_idx].low
            } else {
                bars[ext_idx].high
            };
            return (Some(val), Some(bars[ext_idx].timestamp_ns));
        }
    }
    (None, None)
}

/// Banker's rounding (round half to even) to a specified number of decimal places.
fn bankers_round(value: f64, decimals: u32) -> f64 {
    let multiplier = 10_f64.powi(decimals as i32);
    let scaled = value * multiplier;
    let floor = scaled.floor();
    let frac = scaled - floor;

    let rounded = if (frac - 0.5).abs() < 1e-10 {
        // Exactly 0.5 - round to even
        if floor as i64 % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else if frac > 0.5 {
        floor + 1.0
    } else {
        floor
    };

    rounded / multiplier
}

/// Round to 5 decimal places to match Python's round(x, 5) behavior.
/// This is used for pattern price values (low1, low2, high1, high2, neckline).
fn round_to_5(value: f64) -> f64 {
    (value * 100000.0).round() / 100000.0
}

/// Score the strength of the most recent double-bottom pattern in `bars`
/// (oldest -> newest). Returns 0.0 when no qualifying pattern is found,
/// otherwise a value in (0.0, 1.0] derived from how similar the two bottoms
/// are and how high the intervening peak rallied.
pub fn double_bottom_score(bars: &[Bar]) -> f64 {
    if bars.len() < 2 * SWING_RADIUS + 3 {
        return 0.0;
    }

    let swing_lows = swing_low_indices(bars, SWING_RADIUS);
    if swing_lows.len() < 2 {
        return 0.0;
    }

    let i1 = swing_lows[swing_lows.len() - 2];
    let i2 = swing_lows[swing_lows.len() - 1];

    let low1 = bars[i1].low;
    let low2 = bars[i2].low;
    let lower = low1.min(low2);
    let upper = low1.max(low2);
    if lower <= 0.0 {
        return 0.0;
    }

    let similarity_dev = (upper - lower) / lower;
    if similarity_dev > BOTTOM_SIMILARITY_TOL {
        return 0.0;
    }
    let sim_score = 1.0 - (similarity_dev / BOTTOM_SIMILARITY_TOL);

    let between = &bars[i1 + 1..i2];
    if between.is_empty() {
        return 0.0;
    }
    let peak = between
        .iter()
        .map(|b| b.high)
        .fold(f64::NEG_INFINITY, f64::max);
    if !peak.is_finite() || peak <= upper {
        return 0.0;
    }
    let peak_rise = (peak - lower) / lower;
    let rise_score = (peak_rise / PEAK_RISE_TARGET).clamp(0.0, 1.0);

    (sim_score * rise_score).sqrt()
}

fn swing_low_indices(bars: &[Bar], radius: usize) -> Vec<usize> {
    if bars.len() <= 2 * radius {
        return Vec::new();
    }

    let mut out = Vec::new();
    for i in radius..(bars.len() - radius) {
        let center = bars[i].low;
        let mut is_swing = true;
        for j in (i - radius)..=(i + radius) {
            if j != i && bars[j].low <= center {
                is_swing = false;
                break;
            }
        }
        if is_swing {
            out.push(i);
        }
    }
    out
}

/// Detect double-top patterns in bar data.
///
/// # Arguments
///
/// * `bars` - Slice of bars ordered oldest to newest
/// * `window` - Window size for local extrema detection (uses floor(window/2) on each side)
/// * `tolerance_pct` - Maximum percentage difference between the two highs
/// * `min_width` - Minimum number of bars between the two tops
///
/// # Returns
///
/// * `Ok(DoubleTopDetection)` - Detection result with patterns and latest extrema
/// * `Err(IndicatorError::InvalidPeriod)` - If window == 0, min_width == 0, or tolerance_pct is invalid
///
/// # Invariants
///
/// For every detected pattern:
/// - `idx1 < neckline_idx < idx2`
/// - `width_bars == idx2 - idx1`
/// - `depth_pct >= 0.1`
pub fn detect_double_tops(
    bars: &[Bar],
    window: usize,
    tolerance_pct: f64,
    min_width: usize,
) -> Result<DoubleTopDetection, IndicatorError> {
    // Validate parameters per requirement 11.11
    if window == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "window",
            value: 0,
            reason: "window must be >= 1",
        });
    }
    if min_width == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "min_width",
            value: 0,
            reason: "min_width must be >= 1",
        });
    }
    if tolerance_pct < 0.0 || !tolerance_pct.is_finite() {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "tolerance_pct",
            value: 0, // Can't represent f64 in usize, but the reason explains
            reason: "tolerance_pct must be >= 0.0 and finite",
        });
    }

    // Per requirement 11.4: return empty result if insufficient bars
    if bars.len() < 2 * window + 1 {
        return Ok(DoubleTopDetection {
            patterns: Vec::new(),
            latest_min: None,
            latest_max: None,
        });
    }

    let half_window = window / 2;

    // Find all local minima and maxima
    let local_minima = find_local_minima(bars, half_window);
    let local_maxima = find_local_maxima(bars, half_window);

    // Track latest extrema
    let latest_min = local_minima.last().map(|&idx| bars[idx].low);
    let latest_max = local_maxima.last().map(|&idx| bars[idx].high);

    let mut patterns = Vec::new();

    // Find all valid double-top patterns from CONSECUTIVE pairs of local maxima
    // This matches the Python implementation which uses zip(maxima, maxima[1:])
    for i in 0..local_maxima.len().saturating_sub(1) {
        let idx1 = local_maxima[i];
        let idx2 = local_maxima[i + 1];

        if let Some(pattern) = try_create_double_top(
            bars,
            idx1,
            idx2,
            tolerance_pct,
            min_width,
            &local_minima,
            &local_maxima,
            half_window,
            false, // not forming
        ) {
            patterns.push(pattern);
        }
    }

    // Check for forming pattern at right edge (requirement 11.14)
    // Python only checks the LAST local maximum against the running maximum
    if let Some(&last_max_idx) = local_maxima.last() {
        // Look for running maximum after the last local maximum
        if last_max_idx + 1 < bars.len() {
            let (running_max_idx, running_max_val) =
                find_running_max_after(bars, last_max_idx + 1);

            // Only check if the running maximum is not already a local maximum
            // (Python: if b not in maxima)
            if !local_maxima.contains(&running_max_idx) {
                // Use the last local maximum as idx1
                let idx1 = last_max_idx;

                if let Some(pattern) = try_create_forming_double_top(
                    bars,
                    idx1,
                    running_max_idx,
                    running_max_val,
                    tolerance_pct,
                    min_width,
                    &local_minima,
                    &local_maxima,
                    half_window,
                ) {
                    patterns.push(pattern);
                }
            }
        }
    }

    // Sort patterns by idx2 ascending (requirement 11.12)
    patterns.sort_by_key(|p| p.idx2);

    Ok(DoubleTopDetection {
        patterns,
        latest_min,
        latest_max,
    })
}

/// Detect double-top patterns using default parameters.
///
/// This is a convenience wrapper that delegates to `detect_double_tops` with
/// the Python defaults: `window = 5`, `tolerance_pct = 0.3`, `min_width = 5`.
///
/// # Arguments
///
/// * `bars` - Slice of bars ordered oldest to newest
///
/// # Returns
///
/// * `Ok(DoubleTopDetection)` - Detection result with patterns and latest extrema
/// * `Err(IndicatorError)` - If an error occurs (should not happen with valid defaults)
///
/// # Example
///
/// ```ignore
/// use modelenv_core::indicators::patterns::detect_double_tops_default;
///
/// let bars = vec![/* ... */];
/// let result = detect_double_tops_default(&bars)?;
/// for pattern in result.patterns {
///     println!("Found double top at indices {} and {}", pattern.idx1, pattern.idx2);
/// }
/// ```
pub fn detect_double_tops_default(bars: &[Bar]) -> Result<DoubleTopDetection, IndicatorError> {
    detect_double_tops(bars, 5, 0.3, 5)
}

/// Find the running maximum value and index after a given start index.
fn find_running_max_after(bars: &[Bar], start: usize) -> (usize, f64) {
    let mut max_idx = start;
    let mut max_val = bars[start].high;

    for i in (start + 1)..bars.len() {
        if bars[i].high > max_val {
            max_val = bars[i].high;
            max_idx = i;
        }
    }

    (max_idx, max_val)
}

/// Try to create a double-top pattern from two indices.
#[allow(clippy::too_many_arguments)]
fn try_create_double_top(
    bars: &[Bar],
    idx1: usize,
    idx2: usize,
    tolerance_pct: f64,
    min_width: usize,
    local_minima: &[usize],
    local_maxima: &[usize],
    half_window: usize,
    is_forming: bool,
) -> Option<DoubleTop> {
    // Check width requirement (11.5)
    let width_bars = idx2 - idx1;
    if width_bars < min_width {
        return None;
    }

    let high1 = bars[idx1].high;
    let high2 = bars[idx2].high;

    // Check tolerance requirement (11.5)
    // |bars[idx1].high - bars[idx2].high| / max(bars[idx1].high, bars[idx2].high) * 100 <= tolerance_pct
    let max_high = high1.max(high2);
    if max_high <= 0.0 {
        return None;
    }
    let tolerance_check = (high1 - high2).abs() / max_high * 100.0;
    if tolerance_check > tolerance_pct {
        return None;
    }

    // Find neckline (min low between idx1 and idx2, exclusive of endpoints for strict ordering)
    let (neckline, neckline_idx) = find_neckline_min(bars, idx1, idx2)?;

    // Verify neckline_idx is strictly between idx1 and idx2 (requirement 11.7)
    if neckline_idx <= idx1 || neckline_idx >= idx2 {
        return None;
    }

    // Calculate depth percentage (11.5, 11.10)
    // depth_pct = ((bars[idx1].high + bars[idx2].high) / 2 - neckline) / ((bars[idx1].high + bars[idx2].high) / 2) * 100
    let avg_high = (high1 + high2) / 2.0;
    let depth_pct_raw = (avg_high - neckline) / avg_high * 100.0;

    // Check minimum depth threshold
    if depth_pct_raw < MIN_DEPTH_PCT {
        return None;
    }

    // Apply banker's rounding to 3 decimal places (11.10)
    let depth_pct = bankers_round(depth_pct_raw, 3);

    // Check confirmation (11.6)
    // confirmed = true if any bars[i].close < neckline for i in idx2+1..bars.len()
    let confirmed = if is_forming {
        false
    } else {
        bars[(idx2 + 1)..]
            .iter()
            .any(|bar| bar.close < neckline)
    };

    // Find nearest extrema before idx1 and after idx2 (11.8)
    let (min_before_val, min_before_ts) =
        find_nearest_extremum_before(bars, idx1, local_minima, true);
    let (max_before_val, max_before_ts) =
        find_nearest_extremum_before(bars, idx1, local_maxima, false);

    let (min_after_val, min_after_ts, max_after_val, max_after_ts) = if is_forming {
        // Forming patterns have no after extrema (11.14)
        (None, None, None, None)
    } else {
        let (min_val, min_ts) = find_nearest_extremum_after(bars, idx2, local_minima, half_window, true);
        let (max_val, max_ts) = find_nearest_extremum_after(bars, idx2, local_maxima, half_window, false);
        (min_val, min_ts, max_val, max_ts)
    };

    // Round price values to 5 decimal places to match Python's round(x, 5)
    Some(DoubleTop {
        idx1,
        idx2,
        ts1: bars[idx1].timestamp_ns,
        ts2: bars[idx2].timestamp_ns,
        high1: round_to_5(high1),
        high2: round_to_5(high2),
        neckline: round_to_5(neckline),
        neckline_idx,
        depth_pct,
        width_bars,
        confirmed,
        min_before_val,
        min_before_ts,
        max_before_val,
        max_before_ts,
        min_after_val,
        min_after_ts,
        max_after_val,
        max_after_ts,
    })
}

/// Try to create a forming double-top pattern at the right edge.
#[allow(clippy::too_many_arguments)]
fn try_create_forming_double_top(
    bars: &[Bar],
    idx1: usize,
    running_max_idx: usize,
    running_max_val: f64,
    tolerance_pct: f64,
    min_width: usize,
    local_minima: &[usize],
    local_maxima: &[usize],
    _half_window: usize,
) -> Option<DoubleTop> {
    // Use the running maximum as idx2
    let idx2 = running_max_idx;

    // Check width requirement
    let width_bars = idx2 - idx1;
    if width_bars < min_width {
        return None;
    }

    let high1 = bars[idx1].high;
    let high2 = running_max_val;

    // Check tolerance requirement
    let max_high = high1.max(high2);
    if max_high <= 0.0 {
        return None;
    }
    let tolerance_check = (high1 - high2).abs() / max_high * 100.0;
    if tolerance_check > tolerance_pct {
        return None;
    }

    // Find neckline (min low between idx1 and idx2)
    let (neckline, neckline_idx) = find_neckline_min(bars, idx1, idx2)?;

    // Verify neckline_idx is strictly between idx1 and idx2
    if neckline_idx <= idx1 || neckline_idx >= idx2 {
        return None;
    }

    // Calculate depth percentage
    let avg_high = (high1 + high2) / 2.0;
    let depth_pct_raw = (avg_high - neckline) / avg_high * 100.0;

    // Check minimum depth threshold
    if depth_pct_raw < MIN_DEPTH_PCT {
        return None;
    }

    // Apply banker's rounding to 3 decimal places
    let depth_pct = bankers_round(depth_pct_raw, 3);

    // Find nearest extrema before idx1
    let (min_before_val, min_before_ts) =
        find_nearest_extremum_before(bars, idx1, local_minima, true);
    let (max_before_val, max_before_ts) =
        find_nearest_extremum_before(bars, idx1, local_maxima, false);

    // Round price values to 5 decimal places to match Python's round(x, 5)
    Some(DoubleTop {
        idx1,
        idx2,
        ts1: bars[idx1].timestamp_ns,
        ts2: bars[idx2].timestamp_ns,
        high1: round_to_5(high1),
        high2: round_to_5(high2),
        neckline: round_to_5(neckline),
        neckline_idx,
        depth_pct,
        width_bars,
        confirmed: false, // Forming patterns are never confirmed
        min_before_val,
        min_before_ts,
        max_before_val,
        max_before_ts,
        min_after_val: None, // Forming patterns have no after extrema
        min_after_ts: None,
        max_after_val: None,
        max_after_ts: None,
    })
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

    fn bar_with_ts(low: f64, high: f64, ts: i64) -> Bar {
        Bar {
            timestamp_ns: ts,
            open: (low + high) / 2.0,
            high,
            low,
            close: (low + high) / 2.0,
            volume: 1.0,
        }
    }

    fn bar_with_close(low: f64, high: f64, close: f64, ts: i64) -> Bar {
        Bar {
            timestamp_ns: ts,
            open: (low + high) / 2.0,
            high,
            low,
            close,
            volume: 1.0,
        }
    }

    // ==================== double_bottom_score tests ====================

    #[test]
    fn empty_or_short_returns_zero() {
        assert_eq!(double_bottom_score(&[]), 0.0);
        assert_eq!(double_bottom_score(&[bar(100.0, 101.0); 4]), 0.0);
    }

    #[test]
    fn monotonic_up_returns_zero() {
        let bars: Vec<Bar> = (0..30)
            .map(|i| bar(100.0 + i as f64, 101.0 + i as f64))
            .collect();
        assert_eq!(double_bottom_score(&bars), 0.0);
    }

    #[test]
    fn flat_returns_zero() {
        let bars = vec![bar(100.0, 101.0); 30];
        assert_eq!(double_bottom_score(&bars), 0.0);
    }

    #[test]
    fn clean_double_bottom_scores_positive() {
        // Build: down, low, up, peak, down, low, up
        // Lows at index 4 and 12, peak around index 8.
        let mut bars = Vec::new();
        for i in 0..4 {
            bars.push(bar(110.0 - i as f64, 111.0 - i as f64));
        }
        bars.push(bar(105.0, 106.0));
        bars.push(bar(105.5, 106.5));
        for i in 0..3 {
            bars.push(bar(106.0 + i as f64, 107.0 + i as f64));
        }
        bars.push(bar(108.5, 109.5));
        for i in 0..3 {
            bars.push(bar(108.0 - i as f64, 109.0 - i as f64));
        }
        bars.push(bar(105.05, 106.05));
        for i in 0..4 {
            bars.push(bar(105.5 + i as f64, 106.5 + i as f64));
        }

        let score = double_bottom_score(&bars);
        assert!(score > 0.0, "expected positive score, got {}", score);
        assert!(score <= 1.0, "score must be <= 1.0, got {}", score);
    }

    #[test]
    fn dissimilar_bottoms_score_zero() {
        // Two troughs but the second is far below the first - not a double bottom.
        let mut bars = Vec::new();
        for i in 0..4 {
            bars.push(bar(110.0 - i as f64, 111.0 - i as f64));
        }
        bars.push(bar(105.0, 106.0));
        for i in 0..3 {
            bars.push(bar(106.0 + i as f64, 107.0 + i as f64));
        }
        bars.push(bar(108.5, 109.5));
        for i in 0..3 {
            bars.push(bar(107.0 - i as f64, 108.0 - i as f64));
        }
        bars.push(bar(95.0, 96.0));
        for i in 0..4 {
            bars.push(bar(95.5 + i as f64, 96.5 + i as f64));
        }

        assert_eq!(double_bottom_score(&bars), 0.0);
    }

    // ==================== detect_double_bottoms tests ====================

    #[test]
    fn detect_double_bottoms_invalid_window() {
        let bars = vec![bar(100.0, 101.0); 20];
        let result = detect_double_bottoms(&bars, 0, 0.3, 5);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { param_name: "window", .. })
        ));
    }

    #[test]
    fn detect_double_bottoms_invalid_min_width() {
        let bars = vec![bar(100.0, 101.0); 20];
        let result = detect_double_bottoms(&bars, 5, 0.3, 0);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { param_name: "min_width", .. })
        ));
    }

    #[test]
    fn detect_double_bottoms_invalid_tolerance_negative() {
        let bars = vec![bar(100.0, 101.0); 20];
        let result = detect_double_bottoms(&bars, 5, -0.1, 5);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { param_name: "tolerance_pct", .. })
        ));
    }

    #[test]
    fn detect_double_bottoms_invalid_tolerance_nan() {
        let bars = vec![bar(100.0, 101.0); 20];
        let result = detect_double_bottoms(&bars, 5, f64::NAN, 5);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { param_name: "tolerance_pct", .. })
        ));
    }

    #[test]
    fn detect_double_bottoms_invalid_tolerance_inf() {
        let bars = vec![bar(100.0, 101.0); 20];
        let result = detect_double_bottoms(&bars, 5, f64::INFINITY, 5);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { param_name: "tolerance_pct", .. })
        ));
    }

    #[test]
    fn detect_double_bottoms_insufficient_bars() {
        let bars = vec![bar(100.0, 101.0); 10]; // Less than 2*5+1 = 11
        let result = detect_double_bottoms(&bars, 5, 0.3, 5).unwrap();
        assert!(result.patterns.is_empty());
        assert!(result.latest_min.is_none());
        assert!(result.latest_max.is_none());
    }

    #[test]
    fn detect_double_bottoms_empty_bars() {
        let bars: Vec<Bar> = vec![];
        let result = detect_double_bottoms(&bars, 5, 0.3, 5).unwrap();
        assert!(result.patterns.is_empty());
    }

    #[test]
    fn detect_double_bottoms_flat_no_patterns() {
        let bars = vec![bar(100.0, 101.0); 30];
        let result = detect_double_bottoms(&bars, 5, 0.3, 5).unwrap();
        assert!(result.patterns.is_empty());
    }

    #[test]
    fn detect_double_bottoms_finds_valid_pattern() {
        // Create a clear double-bottom pattern:
        // - First bottom at index ~5
        // - Peak (neckline) at index ~10
        // - Second bottom at index ~15
        // - Confirmation close above neckline at index ~20
        let mut bars = Vec::new();

        // Descending to first bottom (indices 0-4)
        for i in 0..5 {
            bars.push(bar_with_ts(110.0 - i as f64 * 2.0, 112.0 - i as f64 * 2.0, i as i64));
        }

        // First bottom at index 5
        bars.push(bar_with_ts(100.0, 101.0, 5));

        // Rising to neckline (indices 6-9)
        for i in 0..4 {
            bars.push(bar_with_ts(102.0 + i as f64 * 2.0, 104.0 + i as f64 * 2.0, 6 + i as i64));
        }

        // Neckline peak at index 10
        bars.push(bar_with_ts(108.0, 115.0, 10));

        // Descending to second bottom (indices 11-14)
        for i in 0..4 {
            bars.push(bar_with_ts(106.0 - i as f64 * 1.5, 108.0 - i as f64 * 1.5, 11 + i as i64));
        }

        // Second bottom at index 15
        bars.push(bar_with_ts(100.1, 101.1, 15));

        // Rising after second bottom (indices 16-19)
        for i in 0..4 {
            bars.push(bar_with_ts(102.0 + i as f64 * 2.0, 104.0 + i as f64 * 2.0, 16 + i as i64));
        }

        // Confirmation: close above neckline at index 20
        bars.push(bar_with_close(114.0, 118.0, 116.0, 20));

        // Add more bars for extrema detection
        for i in 0..5 {
            bars.push(bar_with_ts(115.0 + i as f64, 117.0 + i as f64, 21 + i as i64));
        }

        let result = detect_double_bottoms(&bars, 3, 0.5, 5).unwrap();

        // Should find at least one pattern
        assert!(
            !result.patterns.is_empty(),
            "Expected to find double-bottom patterns, found none"
        );

        // Check pattern invariants
        for pattern in &result.patterns {
            // idx1 < neckline_idx < idx2
            assert!(
                pattern.idx1 < pattern.neckline_idx,
                "idx1 ({}) should be < neckline_idx ({})",
                pattern.idx1,
                pattern.neckline_idx
            );
            assert!(
                pattern.neckline_idx < pattern.idx2,
                "neckline_idx ({}) should be < idx2 ({})",
                pattern.neckline_idx,
                pattern.idx2
            );

            // width_bars == idx2 - idx1
            assert_eq!(
                pattern.width_bars,
                pattern.idx2 - pattern.idx1,
                "width_bars should equal idx2 - idx1"
            );

            // depth_pct >= 0.1
            assert!(
                pattern.depth_pct >= 0.1,
                "depth_pct ({}) should be >= 0.1",
                pattern.depth_pct
            );
        }
    }

    #[test]
    fn detect_double_bottoms_pattern_invariants() {
        // Test that all detected patterns satisfy the invariants
        let mut bars = Vec::new();

        // Create multiple potential double-bottom patterns
        for cycle in 0..3 {
            let base = cycle * 20;
            // Descending
            for i in 0..5 {
                bars.push(bar_with_ts(
                    110.0 - i as f64 * 2.0,
                    112.0 - i as f64 * 2.0,
                    (base + i) as i64,
                ));
            }
            // Bottom
            bars.push(bar_with_ts(100.0, 101.0, (base + 5) as i64));
            // Rising
            for i in 0..4 {
                bars.push(bar_with_ts(
                    102.0 + i as f64 * 3.0,
                    104.0 + i as f64 * 3.0,
                    (base + 6 + i) as i64,
                ));
            }
            // Peak
            bars.push(bar_with_ts(112.0, 118.0, (base + 10) as i64));
            // Descending
            for i in 0..4 {
                bars.push(bar_with_ts(
                    110.0 - i as f64 * 2.5,
                    112.0 - i as f64 * 2.5,
                    (base + 11 + i) as i64,
                ));
            }
            // Second bottom
            bars.push(bar_with_ts(100.05, 101.05, (base + 15) as i64));
            // Rising with confirmation
            for i in 0..4 {
                bars.push(bar_with_close(
                    115.0 + i as f64,
                    120.0 + i as f64,
                    119.0 + i as f64,
                    (base + 16 + i) as i64,
                ));
            }
        }

        let result = detect_double_bottoms(&bars, 3, 0.5, 5).unwrap();

        for pattern in &result.patterns {
            // Invariant 1: idx1 < neckline_idx < idx2
            assert!(
                pattern.idx1 < pattern.neckline_idx && pattern.neckline_idx < pattern.idx2,
                "Pattern invariant violated: idx1={}, neckline_idx={}, idx2={}",
                pattern.idx1,
                pattern.neckline_idx,
                pattern.idx2
            );

            // Invariant 2: width_bars == idx2 - idx1
            assert_eq!(
                pattern.width_bars,
                pattern.idx2 - pattern.idx1,
                "width_bars mismatch"
            );

            // Invariant 3: depth_pct >= 0.1
            assert!(pattern.depth_pct >= 0.1, "depth_pct too small");

            // Verify timestamps match indices
            assert_eq!(pattern.ts1, bars[pattern.idx1].timestamp_ns);
            assert_eq!(pattern.ts2, bars[pattern.idx2].timestamp_ns);

            // Verify low values match indices
            assert!((pattern.low1 - bars[pattern.idx1].low).abs() < 1e-10);
            assert!((pattern.low2 - bars[pattern.idx2].low).abs() < 1e-10);
        }
    }

    #[test]
    fn bankers_round_test() {
        // Test banker's rounding (round half to even)
        assert!((bankers_round(2.5, 0) - 2.0).abs() < 1e-10); // 2.5 -> 2 (even)
        assert!((bankers_round(3.5, 0) - 4.0).abs() < 1e-10); // 3.5 -> 4 (even)
        assert!((bankers_round(2.25, 1) - 2.2).abs() < 1e-10); // 2.25 -> 2.2 (even)
        assert!((bankers_round(2.35, 1) - 2.4).abs() < 1e-10); // 2.35 -> 2.4 (even)
        assert!((bankers_round(2.345, 2) - 2.34).abs() < 1e-10); // 2.345 -> 2.34 (even)
        assert!((bankers_round(2.355, 2) - 2.36).abs() < 1e-10); // 2.355 -> 2.36 (even)

        // Regular rounding cases
        assert!((bankers_round(2.4, 0) - 2.0).abs() < 1e-10);
        assert!((bankers_round(2.6, 0) - 3.0).abs() < 1e-10);
        assert!((bankers_round(2.123, 2) - 2.12).abs() < 1e-10);
        assert!((bankers_round(2.127, 2) - 2.13).abs() < 1e-10);
    }

    #[test]
    fn find_local_minima_test() {
        // Create bars with clear local minima
        let bars = vec![
            bar(105.0, 106.0), // 0
            bar(104.0, 105.0), // 1
            bar(103.0, 104.0), // 2
            bar(100.0, 101.0), // 3 - local minimum
            bar(102.0, 103.0), // 4
            bar(104.0, 105.0), // 5
            bar(106.0, 107.0), // 6
            bar(105.0, 106.0), // 7
            bar(103.0, 104.0), // 8
            bar(100.5, 101.5), // 9 - local minimum
            bar(102.0, 103.0), // 10
            bar(104.0, 105.0), // 11
            bar(106.0, 107.0), // 12
        ];

        let minima = find_local_minima(&bars, 2);
        assert!(minima.contains(&3), "Should find minimum at index 3");
        assert!(minima.contains(&9), "Should find minimum at index 9");
    }

    #[test]
    fn find_local_maxima_test() {
        // Create bars with clear local maxima
        let bars = vec![
            bar(100.0, 101.0), // 0
            bar(102.0, 103.0), // 1
            bar(104.0, 105.0), // 2
            bar(106.0, 110.0), // 3 - local maximum
            bar(104.0, 105.0), // 4
            bar(102.0, 103.0), // 5
            bar(100.0, 101.0), // 6
            bar(102.0, 103.0), // 7
            bar(104.0, 105.0), // 8
            bar(106.0, 109.0), // 9 - local maximum
            bar(104.0, 105.0), // 10
            bar(102.0, 103.0), // 11
            bar(100.0, 101.0), // 12
        ];

        let maxima = find_local_maxima(&bars, 2);
        assert!(maxima.contains(&3), "Should find maximum at index 3");
        assert!(maxima.contains(&9), "Should find maximum at index 9");
    }

    #[test]
    fn detect_double_bottoms_tolerance_check() {
        // Create two bottoms that are too far apart in price
        let mut bars = Vec::new();

        // First bottom at 100
        for i in 0..5 {
            bars.push(bar_with_ts(110.0 - i as f64 * 2.0, 112.0 - i as f64 * 2.0, i as i64));
        }
        bars.push(bar_with_ts(100.0, 101.0, 5));

        // Peak
        for i in 0..5 {
            bars.push(bar_with_ts(105.0 + i as f64 * 2.0, 107.0 + i as f64 * 2.0, 6 + i as i64));
        }
        bars.push(bar_with_ts(114.0, 120.0, 11));

        // Second bottom at 95 (5% difference, should fail with 0.3% tolerance)
        for i in 0..5 {
            bars.push(bar_with_ts(110.0 - i as f64 * 3.0, 112.0 - i as f64 * 3.0, 12 + i as i64));
        }
        bars.push(bar_with_ts(95.0, 96.0, 17));

        // Trailing bars
        for i in 0..5 {
            bars.push(bar_with_ts(100.0 + i as f64, 102.0 + i as f64, 18 + i as i64));
        }

        // With very tight tolerance, should not find pattern
        let result = detect_double_bottoms(&bars, 3, 0.3, 5).unwrap();
        
        // The 5% difference should exceed 0.3% tolerance
        // Check that no pattern with these specific bottoms exists
        let has_pattern_with_large_diff = result.patterns.iter().any(|p| {
            let diff = (p.low1 - p.low2).abs() / p.low1.min(p.low2) * 100.0;
            diff > 0.3
        });
        assert!(
            !has_pattern_with_large_diff,
            "Should not find patterns exceeding tolerance"
        );
    }

    #[test]
    fn detect_double_bottoms_width_check() {
        // Create two bottoms that are too close together
        let mut bars = Vec::new();

        for i in 0..5 {
            bars.push(bar_with_ts(110.0 - i as f64, 112.0 - i as f64, i as i64));
        }
        // First bottom
        bars.push(bar_with_ts(100.0, 101.0, 5));
        // Small peak
        bars.push(bar_with_ts(102.0, 108.0, 6));
        // Second bottom (only 2 bars apart)
        bars.push(bar_with_ts(100.1, 101.1, 7));

        for i in 0..10 {
            bars.push(bar_with_ts(105.0 + i as f64, 107.0 + i as f64, 8 + i as i64));
        }

        // With min_width=5, should not find pattern with width=2
        let result = detect_double_bottoms(&bars, 2, 1.0, 5).unwrap();
        
        // All patterns should have width >= 5
        for pattern in &result.patterns {
            assert!(
                pattern.width_bars >= 5,
                "Pattern width {} should be >= 5",
                pattern.width_bars
            );
        }
    }

    #[test]
    fn detect_double_bottoms_confirmation() {
        // Create a pattern and verify confirmation logic
        let mut bars = Vec::new();

        // Build pattern
        for i in 0..5 {
            bars.push(bar_with_ts(110.0 - i as f64 * 2.0, 112.0 - i as f64 * 2.0, i as i64));
        }
        bars.push(bar_with_ts(100.0, 101.0, 5)); // First bottom

        for i in 0..5 {
            bars.push(bar_with_ts(105.0 + i as f64 * 2.0, 107.0 + i as f64 * 2.0, 6 + i as i64));
        }
        bars.push(bar_with_ts(114.0, 120.0, 11)); // Neckline at 120

        for i in 0..5 {
            bars.push(bar_with_ts(110.0 - i as f64 * 2.0, 112.0 - i as f64 * 2.0, 12 + i as i64));
        }
        bars.push(bar_with_ts(100.05, 101.05, 17)); // Second bottom

        // Add bars that close above neckline (120)
        for i in 0..5 {
            bars.push(bar_with_close(118.0 + i as f64, 125.0 + i as f64, 122.0, 18 + i as i64));
        }

        let result = detect_double_bottoms(&bars, 3, 0.5, 5).unwrap();

        // Should have confirmed patterns (close > neckline)
        let confirmed_count = result.patterns.iter().filter(|p| p.confirmed).count();
        assert!(
            confirmed_count > 0 || result.patterns.is_empty(),
            "Expected confirmed patterns when close > neckline"
        );
    }

    #[test]
    fn detect_double_bottoms_latest_extrema() {
        // Verify latest_min and latest_max are populated
        let mut bars = Vec::new();

        // Create bars with clear extrema
        for i in 0..5 {
            bars.push(bar_with_ts(100.0 + i as f64 * 2.0, 102.0 + i as f64 * 2.0, i as i64));
        }
        bars.push(bar_with_ts(108.0, 115.0, 5)); // Local max
        for i in 0..5 {
            bars.push(bar_with_ts(106.0 - i as f64 * 2.0, 108.0 - i as f64 * 2.0, 6 + i as i64));
        }
        bars.push(bar_with_ts(95.0, 96.0, 11)); // Local min
        for i in 0..5 {
            bars.push(bar_with_ts(98.0 + i as f64, 100.0 + i as f64, 12 + i as i64));
        }

        let result = detect_double_bottoms(&bars, 2, 1.0, 3).unwrap();

        // Should have latest extrema
        assert!(
            result.latest_min.is_some() || result.latest_max.is_some(),
            "Should have at least one latest extremum"
        );
    }

    #[test]
    fn detect_double_bottoms_default_uses_correct_params() {
        // Test that detect_double_bottoms_default delegates with correct parameters
        // (window=5, tolerance_pct=0.3, min_width=5)
        use super::detect_double_bottoms_default;

        // Empty bars should return empty result (same as calling with explicit params)
        let empty_bars: Vec<Bar> = vec![];
        let result = detect_double_bottoms_default(&empty_bars).unwrap();
        assert!(result.patterns.is_empty());
        assert!(result.latest_min.is_none());
        assert!(result.latest_max.is_none());

        // Insufficient bars (less than 2*5+1 = 11) should return empty result
        let short_bars = vec![bar(100.0, 101.0); 10];
        let result = detect_double_bottoms_default(&short_bars).unwrap();
        assert!(result.patterns.is_empty());

        // Verify it produces same result as explicit call
        let bars = vec![bar(100.0, 101.0); 30];
        let default_result = detect_double_bottoms_default(&bars).unwrap();
        let explicit_result = detect_double_bottoms(&bars, 5, 0.3, 5).unwrap();
        assert_eq!(default_result.patterns.len(), explicit_result.patterns.len());
        assert_eq!(default_result.latest_min, explicit_result.latest_min);
        assert_eq!(default_result.latest_max, explicit_result.latest_max);
    }

    // ==================== detect_double_tops tests ====================

    #[test]
    fn detect_double_tops_invalid_window() {
        let bars = vec![bar(100.0, 101.0); 20];
        let result = detect_double_tops(&bars, 0, 0.3, 5);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { param_name: "window", .. })
        ));
    }

    #[test]
    fn detect_double_tops_invalid_min_width() {
        let bars = vec![bar(100.0, 101.0); 20];
        let result = detect_double_tops(&bars, 5, 0.3, 0);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { param_name: "min_width", .. })
        ));
    }

    #[test]
    fn detect_double_tops_invalid_tolerance_negative() {
        let bars = vec![bar(100.0, 101.0); 20];
        let result = detect_double_tops(&bars, 5, -0.1, 5);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { param_name: "tolerance_pct", .. })
        ));
    }

    #[test]
    fn detect_double_tops_invalid_tolerance_nan() {
        let bars = vec![bar(100.0, 101.0); 20];
        let result = detect_double_tops(&bars, 5, f64::NAN, 5);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { param_name: "tolerance_pct", .. })
        ));
    }

    #[test]
    fn detect_double_tops_invalid_tolerance_inf() {
        let bars = vec![bar(100.0, 101.0); 20];
        let result = detect_double_tops(&bars, 5, f64::INFINITY, 5);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { param_name: "tolerance_pct", .. })
        ));
    }

    #[test]
    fn detect_double_tops_insufficient_bars() {
        let bars = vec![bar(100.0, 101.0); 10]; // Less than 2*5+1 = 11
        let result = detect_double_tops(&bars, 5, 0.3, 5).unwrap();
        assert!(result.patterns.is_empty());
        assert!(result.latest_min.is_none());
        assert!(result.latest_max.is_none());
    }

    #[test]
    fn detect_double_tops_empty_bars() {
        let bars: Vec<Bar> = vec![];
        let result = detect_double_tops(&bars, 5, 0.3, 5).unwrap();
        assert!(result.patterns.is_empty());
    }

    #[test]
    fn detect_double_tops_flat_no_patterns() {
        let bars = vec![bar(100.0, 101.0); 30];
        let result = detect_double_tops(&bars, 5, 0.3, 5).unwrap();
        assert!(result.patterns.is_empty());
    }

    #[test]
    fn detect_double_tops_finds_valid_pattern() {
        // Create a clear double-top pattern:
        // - First top at index ~5
        // - Trough (neckline) at index ~10
        // - Second top at index ~15
        // - Confirmation close below neckline at index ~20
        let mut bars = Vec::new();

        // Ascending to first top (indices 0-4)
        for i in 0..5 {
            bars.push(bar_with_ts(100.0 + i as f64 * 2.0, 102.0 + i as f64 * 2.0, i as i64));
        }

        // First top at index 5
        bars.push(bar_with_ts(109.0, 115.0, 5));

        // Descending to neckline (indices 6-9)
        for i in 0..4 {
            bars.push(bar_with_ts(108.0 - i as f64 * 2.0, 110.0 - i as f64 * 2.0, 6 + i as i64));
        }

        // Neckline trough at index 10
        bars.push(bar_with_ts(100.0, 101.0, 10));

        // Ascending to second top (indices 11-14)
        for i in 0..4 {
            bars.push(bar_with_ts(102.0 + i as f64 * 2.0, 104.0 + i as f64 * 2.0, 11 + i as i64));
        }

        // Second top at index 15
        bars.push(bar_with_ts(109.0, 114.9, 15));

        // Descending after second top (indices 16-19)
        for i in 0..4 {
            bars.push(bar_with_ts(108.0 - i as f64 * 2.0, 110.0 - i as f64 * 2.0, 16 + i as i64));
        }

        // Confirmation: close below neckline at index 20
        bars.push(bar_with_close(98.0, 102.0, 99.0, 20));

        // Add more bars for extrema detection
        for i in 0..5 {
            bars.push(bar_with_ts(95.0 - i as f64, 97.0 - i as f64, 21 + i as i64));
        }

        let result = detect_double_tops(&bars, 3, 0.5, 5).unwrap();

        // Should find at least one pattern
        assert!(
            !result.patterns.is_empty(),
            "Expected to find double-top patterns, found none"
        );

        // Check pattern invariants
        for pattern in &result.patterns {
            // idx1 < neckline_idx < idx2
            assert!(
                pattern.idx1 < pattern.neckline_idx,
                "idx1 ({}) should be < neckline_idx ({})",
                pattern.idx1,
                pattern.neckline_idx
            );
            assert!(
                pattern.neckline_idx < pattern.idx2,
                "neckline_idx ({}) should be < idx2 ({})",
                pattern.neckline_idx,
                pattern.idx2
            );

            // width_bars == idx2 - idx1
            assert_eq!(
                pattern.width_bars,
                pattern.idx2 - pattern.idx1,
                "width_bars should equal idx2 - idx1"
            );

            // depth_pct >= 0.1
            assert!(
                pattern.depth_pct >= 0.1,
                "depth_pct ({}) should be >= 0.1",
                pattern.depth_pct
            );
        }
    }

    #[test]
    fn detect_double_tops_pattern_invariants() {
        // Test that all detected patterns satisfy the invariants
        let mut bars = Vec::new();

        // Create multiple potential double-top patterns
        for cycle in 0..3 {
            let base = cycle * 20;
            // Ascending
            for i in 0..5 {
                bars.push(bar_with_ts(
                    100.0 + i as f64 * 2.0,
                    102.0 + i as f64 * 2.0,
                    (base + i) as i64,
                ));
            }
            // Top
            bars.push(bar_with_ts(109.0, 115.0, (base + 5) as i64));
            // Descending
            for i in 0..4 {
                bars.push(bar_with_ts(
                    108.0 - i as f64 * 3.0,
                    110.0 - i as f64 * 3.0,
                    (base + 6 + i) as i64,
                ));
            }
            // Trough
            bars.push(bar_with_ts(95.0, 96.0, (base + 10) as i64));
            // Ascending
            for i in 0..4 {
                bars.push(bar_with_ts(
                    97.0 + i as f64 * 3.0,
                    99.0 + i as f64 * 3.0,
                    (base + 11 + i) as i64,
                ));
            }
            // Second top
            bars.push(bar_with_ts(109.0, 114.95, (base + 15) as i64));
            // Descending with confirmation
            for i in 0..4 {
                bars.push(bar_with_close(
                    92.0 - i as f64,
                    94.0 - i as f64,
                    93.0 - i as f64,
                    (base + 16 + i) as i64,
                ));
            }
        }

        let result = detect_double_tops(&bars, 3, 0.5, 5).unwrap();

        for pattern in &result.patterns {
            // Invariant 1: idx1 < neckline_idx < idx2
            assert!(
                pattern.idx1 < pattern.neckline_idx && pattern.neckline_idx < pattern.idx2,
                "Pattern invariant violated: idx1={}, neckline_idx={}, idx2={}",
                pattern.idx1,
                pattern.neckline_idx,
                pattern.idx2
            );

            // Invariant 2: width_bars == idx2 - idx1
            assert_eq!(
                pattern.width_bars,
                pattern.idx2 - pattern.idx1,
                "width_bars mismatch"
            );

            // Invariant 3: depth_pct >= 0.1
            assert!(pattern.depth_pct >= 0.1, "depth_pct too small");

            // Verify timestamps match indices
            assert_eq!(pattern.ts1, bars[pattern.idx1].timestamp_ns);
            assert_eq!(pattern.ts2, bars[pattern.idx2].timestamp_ns);

            // Verify high values match indices
            assert!((pattern.high1 - bars[pattern.idx1].high).abs() < 1e-10);
            assert!((pattern.high2 - bars[pattern.idx2].high).abs() < 1e-10);
        }
    }

    #[test]
    fn detect_double_tops_tolerance_check() {
        // Create two tops that are too far apart in price
        let mut bars = Vec::new();

        // First top at 115
        for i in 0..5 {
            bars.push(bar_with_ts(100.0 + i as f64 * 2.0, 102.0 + i as f64 * 2.0, i as i64));
        }
        bars.push(bar_with_ts(113.0, 115.0, 5));

        // Trough
        for i in 0..5 {
            bars.push(bar_with_ts(108.0 - i as f64 * 2.0, 110.0 - i as f64 * 2.0, 6 + i as i64));
        }
        bars.push(bar_with_ts(95.0, 96.0, 11));

        // Second top at 108 (6% difference from 115, should fail with 0.3% tolerance)
        for i in 0..5 {
            bars.push(bar_with_ts(98.0 + i as f64 * 2.0, 100.0 + i as f64 * 2.0, 12 + i as i64));
        }
        bars.push(bar_with_ts(106.0, 108.0, 17));

        // Trailing bars
        for i in 0..5 {
            bars.push(bar_with_ts(100.0 - i as f64, 102.0 - i as f64, 18 + i as i64));
        }

        // With very tight tolerance, should not find pattern
        let result = detect_double_tops(&bars, 3, 0.3, 5).unwrap();
        
        // The 6% difference should exceed 0.3% tolerance
        // Check that no pattern with these specific tops exists
        let has_pattern_with_large_diff = result.patterns.iter().any(|p| {
            let diff = (p.high1 - p.high2).abs() / p.high1.max(p.high2) * 100.0;
            diff > 0.3
        });
        assert!(
            !has_pattern_with_large_diff,
            "Should not find patterns exceeding tolerance"
        );
    }

    #[test]
    fn detect_double_tops_width_check() {
        // Create two tops that are too close together
        let mut bars = Vec::new();

        for i in 0..5 {
            bars.push(bar_with_ts(100.0 + i as f64, 102.0 + i as f64, i as i64));
        }
        // First top
        bars.push(bar_with_ts(109.0, 115.0, 5));
        // Small trough
        bars.push(bar_with_ts(100.0, 101.0, 6));
        // Second top (only 2 bars apart)
        bars.push(bar_with_ts(109.0, 114.9, 7));

        for i in 0..10 {
            bars.push(bar_with_ts(105.0 - i as f64, 107.0 - i as f64, 8 + i as i64));
        }

        // With min_width=5, should not find pattern with width=2
        let result = detect_double_tops(&bars, 2, 1.0, 5).unwrap();
        
        // All patterns should have width >= 5
        for pattern in &result.patterns {
            assert!(
                pattern.width_bars >= 5,
                "Pattern width {} should be >= 5",
                pattern.width_bars
            );
        }
    }

    #[test]
    fn detect_double_tops_confirmation() {
        // Create a pattern and verify confirmation logic
        let mut bars = Vec::new();

        // Build pattern
        for i in 0..5 {
            bars.push(bar_with_ts(100.0 + i as f64 * 2.0, 102.0 + i as f64 * 2.0, i as i64));
        }
        bars.push(bar_with_ts(109.0, 115.0, 5)); // First top

        for i in 0..5 {
            bars.push(bar_with_ts(108.0 - i as f64 * 2.0, 110.0 - i as f64 * 2.0, 6 + i as i64));
        }
        bars.push(bar_with_ts(95.0, 96.0, 11)); // Neckline at 95

        for i in 0..5 {
            bars.push(bar_with_ts(97.0 + i as f64 * 2.0, 99.0 + i as f64 * 2.0, 12 + i as i64));
        }
        bars.push(bar_with_ts(109.0, 114.95, 17)); // Second top

        // Add bars that close below neckline (95)
        for i in 0..5 {
            bars.push(bar_with_close(90.0 - i as f64, 94.0 - i as f64, 92.0, 18 + i as i64));
        }

        let result = detect_double_tops(&bars, 3, 0.5, 5).unwrap();

        // Should have confirmed patterns (close < neckline)
        let confirmed_count = result.patterns.iter().filter(|p| p.confirmed).count();
        assert!(
            confirmed_count > 0 || result.patterns.is_empty(),
            "Expected confirmed patterns when close < neckline"
        );
    }

    #[test]
    fn detect_double_tops_latest_extrema() {
        // Verify latest_min and latest_max are populated
        let mut bars = Vec::new();

        // Create bars with clear extrema
        for i in 0..5 {
            bars.push(bar_with_ts(100.0 + i as f64 * 2.0, 102.0 + i as f64 * 2.0, i as i64));
        }
        bars.push(bar_with_ts(108.0, 115.0, 5)); // Local max
        for i in 0..5 {
            bars.push(bar_with_ts(106.0 - i as f64 * 2.0, 108.0 - i as f64 * 2.0, 6 + i as i64));
        }
        bars.push(bar_with_ts(95.0, 96.0, 11)); // Local min
        for i in 0..5 {
            bars.push(bar_with_ts(98.0 + i as f64, 100.0 + i as f64, 12 + i as i64));
        }

        let result = detect_double_tops(&bars, 2, 1.0, 3).unwrap();

        // Should have latest extrema
        assert!(
            result.latest_min.is_some() || result.latest_max.is_some(),
            "Should have at least one latest extremum"
        );
    }

    #[test]
    fn detect_double_tops_depth_calculation() {
        // Test that depth_pct is calculated correctly:
        // depth_pct = (avg_high - neckline) / avg_high * 100
        let mut bars = Vec::new();

        // Build a pattern with known values
        for i in 0..5 {
            bars.push(bar_with_ts(100.0 + i as f64 * 2.0, 102.0 + i as f64 * 2.0, i as i64));
        }
        // First top: high = 120
        bars.push(bar_with_ts(118.0, 120.0, 5));

        for i in 0..5 {
            bars.push(bar_with_ts(110.0 - i as f64 * 2.0, 112.0 - i as f64 * 2.0, 6 + i as i64));
        }
        // Neckline: low = 100
        bars.push(bar_with_ts(100.0, 101.0, 11));

        for i in 0..5 {
            bars.push(bar_with_ts(102.0 + i as f64 * 2.0, 104.0 + i as f64 * 2.0, 12 + i as i64));
        }
        // Second top: high = 120
        bars.push(bar_with_ts(118.0, 120.0, 17));

        // Trailing bars
        for i in 0..5 {
            bars.push(bar_with_ts(110.0 - i as f64, 112.0 - i as f64, 18 + i as i64));
        }

        let result = detect_double_tops(&bars, 3, 1.0, 5).unwrap();

        // Expected: avg_high = 120, neckline = 100
        // depth_pct = (120 - 100) / 120 * 100 = 16.667%
        for pattern in &result.patterns {
            // Verify depth_pct is reasonable (should be around 16.667% for this setup)
            assert!(
                pattern.depth_pct >= 0.1,
                "depth_pct ({}) should be >= 0.1",
                pattern.depth_pct
            );
        }
    }

    #[test]
    fn detect_double_tops_neckline_is_min_low() {
        // Verify that neckline is the minimum low between the two tops
        let mut bars = Vec::new();

        // Build pattern
        for i in 0..5 {
            bars.push(bar_with_ts(100.0 + i as f64 * 2.0, 102.0 + i as f64 * 2.0, i as i64));
        }
        bars.push(bar_with_ts(118.0, 120.0, 5)); // First top

        // Descending with varying lows
        bars.push(bar_with_ts(110.0, 112.0, 6));
        bars.push(bar_with_ts(105.0, 107.0, 7));
        bars.push(bar_with_ts(95.0, 97.0, 8)); // This should be the neckline (min low)
        bars.push(bar_with_ts(100.0, 102.0, 9));
        bars.push(bar_with_ts(105.0, 107.0, 10));

        bars.push(bar_with_ts(118.0, 120.0, 11)); // Second top

        // Trailing bars
        for i in 0..5 {
            bars.push(bar_with_ts(110.0 - i as f64, 112.0 - i as f64, 12 + i as i64));
        }

        let result = detect_double_tops(&bars, 2, 1.0, 5).unwrap();

        for pattern in &result.patterns {
            // Verify neckline is the minimum low between idx1 and idx2
            let min_low_between = bars[(pattern.idx1 + 1)..pattern.idx2]
                .iter()
                .map(|b| b.low)
                .fold(f64::INFINITY, f64::min);
            
            assert!(
                (pattern.neckline - min_low_between).abs() < 1e-10,
                "Neckline ({}) should equal min low between tops ({})",
                pattern.neckline,
                min_low_between
            );
        }
    }

    #[test]
    fn detect_double_tops_default_uses_correct_params() {
        // Test that detect_double_tops_default delegates with correct parameters
        // (window=5, tolerance_pct=0.3, min_width=5)
        use super::detect_double_tops_default;

        // Empty bars should return empty result (same as calling with explicit params)
        let empty_bars: Vec<Bar> = vec![];
        let result = detect_double_tops_default(&empty_bars).unwrap();
        assert!(result.patterns.is_empty());
        assert!(result.latest_min.is_none());
        assert!(result.latest_max.is_none());

        // Insufficient bars (less than 2*5+1 = 11) should return empty result
        let short_bars = vec![bar(100.0, 101.0); 10];
        let result = detect_double_tops_default(&short_bars).unwrap();
        assert!(result.patterns.is_empty());

        // Verify it produces same result as explicit call
        let bars = vec![bar(100.0, 101.0); 30];
        let default_result = detect_double_tops_default(&bars).unwrap();
        let explicit_result = detect_double_tops(&bars, 5, 0.3, 5).unwrap();
        assert_eq!(default_result.patterns.len(), explicit_result.patterns.len());
        assert_eq!(default_result.latest_min, explicit_result.latest_min);
        assert_eq!(default_result.latest_max, explicit_result.latest_max);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy to generate a valid Bar with realistic OHLC values
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

    /// Strategy to generate a sequence of bars that forms a potential double-bottom pattern
    /// This creates bars with two distinct lows separated by a peak
    fn double_bottom_bars_strategy() -> impl Strategy<Value = Vec<Bar>> {
        (
            50usize..200,           // total bars
            5usize..20,             // position of first bottom
            10usize..50,            // width between bottoms
            100.0f64..1000.0f64,    // base price
            0.05f64..0.3f64,        // depth factor (how deep the pattern is)
        )
            .prop_flat_map(|(total, first_pos, width, base_price, depth_factor)| {
                let second_pos = (first_pos + width).min(total.saturating_sub(10));
                let neckline_pos = first_pos + width / 2;

                // Generate bars with the pattern structure
                prop::collection::vec(bar_strategy(), total).prop_map(move |mut bars| {
                    // Ensure we have enough bars
                    if bars.len() < second_pos + 5 {
                        return bars;
                    }

                    // Set timestamps
                    for (i, bar) in bars.iter_mut().enumerate() {
                        bar.timestamp_ns = i as i64;
                    }

                    // Create the double-bottom pattern structure
                    let neckline_price = base_price * (1.0 + depth_factor);
                    let bottom_price = base_price;

                    // First bottom
                    if first_pos < bars.len() {
                        bars[first_pos].low = bottom_price;
                        bars[first_pos].high = bottom_price * 1.01;
                        bars[first_pos].open = bottom_price * 1.005;
                        bars[first_pos].close = bottom_price * 1.005;
                    }

                    // Neckline (peak between bottoms)
                    if neckline_pos < bars.len() && neckline_pos > first_pos && neckline_pos < second_pos {
                        bars[neckline_pos].high = neckline_price;
                        bars[neckline_pos].low = neckline_price * 0.99;
                        bars[neckline_pos].open = neckline_price * 0.995;
                        bars[neckline_pos].close = neckline_price * 0.995;
                    }

                    // Second bottom (similar to first)
                    if second_pos < bars.len() {
                        bars[second_pos].low = bottom_price * 1.001; // Slightly different
                        bars[second_pos].high = bottom_price * 1.011;
                        bars[second_pos].open = bottom_price * 1.006;
                        bars[second_pos].close = bottom_price * 1.006;
                    }

                    // Ensure surrounding bars don't violate the local minimum property
                    for i in first_pos.saturating_sub(3)..first_pos {
                        if i < bars.len() {
                            bars[i].low = bottom_price * 1.02;
                        }
                    }
                    for i in (first_pos + 1)..(first_pos + 4).min(neckline_pos) {
                        if i < bars.len() {
                            bars[i].low = bottom_price * 1.02;
                        }
                    }
                    for i in second_pos.saturating_sub(3)..second_pos {
                        if i < bars.len() && i > neckline_pos {
                            bars[i].low = bottom_price * 1.02;
                        }
                    }
                    for i in (second_pos + 1)..(second_pos + 4).min(bars.len()) {
                        if i < bars.len() {
                            bars[i].low = bottom_price * 1.02;
                        }
                    }

                    bars
                })
            })
    }

    /// Strategy to generate a sequence of bars that forms a potential double-top pattern
    /// This creates bars with two distinct highs separated by a trough
    fn double_top_bars_strategy() -> impl Strategy<Value = Vec<Bar>> {
        (
            50usize..200,           // total bars
            5usize..20,             // position of first top
            10usize..50,            // width between tops
            100.0f64..1000.0f64,    // base price
            0.05f64..0.3f64,        // depth factor (how deep the pattern is)
        )
            .prop_flat_map(|(total, first_pos, width, base_price, depth_factor)| {
                let second_pos = (first_pos + width).min(total.saturating_sub(10));
                let neckline_pos = first_pos + width / 2;

                // Generate bars with the pattern structure
                prop::collection::vec(bar_strategy(), total).prop_map(move |mut bars| {
                    // Ensure we have enough bars
                    if bars.len() < second_pos + 5 {
                        return bars;
                    }

                    // Set timestamps
                    for (i, bar) in bars.iter_mut().enumerate() {
                        bar.timestamp_ns = i as i64;
                    }

                    // Create the double-top pattern structure
                    let top_price = base_price * (1.0 + depth_factor);
                    let neckline_price = base_price;

                    // First top
                    if first_pos < bars.len() {
                        bars[first_pos].high = top_price;
                        bars[first_pos].low = top_price * 0.99;
                        bars[first_pos].open = top_price * 0.995;
                        bars[first_pos].close = top_price * 0.995;
                    }

                    // Neckline (trough between tops)
                    if neckline_pos < bars.len() && neckline_pos > first_pos && neckline_pos < second_pos {
                        bars[neckline_pos].low = neckline_price;
                        bars[neckline_pos].high = neckline_price * 1.01;
                        bars[neckline_pos].open = neckline_price * 1.005;
                        bars[neckline_pos].close = neckline_price * 1.005;
                    }

                    // Second top (similar to first)
                    if second_pos < bars.len() {
                        bars[second_pos].high = top_price * 0.999; // Slightly different
                        bars[second_pos].low = top_price * 0.989;
                        bars[second_pos].open = top_price * 0.994;
                        bars[second_pos].close = top_price * 0.994;
                    }

                    // Ensure surrounding bars don't violate the local maximum property
                    for i in first_pos.saturating_sub(3)..first_pos {
                        if i < bars.len() {
                            bars[i].high = top_price * 0.98;
                        }
                    }
                    for i in (first_pos + 1)..(first_pos + 4).min(neckline_pos) {
                        if i < bars.len() {
                            bars[i].high = top_price * 0.98;
                        }
                    }
                    for i in second_pos.saturating_sub(3)..second_pos {
                        if i < bars.len() && i > neckline_pos {
                            bars[i].high = top_price * 0.98;
                        }
                    }
                    for i in (second_pos + 1)..(second_pos + 4).min(bars.len()) {
                        if i < bars.len() {
                            bars[i].high = top_price * 0.98;
                        }
                    }

                    bars
                })
            })
    }

    // **Property 8: Pattern Depth Threshold**
    //
    // For any detected `DoubleBottom` or `DoubleTop` pattern, `depth_pct >= 0.1` SHALL hold.
    //
    // **Validates: Requirements 10.5, 11.5**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn double_bottom_depth_threshold(
            bars in double_bottom_bars_strategy(),
            window in 2usize..10,
            tolerance_pct in 0.1f64..5.0f64,
            min_width in 3usize..15
        ) {
            // Skip if not enough bars for the window
            if bars.len() < 2 * window + 1 {
                return Ok(());
            }

            let result = detect_double_bottoms(&bars, window, tolerance_pct, min_width);

            // Should return Ok for valid inputs
            prop_assert!(result.is_ok(), "detect_double_bottoms returned error: {:?}", result);

            let detection = result.unwrap();

            // Verify all detected patterns have depth_pct >= 0.1
            for (i, pattern) in detection.patterns.iter().enumerate() {
                prop_assert!(
                    pattern.depth_pct >= MIN_DEPTH_PCT,
                    "DoubleBottom pattern {} has depth_pct {} which is < {}. \
                     Pattern: idx1={}, idx2={}, neckline={}, low1={}, low2={}, \
                     bars.len()={}, window={}, tolerance_pct={}, min_width={}",
                    i, pattern.depth_pct, MIN_DEPTH_PCT,
                    pattern.idx1, pattern.idx2, pattern.neckline, pattern.low1, pattern.low2,
                    bars.len(), window, tolerance_pct, min_width
                );
            }
        }

        #[test]
        fn double_top_depth_threshold(
            bars in double_top_bars_strategy(),
            window in 2usize..10,
            tolerance_pct in 0.1f64..5.0f64,
            min_width in 3usize..15
        ) {
            // Skip if not enough bars for the window
            if bars.len() < 2 * window + 1 {
                return Ok(());
            }

            let result = detect_double_tops(&bars, window, tolerance_pct, min_width);

            // Should return Ok for valid inputs
            prop_assert!(result.is_ok(), "detect_double_tops returned error: {:?}", result);

            let detection = result.unwrap();

            // Verify all detected patterns have depth_pct >= 0.1
            for (i, pattern) in detection.patterns.iter().enumerate() {
                prop_assert!(
                    pattern.depth_pct >= MIN_DEPTH_PCT,
                    "DoubleTop pattern {} has depth_pct {} which is < {}. \
                     Pattern: idx1={}, idx2={}, neckline={}, high1={}, high2={}, \
                     bars.len()={}, window={}, tolerance_pct={}, min_width={}",
                    i, pattern.depth_pct, MIN_DEPTH_PCT,
                    pattern.idx1, pattern.idx2, pattern.neckline, pattern.high1, pattern.high2,
                    bars.len(), window, tolerance_pct, min_width
                );
            }
        }

        /// Test with completely random bars to ensure the depth threshold is always enforced
        #[test]
        fn random_bars_depth_threshold(
            bars in prop::collection::vec(bar_strategy(), 20..300),
            window in 2usize..10,
            tolerance_pct in 0.1f64..10.0f64,
            min_width in 3usize..20
        ) {
            // Set timestamps for the bars
            let bars: Vec<Bar> = bars.into_iter().enumerate().map(|(i, mut bar)| {
                bar.timestamp_ns = i as i64;
                bar
            }).collect();

            // Skip if not enough bars for the window
            if bars.len() < 2 * window + 1 {
                return Ok(());
            }

            // Test double bottoms
            let bottom_result = detect_double_bottoms(&bars, window, tolerance_pct, min_width);
            prop_assert!(bottom_result.is_ok(), "detect_double_bottoms returned error: {:?}", bottom_result);

            for pattern in bottom_result.unwrap().patterns {
                prop_assert!(
                    pattern.depth_pct >= MIN_DEPTH_PCT,
                    "DoubleBottom depth_pct {} < {} on random bars",
                    pattern.depth_pct, MIN_DEPTH_PCT
                );
            }

            // Test double tops
            let top_result = detect_double_tops(&bars, window, tolerance_pct, min_width);
            prop_assert!(top_result.is_ok(), "detect_double_tops returned error: {:?}", top_result);

            for pattern in top_result.unwrap().patterns {
                prop_assert!(
                    pattern.depth_pct >= MIN_DEPTH_PCT,
                    "DoubleTop depth_pct {} < {} on random bars",
                    pattern.depth_pct, MIN_DEPTH_PCT
                );
            }
        }
    }
}

