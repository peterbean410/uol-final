use modelenv_proto::Bar;

/// Number of indicator scalars produced per time interval. Increment when
/// adding a new indicator and append the new scalar to
/// [`compute_interval_indicators`] in the documented order.
pub const INDICATORS_PER_INTERVAL: usize = 1;

/// Compute the per-interval indicator block for `bars` (oldest -> newest).
///
/// Layout:
///   [0] double-bottom pattern score in [0.0, 1.0]
pub fn compute_interval_indicators(bars: &[Bar]) -> [f64; INDICATORS_PER_INTERVAL] {
    [double_bottom_score(bars)]
}

const SWING_RADIUS: usize = 2;
const BOTTOM_SIMILARITY_TOL: f64 = 0.005;
const PEAK_RISE_TARGET: f64 = 0.01;

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

    #[test]
    fn compute_interval_indicators_returns_fixed_length() {
        let result = compute_interval_indicators(&[]);
        assert_eq!(result.len(), INDICATORS_PER_INTERVAL);
        assert_eq!(result[0], 0.0);
    }
}
