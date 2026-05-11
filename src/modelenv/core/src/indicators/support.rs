//! Support indicators: Fibonacci retracements
//!
//! These indicators identify potential support and resistance levels.

use super::IndicatorError;

/// Fibonacci retracement ratios used for computing support/resistance levels.
const FIBO_RATIOS: [f64; 7] = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0];

/// Output struct containing all seven Fibonacci retracement levels.
///
/// Each field is a `Vec<f64>` aligned index-for-index with the input slices.
/// The levels are computed from rolling high/low values over the specified window.
#[derive(Debug, Clone)]
pub struct FibonacciRetracementsOutput {
    /// 0.0% retracement level (rolling low)
    pub fr_000: Vec<f64>,
    /// 23.6% retracement level
    pub fr_236: Vec<f64>,
    /// 38.2% retracement level
    pub fr_382: Vec<f64>,
    /// 50.0% retracement level
    pub fr_500: Vec<f64>,
    /// 61.8% retracement level
    pub fr_618: Vec<f64>,
    /// 78.6% retracement level
    pub fr_786: Vec<f64>,
    /// 100.0% retracement level (rolling high)
    pub fr_1000: Vec<f64>,
}

/// Compute rolling Fibonacci retracement levels.
///
/// Uses `min_periods=1` behavior so all indices are non-NaN when inputs are finite.
/// The function computes rolling high and low values over the specified window,
/// then applies Fibonacci ratios to calculate retracement levels.
///
/// # Arguments
///
/// * `high` - Slice of high prices
/// * `low` - Slice of low prices
/// * `window` - Rolling window size for computing high/low
///
/// # Returns
///
/// * `Ok(FibonacciRetracementsOutput)` - Seven vectors of retracement levels
/// * `Err(IndicatorError::LengthMismatch)` - If `high.len() != low.len()`
/// * `Err(IndicatorError::InvalidPeriod)` - If `window == 0`
///
/// # Invariants
///
/// For every index `i` where `roll_high[i] >= roll_low[i]`:
/// `fr_000[i] <= fr_236[i] <= fr_382[i] <= fr_500[i] <= fr_618[i] <= fr_786[i] <= fr_1000[i]`
/// within `1e-12` absolute slack per inequality.
///
/// # Example
///
/// ```
/// use modelenv_core::indicators::support::fibonacci_retracements;
///
/// let high = vec![105.0, 110.0, 108.0, 112.0, 107.0];
/// let low = vec![100.0, 102.0, 101.0, 105.0, 103.0];
/// let result = fibonacci_retracements(&high, &low, 3).unwrap();
/// assert_eq!(result.fr_000.len(), 5);
/// ```
pub fn fibonacci_retracements(
    high: &[f64],
    low: &[f64],
    window: usize,
) -> Result<FibonacciRetracementsOutput, IndicatorError> {
    // Validate window parameter
    if window == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "window",
            value: 0,
            reason: "window must be >= 1",
        });
    }

    // Validate input lengths match
    if high.len() != low.len() {
        return Err(IndicatorError::LengthMismatch {
            expected: high.len(),
            actual: low.len(),
            param_name: "low",
        });
    }

    let n = high.len();

    // Handle empty input case
    if n == 0 {
        return Ok(FibonacciRetracementsOutput {
            fr_000: Vec::new(),
            fr_236: Vec::new(),
            fr_382: Vec::new(),
            fr_500: Vec::new(),
            fr_618: Vec::new(),
            fr_786: Vec::new(),
            fr_1000: Vec::new(),
        });
    }

    // Pre-allocate output vectors
    let mut fr_000 = Vec::with_capacity(n);
    let mut fr_236 = Vec::with_capacity(n);
    let mut fr_382 = Vec::with_capacity(n);
    let mut fr_500 = Vec::with_capacity(n);
    let mut fr_618 = Vec::with_capacity(n);
    let mut fr_786 = Vec::with_capacity(n);
    let mut fr_1000 = Vec::with_capacity(n);

    // Compute rolling high/low with min_periods=1 behavior
    // This means we use whatever data is available from the start
    for i in 0..n {
        // Calculate the start of the window (saturating_sub handles the case where i < window - 1)
        let start = i.saturating_sub(window - 1);

        // Find rolling max of high and rolling min of low in the window [start, i]
        let mut roll_high = f64::NEG_INFINITY;
        let mut roll_low = f64::INFINITY;

        for j in start..=i {
            let h = high[j];
            let l = low[j];

            // Handle NaN propagation: if any value in window is NaN, result is NaN
            if h.is_nan() || l.is_nan() {
                roll_high = f64::NAN;
                roll_low = f64::NAN;
                break;
            }

            if h > roll_high {
                roll_high = h;
            }
            if l < roll_low {
                roll_low = l;
            }
        }

        // Compute the range
        let range = roll_high - roll_low;

        // Apply Fibonacci ratios: level = roll_low + range * ratio
        // fr_000 = roll_low + range * 0.0 = roll_low
        // fr_1000 = roll_low + range * 1.0 = roll_high
        fr_000.push(roll_low + range * FIBO_RATIOS[0]);
        fr_236.push(roll_low + range * FIBO_RATIOS[1]);
        fr_382.push(roll_low + range * FIBO_RATIOS[2]);
        fr_500.push(roll_low + range * FIBO_RATIOS[3]);
        fr_618.push(roll_low + range * FIBO_RATIOS[4]);
        fr_786.push(roll_low + range * FIBO_RATIOS[5]);
        fr_1000.push(roll_low + range * FIBO_RATIOS[6]);
    }

    Ok(FibonacciRetracementsOutput {
        fr_000,
        fr_236,
        fr_382,
        fr_500,
        fr_618,
        fr_786,
        fr_1000,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_empty_input() {
        let result = fibonacci_retracements(&[], &[], 5).unwrap();
        assert!(result.fr_000.is_empty());
        assert!(result.fr_236.is_empty());
        assert!(result.fr_382.is_empty());
        assert!(result.fr_500.is_empty());
        assert!(result.fr_618.is_empty());
        assert!(result.fr_786.is_empty());
        assert!(result.fr_1000.is_empty());
    }

    #[test]
    fn test_fibonacci_window_zero_error() {
        let high = vec![100.0, 110.0];
        let low = vec![90.0, 95.0];
        let result = fibonacci_retracements(&high, &low, 0);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { param_name: "window", value: 0, .. })
        ));
    }

    #[test]
    fn test_fibonacci_length_mismatch_error() {
        let high = vec![100.0, 110.0, 105.0];
        let low = vec![90.0, 95.0];
        let result = fibonacci_retracements(&high, &low, 3);
        assert!(matches!(
            result,
            Err(IndicatorError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn test_fibonacci_output_length() {
        let high = vec![105.0, 110.0, 108.0, 112.0, 107.0];
        let low = vec![100.0, 102.0, 101.0, 105.0, 103.0];
        let result = fibonacci_retracements(&high, &low, 3).unwrap();

        assert_eq!(result.fr_000.len(), 5);
        assert_eq!(result.fr_236.len(), 5);
        assert_eq!(result.fr_382.len(), 5);
        assert_eq!(result.fr_500.len(), 5);
        assert_eq!(result.fr_618.len(), 5);
        assert_eq!(result.fr_786.len(), 5);
        assert_eq!(result.fr_1000.len(), 5);
    }

    #[test]
    fn test_fibonacci_ordering_invariant() {
        let high = vec![105.0, 110.0, 108.0, 112.0, 107.0, 115.0, 109.0];
        let low = vec![100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 104.0];
        let result = fibonacci_retracements(&high, &low, 3).unwrap();

        const SLACK: f64 = 1e-12;

        for i in 0..high.len() {
            // Verify ordering: fr_000 <= fr_236 <= fr_382 <= fr_500 <= fr_618 <= fr_786 <= fr_1000
            assert!(
                result.fr_000[i] <= result.fr_236[i] + SLACK,
                "fr_000[{}] > fr_236[{}]: {} > {}",
                i, i, result.fr_000[i], result.fr_236[i]
            );
            assert!(
                result.fr_236[i] <= result.fr_382[i] + SLACK,
                "fr_236[{}] > fr_382[{}]: {} > {}",
                i, i, result.fr_236[i], result.fr_382[i]
            );
            assert!(
                result.fr_382[i] <= result.fr_500[i] + SLACK,
                "fr_382[{}] > fr_500[{}]: {} > {}",
                i, i, result.fr_382[i], result.fr_500[i]
            );
            assert!(
                result.fr_500[i] <= result.fr_618[i] + SLACK,
                "fr_500[{}] > fr_618[{}]: {} > {}",
                i, i, result.fr_500[i], result.fr_618[i]
            );
            assert!(
                result.fr_618[i] <= result.fr_786[i] + SLACK,
                "fr_618[{}] > fr_786[{}]: {} > {}",
                i, i, result.fr_618[i], result.fr_786[i]
            );
            assert!(
                result.fr_786[i] <= result.fr_1000[i] + SLACK,
                "fr_786[{}] > fr_1000[{}]: {} > {}",
                i, i, result.fr_786[i], result.fr_1000[i]
            );
        }
    }

    #[test]
    fn test_fibonacci_min_periods_1_behavior() {
        // With min_periods=1, even the first element should have valid values
        let high = vec![110.0, 115.0, 112.0];
        let low = vec![100.0, 105.0, 103.0];
        let result = fibonacci_retracements(&high, &low, 5).unwrap();

        // First element: window is just [0..=0], so roll_high=110, roll_low=100
        assert!(!result.fr_000[0].is_nan());
        assert!(!result.fr_1000[0].is_nan());

        // fr_000 should be roll_low, fr_1000 should be roll_high
        assert!((result.fr_000[0] - 100.0).abs() < 1e-9);
        assert!((result.fr_1000[0] - 110.0).abs() < 1e-9);
    }

    #[test]
    fn test_fibonacci_known_values() {
        // Simple case: constant range
        let high = vec![110.0, 110.0, 110.0];
        let low = vec![100.0, 100.0, 100.0];
        let result = fibonacci_retracements(&high, &low, 3).unwrap();

        // Range is 10.0 for all indices
        // fr_000 = 100 + 10 * 0.0 = 100
        // fr_236 = 100 + 10 * 0.236 = 102.36
        // fr_382 = 100 + 10 * 0.382 = 103.82
        // fr_500 = 100 + 10 * 0.5 = 105
        // fr_618 = 100 + 10 * 0.618 = 106.18
        // fr_786 = 100 + 10 * 0.786 = 107.86
        // fr_1000 = 100 + 10 * 1.0 = 110

        for i in 0..3 {
            assert!((result.fr_000[i] - 100.0).abs() < 1e-9);
            assert!((result.fr_236[i] - 102.36).abs() < 1e-9);
            assert!((result.fr_382[i] - 103.82).abs() < 1e-9);
            assert!((result.fr_500[i] - 105.0).abs() < 1e-9);
            assert!((result.fr_618[i] - 106.18).abs() < 1e-9);
            assert!((result.fr_786[i] - 107.86).abs() < 1e-9);
            assert!((result.fr_1000[i] - 110.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_fibonacci_nan_propagation() {
        let high = vec![110.0, f64::NAN, 112.0];
        let low = vec![100.0, 102.0, 101.0];
        let result = fibonacci_retracements(&high, &low, 2).unwrap();

        // Index 0: window [0..=0], no NaN
        assert!(!result.fr_000[0].is_nan());

        // Index 1: window [0..=1], contains NaN at high[1]
        assert!(result.fr_000[1].is_nan());
        assert!(result.fr_1000[1].is_nan());

        // Index 2: window [1..=2], contains NaN at high[1]
        assert!(result.fr_000[2].is_nan());
        assert!(result.fr_1000[2].is_nan());
    }

    #[test]
    fn test_fibonacci_single_element() {
        let high = vec![110.0];
        let low = vec![100.0];
        let result = fibonacci_retracements(&high, &low, 5).unwrap();

        assert_eq!(result.fr_000.len(), 1);
        assert!((result.fr_000[0] - 100.0).abs() < 1e-9);
        assert!((result.fr_1000[0] - 110.0).abs() < 1e-9);
    }

    #[test]
    fn test_fibonacci_window_larger_than_input() {
        // Window of 10, but only 3 elements - should still work with min_periods=1
        let high = vec![110.0, 115.0, 112.0];
        let low = vec![100.0, 105.0, 103.0];
        let result = fibonacci_retracements(&high, &low, 10).unwrap();

        assert_eq!(result.fr_000.len(), 3);

        // All values should be non-NaN
        for i in 0..3 {
            assert!(!result.fr_000[i].is_nan());
            assert!(!result.fr_1000[i].is_nan());
        }

        // Last element should use all available data [0..=2]
        // roll_high = max(110, 115, 112) = 115
        // roll_low = min(100, 105, 103) = 100
        assert!((result.fr_000[2] - 100.0).abs() < 1e-9);
        assert!((result.fr_1000[2] - 115.0).abs() < 1e-9);
    }
}
