//! Momentum indicators: RSI, CCI
//!
//! These indicators measure the rate of price change and are used to identify
//! overbought/oversold conditions.

use crate::indicators::IndicatorError;
use talib_rs::momentum::cci as talib_cci;
use talib_rs::momentum::rsi as talib_rsi;

/// Compute Relative Strength Index (RSI).
///
/// Returns a vector of length `close.len()` where indices `[0, period)` are NaN
/// and indices `[period, n)` contain RSI values in `[0.0, 100.0]`.
///
/// # Arguments
///
/// * `close` - Slice of closing prices, ordered oldest to newest
/// * `period` - The lookback period for RSI calculation
///
/// # Returns
///
/// A `Vec<f64>` aligned index-for-index with `close`. Leading `period` elements
/// are `f64::NAN`, and subsequent elements are RSI values in `[0.0, 100.0]`.
///
/// # Edge Cases
///
/// - If `period == 0`: triggers `debug_assert!` in debug builds, returns all-NaN in release
/// - If `close.len() <= period`: returns all-NaN vector
/// - If `close` is empty: returns empty vector
///
/// # Example
///
/// ```
/// use modelenv_core::indicators::momentum::rsi;
///
/// let close = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5, 44.0, 44.5];
/// let result = rsi(&close, 5);
/// assert_eq!(result.len(), close.len());
/// assert!(result[4].is_nan()); // First 5 elements are NaN
/// assert!(!result[5].is_nan()); // RSI computed from index 5 onwards
/// ```
pub fn rsi(close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();

    // Handle period == 0: debug_assert in debug builds, return all-NaN in release
    debug_assert!(period > 0, "RSI period must be greater than 0");
    if period == 0 {
        return vec![f64::NAN; n];
    }

    // Handle empty input
    if n == 0 {
        return Vec::new();
    }

    // Handle insufficient data: close.len() <= period
    if n <= period {
        return vec![f64::NAN; n];
    }

    // Call talib-rs RSI function
    // talib-rs returns a vector where the first `period` elements are NaN
    // and the rest are RSI values
    match talib_rsi(close, period) {
        Ok(result) => {
            // talib-rs should return a vector of the same length as input
            // with leading NaNs for the warm-up period
            if result.len() == n {
                result
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
                output
            }
        }
        Err(_) => {
            // On any error from talib-rs, return all-NaN
            // This handles edge cases like insufficient data
            vec![f64::NAN; n]
        }
    }
}

/// Compute Commodity Channel Index (CCI).
///
/// Returns a vector of length `close.len()` where indices `[0, period - 1)` are NaN
/// and indices `[period - 1, n)` contain CCI values.
///
/// # Arguments
///
/// * `high` - Slice of high prices, ordered oldest to newest
/// * `low` - Slice of low prices, ordered oldest to newest
/// * `close` - Slice of closing prices, ordered oldest to newest
/// * `period` - The lookback period for CCI calculation
///
/// # Returns
///
/// A `Result<Vec<f64>, IndicatorError>` aligned index-for-index with the input slices.
/// Leading `period - 1` elements are `f64::NAN`, and subsequent elements are CCI values.
///
/// # Errors
///
/// - `IndicatorError::LengthMismatch` if `high`, `low`, and `close` have different lengths
/// - `IndicatorError::InvalidPeriod` if `period == 0`
///
/// # Edge Cases
///
/// - If `close.len() < period`: returns `Ok` with all-NaN vector
/// - If all inputs are empty: returns `Ok` with empty vector
///
/// # Example
///
/// ```
/// use modelenv_core::indicators::momentum::cci;
///
/// let high = vec![45.0, 45.5, 46.0, 45.5, 46.0, 45.5, 45.0, 44.5, 45.0, 45.5];
/// let low = vec![44.0, 44.5, 45.0, 44.5, 45.0, 44.5, 44.0, 43.5, 44.0, 44.5];
/// let close = vec![44.5, 45.0, 45.5, 45.0, 45.5, 45.0, 44.5, 44.0, 44.5, 45.0];
/// let result = cci(&high, &low, &close, 5).unwrap();
/// assert_eq!(result.len(), close.len());
/// assert!(result[3].is_nan()); // First period-1 elements are NaN
/// assert!(!result[4].is_nan()); // CCI computed from index period-1 onwards
/// ```
pub fn cci(
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

    // Handle insufficient data: close.len() < period
    // Return all-NaN vector per requirements
    if n < period {
        return Ok(vec![f64::NAN; n]);
    }

    // Call talib-rs CCI function
    // Note: talib-rs requires period >= 2, so we handle period == 1 specially
    if period == 1 {
        // For period == 1, CCI is always 0 (no deviation from single-point mean)
        // The typical price equals the mean, so (TP - mean) = 0
        return Ok(vec![0.0; n]);
    }

    match talib_cci(high, low, close, period) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rsi_returns_correct_length() {
        let close = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5, 44.0, 44.5];
        let result = rsi(&close, 5);
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn rsi_leading_nans() {
        let close = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5, 44.0, 44.5];
        let result = rsi(&close, 5);
        // First `period` elements should be NaN
        for i in 0..5 {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }
    }

    #[test]
    fn rsi_values_in_range() {
        let close = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5, 44.0, 44.5];
        let result = rsi(&close, 5);
        // RSI values should be in [0.0, 100.0]
        for i in 5..result.len() {
            assert!(
                !result[i].is_nan(),
                "Expected non-NaN RSI at index {}",
                i
            );
            assert!(
                result[i] >= 0.0 && result[i] <= 100.0,
                "RSI at index {} is {} which is out of range [0, 100]",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn rsi_empty_input() {
        let close: Vec<f64> = vec![];
        let result = rsi(&close, 14);
        assert!(result.is_empty());
    }

    #[test]
    fn rsi_insufficient_data() {
        // close.len() <= period should return all-NaN
        let close = vec![44.0, 44.25, 44.5];
        let result = rsi(&close, 5);
        assert_eq!(result.len(), 3);
        for val in &result {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn rsi_period_equals_length() {
        // close.len() == period should return all-NaN
        let close = vec![44.0, 44.25, 44.5, 43.75, 44.5];
        let result = rsi(&close, 5);
        assert_eq!(result.len(), 5);
        for val in &result {
            assert!(val.is_nan());
        }
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn rsi_period_zero_returns_all_nan() {
        let close = vec![44.0, 44.25, 44.5];
        let result = rsi(&close, 0);
        assert_eq!(result.len(), 3);
        for val in &result {
            assert!(val.is_nan());
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "RSI period must be greater than 0")]
    fn rsi_period_zero_panics_in_debug() {
        let close = vec![44.0, 44.25, 44.5];
        let _ = rsi(&close, 0);
    }

    // CCI Tests

    #[test]
    fn cci_returns_correct_length() {
        let high = vec![45.0, 45.5, 46.0, 45.5, 46.0, 45.5, 45.0, 44.5, 45.0, 45.5];
        let low = vec![44.0, 44.5, 45.0, 44.5, 45.0, 44.5, 44.0, 43.5, 44.0, 44.5];
        let close = vec![44.5, 45.0, 45.5, 45.0, 45.5, 45.0, 44.5, 44.0, 44.5, 45.0];
        let result = cci(&high, &low, &close, 5).unwrap();
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn cci_leading_nans() {
        let high = vec![45.0, 45.5, 46.0, 45.5, 46.0, 45.5, 45.0, 44.5, 45.0, 45.5];
        let low = vec![44.0, 44.5, 45.0, 44.5, 45.0, 44.5, 44.0, 43.5, 44.0, 44.5];
        let close = vec![44.5, 45.0, 45.5, 45.0, 45.5, 45.0, 44.5, 44.0, 44.5, 45.0];
        let result = cci(&high, &low, &close, 5).unwrap();
        // First `period - 1` elements should be NaN
        for i in 0..4 {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }
        // From index period - 1 onwards, should have values
        assert!(!result[4].is_nan(), "Expected non-NaN CCI at index 4");
    }

    #[test]
    fn cci_empty_input() {
        let high: Vec<f64> = vec![];
        let low: Vec<f64> = vec![];
        let close: Vec<f64> = vec![];
        let result = cci(&high, &low, &close, 14).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn cci_insufficient_data() {
        // close.len() < period should return all-NaN
        let high = vec![45.0, 45.5, 46.0];
        let low = vec![44.0, 44.5, 45.0];
        let close = vec![44.5, 45.0, 45.5];
        let result = cci(&high, &low, &close, 5).unwrap();
        assert_eq!(result.len(), 3);
        for val in &result {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn cci_period_zero_returns_error() {
        let high = vec![45.0, 45.5, 46.0];
        let low = vec![44.0, 44.5, 45.0];
        let close = vec![44.5, 45.0, 45.5];
        let result = cci(&high, &low, &close, 0);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod { value: 0, .. })
        ));
    }

    #[test]
    fn cci_length_mismatch_high() {
        let high = vec![45.0, 45.5]; // Different length
        let low = vec![44.0, 44.5, 45.0];
        let close = vec![44.5, 45.0, 45.5];
        let result = cci(&high, &low, &close, 2);
        assert!(matches!(
            result,
            Err(IndicatorError::LengthMismatch {
                param_name: "high",
                ..
            })
        ));
    }

    #[test]
    fn cci_length_mismatch_low() {
        let high = vec![45.0, 45.5, 46.0];
        let low = vec![44.0, 44.5]; // Different length
        let close = vec![44.5, 45.0, 45.5];
        let result = cci(&high, &low, &close, 2);
        assert!(matches!(
            result,
            Err(IndicatorError::LengthMismatch {
                param_name: "low",
                ..
            })
        ));
    }

    #[test]
    fn cci_period_one() {
        // Period 1 is a special case - CCI should be 0 (no deviation from single-point mean)
        let high = vec![45.0, 45.5, 46.0, 45.5, 46.0];
        let low = vec![44.0, 44.5, 45.0, 44.5, 45.0];
        let close = vec![44.5, 45.0, 45.5, 45.0, 45.5];
        let result = cci(&high, &low, &close, 1).unwrap();
        assert_eq!(result.len(), 5);
        for val in &result {
            assert_eq!(*val, 0.0);
        }
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // **Property 2: RSI Bounded Range**
    //
    // For any close price slice with no NaN values and `period >= 1`,
    // all non-NaN RSI output values SHALL be in the closed range `[0.0, 100.0]`.
    //
    // **Validates: Requirements 2.3**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn rsi_bounded_range(
            // Generate a vector of positive close prices (no NaN values)
            // Using positive values to simulate realistic price data
            close in prop::collection::vec(1.0f64..10000.0f64, 1..500),
            // Period must be >= 1
            period in 1usize..50
        ) {
            let result = rsi(&close, period);

            // Verify output length matches input length
            prop_assert_eq!(result.len(), close.len());

            // Verify all non-NaN RSI values are in [0.0, 100.0]
            for (i, &val) in result.iter().enumerate() {
                if !val.is_nan() {
                    prop_assert!(
                        val >= 0.0 && val <= 100.0,
                        "RSI at index {} is {} which is outside [0.0, 100.0]. \
                         Input length: {}, period: {}",
                        i, val, close.len(), period
                    );
                }
            }
        }
    }

    // **Property 14: Length Mismatch Error**
    //
    // For any indicator function that accepts multiple input slices, calling it
    // with slices of differing lengths SHALL return `Err(IndicatorError::LengthMismatch)`
    // without panicking.
    //
    // **Validates: Requirements 3.2, 4.5, 7.10, 9.8, 16.2**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Test CCI returns LengthMismatch when high slice has different length than close
        #[test]
        fn cci_length_mismatch_high_slice(
            // Generate close slice with arbitrary length
            close in prop::collection::vec(1.0f64..10000.0f64, 1..100),
            // Generate high slice with a DIFFERENT length (either shorter or longer)
            high_len_diff in 1usize..50,
            high_shorter in proptest::bool::ANY,
            // Period must be >= 1
            period in 1usize..20
        ) {
            let close_len = close.len();
            // Ensure high has a different length
            let high_len = if high_shorter {
                close_len.saturating_sub(high_len_diff).max(0)
            } else {
                close_len + high_len_diff
            };

            // Skip if lengths happen to be equal
            prop_assume!(high_len != close_len);

            let high: Vec<f64> = (0..high_len).map(|i| 100.0 + (i as f64)).collect();
            let low: Vec<f64> = (0..close_len).map(|i| 90.0 + (i as f64)).collect();

            let result = cci(&high, &low, &close, period);

            // Verify LengthMismatch is returned without panicking
            prop_assert!(
                matches!(
                    result,
                    Err(IndicatorError::LengthMismatch {
                        param_name: "high",
                        expected,
                        actual,
                    }) if expected == close_len && actual == high_len
                ),
                "Expected LengthMismatch for high slice, got {:?}. \
                 close_len: {}, high_len: {}",
                result, close_len, high_len
            );
        }

        /// Test CCI returns LengthMismatch when low slice has different length than close
        #[test]
        fn cci_length_mismatch_low_slice(
            // Generate close slice with arbitrary length
            close in prop::collection::vec(1.0f64..10000.0f64, 1..100),
            // Generate low slice with a DIFFERENT length (either shorter or longer)
            low_len_diff in 1usize..50,
            low_shorter in proptest::bool::ANY,
            // Period must be >= 1
            period in 1usize..20
        ) {
            let close_len = close.len();
            // Ensure low has a different length
            let low_len = if low_shorter {
                close_len.saturating_sub(low_len_diff).max(0)
            } else {
                close_len + low_len_diff
            };

            // Skip if lengths happen to be equal
            prop_assume!(low_len != close_len);

            // High has same length as close (so it passes validation)
            let high: Vec<f64> = (0..close_len).map(|i| 100.0 + (i as f64)).collect();
            let low: Vec<f64> = (0..low_len).map(|i| 90.0 + (i as f64)).collect();

            let result = cci(&high, &low, &close, period);

            // Verify LengthMismatch is returned without panicking
            prop_assert!(
                matches!(
                    result,
                    Err(IndicatorError::LengthMismatch {
                        param_name: "low",
                        expected,
                        actual,
                    }) if expected == close_len && actual == low_len
                ),
                "Expected LengthMismatch for low slice, got {:?}. \
                 close_len: {}, low_len: {}",
                result, close_len, low_len
            );
        }

        /// Test CCI returns LengthMismatch when both high and low have different lengths
        /// (high is checked first, so we expect high mismatch error)
        #[test]
        fn cci_length_mismatch_both_slices(
            // Generate close slice with arbitrary length
            close in prop::collection::vec(1.0f64..10000.0f64, 2..100),
            // Generate different lengths for high and low
            high_len_diff in 1usize..20,
            low_len_diff in 1usize..20,
            // Period must be >= 1
            period in 1usize..20
        ) {
            let close_len = close.len();
            // Ensure both high and low have different lengths than close
            let high_len = close_len.saturating_sub(high_len_diff).max(0);
            let low_len = close_len + low_len_diff;

            // Skip if high length happens to equal close length
            prop_assume!(high_len != close_len);

            let high: Vec<f64> = (0..high_len).map(|i| 100.0 + (i as f64)).collect();
            let low: Vec<f64> = (0..low_len).map(|i| 90.0 + (i as f64)).collect();

            let result = cci(&high, &low, &close, period);

            // High is checked first, so we expect high mismatch error
            prop_assert!(
                matches!(
                    result,
                    Err(IndicatorError::LengthMismatch {
                        param_name: "high",
                        expected,
                        actual,
                    }) if expected == close_len && actual == high_len
                ),
                "Expected LengthMismatch for high slice (checked first), got {:?}. \
                 close_len: {}, high_len: {}, low_len: {}",
                result, close_len, high_len, low_len
            );
        }
    }

    // **Property 15: Invalid Period Error**
    //
    // For any indicator function with a `period` parameter, calling it with
    // `period == 0` SHALL return `Err(IndicatorError::InvalidPeriod)` without panicking.
    //
    // **Validates: Requirements 3.6, 5.5, 6.7, 7.9, 8.6, 9.7, 10.11, 11.11, 16.3**
    //
    // This test covers all indicator functions that accept a period parameter:
    // - CCI (momentum.rs) - Requirement 3.6
    // - MACD (trend.rs) - Requirement 5.5 (fast, slow, signal params)
    // - moving_average (trend.rs) - Requirement 6.7
    // - bollinger_bands (volatility.rs) - Requirement 8.6
    //
    // Note: The following indicators are not yet implemented and will be tested
    // when they are added:
    // - ichimoku (trend.rs) - Requirement 7.9 (tenkan, kijun, senkou_b_period params)
    // - fibonacci_retracements (support.rs) - Requirement 9.7
    // - detect_double_bottoms (patterns.rs) - Requirement 10.11
    // - detect_double_tops (patterns.rs) - Requirement 11.11
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Test CCI returns InvalidPeriod when period == 0
        #[test]
        fn cci_invalid_period_zero(
            // Generate valid input slices of the same length
            close in prop::collection::vec(1.0f64..10000.0f64, 1..100)
        ) {
            let n = close.len();
            let high: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64)).collect();
            let low: Vec<f64> = (0..n).map(|i| 90.0 + (i as f64)).collect();

            let result = cci(&high, &low, &close, 0);

            // Verify InvalidPeriod is returned without panicking
            prop_assert!(
                matches!(
                    result,
                    Err(IndicatorError::InvalidPeriod {
                        param_name: "period",
                        value: 0,
                        ..
                    })
                ),
                "Expected InvalidPeriod for period == 0, got {:?}",
                result
            );
        }
    }
}

#[cfg(test)]
mod invalid_period_tests {
    use super::*;
    use crate::indicators::trend::{adx, macd, moving_average, MovingAverageKind};
    use crate::indicators::volatility::bollinger_bands;
    use proptest::prelude::*;

    // **Property 15: Invalid Period Error**
    //
    // For any indicator function with a `period` parameter, calling it with
    // `period == 0` SHALL return `Err(IndicatorError::InvalidPeriod)` without panicking.
    //
    // **Validates: Requirements 3.6, 5.5, 6.7, 7.9, 8.6, 9.7, 10.11, 11.11, 16.3**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Test ADX returns InvalidPeriod when period == 0
        /// **Validates: Requirement 4.5 (period == 0 case)**
        #[test]
        fn adx_invalid_period_zero(
            // Generate valid input slices of the same length
            close in prop::collection::vec(1.0f64..10000.0f64, 1..100)
        ) {
            let n = close.len();
            let high: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64)).collect();
            let low: Vec<f64> = (0..n).map(|i| 90.0 + (i as f64)).collect();

            let result = adx(&high, &low, &close, 0);

            // Verify InvalidPeriod is returned without panicking
            prop_assert!(
                matches!(
                    result,
                    Err(IndicatorError::InvalidPeriod {
                        param_name: "period",
                        value: 0,
                        ..
                    })
                ),
                "Expected InvalidPeriod for period == 0, got {:?}",
                result
            );
        }

        /// Test MACD returns InvalidPeriod when fast == 0
        /// **Validates: Requirement 5.5**
        #[test]
        fn macd_invalid_period_fast_zero(
            // Generate valid close prices
            close in prop::collection::vec(1.0f64..10000.0f64, 1..100),
            // Valid slow and signal periods
            slow in 2usize..50,
            signal in 1usize..20
        ) {
            let result = macd(&close, 0, slow, signal);

            // Verify InvalidPeriod is returned without panicking
            prop_assert!(
                matches!(
                    result,
                    Err(IndicatorError::InvalidPeriod {
                        param_name: "fast",
                        value: 0,
                        ..
                    })
                ),
                "Expected InvalidPeriod for fast == 0, got {:?}",
                result
            );
        }

        /// Test MACD returns InvalidPeriod when slow == 0
        /// **Validates: Requirement 5.5**
        #[test]
        fn macd_invalid_period_slow_zero(
            // Generate valid close prices
            close in prop::collection::vec(1.0f64..10000.0f64, 1..100),
            // Valid fast and signal periods
            fast in 1usize..20,
            signal in 1usize..20
        ) {
            let result = macd(&close, fast, 0, signal);

            // Verify InvalidPeriod is returned without panicking
            prop_assert!(
                matches!(
                    result,
                    Err(IndicatorError::InvalidPeriod {
                        param_name: "slow",
                        value: 0,
                        ..
                    })
                ),
                "Expected InvalidPeriod for slow == 0, got {:?}",
                result
            );
        }

        /// Test MACD returns InvalidPeriod when signal == 0
        /// **Validates: Requirement 5.5**
        #[test]
        fn macd_invalid_period_signal_zero(
            // Generate valid close prices
            close in prop::collection::vec(1.0f64..10000.0f64, 1..100),
            // Valid fast and slow periods (fast < slow)
            fast in 1usize..20,
            slow in 21usize..50
        ) {
            let result = macd(&close, fast, slow, 0);

            // Verify InvalidPeriod is returned without panicking
            prop_assert!(
                matches!(
                    result,
                    Err(IndicatorError::InvalidPeriod {
                        param_name: "signal",
                        value: 0,
                        ..
                    })
                ),
                "Expected InvalidPeriod for signal == 0, got {:?}",
                result
            );
        }

        /// Test moving_average returns InvalidPeriod when period == 0 for all MA kinds
        /// **Validates: Requirement 6.7**
        #[test]
        fn moving_average_invalid_period_zero(
            // Generate valid close prices
            close in prop::collection::vec(1.0f64..10000.0f64, 1..100),
            // Test all MA kinds
            kind_idx in 0usize..7
        ) {
            let kinds = [
                MovingAverageKind::Sma,
                MovingAverageKind::Ema,
                MovingAverageKind::Wma,
                MovingAverageKind::Dema,
                MovingAverageKind::Tema,
                MovingAverageKind::Kama,
                MovingAverageKind::Trima,
            ];
            let kind = kinds[kind_idx];

            let result = moving_average(&close, kind, 0);

            // Verify InvalidPeriod is returned without panicking
            prop_assert!(
                matches!(
                    result,
                    Err(IndicatorError::InvalidPeriod {
                        param_name: "period",
                        value: 0,
                        ..
                    })
                ),
                "Expected InvalidPeriod for period == 0 with {:?}, got {:?}",
                kind, result
            );
        }

        /// Test bollinger_bands returns InvalidPeriod when period == 0
        /// **Validates: Requirement 8.6**
        #[test]
        fn bollinger_bands_invalid_period_zero(
            // Generate valid close prices
            close in prop::collection::vec(1.0f64..10000.0f64, 1..100),
            // Valid nbdev
            nbdev in 0.0f64..5.0f64
        ) {
            let result = bollinger_bands(&close, 0, nbdev);

            // Verify InvalidPeriod is returned without panicking
            prop_assert!(
                matches!(
                    result,
                    Err(IndicatorError::InvalidPeriod {
                        param_name: "period",
                        value: 0,
                        ..
                    })
                ),
                "Expected InvalidPeriod for period == 0, got {:?}",
                result
            );
        }
    }
}
