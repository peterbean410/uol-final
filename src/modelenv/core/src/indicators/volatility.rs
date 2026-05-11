//! Volatility indicators: Bollinger Bands
//!
//! These indicators measure the degree of price variation over time.

use crate::indicators::IndicatorError;
use talib_rs::ma_type::MaType;
use talib_rs::overlap::bbands as talib_bbands;

/// Output structure for Bollinger Bands indicator.
///
/// Contains three vectors aligned index-for-index with the input:
/// - `upper`: The upper band (middle + nbdev * stddev)
/// - `middle`: The middle band (SMA of close prices)
/// - `lower`: The lower band (middle - nbdev * stddev)
///
/// # Invariants
///
/// For all finite indices `i`: `lower[i] <= middle[i] <= upper[i]`
#[derive(Debug, Clone, PartialEq)]
pub struct BollingerBandsOutput {
    /// Upper band: middle + nbdev * standard deviation
    pub upper: Vec<f64>,
    /// Middle band: Simple Moving Average of close prices
    pub middle: Vec<f64>,
    /// Lower band: middle - nbdev * standard deviation
    pub lower: Vec<f64>,
}

/// Compute Bollinger Bands.
///
/// Returns a `BollingerBandsOutput` struct containing three vectors of length `close.len()`:
/// - `upper`: The upper band (middle + nbdev * stddev)
/// - `middle`: The middle band (SMA of close prices)
/// - `lower`: The lower band (middle - nbdev * stddev)
///
/// Leading elements up to index `period - 1` are NaN.
///
/// # Arguments
///
/// * `close` - Slice of closing prices, ordered oldest to newest
/// * `period` - The lookback period for the moving average and standard deviation
/// * `nbdev` - The number of standard deviations for the upper and lower bands
///
/// # Returns
///
/// A `Result<BollingerBandsOutput, IndicatorError>` where each vector is aligned
/// index-for-index with `close`. Leading `period - 1` elements are `f64::NAN`.
///
/// # Errors
///
/// - `IndicatorError::InvalidPeriod` if `period == 0`
/// - `IndicatorError::InvalidPeriod` if `nbdev.is_nan() || nbdev < 0.0 || !nbdev.is_finite()`
///
/// # Edge Cases
///
/// - If `close` is empty: returns `Ok` with empty vectors
/// - If `close.len() < period`: returns `Ok` with all-NaN vectors
///
/// # Invariants
///
/// For all finite indices `i`: `lower[i] <= middle[i] <= upper[i]`
///
/// # Example
///
/// ```
/// use modelenv_core::indicators::volatility::bollinger_bands;
///
/// let close: Vec<f64> = (1..=30).map(|x| x as f64).collect();
/// let result = bollinger_bands(&close, 20, 2.0).unwrap();
/// assert_eq!(result.upper.len(), close.len());
/// assert_eq!(result.middle.len(), close.len());
/// assert_eq!(result.lower.len(), close.len());
/// // First period - 1 = 19 elements are NaN
/// assert!(result.upper[18].is_nan());
/// assert!(!result.upper[19].is_nan());
/// ```
pub fn bollinger_bands(
    close: &[f64],
    period: usize,
    nbdev: f64,
) -> Result<BollingerBandsOutput, IndicatorError> {
    let n = close.len();

    // Check for invalid period
    if period == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "period",
            value: 0,
            reason: "period must be greater than 0",
        });
    }

    // Check for invalid nbdev: must be finite and non-negative
    // Per requirement 8.7: IF nbdev.is_nan() || nbdev < 0.0 || !nbdev.is_finite()
    if nbdev.is_nan() || nbdev < 0.0 || !nbdev.is_finite() {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "nbdev",
            value: 0, // Use 0 as placeholder since value is usize but nbdev is f64
            reason: "nbdev must be finite and non-negative",
        });
    }

    // Handle empty input - return empty vectors per requirement 8.2
    if n == 0 {
        return Ok(BollingerBandsOutput {
            upper: Vec::new(),
            middle: Vec::new(),
            lower: Vec::new(),
        });
    }

    // Handle insufficient data: close.len() < period
    // Return all-NaN vectors per requirements
    if n < period {
        return Ok(BollingerBandsOutput {
            upper: vec![f64::NAN; n],
            middle: vec![f64::NAN; n],
            lower: vec![f64::NAN; n],
        });
    }

    // Call talib-rs bbands function
    // talib-rs bbands signature: bbands(input, timeperiod, nbdevup, nbdevdn, matype)
    // Returns (upperband, middleband, lowerband)
    // We use SMA (MaType::Sma) as the moving average type to match TA-Lib BBANDS default
    match talib_bbands(close, period, nbdev, nbdev, MaType::Sma) {
        Ok((upper, middle, lower)) => {
            // talib-rs should return vectors of the same length as input
            // with leading NaNs for the warm-up period
            if upper.len() == n && middle.len() == n && lower.len() == n {
                Ok(BollingerBandsOutput {
                    upper,
                    middle,
                    lower,
                })
            } else {
                // If talib-rs returns different lengths, pad with NaNs
                // This shouldn't happen with talib-rs 0.1.2, but handle it defensively
                let mut upper_out = vec![f64::NAN; n];
                let mut middle_out = vec![f64::NAN; n];
                let mut lower_out = vec![f64::NAN; n];

                let start_idx = n.saturating_sub(upper.len());
                for (i, &val) in upper.iter().enumerate() {
                    if start_idx + i < n {
                        upper_out[start_idx + i] = val;
                    }
                }

                let start_idx = n.saturating_sub(middle.len());
                for (i, &val) in middle.iter().enumerate() {
                    if start_idx + i < n {
                        middle_out[start_idx + i] = val;
                    }
                }

                let start_idx = n.saturating_sub(lower.len());
                for (i, &val) in lower.iter().enumerate() {
                    if start_idx + i < n {
                        lower_out[start_idx + i] = val;
                    }
                }

                Ok(BollingerBandsOutput {
                    upper: upper_out,
                    middle: middle_out,
                    lower: lower_out,
                })
            }
        }
        Err(talib_rs::error::TaError::InsufficientData { .. }) => {
            // If talib-rs says insufficient data, return all-NaN
            Ok(BollingerBandsOutput {
                upper: vec![f64::NAN; n],
                middle: vec![f64::NAN; n],
                lower: vec![f64::NAN; n],
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
            Ok(BollingerBandsOutput {
                upper: vec![f64::NAN; n],
                middle: vec![f64::NAN; n],
                lower: vec![f64::NAN; n],
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bollinger_bands_returns_correct_length() {
        let close: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let result = bollinger_bands(&close, 20, 2.0).unwrap();
        assert_eq!(result.upper.len(), close.len());
        assert_eq!(result.middle.len(), close.len());
        assert_eq!(result.lower.len(), close.len());
    }

    #[test]
    fn bollinger_bands_leading_nans() {
        let close: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let result = bollinger_bands(&close, 20, 2.0).unwrap();
        // First period - 1 = 19 elements should be NaN
        let lookback = 20 - 1;
        for i in 0..lookback {
            assert!(
                result.upper[i].is_nan(),
                "Expected NaN upper at index {}",
                i
            );
            assert!(
                result.middle[i].is_nan(),
                "Expected NaN middle at index {}",
                i
            );
            assert!(
                result.lower[i].is_nan(),
                "Expected NaN lower at index {}",
                i
            );
        }
        // From index lookback onwards, should have values
        assert!(
            !result.upper[lookback].is_nan(),
            "Expected non-NaN upper at index {}",
            lookback
        );
        assert!(
            !result.middle[lookback].is_nan(),
            "Expected non-NaN middle at index {}",
            lookback
        );
        assert!(
            !result.lower[lookback].is_nan(),
            "Expected non-NaN lower at index {}",
            lookback
        );
    }

    #[test]
    fn bollinger_bands_ordering() {
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = bollinger_bands(&close, 20, 2.0).unwrap();
        // Verify lower <= middle <= upper for all finite indices
        for i in 0..result.upper.len() {
            if !result.upper[i].is_nan()
                && !result.middle[i].is_nan()
                && !result.lower[i].is_nan()
            {
                assert!(
                    result.lower[i] <= result.middle[i],
                    "Ordering violated at index {}: lower={} > middle={}",
                    i,
                    result.lower[i],
                    result.middle[i]
                );
                assert!(
                    result.middle[i] <= result.upper[i],
                    "Ordering violated at index {}: middle={} > upper={}",
                    i,
                    result.middle[i],
                    result.upper[i]
                );
            }
        }
    }

    #[test]
    fn bollinger_bands_empty_input() {
        let close: Vec<f64> = vec![];
        let result = bollinger_bands(&close, 20, 2.0).unwrap();
        assert!(result.upper.is_empty());
        assert!(result.middle.is_empty());
        assert!(result.lower.is_empty());
    }

    #[test]
    fn bollinger_bands_insufficient_data() {
        // close.len() < period should return all-NaN
        let close: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let result = bollinger_bands(&close, 20, 2.0).unwrap();
        assert_eq!(result.upper.len(), 10);
        assert_eq!(result.middle.len(), 10);
        assert_eq!(result.lower.len(), 10);
        for i in 0..10 {
            assert!(result.upper[i].is_nan());
            assert!(result.middle[i].is_nan());
            assert!(result.lower[i].is_nan());
        }
    }

    #[test]
    fn bollinger_bands_period_zero_returns_error() {
        let close: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let result = bollinger_bands(&close, 0, 2.0);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod {
                param_name: "period",
                value: 0,
                ..
            })
        ));
    }

    #[test]
    fn bollinger_bands_nbdev_nan_returns_error() {
        let close: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let result = bollinger_bands(&close, 20, f64::NAN);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod {
                param_name: "nbdev",
                ..
            })
        ));
    }

    #[test]
    fn bollinger_bands_nbdev_negative_returns_error() {
        let close: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let result = bollinger_bands(&close, 20, -1.0);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod {
                param_name: "nbdev",
                ..
            })
        ));
    }

    #[test]
    fn bollinger_bands_nbdev_infinity_returns_error() {
        let close: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let result = bollinger_bands(&close, 20, f64::INFINITY);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod {
                param_name: "nbdev",
                ..
            })
        ));
    }

    #[test]
    fn bollinger_bands_nbdev_neg_infinity_returns_error() {
        let close: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let result = bollinger_bands(&close, 20, f64::NEG_INFINITY);
        assert!(matches!(
            result,
            Err(IndicatorError::InvalidPeriod {
                param_name: "nbdev",
                ..
            })
        ));
    }

    #[test]
    fn bollinger_bands_nbdev_zero_is_valid() {
        // nbdev = 0 is valid (bands collapse to middle line)
        let close: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let result = bollinger_bands(&close, 20, 0.0).unwrap();
        assert_eq!(result.upper.len(), close.len());
        // With nbdev = 0, upper == middle == lower for all finite indices
        for i in 0..result.upper.len() {
            if !result.upper[i].is_nan() {
                let diff_upper = (result.upper[i] - result.middle[i]).abs();
                let diff_lower = (result.lower[i] - result.middle[i]).abs();
                assert!(
                    diff_upper <= 1e-9,
                    "With nbdev=0, upper should equal middle at index {}",
                    i
                );
                assert!(
                    diff_lower <= 1e-9,
                    "With nbdev=0, lower should equal middle at index {}",
                    i
                );
            }
        }
    }

    #[test]
    fn bollinger_bands_custom_period() {
        // Test with different period
        let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = bollinger_bands(&close, 10, 1.5).unwrap();
        assert_eq!(result.upper.len(), 50);
        // lookback = period - 1 = 9
        let lookback = 10 - 1;
        assert!(result.upper[lookback - 1].is_nan());
        assert!(!result.upper[lookback].is_nan());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // **Property 5: Bollinger Bands Ordering**
    //
    // For any close price slice with no NaN values and valid parameters,
    // at every index `i` where all three bands are finite, the ordering
    // `lower[i] <= middle[i] <= upper[i]` SHALL hold.
    //
    // **Validates: Requirements 8.4**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn bollinger_bands_ordering(
            // Generate a vector of positive close prices (no NaN values)
            // Using positive values to simulate realistic price data
            close in prop::collection::vec(1.0f64..10000.0f64, 1..500),
            // Period must be >= 1
            period in 1usize..50,
            // nbdev must be finite and non-negative (0.0 to 5.0 is a reasonable range)
            nbdev in 0.0f64..5.0f64
        ) {
            let result = bollinger_bands(&close, period, nbdev);

            // Bollinger Bands should return Ok for valid inputs
            prop_assert!(result.is_ok(), "Bollinger Bands returned error: {:?}", result);

            let output = result.unwrap();

            // Verify output lengths match input length
            let n = close.len();
            prop_assert_eq!(output.upper.len(), n);
            prop_assert_eq!(output.middle.len(), n);
            prop_assert_eq!(output.lower.len(), n);

            // Verify ordering: lower[i] <= middle[i] <= upper[i] for all finite indices
            for i in 0..n {
                let lower = output.lower[i];
                let middle = output.middle[i];
                let upper = output.upper[i];

                // Only check ordering when all three values are finite
                if !lower.is_nan() && !middle.is_nan() && !upper.is_nan() {
                    prop_assert!(
                        lower <= middle,
                        "Bollinger Bands ordering violated at index {}: lower ({}) > middle ({}). \
                         Input length: {}, period: {}, nbdev: {}",
                        i, lower, middle, n, period, nbdev
                    );
                    prop_assert!(
                        middle <= upper,
                        "Bollinger Bands ordering violated at index {}: middle ({}) > upper ({}). \
                         Input length: {}, period: {}, nbdev: {}",
                        i, middle, upper, n, period, nbdev
                    );
                }
            }
        }
    }
}
