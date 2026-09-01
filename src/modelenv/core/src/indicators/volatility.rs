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

        if period == 0 {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "period",
            value: 0,
            reason: "period must be greater than 0",
        });
    }

            if nbdev.is_nan() || nbdev < 0.0 || !nbdev.is_finite() {
        return Err(IndicatorError::InvalidPeriod {
            param_name: "nbdev",
            value: 0,
            reason: "nbdev must be finite and non-negative",
        });
    }

        if n == 0 {
        return Ok(BollingerBandsOutput {
            upper: Vec::new(),
            middle: Vec::new(),
            lower: Vec::new(),
        });
    }

            if n < period {
        return Ok(BollingerBandsOutput {
            upper: vec![f64::NAN; n],
            middle: vec![f64::NAN; n],
            lower: vec![f64::NAN; n],
        });
    }

                    match talib_bbands(close, period, nbdev, nbdev, MaType::Sma) {
        Ok((upper, middle, lower)) => {
                                    if upper.len() == n && middle.len() == n && lower.len() == n {
                Ok(BollingerBandsOutput {
                    upper,
                    middle,
                    lower,
                })
            } else {
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
                        Ok(BollingerBandsOutput {
                upper: vec![f64::NAN; n],
                middle: vec![f64::NAN; n],
                lower: vec![f64::NAN; n],
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
                let close: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let result = bollinger_bands(&close, 20, 0.0).unwrap();
        assert_eq!(result.upper.len(), close.len());
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
                let close: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = bollinger_bands(&close, 10, 1.5).unwrap();
        assert_eq!(result.upper.len(), 50);
                let lookback = 10 - 1;
        assert!(result.upper[lookback - 1].is_nan());
        assert!(!result.upper[lookback].is_nan());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

                                proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn bollinger_bands_ordering(
                                    close in prop::collection::vec(1.0f64..10000.0f64, 1..500),
                        period in 1usize..50,
                        nbdev in 0.0f64..5.0f64
        ) {
            let result = bollinger_bands(&close, period, nbdev);

                        prop_assert!(result.is_ok(), "Bollinger Bands returned error: {:?}", result);

            let output = result.unwrap();

                        let n = close.len();
            prop_assert_eq!(output.upper.len(), n);
            prop_assert_eq!(output.middle.len(), n);
            prop_assert_eq!(output.lower.len(), n);

                        for i in 0..n {
                let lower = output.lower[i];
                let middle = output.middle[i];
                let upper = output.upper[i];

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
