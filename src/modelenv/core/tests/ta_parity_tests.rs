//! Parity test infrastructure for TA indicators.
//!
//! This module provides test helpers for verifying that Rust indicator implementations
//! match Python `ta/` package outputs within the specified tolerance.
//!
//! # Fixture Structure
//!
//! JSON fixtures are stored in `tests/fixtures/ta-indicators/` with the following format:
//!
//! ```json
//! {
//!   "input": {
//!     "timestamp_ns": [...],
//!     "open": [...],
//!     "high": [...],
//!     "low": [...],
//!     "close": [...],
//!     "volume": [...]
//!   },
//!   "params": { ... },
//!   "output": {
//!     "values": [...],
//!     "nan_indices": [...]
//!   }
//! }
//! ```
//!
//! # Parity Tolerance
//!
//! All comparisons use `1e-9` absolute tolerance as specified in the design document.

use modelenv_proto::Bar;
use serde::{Deserialize, Deserializer};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

/// Custom deserializer for i64 that can handle both string and integer formats.
/// This is needed because JSON doesn't handle large integers well, so timestamps
/// are often serialized as strings.
fn deserialize_i64_from_string_or_int<'de, D>(deserializer: D) -> Result<i64, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;

    struct I64Visitor;

    impl<'de> Visitor<'de> for I64Visitor {
        type Value = i64;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("an integer or a string containing an integer")
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(value)
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(value as i64)
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            value.parse::<i64>().map_err(de::Error::custom)
        }
    }

    deserializer.deserialize_any(I64Visitor)
}

/// Custom deserializer for Option<i64> that can handle both string and integer formats.
fn deserialize_optional_i64_from_string_or_int<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;

    struct OptionalI64Visitor;

    impl<'de> Visitor<'de> for OptionalI64Visitor {
        type Value = Option<i64>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("null, an integer, or a string containing an integer")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserialize_i64_from_string_or_int(deserializer).map(Some)
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(value))
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(value as i64))
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            value.parse::<i64>().map(Some).map_err(de::Error::custom)
        }
    }

    deserializer.deserialize_any(OptionalI64Visitor)
}

/// The absolute tolerance for floating-point comparisons.
/// This matches the parity tolerance specified in the design document.
pub const PARITY_TOLERANCE: f64 = 1e-9;

/// Input data structure for fixtures.
#[derive(Debug, Deserialize)]
pub struct FixtureInput {
    pub timestamp_ns: Vec<i64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

impl FixtureInput {
    /// Convert fixture input to a vector of `Bar` structs.
    pub fn to_bars(&self) -> Vec<Bar> {
        let n = self.timestamp_ns.len();
        (0..n)
            .map(|i| Bar {
                timestamp_ns: self.timestamp_ns[i],
                open: self.open[i],
                high: self.high[i],
                low: self.low[i],
                close: self.close[i],
                volume: self.volume[i],
            })
            .collect()
    }
}

/// Output data structure for scalar indicator fixtures (RSI, CCI, ADX, etc.).
#[derive(Debug, Deserialize)]
pub struct ScalarOutput {
    /// Output values where `null` in JSON becomes `None`.
    pub values: Vec<Option<f64>>,
}

/// Output data structure for MACD fixtures.
#[derive(Debug, Deserialize)]
pub struct MacdFixtureOutput {
    pub macd: Vec<Option<f64>>,
    pub signal: Vec<Option<f64>>,
    pub hist: Vec<Option<f64>>,
}

/// Output data structure for Bollinger Bands fixtures.
#[derive(Debug, Deserialize)]
pub struct BollingerBandsFixtureOutput {
    pub upper: Vec<Option<f64>>,
    pub middle: Vec<Option<f64>>,
    pub lower: Vec<Option<f64>>,
}

/// Output data structure for Ichimoku fixtures.
#[derive(Debug, Deserialize)]
pub struct IchimokuFixtureOutput {
    pub tenkan: Vec<Option<f64>>,
    pub kijun: Vec<Option<f64>>,
    pub senkou_a: Vec<Option<f64>>,
    pub senkou_b: Vec<Option<f64>>,
    pub chikou: Vec<Option<f64>>,
}

/// Output data structure for Fibonacci retracement fixtures.
#[derive(Debug, Deserialize)]
pub struct FibonacciFixtureOutput {
    pub fr_000: Vec<Option<f64>>,
    pub fr_236: Vec<Option<f64>>,
    pub fr_382: Vec<Option<f64>>,
    pub fr_500: Vec<Option<f64>>,
    pub fr_618: Vec<Option<f64>>,
    pub fr_786: Vec<Option<f64>>,
    pub fr_1000: Vec<Option<f64>>,
}

/// Pattern structure for double bottom/top fixtures.
#[derive(Debug, Deserialize)]
pub struct PatternFixture {
    pub idx1: usize,
    pub idx2: usize,
    #[serde(deserialize_with = "deserialize_i64_from_string_or_int")]
    pub ts1: i64,
    #[serde(deserialize_with = "deserialize_i64_from_string_or_int")]
    pub ts2: i64,
    pub neckline: f64,
    pub neckline_idx: usize,
    pub depth_pct: f64,
    pub width_bars: usize,
    pub confirmed: bool,
    pub min_before_val: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_optional_i64_from_string_or_int")]
    pub min_before_ts: Option<i64>,
    pub max_before_val: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_optional_i64_from_string_or_int")]
    pub max_before_ts: Option<i64>,
    pub min_after_val: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_optional_i64_from_string_or_int")]
    pub min_after_ts: Option<i64>,
    pub max_after_val: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_optional_i64_from_string_or_int")]
    pub max_after_ts: Option<i64>,
    // Double bottom specific
    pub low1: Option<f64>,
    pub low2: Option<f64>,
    // Double top specific
    pub high1: Option<f64>,
    pub high2: Option<f64>,
}

/// Output data structure for pattern detection fixtures.
#[derive(Debug, Deserialize)]
pub struct PatternDetectionFixtureOutput {
    pub patterns: Vec<PatternFixture>,
    pub latest_min: Option<f64>,
    pub latest_max: Option<f64>,
}

/// Generic fixture structure for scalar indicators.
#[derive(Debug, Deserialize)]
pub struct ScalarFixture {
    pub input: FixtureInput,
    pub params: serde_json::Value,
    pub output: ScalarOutput,
}

/// Fixture structure for MACD indicator.
#[derive(Debug, Deserialize)]
pub struct MacdFixture {
    pub input: FixtureInput,
    pub params: MacdParams,
    pub output: MacdFixtureOutput,
}

#[derive(Debug, Deserialize)]
pub struct MacdParams {
    pub fast: usize,
    pub slow: usize,
    pub signal: usize,
}

/// Fixture structure for Bollinger Bands indicator.
#[derive(Debug, Deserialize)]
pub struct BollingerBandsFixture {
    pub input: FixtureInput,
    pub params: BollingerBandsParams,
    pub output: BollingerBandsFixtureOutput,
}

#[derive(Debug, Deserialize)]
pub struct BollingerBandsParams {
    pub period: usize,
    pub nbdev: f64,
}

/// Fixture structure for Ichimoku indicator.
#[derive(Debug, Deserialize)]
pub struct IchimokuFixture {
    pub input: FixtureInput,
    pub params: IchimokuParams,
    pub output: IchimokuFixtureOutput,
}

#[derive(Debug, Deserialize)]
pub struct IchimokuParams {
    pub tenkan: usize,
    pub kijun: usize,
    pub senkou_b_period: usize,
}

/// Fixture structure for Fibonacci retracements.
#[derive(Debug, Deserialize)]
pub struct FibonacciFixture {
    pub input: FixtureInput,
    pub params: FibonacciParams,
    pub output: FibonacciFixtureOutput,
}

#[derive(Debug, Deserialize)]
pub struct FibonacciParams {
    pub window: usize,
}

/// Fixture structure for pattern detection.
#[derive(Debug, Deserialize)]
pub struct PatternDetectionFixture {
    pub input: FixtureInput,
    pub params: PatternDetectionParams,
    pub output: PatternDetectionFixtureOutput,
}

#[derive(Debug, Deserialize)]
pub struct PatternDetectionParams {
    pub window: usize,
    pub tolerance_pct: f64,
    pub min_width: usize,
}

/// Load a JSON fixture file from the fixtures directory.
///
/// # Arguments
///
/// * `fixture_name` - The name of the fixture file (e.g., "rsi_14.json")
///
/// # Returns
///
/// The raw JSON string content of the fixture file.
///
/// # Panics
///
/// Panics if the fixture file cannot be read.
pub fn load_fixture_raw(fixture_name: &str) -> String {
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("ta-indicators")
        .join(fixture_name);

    fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("Failed to read fixture {}: {}", fixture_path.display(), e))
}

/// Load and parse a scalar indicator fixture.
///
/// # Arguments
///
/// * `fixture_name` - The name of the fixture file (e.g., "rsi_14.json")
///
/// # Returns
///
/// The parsed `ScalarFixture` struct.
pub fn load_scalar_fixture(fixture_name: &str) -> ScalarFixture {
    let content = load_fixture_raw(fixture_name);
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse fixture {}: {}", fixture_name, e))
}

/// Load and parse a MACD fixture.
pub fn load_macd_fixture(fixture_name: &str) -> MacdFixture {
    let content = load_fixture_raw(fixture_name);
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse MACD fixture {}: {}", fixture_name, e))
}

/// Load and parse a Bollinger Bands fixture.
pub fn load_bollinger_bands_fixture(fixture_name: &str) -> BollingerBandsFixture {
    let content = load_fixture_raw(fixture_name);
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse Bollinger Bands fixture {}: {}", fixture_name, e))
}

/// Load and parse an Ichimoku fixture.
pub fn load_ichimoku_fixture(fixture_name: &str) -> IchimokuFixture {
    let content = load_fixture_raw(fixture_name);
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse Ichimoku fixture {}: {}", fixture_name, e))
}

/// Load and parse a Fibonacci fixture.
pub fn load_fibonacci_fixture(fixture_name: &str) -> FibonacciFixture {
    let content = load_fixture_raw(fixture_name);
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse Fibonacci fixture {}: {}", fixture_name, e))
}

/// Load and parse a pattern detection fixture.
pub fn load_pattern_fixture(fixture_name: &str) -> PatternDetectionFixture {
    let content = load_fixture_raw(fixture_name);
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse pattern fixture {}: {}", fixture_name, e))
}

/// Result of comparing two output vectors.
#[derive(Debug)]
pub struct ComparisonResult {
    /// Whether all comparisons passed.
    pub passed: bool,
    /// Total number of elements compared.
    pub total_elements: usize,
    /// Number of elements that matched within tolerance.
    pub matched_elements: usize,
    /// Indices where values differed beyond tolerance.
    pub mismatched_indices: Vec<usize>,
    /// Details of mismatches for debugging.
    pub mismatch_details: Vec<MismatchDetail>,
}

/// Details of a single mismatch.
#[derive(Debug)]
pub struct MismatchDetail {
    pub index: usize,
    pub expected: Option<f64>,
    pub actual: f64,
    pub difference: Option<f64>,
}

/// Compare two output vectors with the specified tolerance.
///
/// This function compares a Rust output vector against expected values from a fixture,
/// handling NaN values correctly (NaN == NaN for comparison purposes).
///
/// # Arguments
///
/// * `actual` - The actual output from the Rust implementation
/// * `expected` - The expected output from the fixture (None represents NaN)
/// * `tolerance` - The absolute tolerance for floating-point comparisons
///
/// # Returns
///
/// A `ComparisonResult` containing details about the comparison.
pub fn compare_outputs(
    actual: &[f64],
    expected: &[Option<f64>],
    tolerance: f64,
) -> ComparisonResult {
    let total_elements = expected.len();
    let mut matched_elements = 0;
    let mut mismatched_indices = Vec::new();
    let mut mismatch_details = Vec::new();

    // Check length match
    if actual.len() != expected.len() {
        return ComparisonResult {
            passed: false,
            total_elements,
            matched_elements: 0,
            mismatched_indices: vec![],
            mismatch_details: vec![MismatchDetail {
                index: 0,
                expected: None,
                actual: 0.0,
                difference: None,
            }],
        };
    }

    for (i, (act, exp)) in actual.iter().zip(expected.iter()).enumerate() {
        let matches = match exp {
            None => {
                // Expected NaN - actual should also be NaN
                act.is_nan()
            }
            Some(exp_val) => {
                if exp_val.is_nan() {
                    // Expected NaN - actual should also be NaN
                    act.is_nan()
                } else if act.is_nan() {
                    // Expected finite but got NaN
                    false
                } else {
                    // Both finite - compare with tolerance
                    (act - exp_val).abs() <= tolerance
                }
            }
        };

        if matches {
            matched_elements += 1;
        } else {
            mismatched_indices.push(i);
            mismatch_details.push(MismatchDetail {
                index: i,
                expected: *exp,
                actual: *act,
                difference: exp.map(|e| (act - e).abs()),
            });
        }
    }

    ComparisonResult {
        passed: mismatched_indices.is_empty(),
        total_elements,
        matched_elements,
        mismatched_indices,
        mismatch_details,
    }
}

/// Compare outputs using the default parity tolerance (1e-9).
pub fn compare_outputs_default(actual: &[f64], expected: &[Option<f64>]) -> ComparisonResult {
    compare_outputs(actual, expected, PARITY_TOLERANCE)
}

/// Verify that NaN placement in the actual output matches the expected NaN positions.
///
/// # Arguments
///
/// * `actual` - The actual output from the Rust implementation
/// * `expected` - The expected output from the fixture (None represents NaN)
///
/// # Returns
///
/// A `NaNPlacementResult` containing details about NaN placement verification.
#[derive(Debug)]
pub struct NaNPlacementResult {
    /// Whether all NaN placements match.
    pub passed: bool,
    /// Indices where NaN was expected but a finite value was found.
    pub missing_nans: Vec<usize>,
    /// Indices where NaN was found but a finite value was expected.
    pub unexpected_nans: Vec<usize>,
    /// Total number of expected NaN positions.
    pub expected_nan_count: usize,
    /// Total number of actual NaN positions.
    pub actual_nan_count: usize,
}

/// Verify NaN placement matches between actual and expected outputs.
pub fn verify_nan_placement(actual: &[f64], expected: &[Option<f64>]) -> NaNPlacementResult {
    let mut missing_nans = Vec::new();
    let mut unexpected_nans = Vec::new();
    let mut expected_nan_count = 0;
    let mut actual_nan_count = 0;

    // Handle length mismatch
    if actual.len() != expected.len() {
        return NaNPlacementResult {
            passed: false,
            missing_nans: vec![],
            unexpected_nans: vec![],
            expected_nan_count: 0,
            actual_nan_count: 0,
        };
    }

    for (i, (act, exp)) in actual.iter().zip(expected.iter()).enumerate() {
        let expected_nan = exp.is_none() || exp.map(|v| v.is_nan()).unwrap_or(false);
        let actual_nan = act.is_nan();

        if expected_nan {
            expected_nan_count += 1;
        }
        if actual_nan {
            actual_nan_count += 1;
        }

        if expected_nan && !actual_nan {
            missing_nans.push(i);
        } else if !expected_nan && actual_nan {
            unexpected_nans.push(i);
        }
    }

    NaNPlacementResult {
        passed: missing_nans.is_empty() && unexpected_nans.is_empty(),
        missing_nans,
        unexpected_nans,
        expected_nan_count,
        actual_nan_count,
    }
}

/// Get the set of NaN indices from an expected output vector.
pub fn get_expected_nan_indices(expected: &[Option<f64>]) -> HashSet<usize> {
    expected
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_none() || v.map(|x| x.is_nan()).unwrap_or(false))
        .map(|(i, _)| i)
        .collect()
}

/// Get the set of NaN indices from an actual output vector.
pub fn get_actual_nan_indices(actual: &[f64]) -> HashSet<usize> {
    actual
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_nan())
        .map(|(i, _)| i)
        .collect()
}

/// Assert that two output vectors match within the parity tolerance.
///
/// This is a convenience macro for use in tests that provides detailed
/// error messages on failure.
#[macro_export]
macro_rules! assert_parity {
    ($actual:expr, $expected:expr) => {
        assert_parity!($actual, $expected, $crate::PARITY_TOLERANCE)
    };
    ($actual:expr, $expected:expr, $tolerance:expr) => {{
        let result = $crate::compare_outputs($actual, $expected, $tolerance);
        if !result.passed {
            let mut msg = format!(
                "Parity check failed: {}/{} elements matched\n",
                result.matched_elements, result.total_elements
            );
            for detail in result.mismatch_details.iter().take(10) {
                msg.push_str(&format!(
                    "  Index {}: expected {:?}, got {}, diff {:?}\n",
                    detail.index, detail.expected, detail.actual, detail.difference
                ));
            }
            if result.mismatch_details.len() > 10 {
                msg.push_str(&format!(
                    "  ... and {} more mismatches\n",
                    result.mismatch_details.len() - 10
                ));
            }
            panic!("{}", msg);
        }
    }};
}

/// Assert that NaN placement matches between actual and expected outputs.
#[macro_export]
macro_rules! assert_nan_placement {
    ($actual:expr, $expected:expr) => {{
        let result = $crate::verify_nan_placement($actual, $expected);
        if !result.passed {
            let mut msg = format!(
                "NaN placement check failed:\n  Expected {} NaNs, found {} NaNs\n",
                result.expected_nan_count, result.actual_nan_count
            );
            if !result.missing_nans.is_empty() {
                msg.push_str(&format!(
                    "  Missing NaNs at indices: {:?}\n",
                    &result.missing_nans[..result.missing_nans.len().min(10)]
                ));
            }
            if !result.unexpected_nans.is_empty() {
                msg.push_str(&format!(
                    "  Unexpected NaNs at indices: {:?}\n",
                    &result.unexpected_nans[..result.unexpected_nans.len().min(10)]
                ));
            }
            panic!("{}", msg);
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_outputs_exact_match() {
        let actual = vec![1.0, 2.0, 3.0];
        let expected = vec![Some(1.0), Some(2.0), Some(3.0)];
        let result = compare_outputs_default(&actual, &expected);
        assert!(result.passed);
        assert_eq!(result.matched_elements, 3);
    }

    #[test]
    fn test_compare_outputs_within_tolerance() {
        let actual = vec![1.0 + 1e-10, 2.0 - 1e-10, 3.0];
        let expected = vec![Some(1.0), Some(2.0), Some(3.0)];
        let result = compare_outputs_default(&actual, &expected);
        assert!(result.passed);
    }

    #[test]
    fn test_compare_outputs_beyond_tolerance() {
        let actual = vec![1.0 + 1e-8, 2.0, 3.0];
        let expected = vec![Some(1.0), Some(2.0), Some(3.0)];
        let result = compare_outputs_default(&actual, &expected);
        assert!(!result.passed);
        assert_eq!(result.mismatched_indices, vec![0]);
    }

    #[test]
    fn test_compare_outputs_nan_handling() {
        let actual = vec![f64::NAN, 2.0, f64::NAN];
        let expected = vec![None, Some(2.0), None];
        let result = compare_outputs_default(&actual, &expected);
        assert!(result.passed);
    }

    #[test]
    fn test_compare_outputs_nan_mismatch() {
        let actual = vec![1.0, 2.0, 3.0];
        let expected = vec![None, Some(2.0), Some(3.0)];
        let result = compare_outputs_default(&actual, &expected);
        assert!(!result.passed);
        assert_eq!(result.mismatched_indices, vec![0]);
    }

    #[test]
    fn test_verify_nan_placement_match() {
        let actual = vec![f64::NAN, f64::NAN, 3.0, 4.0];
        let expected = vec![None, None, Some(3.0), Some(4.0)];
        let result = verify_nan_placement(&actual, &expected);
        assert!(result.passed);
        assert_eq!(result.expected_nan_count, 2);
        assert_eq!(result.actual_nan_count, 2);
    }

    #[test]
    fn test_verify_nan_placement_missing_nan() {
        let actual = vec![1.0, f64::NAN, 3.0, 4.0];
        let expected = vec![None, None, Some(3.0), Some(4.0)];
        let result = verify_nan_placement(&actual, &expected);
        assert!(!result.passed);
        assert_eq!(result.missing_nans, vec![0]);
    }

    #[test]
    fn test_verify_nan_placement_unexpected_nan() {
        let actual = vec![f64::NAN, f64::NAN, f64::NAN, 4.0];
        let expected = vec![None, None, Some(3.0), Some(4.0)];
        let result = verify_nan_placement(&actual, &expected);
        assert!(!result.passed);
        assert_eq!(result.unexpected_nans, vec![2]);
    }

    #[test]
    fn test_get_nan_indices() {
        let expected = vec![None, Some(2.0), None, Some(4.0)];
        let nan_indices = get_expected_nan_indices(&expected);
        assert!(nan_indices.contains(&0));
        assert!(nan_indices.contains(&2));
        assert!(!nan_indices.contains(&1));
        assert!(!nan_indices.contains(&3));
    }

    #[test]
    fn test_fixture_input_to_bars() {
        let input = FixtureInput {
            timestamp_ns: vec![1000, 2000, 3000],
            open: vec![100.0, 101.0, 102.0],
            high: vec![101.0, 102.0, 103.0],
            low: vec![99.0, 100.0, 101.0],
            close: vec![100.5, 101.5, 102.5],
            volume: vec![1000.0, 2000.0, 3000.0],
        };

        let bars = input.to_bars();
        assert_eq!(bars.len(), 3);
        assert_eq!(bars[0].timestamp_ns, 1000);
        assert_eq!(bars[0].open, 100.0);
        assert_eq!(bars[0].high, 101.0);
        assert_eq!(bars[0].low, 99.0);
        assert_eq!(bars[0].close, 100.5);
        assert_eq!(bars[0].volume, 1000.0);
    }

    #[test]
    fn test_load_fixture_raw() {
        // This test verifies that we can load a fixture file
        // It will fail if the fixture doesn't exist, which is expected
        // during initial development
        let result = std::panic::catch_unwind(|| load_fixture_raw("rsi_14.json"));
        // The fixture should exist based on the directory listing
        assert!(result.is_ok(), "rsi_14.json fixture should exist");
    }

    #[test]
    fn test_load_scalar_fixture() {
        let fixture = load_scalar_fixture("rsi_14.json");
        assert!(!fixture.input.close.is_empty());
        assert!(!fixture.output.values.is_empty());
        assert_eq!(fixture.input.close.len(), fixture.output.values.len());
    }
}


// =============================================================================
// Property 9: talib-rs Parity Tests
// =============================================================================
//
// These tests verify that Rust indicator implementations match Python `ta/`
// package outputs within 1e-9 absolute tolerance.
//
// **Validates: Requirements 2.5, 3.4, 4.4, 5.3, 6.4, 8.3**
//
// Indicators tested:
// - RSI (Requirement 2.5)
// - CCI (Requirement 3.4)
// - ADX (Requirement 4.4)
// - MACD (Requirement 5.3)
// - All MovingAverageKind variants: SMA, EMA, WMA, DEMA, TEMA, KAMA, TRIMA (Requirement 6.4)
// - Bollinger Bands (Requirement 8.3)

#[cfg(test)]
mod talib_parity_tests {
    use super::*;
    use modelenv_core::indicators::momentum::{cci, rsi};
    use modelenv_core::indicators::trend::{adx, macd, moving_average, MovingAverageKind};
    use modelenv_core::indicators::volatility::bollinger_bands;

    // =========================================================================
    // RSI Parity Test
    // =========================================================================
    //
    // **Property 9: talib-rs Parity - RSI**
    // **Validates: Requirement 2.5**
    //
    // WHEN `close` contains no NaN values and `period >= 1`, THE Indicators_Module
    // SHALL produce output whose value at every index `i >= period` differs from
    // `talib.RSI(close, timeperiod=period)[i]` by at most `1e-9` in absolute value.

    #[test]
    fn test_rsi_parity() {
        let fixture = load_scalar_fixture("rsi_14.json");
        let period: usize = fixture.params["period"].as_u64().unwrap() as usize;

        // Compute RSI using Rust implementation
        let actual = rsi(&fixture.input.close, period);

        // Compare with expected output from Python fixture
        let result = compare_outputs_default(&actual, &fixture.output.values);

        assert!(
            result.passed,
            "RSI parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            result.matched_elements,
            result.total_elements,
            result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Also verify NaN placement
        let nan_result = verify_nan_placement(&actual, &fixture.output.values);
        assert!(
            nan_result.passed,
            "RSI NaN placement failed: expected {} NaNs, got {}.\n\
             Missing NaNs at: {:?}\n\
             Unexpected NaNs at: {:?}",
            nan_result.expected_nan_count,
            nan_result.actual_nan_count,
            nan_result.missing_nans.iter().take(10).collect::<Vec<_>>(),
            nan_result.unexpected_nans.iter().take(10).collect::<Vec<_>>()
        );
    }

    // =========================================================================
    // CCI Parity Test
    // =========================================================================
    //
    // **Property 9: talib-rs Parity - CCI**
    // **Validates: Requirement 3.4**
    //
    // WHEN `close` contains no NaN values, `high`, `low`, `close` have equal
    // length, and `period >= 1`, THE Indicators_Module SHALL produce a vector
    // whose value at every index where `talib.CCI(high, low, close, timeperiod=period)`
    // is finite differs from that TA-Lib value by at most `1e-9` in absolute value.

    #[test]
    fn test_cci_parity() {
        let fixture = load_scalar_fixture("cci_14.json");
        let period: usize = fixture.params["period"].as_u64().unwrap() as usize;

        // Compute CCI using Rust implementation
        let actual = cci(
            &fixture.input.high,
            &fixture.input.low,
            &fixture.input.close,
            period,
        )
        .expect("CCI computation should succeed");

        // Compare with expected output from Python fixture
        let result = compare_outputs_default(&actual, &fixture.output.values);

        assert!(
            result.passed,
            "CCI parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            result.matched_elements,
            result.total_elements,
            result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Also verify NaN placement
        let nan_result = verify_nan_placement(&actual, &fixture.output.values);
        assert!(
            nan_result.passed,
            "CCI NaN placement failed: expected {} NaNs, got {}.\n\
             Missing NaNs at: {:?}\n\
             Unexpected NaNs at: {:?}",
            nan_result.expected_nan_count,
            nan_result.actual_nan_count,
            nan_result.missing_nans.iter().take(10).collect::<Vec<_>>(),
            nan_result.unexpected_nans.iter().take(10).collect::<Vec<_>>()
        );
    }

    // =========================================================================
    // ADX Parity Test
    // =========================================================================
    //
    // **Property 9: talib-rs Parity - ADX**
    // **Validates: Requirement 4.4**
    //
    // WHEN `high`, `low`, `close` are equal-length and contain no NaN inputs AND
    // `period >= 1`, THE Indicators_Module SHALL produce output that agrees with
    // `talib.ADX(high, low, close, timeperiod=period)` element-wise: every
    // finite pair differs by at most `1e-9` in absolute value, and every NaN
    // position in the TA-Lib reference is also NaN in the Rust output.

    #[test]
    fn test_adx_parity() {
        let fixture = load_scalar_fixture("adx_14.json");
        let period: usize = fixture.params["period"].as_u64().unwrap() as usize;

        // Compute ADX using Rust implementation
        let actual = adx(
            &fixture.input.high,
            &fixture.input.low,
            &fixture.input.close,
            period,
        )
        .expect("ADX computation should succeed");

        // Compare with expected output from Python fixture
        let result = compare_outputs_default(&actual, &fixture.output.values);

        assert!(
            result.passed,
            "ADX parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            result.matched_elements,
            result.total_elements,
            result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Also verify NaN placement
        let nan_result = verify_nan_placement(&actual, &fixture.output.values);
        assert!(
            nan_result.passed,
            "ADX NaN placement failed: expected {} NaNs, got {}.\n\
             Missing NaNs at: {:?}\n\
             Unexpected NaNs at: {:?}",
            nan_result.expected_nan_count,
            nan_result.actual_nan_count,
            nan_result.missing_nans.iter().take(10).collect::<Vec<_>>(),
            nan_result.unexpected_nans.iter().take(10).collect::<Vec<_>>()
        );
    }

    // =========================================================================
    // MACD Parity Test
    // =========================================================================
    //
    // **Property 9: talib-rs Parity - MACD**
    // **Validates: Requirement 5.3**
    //
    // WHEN `close` contains no NaN values AND `fast`, `slow`, `signal` are each
    // `>= 1` AND `fast < slow` AND `close.len() >= slow + signal - 1`, THE
    // Indicators_Module SHALL produce `macd`, `signal`, and `hist` whose value
    // at every index where the corresponding TA-Lib output is finite differs
    // from that TA-Lib value by at most `1e-9` in absolute value.

    #[test]
    fn test_macd_parity() {
        let fixture = load_macd_fixture("macd_12_26_9.json");

        // Compute MACD using Rust implementation
        let actual = macd(
            &fixture.input.close,
            fixture.params.fast,
            fixture.params.slow,
            fixture.params.signal,
        )
        .expect("MACD computation should succeed");

        // Compare MACD line
        let macd_result = compare_outputs_default(&actual.macd, &fixture.output.macd);
        assert!(
            macd_result.passed,
            "MACD line parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            macd_result.matched_elements,
            macd_result.total_elements,
            macd_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Compare signal line
        let signal_result = compare_outputs_default(&actual.signal, &fixture.output.signal);
        assert!(
            signal_result.passed,
            "MACD signal parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            signal_result.matched_elements,
            signal_result.total_elements,
            signal_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Compare histogram
        let hist_result = compare_outputs_default(&actual.hist, &fixture.output.hist);
        assert!(
            hist_result.passed,
            "MACD histogram parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            hist_result.matched_elements,
            hist_result.total_elements,
            hist_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Verify NaN placement for all three outputs
        let macd_nan = verify_nan_placement(&actual.macd, &fixture.output.macd);
        let signal_nan = verify_nan_placement(&actual.signal, &fixture.output.signal);
        let hist_nan = verify_nan_placement(&actual.hist, &fixture.output.hist);

        assert!(
            macd_nan.passed,
            "MACD line NaN placement failed"
        );
        assert!(
            signal_nan.passed,
            "MACD signal NaN placement failed"
        );
        assert!(
            hist_nan.passed,
            "MACD histogram NaN placement failed"
        );
    }

    // =========================================================================
    // Moving Average Parity Tests
    // =========================================================================
    //
    // **Property 9: talib-rs Parity - Moving Averages**
    // **Validates: Requirement 6.4**
    //
    // WHEN `close` contains no NaN values AND `period >= 1` AND `close.len() >`
    // the TA-Lib warm-up length for that `kind` and `period`, THE
    // Indicators_Module SHALL produce a vector whose value at every index where
    // the corresponding `talib.SMA` / `talib.EMA` / `talib.WMA` / `talib.DEMA` /
    // `talib.TEMA` / `talib.KAMA` / `talib.TRIMA` output is finite differs from
    // that TA-Lib value by at most `1e-9` in absolute value.

    #[test]
    fn test_sma_parity() {
        let fixture = load_scalar_fixture("sma_10.json");
        let period: usize = fixture.params["period"].as_u64().unwrap() as usize;

        // Compute SMA using Rust implementation
        let actual = moving_average(&fixture.input.close, MovingAverageKind::Sma, period)
            .expect("SMA computation should succeed");

        // Compare with expected output from Python fixture
        let result = compare_outputs_default(&actual, &fixture.output.values);

        assert!(
            result.passed,
            "SMA parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            result.matched_elements,
            result.total_elements,
            result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Also verify NaN placement
        let nan_result = verify_nan_placement(&actual, &fixture.output.values);
        assert!(
            nan_result.passed,
            "SMA NaN placement failed: expected {} NaNs, got {}",
            nan_result.expected_nan_count,
            nan_result.actual_nan_count
        );
    }

    #[test]
    fn test_ema_parity() {
        let fixture = load_scalar_fixture("ema_20.json");
        let period: usize = fixture.params["period"].as_u64().unwrap() as usize;

        // Compute EMA using Rust implementation
        let actual = moving_average(&fixture.input.close, MovingAverageKind::Ema, period)
            .expect("EMA computation should succeed");

        // Compare with expected output from Python fixture
        let result = compare_outputs_default(&actual, &fixture.output.values);

        assert!(
            result.passed,
            "EMA parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            result.matched_elements,
            result.total_elements,
            result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Also verify NaN placement
        let nan_result = verify_nan_placement(&actual, &fixture.output.values);
        assert!(
            nan_result.passed,
            "EMA NaN placement failed: expected {} NaNs, got {}",
            nan_result.expected_nan_count,
            nan_result.actual_nan_count
        );
    }

    #[test]
    fn test_wma_parity() {
        let fixture = load_scalar_fixture("wma_50.json");
        let period: usize = fixture.params["period"].as_u64().unwrap() as usize;

        // Compute WMA using Rust implementation
        let actual = moving_average(&fixture.input.close, MovingAverageKind::Wma, period)
            .expect("WMA computation should succeed");

        // Compare with expected output from Python fixture
        let result = compare_outputs_default(&actual, &fixture.output.values);

        assert!(
            result.passed,
            "WMA parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            result.matched_elements,
            result.total_elements,
            result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Also verify NaN placement
        let nan_result = verify_nan_placement(&actual, &fixture.output.values);
        assert!(
            nan_result.passed,
            "WMA NaN placement failed: expected {} NaNs, got {}",
            nan_result.expected_nan_count,
            nan_result.actual_nan_count
        );
    }

    #[test]
    fn test_dema_parity() {
        let fixture = load_scalar_fixture("dema_10.json");
        let period: usize = fixture.params["period"].as_u64().unwrap() as usize;

        // Compute DEMA using Rust implementation
        let actual = moving_average(&fixture.input.close, MovingAverageKind::Dema, period)
            .expect("DEMA computation should succeed");

        // Compare with expected output from Python fixture
        let result = compare_outputs_default(&actual, &fixture.output.values);

        assert!(
            result.passed,
            "DEMA parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            result.matched_elements,
            result.total_elements,
            result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Also verify NaN placement
        let nan_result = verify_nan_placement(&actual, &fixture.output.values);
        assert!(
            nan_result.passed,
            "DEMA NaN placement failed: expected {} NaNs, got {}",
            nan_result.expected_nan_count,
            nan_result.actual_nan_count
        );
    }

    #[test]
    fn test_tema_parity() {
        let fixture = load_scalar_fixture("tema_20.json");
        let period: usize = fixture.params["period"].as_u64().unwrap() as usize;

        // Compute TEMA using Rust implementation
        let actual = moving_average(&fixture.input.close, MovingAverageKind::Tema, period)
            .expect("TEMA computation should succeed");

        // Compare with expected output from Python fixture
        let result = compare_outputs_default(&actual, &fixture.output.values);

        assert!(
            result.passed,
            "TEMA parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            result.matched_elements,
            result.total_elements,
            result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Also verify NaN placement
        let nan_result = verify_nan_placement(&actual, &fixture.output.values);
        assert!(
            nan_result.passed,
            "TEMA NaN placement failed: expected {} NaNs, got {}",
            nan_result.expected_nan_count,
            nan_result.actual_nan_count
        );
    }

    #[test]
    fn test_kama_parity() {
        let fixture = load_scalar_fixture("kama_10.json");
        let period: usize = fixture.params["period"].as_u64().unwrap() as usize;

        // Compute KAMA using Rust implementation
        let actual = moving_average(&fixture.input.close, MovingAverageKind::Kama, period)
            .expect("KAMA computation should succeed");

        // Compare with expected output from Python fixture
        let result = compare_outputs_default(&actual, &fixture.output.values);

        assert!(
            result.passed,
            "KAMA parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            result.matched_elements,
            result.total_elements,
            result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Also verify NaN placement
        let nan_result = verify_nan_placement(&actual, &fixture.output.values);
        assert!(
            nan_result.passed,
            "KAMA NaN placement failed: expected {} NaNs, got {}",
            nan_result.expected_nan_count,
            nan_result.actual_nan_count
        );
    }

    #[test]
    fn test_trima_parity() {
        let fixture = load_scalar_fixture("trima_20.json");
        let period: usize = fixture.params["period"].as_u64().unwrap() as usize;

        // Compute TRIMA using Rust implementation
        let actual = moving_average(&fixture.input.close, MovingAverageKind::Trima, period)
            .expect("TRIMA computation should succeed");

        // Compare with expected output from Python fixture
        let result = compare_outputs_default(&actual, &fixture.output.values);

        assert!(
            result.passed,
            "TRIMA parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            result.matched_elements,
            result.total_elements,
            result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Also verify NaN placement
        let nan_result = verify_nan_placement(&actual, &fixture.output.values);
        assert!(
            nan_result.passed,
            "TRIMA NaN placement failed: expected {} NaNs, got {}",
            nan_result.expected_nan_count,
            nan_result.actual_nan_count
        );
    }

    // =========================================================================
    // Bollinger Bands Parity Test
    // =========================================================================
    //
    // **Property 9: talib-rs Parity - Bollinger Bands**
    // **Validates: Requirement 8.3**
    //
    // WHEN `close` contains no NaN values AND `period >= 1` AND `nbdev.is_finite()
    // && nbdev >= 0.0`, THE Indicators_Module SHALL set indices `[0, period - 1)`
    // of all three vectors to `f64::NAN` and produce values at indices
    // `[period - 1, n)` whose difference from
    // `talib.BBANDS(close, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev)`
    // is at most `1e-9` in absolute value at every index where the TA-Lib
    // reference is finite.

    #[test]
    fn test_bollinger_bands_parity() {
        let fixture = load_bollinger_bands_fixture("bollinger_20_2.json");

        // Compute Bollinger Bands using Rust implementation
        let actual = bollinger_bands(
            &fixture.input.close,
            fixture.params.period,
            fixture.params.nbdev,
        )
        .expect("Bollinger Bands computation should succeed");

        // Compare upper band
        let upper_result = compare_outputs_default(&actual.upper, &fixture.output.upper);
        assert!(
            upper_result.passed,
            "Bollinger Bands upper parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            upper_result.matched_elements,
            upper_result.total_elements,
            upper_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Compare middle band
        let middle_result = compare_outputs_default(&actual.middle, &fixture.output.middle);
        assert!(
            middle_result.passed,
            "Bollinger Bands middle parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            middle_result.matched_elements,
            middle_result.total_elements,
            middle_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Compare lower band
        let lower_result = compare_outputs_default(&actual.lower, &fixture.output.lower);
        assert!(
            lower_result.passed,
            "Bollinger Bands lower parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            lower_result.matched_elements,
            lower_result.total_elements,
            lower_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Verify NaN placement for all three bands
        let upper_nan = verify_nan_placement(&actual.upper, &fixture.output.upper);
        let middle_nan = verify_nan_placement(&actual.middle, &fixture.output.middle);
        let lower_nan = verify_nan_placement(&actual.lower, &fixture.output.lower);

        assert!(
            upper_nan.passed,
            "Bollinger Bands upper NaN placement failed: expected {} NaNs, got {}",
            upper_nan.expected_nan_count,
            upper_nan.actual_nan_count
        );
        assert!(
            middle_nan.passed,
            "Bollinger Bands middle NaN placement failed: expected {} NaNs, got {}",
            middle_nan.expected_nan_count,
            middle_nan.actual_nan_count
        );
        assert!(
            lower_nan.passed,
            "Bollinger Bands lower NaN placement failed: expected {} NaNs, got {}",
            lower_nan.expected_nan_count,
            lower_nan.actual_nan_count
        );
    }
}

// =============================================================================
// Property 10: Ichimoku Parity Test
// =============================================================================
//
// This test verifies that the Rust Ichimoku Cloud implementation matches
// Python `ta/trend/ic.py` output within 1e-9 absolute tolerance.
//
// **Validates: Requirements 7.8**
//
// WHEN `high`, `low`, `close` contain no NaN values, THE Indicators_Module
// SHALL produce outputs whose value at every index where the Python
// `ta/trend/ic.py` output is finite differs from the Python value by at
// most `1e-9` in absolute value.

#[cfg(test)]
mod ichimoku_parity_tests {
    use super::*;
    use modelenv_core::indicators::trend::ichimoku;

    // =========================================================================
    // Ichimoku Parity Test
    // =========================================================================
    //
    // **Property 10: Ichimoku Parity**
    // **Validates: Requirement 7.8**
    //
    // Test against Python `ta/trend/ic.py` output.
    // Verify all 5 lines (tenkan, kijun, senkou_a, senkou_b, chikou) match
    // within 1e-9 tolerance.
    // Verify NaN placement matches the Python reference.

    #[test]
    fn test_ichimoku_parity() {
        let fixture = load_ichimoku_fixture("ichimoku_9_26_52.json");

        // Compute Ichimoku using Rust implementation
        let actual = ichimoku(
            &fixture.input.high,
            &fixture.input.low,
            &fixture.input.close,
            fixture.params.tenkan,
            fixture.params.kijun,
            fixture.params.senkou_b_period,
        )
        .expect("Ichimoku computation should succeed");

        // Compare tenkan line
        let tenkan_result = compare_outputs_default(&actual.tenkan, &fixture.output.tenkan);
        assert!(
            tenkan_result.passed,
            "Ichimoku tenkan parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            tenkan_result.matched_elements,
            tenkan_result.total_elements,
            tenkan_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Compare kijun line
        let kijun_result = compare_outputs_default(&actual.kijun, &fixture.output.kijun);
        assert!(
            kijun_result.passed,
            "Ichimoku kijun parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            kijun_result.matched_elements,
            kijun_result.total_elements,
            kijun_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Compare senkou_a line
        let senkou_a_result = compare_outputs_default(&actual.senkou_a, &fixture.output.senkou_a);
        assert!(
            senkou_a_result.passed,
            "Ichimoku senkou_a parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            senkou_a_result.matched_elements,
            senkou_a_result.total_elements,
            senkou_a_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Compare senkou_b line
        let senkou_b_result = compare_outputs_default(&actual.senkou_b, &fixture.output.senkou_b);
        assert!(
            senkou_b_result.passed,
            "Ichimoku senkou_b parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            senkou_b_result.matched_elements,
            senkou_b_result.total_elements,
            senkou_b_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Compare chikou line
        let chikou_result = compare_outputs_default(&actual.chikou, &fixture.output.chikou);
        assert!(
            chikou_result.passed,
            "Ichimoku chikou parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            chikou_result.matched_elements,
            chikou_result.total_elements,
            chikou_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Verify NaN placement for all five lines
        let tenkan_nan = verify_nan_placement(&actual.tenkan, &fixture.output.tenkan);
        let kijun_nan = verify_nan_placement(&actual.kijun, &fixture.output.kijun);
        let senkou_a_nan = verify_nan_placement(&actual.senkou_a, &fixture.output.senkou_a);
        let senkou_b_nan = verify_nan_placement(&actual.senkou_b, &fixture.output.senkou_b);
        let chikou_nan = verify_nan_placement(&actual.chikou, &fixture.output.chikou);

        assert!(
            tenkan_nan.passed,
            "Ichimoku tenkan NaN placement failed: expected {} NaNs, got {}.\n\
             Missing NaNs at: {:?}\n\
             Unexpected NaNs at: {:?}",
            tenkan_nan.expected_nan_count,
            tenkan_nan.actual_nan_count,
            tenkan_nan.missing_nans.iter().take(10).collect::<Vec<_>>(),
            tenkan_nan.unexpected_nans.iter().take(10).collect::<Vec<_>>()
        );

        assert!(
            kijun_nan.passed,
            "Ichimoku kijun NaN placement failed: expected {} NaNs, got {}.\n\
             Missing NaNs at: {:?}\n\
             Unexpected NaNs at: {:?}",
            kijun_nan.expected_nan_count,
            kijun_nan.actual_nan_count,
            kijun_nan.missing_nans.iter().take(10).collect::<Vec<_>>(),
            kijun_nan.unexpected_nans.iter().take(10).collect::<Vec<_>>()
        );

        assert!(
            senkou_a_nan.passed,
            "Ichimoku senkou_a NaN placement failed: expected {} NaNs, got {}.\n\
             Missing NaNs at: {:?}\n\
             Unexpected NaNs at: {:?}",
            senkou_a_nan.expected_nan_count,
            senkou_a_nan.actual_nan_count,
            senkou_a_nan.missing_nans.iter().take(10).collect::<Vec<_>>(),
            senkou_a_nan.unexpected_nans.iter().take(10).collect::<Vec<_>>()
        );

        assert!(
            senkou_b_nan.passed,
            "Ichimoku senkou_b NaN placement failed: expected {} NaNs, got {}.\n\
             Missing NaNs at: {:?}\n\
             Unexpected NaNs at: {:?}",
            senkou_b_nan.expected_nan_count,
            senkou_b_nan.actual_nan_count,
            senkou_b_nan.missing_nans.iter().take(10).collect::<Vec<_>>(),
            senkou_b_nan.unexpected_nans.iter().take(10).collect::<Vec<_>>()
        );

        assert!(
            chikou_nan.passed,
            "Ichimoku chikou NaN placement failed: expected {} NaNs, got {}.\n\
             Missing NaNs at: {:?}\n\
             Unexpected NaNs at: {:?}",
            chikou_nan.expected_nan_count,
            chikou_nan.actual_nan_count,
            chikou_nan.missing_nans.iter().take(10).collect::<Vec<_>>(),
            chikou_nan.unexpected_nans.iter().take(10).collect::<Vec<_>>()
        );
    }
}

// =============================================================================
// Property 11: Fibonacci Parity Test
// =============================================================================
//
// This test verifies that the Rust Fibonacci retracement implementation matches
// Python `ta/support/fr.py` output within 1e-9 absolute tolerance.
//
// **Validates: Requirements 9.6**
//
// WHEN `high` and `low` contain no NaN values AND `window >= 1`, THE
// Indicators_Module SHALL produce output whose value at every index differs
// from the Python `ta/support/fr.py` output at the same index by at most
// `1e-9` in absolute value.

#[cfg(test)]
mod fibonacci_parity_tests {
    use super::*;
    use modelenv_core::indicators::support::fibonacci_retracements;

    // =========================================================================
    // Fibonacci Retracement Parity Test
    // =========================================================================
    //
    // **Property 11: Fibonacci Parity**
    // **Validates: Requirement 9.6**
    //
    // Test against Python `ta/support/fr.py` output.
    // Verify all 7 retracement levels (fr_000, fr_236, fr_382, fr_500, fr_618,
    // fr_786, fr_1000) match within 1e-9 tolerance.
    // Note: Fibonacci retracements use min_periods=1, so there should be no
    // leading NaNs when inputs are finite.

    #[test]
    fn test_fibonacci_parity() {
        let fixture = load_fibonacci_fixture("fibonacci_50.json");

        // Compute Fibonacci retracements using Rust implementation
        let actual = fibonacci_retracements(
            &fixture.input.high,
            &fixture.input.low,
            fixture.params.window,
        )
        .expect("Fibonacci retracement computation should succeed");

        // Compare fr_000 (0.0% level - rolling low)
        let fr_000_result = compare_outputs_default(&actual.fr_000, &fixture.output.fr_000);
        assert!(
            fr_000_result.passed,
            "Fibonacci fr_000 parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            fr_000_result.matched_elements,
            fr_000_result.total_elements,
            fr_000_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Compare fr_236 (23.6% level)
        let fr_236_result = compare_outputs_default(&actual.fr_236, &fixture.output.fr_236);
        assert!(
            fr_236_result.passed,
            "Fibonacci fr_236 parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            fr_236_result.matched_elements,
            fr_236_result.total_elements,
            fr_236_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Compare fr_382 (38.2% level)
        let fr_382_result = compare_outputs_default(&actual.fr_382, &fixture.output.fr_382);
        assert!(
            fr_382_result.passed,
            "Fibonacci fr_382 parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            fr_382_result.matched_elements,
            fr_382_result.total_elements,
            fr_382_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Compare fr_500 (50.0% level)
        let fr_500_result = compare_outputs_default(&actual.fr_500, &fixture.output.fr_500);
        assert!(
            fr_500_result.passed,
            "Fibonacci fr_500 parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            fr_500_result.matched_elements,
            fr_500_result.total_elements,
            fr_500_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Compare fr_618 (61.8% level)
        let fr_618_result = compare_outputs_default(&actual.fr_618, &fixture.output.fr_618);
        assert!(
            fr_618_result.passed,
            "Fibonacci fr_618 parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            fr_618_result.matched_elements,
            fr_618_result.total_elements,
            fr_618_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Compare fr_786 (78.6% level)
        let fr_786_result = compare_outputs_default(&actual.fr_786, &fixture.output.fr_786);
        assert!(
            fr_786_result.passed,
            "Fibonacci fr_786 parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            fr_786_result.matched_elements,
            fr_786_result.total_elements,
            fr_786_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Compare fr_1000 (100.0% level - rolling high)
        let fr_1000_result = compare_outputs_default(&actual.fr_1000, &fixture.output.fr_1000);
        assert!(
            fr_1000_result.passed,
            "Fibonacci fr_1000 parity test failed: {}/{} elements matched.\n\
             First 10 mismatches: {:?}",
            fr_1000_result.matched_elements,
            fr_1000_result.total_elements,
            fr_1000_result.mismatch_details.iter().take(10).collect::<Vec<_>>()
        );

        // Verify NaN placement for all seven levels
        // Note: With min_periods=1, there should be no NaNs when inputs are finite
        let fr_000_nan = verify_nan_placement(&actual.fr_000, &fixture.output.fr_000);
        let fr_236_nan = verify_nan_placement(&actual.fr_236, &fixture.output.fr_236);
        let fr_382_nan = verify_nan_placement(&actual.fr_382, &fixture.output.fr_382);
        let fr_500_nan = verify_nan_placement(&actual.fr_500, &fixture.output.fr_500);
        let fr_618_nan = verify_nan_placement(&actual.fr_618, &fixture.output.fr_618);
        let fr_786_nan = verify_nan_placement(&actual.fr_786, &fixture.output.fr_786);
        let fr_1000_nan = verify_nan_placement(&actual.fr_1000, &fixture.output.fr_1000);

        assert!(
            fr_000_nan.passed,
            "Fibonacci fr_000 NaN placement failed: expected {} NaNs, got {}.\n\
             Missing NaNs at: {:?}\n\
             Unexpected NaNs at: {:?}",
            fr_000_nan.expected_nan_count,
            fr_000_nan.actual_nan_count,
            fr_000_nan.missing_nans.iter().take(10).collect::<Vec<_>>(),
            fr_000_nan.unexpected_nans.iter().take(10).collect::<Vec<_>>()
        );

        assert!(
            fr_236_nan.passed,
            "Fibonacci fr_236 NaN placement failed: expected {} NaNs, got {}.\n\
             Missing NaNs at: {:?}\n\
             Unexpected NaNs at: {:?}",
            fr_236_nan.expected_nan_count,
            fr_236_nan.actual_nan_count,
            fr_236_nan.missing_nans.iter().take(10).collect::<Vec<_>>(),
            fr_236_nan.unexpected_nans.iter().take(10).collect::<Vec<_>>()
        );

        assert!(
            fr_382_nan.passed,
            "Fibonacci fr_382 NaN placement failed: expected {} NaNs, got {}.\n\
             Missing NaNs at: {:?}\n\
             Unexpected NaNs at: {:?}",
            fr_382_nan.expected_nan_count,
            fr_382_nan.actual_nan_count,
            fr_382_nan.missing_nans.iter().take(10).collect::<Vec<_>>(),
            fr_382_nan.unexpected_nans.iter().take(10).collect::<Vec<_>>()
        );

        assert!(
            fr_500_nan.passed,
            "Fibonacci fr_500 NaN placement failed: expected {} NaNs, got {}.\n\
             Missing NaNs at: {:?}\n\
             Unexpected NaNs at: {:?}",
            fr_500_nan.expected_nan_count,
            fr_500_nan.actual_nan_count,
            fr_500_nan.missing_nans.iter().take(10).collect::<Vec<_>>(),
            fr_500_nan.unexpected_nans.iter().take(10).collect::<Vec<_>>()
        );

        assert!(
            fr_618_nan.passed,
            "Fibonacci fr_618 NaN placement failed: expected {} NaNs, got {}.\n\
             Missing NaNs at: {:?}\n\
             Unexpected NaNs at: {:?}",
            fr_618_nan.expected_nan_count,
            fr_618_nan.actual_nan_count,
            fr_618_nan.missing_nans.iter().take(10).collect::<Vec<_>>(),
            fr_618_nan.unexpected_nans.iter().take(10).collect::<Vec<_>>()
        );

        assert!(
            fr_786_nan.passed,
            "Fibonacci fr_786 NaN placement failed: expected {} NaNs, got {}.\n\
             Missing NaNs at: {:?}\n\
             Unexpected NaNs at: {:?}",
            fr_786_nan.expected_nan_count,
            fr_786_nan.actual_nan_count,
            fr_786_nan.missing_nans.iter().take(10).collect::<Vec<_>>(),
            fr_786_nan.unexpected_nans.iter().take(10).collect::<Vec<_>>()
        );

        assert!(
            fr_1000_nan.passed,
            "Fibonacci fr_1000 NaN placement failed: expected {} NaNs, got {}.\n\
             Missing NaNs at: {:?}\n\
             Unexpected NaNs at: {:?}",
            fr_1000_nan.expected_nan_count,
            fr_1000_nan.actual_nan_count,
            fr_1000_nan.missing_nans.iter().take(10).collect::<Vec<_>>(),
            fr_1000_nan.unexpected_nans.iter().take(10).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_fibonacci_ordering_invariant_with_fixture() {
        // Additional test to verify the ordering invariant holds for fixture data
        let fixture = load_fibonacci_fixture("fibonacci_50.json");

        let actual = fibonacci_retracements(
            &fixture.input.high,
            &fixture.input.low,
            fixture.params.window,
        )
        .expect("Fibonacci retracement computation should succeed");

        // Verify ordering invariant: fr_000 <= fr_236 <= fr_382 <= fr_500 <= fr_618 <= fr_786 <= fr_1000
        // within 1e-12 slack per inequality
        const ORDERING_SLACK: f64 = 1e-12;

        for i in 0..actual.fr_000.len() {
            // Skip if any value is NaN
            if actual.fr_000[i].is_nan()
                || actual.fr_236[i].is_nan()
                || actual.fr_382[i].is_nan()
                || actual.fr_500[i].is_nan()
                || actual.fr_618[i].is_nan()
                || actual.fr_786[i].is_nan()
                || actual.fr_1000[i].is_nan()
            {
                continue;
            }

            assert!(
                actual.fr_000[i] <= actual.fr_236[i] + ORDERING_SLACK,
                "Ordering violation at index {}: fr_000 ({}) > fr_236 ({})",
                i,
                actual.fr_000[i],
                actual.fr_236[i]
            );
            assert!(
                actual.fr_236[i] <= actual.fr_382[i] + ORDERING_SLACK,
                "Ordering violation at index {}: fr_236 ({}) > fr_382 ({})",
                i,
                actual.fr_236[i],
                actual.fr_382[i]
            );
            assert!(
                actual.fr_382[i] <= actual.fr_500[i] + ORDERING_SLACK,
                "Ordering violation at index {}: fr_382 ({}) > fr_500 ({})",
                i,
                actual.fr_382[i],
                actual.fr_500[i]
            );
            assert!(
                actual.fr_500[i] <= actual.fr_618[i] + ORDERING_SLACK,
                "Ordering violation at index {}: fr_500 ({}) > fr_618 ({})",
                i,
                actual.fr_500[i],
                actual.fr_618[i]
            );
            assert!(
                actual.fr_618[i] <= actual.fr_786[i] + ORDERING_SLACK,
                "Ordering violation at index {}: fr_618 ({}) > fr_786 ({})",
                i,
                actual.fr_618[i],
                actual.fr_786[i]
            );
            assert!(
                actual.fr_786[i] <= actual.fr_1000[i] + ORDERING_SLACK,
                "Ordering violation at index {}: fr_786 ({}) > fr_1000 ({})",
                i,
                actual.fr_786[i],
                actual.fr_1000[i]
            );
        }
    }
}

// =============================================================================
// Property 12: Pattern Detection Parity Test
// =============================================================================
//
// This test verifies that the Rust pattern detection implementations match
// Python `ta/patterns/` output.
//
// **Validates: Requirements 10.12, 11.13**
//
// WHEN `bars` is fed into both the Rust `detect_double_bottoms` and the
// Python `detect_double_bottoms` with identical `(low, high, close,
// timestamp)` series and the same `window`, `tolerance_pct`, `min_width`,
// THE Indicators_Module SHALL produce the same number of patterns in the
// same order, with exact equality on `idx1`, `idx2`, `ts1`, `ts2`,
// `neckline_idx`, `width_bars`, and `confirmed`, and `1e-9` absolute
// parity on `low1`, `low2`, `neckline`, `depth_pct`, `min_before_val`,
// `max_before_val`, `min_after_val`, `max_after_val`.

#[cfg(test)]
mod pattern_detection_parity_tests {
    use super::*;
    use modelenv_core::indicators::patterns::{detect_double_bottoms, detect_double_tops};

    /// Compare two optional f64 values with tolerance.
    /// Returns true if both are None, or both are Some and within tolerance.
    fn compare_optional_f64(actual: Option<f64>, expected: Option<f64>, tolerance: f64) -> bool {
        match (actual, expected) {
            (None, None) => true,
            (Some(a), Some(e)) => (a - e).abs() <= tolerance,
            _ => false,
        }
    }

    /// Compare two optional i64 values for exact equality.
    /// Returns true if both are None, or both are Some and equal.
    fn compare_optional_i64(actual: Option<i64>, expected: Option<i64>) -> bool {
        match (actual, expected) {
            (None, None) => true,
            (Some(a), Some(e)) => a == e,
            _ => false,
        }
    }

    // =========================================================================
    // Double Bottom Parity Test
    // =========================================================================
    //
    // **Property 12: Pattern Detection Parity - Double Bottom**
    // **Validates: Requirement 10.12**
    //
    // Test double-bottom detection against Python `ta/patterns/` output.
    // Verify same number of patterns in same order.
    // Verify exact equality on integer/boolean fields.
    // Verify `1e-9` tolerance on floating-point fields.

    #[test]
    fn test_double_bottom_parity() {
        let fixture = load_pattern_fixture("double_bottom.json");
        let bars = fixture.input.to_bars();

        // Compute double-bottom patterns using Rust implementation
        let actual = detect_double_bottoms(
            &bars,
            fixture.params.window,
            fixture.params.tolerance_pct,
            fixture.params.min_width,
        )
        .expect("Double bottom detection should succeed");

        // Verify same number of patterns
        assert_eq!(
            actual.patterns.len(),
            fixture.output.patterns.len(),
            "Pattern count mismatch: Rust detected {} patterns, Python detected {}",
            actual.patterns.len(),
            fixture.output.patterns.len()
        );

        // Compare each pattern
        for (i, (rust_pattern, py_pattern)) in actual
            .patterns
            .iter()
            .zip(fixture.output.patterns.iter())
            .enumerate()
        {
            // Verify exact equality on integer fields
            assert_eq!(
                rust_pattern.idx1, py_pattern.idx1,
                "Pattern {}: idx1 mismatch - Rust: {}, Python: {}",
                i, rust_pattern.idx1, py_pattern.idx1
            );
            assert_eq!(
                rust_pattern.idx2, py_pattern.idx2,
                "Pattern {}: idx2 mismatch - Rust: {}, Python: {}",
                i, rust_pattern.idx2, py_pattern.idx2
            );
            assert_eq!(
                rust_pattern.neckline_idx, py_pattern.neckline_idx,
                "Pattern {}: neckline_idx mismatch - Rust: {}, Python: {}",
                i, rust_pattern.neckline_idx, py_pattern.neckline_idx
            );
            assert_eq!(
                rust_pattern.width_bars, py_pattern.width_bars,
                "Pattern {}: width_bars mismatch - Rust: {}, Python: {}",
                i, rust_pattern.width_bars, py_pattern.width_bars
            );

            // Verify exact equality on boolean fields
            assert_eq!(
                rust_pattern.confirmed, py_pattern.confirmed,
                "Pattern {}: confirmed mismatch - Rust: {}, Python: {}",
                i, rust_pattern.confirmed, py_pattern.confirmed
            );

            // Verify timestamps (parsed from string in fixture)
            let expected_ts1: i64 = py_pattern.ts1;
            let expected_ts2: i64 = py_pattern.ts2;
            assert_eq!(
                rust_pattern.ts1, expected_ts1,
                "Pattern {}: ts1 mismatch - Rust: {}, Python: {}",
                i, rust_pattern.ts1, expected_ts1
            );
            assert_eq!(
                rust_pattern.ts2, expected_ts2,
                "Pattern {}: ts2 mismatch - Rust: {}, Python: {}",
                i, rust_pattern.ts2, expected_ts2
            );

            // Verify floating-point fields with 1e-9 tolerance
            let low1_expected = py_pattern.low1.expect("Double bottom should have low1");
            let low2_expected = py_pattern.low2.expect("Double bottom should have low2");

            assert!(
                (rust_pattern.low1 - low1_expected).abs() <= PARITY_TOLERANCE,
                "Pattern {}: low1 mismatch - Rust: {}, Python: {}, diff: {}",
                i,
                rust_pattern.low1,
                low1_expected,
                (rust_pattern.low1 - low1_expected).abs()
            );
            assert!(
                (rust_pattern.low2 - low2_expected).abs() <= PARITY_TOLERANCE,
                "Pattern {}: low2 mismatch - Rust: {}, Python: {}, diff: {}",
                i,
                rust_pattern.low2,
                low2_expected,
                (rust_pattern.low2 - low2_expected).abs()
            );
            assert!(
                (rust_pattern.neckline - py_pattern.neckline).abs() <= PARITY_TOLERANCE,
                "Pattern {}: neckline mismatch - Rust: {}, Python: {}, diff: {}",
                i,
                rust_pattern.neckline,
                py_pattern.neckline,
                (rust_pattern.neckline - py_pattern.neckline).abs()
            );
            assert!(
                (rust_pattern.depth_pct - py_pattern.depth_pct).abs() <= PARITY_TOLERANCE,
                "Pattern {}: depth_pct mismatch - Rust: {}, Python: {}, diff: {}",
                i,
                rust_pattern.depth_pct,
                py_pattern.depth_pct,
                (rust_pattern.depth_pct - py_pattern.depth_pct).abs()
            );

            // Verify optional floating-point fields (before/after extrema)
            assert!(
                compare_optional_f64(
                    rust_pattern.min_before_val,
                    py_pattern.min_before_val,
                    PARITY_TOLERANCE
                ),
                "Pattern {}: min_before_val mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.min_before_val,
                py_pattern.min_before_val
            );
            assert!(
                compare_optional_f64(
                    rust_pattern.max_before_val,
                    py_pattern.max_before_val,
                    PARITY_TOLERANCE
                ),
                "Pattern {}: max_before_val mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.max_before_val,
                py_pattern.max_before_val
            );
            assert!(
                compare_optional_f64(
                    rust_pattern.min_after_val,
                    py_pattern.min_after_val,
                    PARITY_TOLERANCE
                ),
                "Pattern {}: min_after_val mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.min_after_val,
                py_pattern.min_after_val
            );
            assert!(
                compare_optional_f64(
                    rust_pattern.max_after_val,
                    py_pattern.max_after_val,
                    PARITY_TOLERANCE
                ),
                "Pattern {}: max_after_val mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.max_after_val,
                py_pattern.max_after_val
            );

            // Verify optional timestamp fields
            assert!(
                compare_optional_i64(rust_pattern.min_before_ts, py_pattern.min_before_ts),
                "Pattern {}: min_before_ts mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.min_before_ts,
                py_pattern.min_before_ts
            );
            assert!(
                compare_optional_i64(rust_pattern.max_before_ts, py_pattern.max_before_ts),
                "Pattern {}: max_before_ts mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.max_before_ts,
                py_pattern.max_before_ts
            );
            assert!(
                compare_optional_i64(rust_pattern.min_after_ts, py_pattern.min_after_ts),
                "Pattern {}: min_after_ts mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.min_after_ts,
                py_pattern.min_after_ts
            );
            assert!(
                compare_optional_i64(rust_pattern.max_after_ts, py_pattern.max_after_ts),
                "Pattern {}: max_after_ts mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.max_after_ts,
                py_pattern.max_after_ts
            );
        }

        // Verify latest_min and latest_max
        assert!(
            compare_optional_f64(actual.latest_min, fixture.output.latest_min, PARITY_TOLERANCE),
            "latest_min mismatch - Rust: {:?}, Python: {:?}",
            actual.latest_min,
            fixture.output.latest_min
        );
        assert!(
            compare_optional_f64(actual.latest_max, fixture.output.latest_max, PARITY_TOLERANCE),
            "latest_max mismatch - Rust: {:?}, Python: {:?}",
            actual.latest_max,
            fixture.output.latest_max
        );
    }

    // =========================================================================
    // Double Top Parity Test
    // =========================================================================
    //
    // **Property 12: Pattern Detection Parity - Double Top**
    // **Validates: Requirement 11.13**
    //
    // Test double-top detection against Python `ta/patterns/` output.
    // Verify same number of patterns in same order.
    // Verify exact equality on integer/boolean fields.
    // Verify `1e-9` tolerance on floating-point fields.

    #[test]
    fn test_double_top_parity() {
        let fixture = load_pattern_fixture("double_top.json");
        let bars = fixture.input.to_bars();

        // Compute double-top patterns using Rust implementation
        let actual = detect_double_tops(
            &bars,
            fixture.params.window,
            fixture.params.tolerance_pct,
            fixture.params.min_width,
        )
        .expect("Double top detection should succeed");

        // Verify same number of patterns
        assert_eq!(
            actual.patterns.len(),
            fixture.output.patterns.len(),
            "Pattern count mismatch: Rust detected {} patterns, Python detected {}",
            actual.patterns.len(),
            fixture.output.patterns.len()
        );

        // Compare each pattern
        for (i, (rust_pattern, py_pattern)) in actual
            .patterns
            .iter()
            .zip(fixture.output.patterns.iter())
            .enumerate()
        {
            // Verify exact equality on integer fields
            assert_eq!(
                rust_pattern.idx1, py_pattern.idx1,
                "Pattern {}: idx1 mismatch - Rust: {}, Python: {}",
                i, rust_pattern.idx1, py_pattern.idx1
            );
            assert_eq!(
                rust_pattern.idx2, py_pattern.idx2,
                "Pattern {}: idx2 mismatch - Rust: {}, Python: {}",
                i, rust_pattern.idx2, py_pattern.idx2
            );
            assert_eq!(
                rust_pattern.neckline_idx, py_pattern.neckline_idx,
                "Pattern {}: neckline_idx mismatch - Rust: {}, Python: {}",
                i, rust_pattern.neckline_idx, py_pattern.neckline_idx
            );
            assert_eq!(
                rust_pattern.width_bars, py_pattern.width_bars,
                "Pattern {}: width_bars mismatch - Rust: {}, Python: {}",
                i, rust_pattern.width_bars, py_pattern.width_bars
            );

            // Verify exact equality on boolean fields
            assert_eq!(
                rust_pattern.confirmed, py_pattern.confirmed,
                "Pattern {}: confirmed mismatch - Rust: {}, Python: {}",
                i, rust_pattern.confirmed, py_pattern.confirmed
            );

            // Verify timestamps (now properly deserialized as i64)
            assert_eq!(
                rust_pattern.ts1, py_pattern.ts1,
                "Pattern {}: ts1 mismatch - Rust: {}, Python: {}",
                i, rust_pattern.ts1, py_pattern.ts1
            );
            assert_eq!(
                rust_pattern.ts2, py_pattern.ts2,
                "Pattern {}: ts2 mismatch - Rust: {}, Python: {}",
                i, rust_pattern.ts2, py_pattern.ts2
            );

            // Verify floating-point fields with 1e-9 tolerance
            let high1_expected = py_pattern.high1.expect("Double top should have high1");
            let high2_expected = py_pattern.high2.expect("Double top should have high2");

            assert!(
                (rust_pattern.high1 - high1_expected).abs() <= PARITY_TOLERANCE,
                "Pattern {}: high1 mismatch - Rust: {}, Python: {}, diff: {}",
                i,
                rust_pattern.high1,
                high1_expected,
                (rust_pattern.high1 - high1_expected).abs()
            );
            assert!(
                (rust_pattern.high2 - high2_expected).abs() <= PARITY_TOLERANCE,
                "Pattern {}: high2 mismatch - Rust: {}, Python: {}, diff: {}",
                i,
                rust_pattern.high2,
                high2_expected,
                (rust_pattern.high2 - high2_expected).abs()
            );
            assert!(
                (rust_pattern.neckline - py_pattern.neckline).abs() <= PARITY_TOLERANCE,
                "Pattern {}: neckline mismatch - Rust: {}, Python: {}, diff: {}",
                i,
                rust_pattern.neckline,
                py_pattern.neckline,
                (rust_pattern.neckline - py_pattern.neckline).abs()
            );
            assert!(
                (rust_pattern.depth_pct - py_pattern.depth_pct).abs() <= PARITY_TOLERANCE,
                "Pattern {}: depth_pct mismatch - Rust: {}, Python: {}, diff: {}",
                i,
                rust_pattern.depth_pct,
                py_pattern.depth_pct,
                (rust_pattern.depth_pct - py_pattern.depth_pct).abs()
            );

            // Verify optional floating-point fields (before/after extrema)
            assert!(
                compare_optional_f64(
                    rust_pattern.min_before_val,
                    py_pattern.min_before_val,
                    PARITY_TOLERANCE
                ),
                "Pattern {}: min_before_val mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.min_before_val,
                py_pattern.min_before_val
            );
            assert!(
                compare_optional_f64(
                    rust_pattern.max_before_val,
                    py_pattern.max_before_val,
                    PARITY_TOLERANCE
                ),
                "Pattern {}: max_before_val mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.max_before_val,
                py_pattern.max_before_val
            );
            assert!(
                compare_optional_f64(
                    rust_pattern.min_after_val,
                    py_pattern.min_after_val,
                    PARITY_TOLERANCE
                ),
                "Pattern {}: min_after_val mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.min_after_val,
                py_pattern.min_after_val
            );
            assert!(
                compare_optional_f64(
                    rust_pattern.max_after_val,
                    py_pattern.max_after_val,
                    PARITY_TOLERANCE
                ),
                "Pattern {}: max_after_val mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.max_after_val,
                py_pattern.max_after_val
            );

            // Verify optional timestamp fields (now properly deserialized as i64)
            assert!(
                compare_optional_i64(rust_pattern.min_before_ts, py_pattern.min_before_ts),
                "Pattern {}: min_before_ts mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.min_before_ts,
                py_pattern.min_before_ts
            );
            assert!(
                compare_optional_i64(rust_pattern.max_before_ts, py_pattern.max_before_ts),
                "Pattern {}: max_before_ts mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.max_before_ts,
                py_pattern.max_before_ts
            );
            assert!(
                compare_optional_i64(rust_pattern.min_after_ts, py_pattern.min_after_ts),
                "Pattern {}: min_after_ts mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.min_after_ts,
                py_pattern.min_after_ts
            );
            assert!(
                compare_optional_i64(rust_pattern.max_after_ts, py_pattern.max_after_ts),
                "Pattern {}: max_after_ts mismatch - Rust: {:?}, Python: {:?}",
                i,
                rust_pattern.max_after_ts,
                py_pattern.max_after_ts
            );
        }

        // Verify latest_min and latest_max
        assert!(
            compare_optional_f64(actual.latest_min, fixture.output.latest_min, PARITY_TOLERANCE),
            "latest_min mismatch - Rust: {:?}, Python: {:?}",
            actual.latest_min,
            fixture.output.latest_min
        );
        assert!(
            compare_optional_f64(actual.latest_max, fixture.output.latest_max, PARITY_TOLERANCE),
            "latest_max mismatch - Rust: {:?}, Python: {:?}",
            actual.latest_max,
            fixture.output.latest_max
        );
    }

    // =========================================================================
    // Pattern Index Ordering Invariant Test (with fixture data)
    // =========================================================================
    //
    // **Property 7: Pattern Index Ordering**
    // **Validates: Requirements 10.7, 11.7**
    //
    // For any detected DoubleBottom or DoubleTop pattern, the invariants
    // `idx1 < neckline_idx < idx2` AND `width_bars == idx2 - idx1` SHALL hold.

    #[test]
    fn test_pattern_index_ordering_with_fixture() {
        // Test double bottom patterns
        let db_fixture = load_pattern_fixture("double_bottom.json");
        let db_bars = db_fixture.input.to_bars();
        let db_result = detect_double_bottoms(
            &db_bars,
            db_fixture.params.window,
            db_fixture.params.tolerance_pct,
            db_fixture.params.min_width,
        )
        .expect("Double bottom detection should succeed");

        for (i, pattern) in db_result.patterns.iter().enumerate() {
            assert!(
                pattern.idx1 < pattern.neckline_idx,
                "Double bottom pattern {}: idx1 ({}) should be < neckline_idx ({})",
                i,
                pattern.idx1,
                pattern.neckline_idx
            );
            assert!(
                pattern.neckline_idx < pattern.idx2,
                "Double bottom pattern {}: neckline_idx ({}) should be < idx2 ({})",
                i,
                pattern.neckline_idx,
                pattern.idx2
            );
            assert_eq!(
                pattern.width_bars,
                pattern.idx2 - pattern.idx1,
                "Double bottom pattern {}: width_bars ({}) should equal idx2 - idx1 ({})",
                i,
                pattern.width_bars,
                pattern.idx2 - pattern.idx1
            );
        }

        // Test double top patterns
        let dt_fixture = load_pattern_fixture("double_top.json");
        let dt_bars = dt_fixture.input.to_bars();
        let dt_result = detect_double_tops(
            &dt_bars,
            dt_fixture.params.window,
            dt_fixture.params.tolerance_pct,
            dt_fixture.params.min_width,
        )
        .expect("Double top detection should succeed");

        for (i, pattern) in dt_result.patterns.iter().enumerate() {
            assert!(
                pattern.idx1 < pattern.neckline_idx,
                "Double top pattern {}: idx1 ({}) should be < neckline_idx ({})",
                i,
                pattern.idx1,
                pattern.neckline_idx
            );
            assert!(
                pattern.neckline_idx < pattern.idx2,
                "Double top pattern {}: neckline_idx ({}) should be < idx2 ({})",
                i,
                pattern.neckline_idx,
                pattern.idx2
            );
            assert_eq!(
                pattern.width_bars,
                pattern.idx2 - pattern.idx1,
                "Double top pattern {}: width_bars ({}) should equal idx2 - idx1 ({})",
                i,
                pattern.width_bars,
                pattern.idx2 - pattern.idx1
            );
        }
    }

    // =========================================================================
    // Pattern Depth Threshold Test (with fixture data)
    // =========================================================================
    //
    // **Property 8: Pattern Depth Threshold**
    // **Validates: Requirements 10.5, 11.5**
    //
    // For any detected DoubleBottom or DoubleTop pattern, `depth_pct >= 0.1` SHALL hold.

    #[test]
    fn test_pattern_depth_threshold_with_fixture() {
        const MIN_DEPTH_PCT: f64 = 0.1;

        // Test double bottom patterns
        let db_fixture = load_pattern_fixture("double_bottom.json");
        let db_bars = db_fixture.input.to_bars();
        let db_result = detect_double_bottoms(
            &db_bars,
            db_fixture.params.window,
            db_fixture.params.tolerance_pct,
            db_fixture.params.min_width,
        )
        .expect("Double bottom detection should succeed");

        for (i, pattern) in db_result.patterns.iter().enumerate() {
            assert!(
                pattern.depth_pct >= MIN_DEPTH_PCT,
                "Double bottom pattern {}: depth_pct ({}) should be >= {}",
                i,
                pattern.depth_pct,
                MIN_DEPTH_PCT
            );
        }

        // Test double top patterns
        let dt_fixture = load_pattern_fixture("double_top.json");
        let dt_bars = dt_fixture.input.to_bars();
        let dt_result = detect_double_tops(
            &dt_bars,
            dt_fixture.params.window,
            dt_fixture.params.tolerance_pct,
            dt_fixture.params.min_width,
        )
        .expect("Double top detection should succeed");

        for (i, pattern) in dt_result.patterns.iter().enumerate() {
            assert!(
                pattern.depth_pct >= MIN_DEPTH_PCT,
                "Double top pattern {}: depth_pct ({}) should be >= {}",
                i,
                pattern.depth_pct,
                MIN_DEPTH_PCT
            );
        }
    }
}

// =============================================================================
// Property 13: NaN Placement Parity Tests
// =============================================================================
//
// This test module verifies that leading NaN positions in Rust outputs match
// the corresponding TA-Lib or Python reference exactly.
//
// **Validates: Requirements 12.1, 12.2**
//
// Requirement 12.1: FOR every indicator function that wraps TA-Lib (RSI, CCI,
// ADX, MACD, every MovingAverageKind variant, Bollinger Bands), THE
// Indicators_Module SHALL place f64::NAN at exactly the same leading indices
// as the corresponding TA-Lib call produces, and every finite output cell
// SHALL correspond to a finite TA-Lib cell at the same index.
//
// Requirement 12.2: FOR the Ichimoku Cloud, THE Indicators_Module SHALL place
// f64::NAN at the same indices as the Python ta/trend/ic.py implementation
// for each of the five lines tenkan, kijun, senkou_a, senkou_b, and chikou.

#[cfg(test)]
mod nan_placement_parity_tests {
    use super::*;
    use modelenv_core::indicators::momentum::{cci, rsi};
    use modelenv_core::indicators::trend::{adx, ichimoku, macd, moving_average, MovingAverageKind};
    use modelenv_core::indicators::volatility::bollinger_bands;

    /// Helper struct to collect NaN placement verification results for reporting.
    #[derive(Debug)]
    struct NaNPlacementReport {
        indicator_name: &'static str,
        field_name: Option<&'static str>,
        passed: bool,
        expected_nan_count: usize,
        actual_nan_count: usize,
        missing_nans: Vec<usize>,
        unexpected_nans: Vec<usize>,
    }

    impl NaNPlacementReport {
        fn from_result(
            indicator_name: &'static str,
            field_name: Option<&'static str>,
            result: &NaNPlacementResult,
        ) -> Self {
            Self {
                indicator_name,
                field_name,
                passed: result.passed,
                expected_nan_count: result.expected_nan_count,
                actual_nan_count: result.actual_nan_count,
                missing_nans: result.missing_nans.clone(),
                unexpected_nans: result.unexpected_nans.clone(),
            }
        }

        fn format_error(&self) -> String {
            let field_suffix = self.field_name.map_or(String::new(), |f| format!(" ({})", f));
            format!(
                "{}{}: NaN placement mismatch\n\
                 \x20 Expected {} NaNs, found {} NaNs\n\
                 \x20 Missing NaNs at indices: {:?}\n\
                 \x20 Unexpected NaNs at indices: {:?}",
                self.indicator_name,
                field_suffix,
                self.expected_nan_count,
                self.actual_nan_count,
                &self.missing_nans[..self.missing_nans.len().min(10)],
                &self.unexpected_nans[..self.unexpected_nans.len().min(10)]
            )
        }
    }

    // =========================================================================
    // Comprehensive NaN Placement Parity Test for TA-Lib Indicators
    // =========================================================================
    //
    // **Property 13: NaN Placement Parity - TA-Lib Indicators**
    // **Validates: Requirement 12.1**
    //
    // This test verifies that all TA-Lib-backed indicators (RSI, CCI, ADX, MACD,
    // SMA, EMA, WMA, DEMA, TEMA, KAMA, TRIMA, Bollinger Bands) place NaN values
    // at exactly the same indices as the TA-Lib/Python reference.

    #[test]
    fn test_nan_placement_parity_talib_indicators() {
        let mut reports: Vec<NaNPlacementReport> = Vec::new();

        // =====================================================================
        // RSI NaN Placement
        // =====================================================================
        {
            let fixture = load_scalar_fixture("rsi_14.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = rsi(&fixture.input.close, period);
            let result = verify_nan_placement(&actual, &fixture.output.values);
            reports.push(NaNPlacementReport::from_result("RSI", None, &result));
        }

        // =====================================================================
        // CCI NaN Placement
        // =====================================================================
        {
            let fixture = load_scalar_fixture("cci_14.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = cci(
                &fixture.input.high,
                &fixture.input.low,
                &fixture.input.close,
                period,
            )
            .expect("CCI computation should succeed");
            let result = verify_nan_placement(&actual, &fixture.output.values);
            reports.push(NaNPlacementReport::from_result("CCI", None, &result));
        }

        // =====================================================================
        // ADX NaN Placement
        // =====================================================================
        {
            let fixture = load_scalar_fixture("adx_14.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = adx(
                &fixture.input.high,
                &fixture.input.low,
                &fixture.input.close,
                period,
            )
            .expect("ADX computation should succeed");
            let result = verify_nan_placement(&actual, &fixture.output.values);
            reports.push(NaNPlacementReport::from_result("ADX", None, &result));
        }

        // =====================================================================
        // MACD NaN Placement (all three outputs)
        // =====================================================================
        {
            let fixture = load_macd_fixture("macd_12_26_9.json");
            let actual = macd(
                &fixture.input.close,
                fixture.params.fast,
                fixture.params.slow,
                fixture.params.signal,
            )
            .expect("MACD computation should succeed");

            let macd_result = verify_nan_placement(&actual.macd, &fixture.output.macd);
            reports.push(NaNPlacementReport::from_result("MACD", Some("macd_line"), &macd_result));

            let signal_result = verify_nan_placement(&actual.signal, &fixture.output.signal);
            reports.push(NaNPlacementReport::from_result("MACD", Some("signal_line"), &signal_result));

            let hist_result = verify_nan_placement(&actual.hist, &fixture.output.hist);
            reports.push(NaNPlacementReport::from_result("MACD", Some("histogram"), &hist_result));
        }

        // =====================================================================
        // SMA NaN Placement
        // =====================================================================
        {
            let fixture = load_scalar_fixture("sma_10.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = moving_average(&fixture.input.close, MovingAverageKind::Sma, period)
                .expect("SMA computation should succeed");
            let result = verify_nan_placement(&actual, &fixture.output.values);
            reports.push(NaNPlacementReport::from_result("SMA", None, &result));
        }

        // =====================================================================
        // EMA NaN Placement
        // =====================================================================
        {
            let fixture = load_scalar_fixture("ema_20.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = moving_average(&fixture.input.close, MovingAverageKind::Ema, period)
                .expect("EMA computation should succeed");
            let result = verify_nan_placement(&actual, &fixture.output.values);
            reports.push(NaNPlacementReport::from_result("EMA", None, &result));
        }

        // =====================================================================
        // WMA NaN Placement
        // =====================================================================
        {
            let fixture = load_scalar_fixture("wma_50.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = moving_average(&fixture.input.close, MovingAverageKind::Wma, period)
                .expect("WMA computation should succeed");
            let result = verify_nan_placement(&actual, &fixture.output.values);
            reports.push(NaNPlacementReport::from_result("WMA", None, &result));
        }

        // =====================================================================
        // DEMA NaN Placement
        // =====================================================================
        {
            let fixture = load_scalar_fixture("dema_10.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = moving_average(&fixture.input.close, MovingAverageKind::Dema, period)
                .expect("DEMA computation should succeed");
            let result = verify_nan_placement(&actual, &fixture.output.values);
            reports.push(NaNPlacementReport::from_result("DEMA", None, &result));
        }

        // =====================================================================
        // TEMA NaN Placement
        // =====================================================================
        {
            let fixture = load_scalar_fixture("tema_20.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = moving_average(&fixture.input.close, MovingAverageKind::Tema, period)
                .expect("TEMA computation should succeed");
            let result = verify_nan_placement(&actual, &fixture.output.values);
            reports.push(NaNPlacementReport::from_result("TEMA", None, &result));
        }

        // =====================================================================
        // KAMA NaN Placement
        // =====================================================================
        {
            let fixture = load_scalar_fixture("kama_10.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = moving_average(&fixture.input.close, MovingAverageKind::Kama, period)
                .expect("KAMA computation should succeed");
            let result = verify_nan_placement(&actual, &fixture.output.values);
            reports.push(NaNPlacementReport::from_result("KAMA", None, &result));
        }

        // =====================================================================
        // TRIMA NaN Placement
        // =====================================================================
        {
            let fixture = load_scalar_fixture("trima_20.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = moving_average(&fixture.input.close, MovingAverageKind::Trima, period)
                .expect("TRIMA computation should succeed");
            let result = verify_nan_placement(&actual, &fixture.output.values);
            reports.push(NaNPlacementReport::from_result("TRIMA", None, &result));
        }

        // =====================================================================
        // Bollinger Bands NaN Placement (all three bands)
        // =====================================================================
        {
            let fixture = load_bollinger_bands_fixture("bollinger_20_2.json");
            let actual = bollinger_bands(
                &fixture.input.close,
                fixture.params.period,
                fixture.params.nbdev,
            )
            .expect("Bollinger Bands computation should succeed");

            let upper_result = verify_nan_placement(&actual.upper, &fixture.output.upper);
            reports.push(NaNPlacementReport::from_result("Bollinger Bands", Some("upper"), &upper_result));

            let middle_result = verify_nan_placement(&actual.middle, &fixture.output.middle);
            reports.push(NaNPlacementReport::from_result("Bollinger Bands", Some("middle"), &middle_result));

            let lower_result = verify_nan_placement(&actual.lower, &fixture.output.lower);
            reports.push(NaNPlacementReport::from_result("Bollinger Bands", Some("lower"), &lower_result));
        }

        // =====================================================================
        // Aggregate Results and Report
        // =====================================================================
        let failed_reports: Vec<_> = reports.iter().filter(|r| !r.passed).collect();

        if !failed_reports.is_empty() {
            let mut error_msg = format!(
                "NaN Placement Parity Test Failed for TA-Lib Indicators\n\
                 ========================================================\n\
                 {} of {} indicator checks failed:\n\n",
                failed_reports.len(),
                reports.len()
            );

            for report in &failed_reports {
                error_msg.push_str(&report.format_error());
                error_msg.push_str("\n\n");
            }

            panic!("{}", error_msg);
        }

        // Print summary on success
        println!(
            "NaN Placement Parity Test Passed for TA-Lib Indicators\n\
             ========================================================\n\
             All {} indicator checks passed.\n\
             Indicators verified: RSI, CCI, ADX, MACD (3 outputs), SMA, EMA, WMA, DEMA, TEMA, KAMA, TRIMA, Bollinger Bands (3 outputs)",
            reports.len()
        );
    }

    // =========================================================================
    // Comprehensive NaN Placement Parity Test for Ichimoku Cloud
    // =========================================================================
    //
    // **Property 13: NaN Placement Parity - Ichimoku Cloud**
    // **Validates: Requirement 12.2**
    //
    // This test verifies that the Ichimoku Cloud implementation places NaN values
    // at exactly the same indices as the Python ta/trend/ic.py implementation
    // for each of the five lines: tenkan, kijun, senkou_a, senkou_b, and chikou.

    #[test]
    fn test_nan_placement_parity_ichimoku() {
        let fixture = load_ichimoku_fixture("ichimoku_9_26_52.json");

        let actual = ichimoku(
            &fixture.input.high,
            &fixture.input.low,
            &fixture.input.close,
            fixture.params.tenkan,
            fixture.params.kijun,
            fixture.params.senkou_b_period,
        )
        .expect("Ichimoku computation should succeed");

        let mut reports: Vec<NaNPlacementReport> = Vec::new();

        // Verify NaN placement for tenkan line
        let tenkan_result = verify_nan_placement(&actual.tenkan, &fixture.output.tenkan);
        reports.push(NaNPlacementReport::from_result("Ichimoku", Some("tenkan"), &tenkan_result));

        // Verify NaN placement for kijun line
        let kijun_result = verify_nan_placement(&actual.kijun, &fixture.output.kijun);
        reports.push(NaNPlacementReport::from_result("Ichimoku", Some("kijun"), &kijun_result));

        // Verify NaN placement for senkou_a line
        let senkou_a_result = verify_nan_placement(&actual.senkou_a, &fixture.output.senkou_a);
        reports.push(NaNPlacementReport::from_result("Ichimoku", Some("senkou_a"), &senkou_a_result));

        // Verify NaN placement for senkou_b line
        let senkou_b_result = verify_nan_placement(&actual.senkou_b, &fixture.output.senkou_b);
        reports.push(NaNPlacementReport::from_result("Ichimoku", Some("senkou_b"), &senkou_b_result));

        // Verify NaN placement for chikou line
        let chikou_result = verify_nan_placement(&actual.chikou, &fixture.output.chikou);
        reports.push(NaNPlacementReport::from_result("Ichimoku", Some("chikou"), &chikou_result));

        // Aggregate Results and Report
        let failed_reports: Vec<_> = reports.iter().filter(|r| !r.passed).collect();

        if !failed_reports.is_empty() {
            let mut error_msg = format!(
                "NaN Placement Parity Test Failed for Ichimoku Cloud\n\
                 ====================================================\n\
                 {} of {} line checks failed:\n\n",
                failed_reports.len(),
                reports.len()
            );

            for report in &failed_reports {
                error_msg.push_str(&report.format_error());
                error_msg.push_str("\n\n");
            }

            panic!("{}", error_msg);
        }

        // Print summary on success
        println!(
            "NaN Placement Parity Test Passed for Ichimoku Cloud\n\
             ====================================================\n\
             All {} line checks passed.\n\
             Lines verified: tenkan, kijun, senkou_a, senkou_b, chikou",
            reports.len()
        );
    }

    // =========================================================================
    // Individual NaN Placement Tests for Detailed Diagnostics
    // =========================================================================
    //
    // These individual tests provide more granular failure information when
    // debugging specific indicator NaN placement issues.

    #[test]
    fn test_nan_placement_rsi() {
        let fixture = load_scalar_fixture("rsi_14.json");
        let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
        let actual = rsi(&fixture.input.close, period);

        assert_nan_placement!(&actual, &fixture.output.values);

        // Verify expected NaN count matches period - 1 (RSI warm-up)
        let expected_nans = get_expected_nan_indices(&fixture.output.values);
        let actual_nans = get_actual_nan_indices(&actual);
        assert_eq!(
            expected_nans, actual_nans,
            "RSI NaN indices mismatch: expected {:?}, got {:?}",
            expected_nans, actual_nans
        );
    }

    #[test]
    fn test_nan_placement_cci() {
        let fixture = load_scalar_fixture("cci_14.json");
        let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
        let actual = cci(
            &fixture.input.high,
            &fixture.input.low,
            &fixture.input.close,
            period,
        )
        .expect("CCI computation should succeed");

        assert_nan_placement!(&actual, &fixture.output.values);
    }

    #[test]
    fn test_nan_placement_adx() {
        let fixture = load_scalar_fixture("adx_14.json");
        let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
        let actual = adx(
            &fixture.input.high,
            &fixture.input.low,
            &fixture.input.close,
            period,
        )
        .expect("ADX computation should succeed");

        assert_nan_placement!(&actual, &fixture.output.values);

        // ADX has a longer warm-up period (2*period - 1)
        let expected_nans = get_expected_nan_indices(&fixture.output.values);
        let actual_nans = get_actual_nan_indices(&actual);
        assert_eq!(
            expected_nans, actual_nans,
            "ADX NaN indices mismatch: expected {:?}, got {:?}",
            expected_nans, actual_nans
        );
    }

    #[test]
    fn test_nan_placement_macd() {
        let fixture = load_macd_fixture("macd_12_26_9.json");
        let actual = macd(
            &fixture.input.close,
            fixture.params.fast,
            fixture.params.slow,
            fixture.params.signal,
        )
        .expect("MACD computation should succeed");

        // Verify MACD line NaN placement
        assert_nan_placement!(&actual.macd, &fixture.output.macd);

        // Verify signal line NaN placement
        assert_nan_placement!(&actual.signal, &fixture.output.signal);

        // Verify histogram NaN placement
        assert_nan_placement!(&actual.hist, &fixture.output.hist);
    }

    #[test]
    fn test_nan_placement_moving_averages() {
        // SMA
        {
            let fixture = load_scalar_fixture("sma_10.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = moving_average(&fixture.input.close, MovingAverageKind::Sma, period)
                .expect("SMA computation should succeed");
            assert_nan_placement!(&actual, &fixture.output.values);
        }

        // EMA
        {
            let fixture = load_scalar_fixture("ema_20.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = moving_average(&fixture.input.close, MovingAverageKind::Ema, period)
                .expect("EMA computation should succeed");
            assert_nan_placement!(&actual, &fixture.output.values);
        }

        // WMA
        {
            let fixture = load_scalar_fixture("wma_50.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = moving_average(&fixture.input.close, MovingAverageKind::Wma, period)
                .expect("WMA computation should succeed");
            assert_nan_placement!(&actual, &fixture.output.values);
        }

        // DEMA
        {
            let fixture = load_scalar_fixture("dema_10.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = moving_average(&fixture.input.close, MovingAverageKind::Dema, period)
                .expect("DEMA computation should succeed");
            assert_nan_placement!(&actual, &fixture.output.values);
        }

        // TEMA
        {
            let fixture = load_scalar_fixture("tema_20.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = moving_average(&fixture.input.close, MovingAverageKind::Tema, period)
                .expect("TEMA computation should succeed");
            assert_nan_placement!(&actual, &fixture.output.values);
        }

        // KAMA
        {
            let fixture = load_scalar_fixture("kama_10.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = moving_average(&fixture.input.close, MovingAverageKind::Kama, period)
                .expect("KAMA computation should succeed");
            assert_nan_placement!(&actual, &fixture.output.values);
        }

        // TRIMA
        {
            let fixture = load_scalar_fixture("trima_20.json");
            let period: usize = fixture.params["period"].as_u64().unwrap() as usize;
            let actual = moving_average(&fixture.input.close, MovingAverageKind::Trima, period)
                .expect("TRIMA computation should succeed");
            assert_nan_placement!(&actual, &fixture.output.values);
        }
    }

    #[test]
    fn test_nan_placement_bollinger_bands() {
        let fixture = load_bollinger_bands_fixture("bollinger_20_2.json");
        let actual = bollinger_bands(
            &fixture.input.close,
            fixture.params.period,
            fixture.params.nbdev,
        )
        .expect("Bollinger Bands computation should succeed");

        // Verify upper band NaN placement
        assert_nan_placement!(&actual.upper, &fixture.output.upper);

        // Verify middle band NaN placement
        assert_nan_placement!(&actual.middle, &fixture.output.middle);

        // Verify lower band NaN placement
        assert_nan_placement!(&actual.lower, &fixture.output.lower);

        // All three bands should have the same NaN indices
        let upper_nans = get_actual_nan_indices(&actual.upper);
        let middle_nans = get_actual_nan_indices(&actual.middle);
        let lower_nans = get_actual_nan_indices(&actual.lower);

        assert_eq!(
            upper_nans, middle_nans,
            "Bollinger Bands: upper and middle bands have different NaN indices"
        );
        assert_eq!(
            middle_nans, lower_nans,
            "Bollinger Bands: middle and lower bands have different NaN indices"
        );
    }

    #[test]
    fn test_nan_placement_ichimoku_individual_lines() {
        let fixture = load_ichimoku_fixture("ichimoku_9_26_52.json");
        let actual = ichimoku(
            &fixture.input.high,
            &fixture.input.low,
            &fixture.input.close,
            fixture.params.tenkan,
            fixture.params.kijun,
            fixture.params.senkou_b_period,
        )
        .expect("Ichimoku computation should succeed");

        // Verify tenkan line NaN placement
        assert_nan_placement!(&actual.tenkan, &fixture.output.tenkan);

        // Verify kijun line NaN placement
        assert_nan_placement!(&actual.kijun, &fixture.output.kijun);

        // Verify senkou_a line NaN placement
        assert_nan_placement!(&actual.senkou_a, &fixture.output.senkou_a);

        // Verify senkou_b line NaN placement
        assert_nan_placement!(&actual.senkou_b, &fixture.output.senkou_b);

        // Verify chikou line NaN placement
        assert_nan_placement!(&actual.chikou, &fixture.output.chikou);

        // Print NaN counts for debugging
        println!(
            "Ichimoku NaN counts:\n\
             \x20 tenkan: {} NaNs\n\
             \x20 kijun: {} NaNs\n\
             \x20 senkou_a: {} NaNs\n\
             \x20 senkou_b: {} NaNs\n\
             \x20 chikou: {} NaNs",
            get_actual_nan_indices(&actual.tenkan).len(),
            get_actual_nan_indices(&actual.kijun).len(),
            get_actual_nan_indices(&actual.senkou_a).len(),
            get_actual_nan_indices(&actual.senkou_b).len(),
            get_actual_nan_indices(&actual.chikou).len()
        );
    }

    // =========================================================================
    // NaN Placement Consistency Tests
    // =========================================================================
    //
    // These tests verify that NaN placement is consistent across multiple
    // computations with the same parameters.

    #[test]
    fn test_nan_placement_consistency() {
        // Verify that running the same computation twice produces identical NaN placement
        let fixture = load_scalar_fixture("rsi_14.json");
        let period: usize = fixture.params["period"].as_u64().unwrap() as usize;

        let actual1 = rsi(&fixture.input.close, period);
        let actual2 = rsi(&fixture.input.close, period);

        let nans1 = get_actual_nan_indices(&actual1);
        let nans2 = get_actual_nan_indices(&actual2);

        assert_eq!(
            nans1, nans2,
            "RSI NaN placement is not consistent across multiple computations"
        );
    }

    // =========================================================================
    // NaN Placement Edge Case Tests
    // =========================================================================
    //
    // These tests verify NaN placement behavior for edge cases like short inputs.

    #[test]
    fn test_nan_placement_short_input() {
        // When input length is less than the warm-up period, all outputs should be NaN
        let short_close: Vec<f64> = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let period = 14;

        let actual = rsi(&short_close, period);

        // All values should be NaN since input length (5) < period (14)
        assert_eq!(
            actual.len(),
            short_close.len(),
            "Output length should match input length"
        );

        for (i, val) in actual.iter().enumerate() {
            assert!(
                val.is_nan(),
                "RSI value at index {} should be NaN for short input, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_nan_placement_exact_warmup_length() {
        // When input length equals the warm-up period, only the last value should be finite
        let fixture = load_scalar_fixture("rsi_14.json");
        let period: usize = fixture.params["period"].as_u64().unwrap() as usize;

        // Take exactly period + 1 values (minimum for one finite RSI output)
        let close: Vec<f64> = fixture.input.close[..period + 1].to_vec();
        let actual = rsi(&close, period);

        assert_eq!(actual.len(), close.len());

        // First `period` values should be NaN
        for i in 0..period {
            assert!(
                actual[i].is_nan(),
                "RSI value at index {} should be NaN during warm-up, got {}",
                i,
                actual[i]
            );
        }

        // Last value should be finite
        assert!(
            !actual[period].is_nan(),
            "RSI value at index {} should be finite after warm-up, got NaN",
            period
        );
    }
}


// =============================================================================
// Property 16: NaN Input Propagation Tests
// =============================================================================
//
// **Property 16: NaN Input Propagation**
//
// For any indicator function and any input containing NaN values, the function
// SHALL return successfully without panicking AND SHALL propagate NaN into every
// output index whose computation window includes a NaN input.
//
// **Validates: Requirements 6.8, 16.1**
//
// This test module verifies that:
// 1. All indicator functions handle NaN inputs without panicking
// 2. NaN values in inputs propagate correctly to outputs
// 3. The output length matches the input length even with NaN inputs

#[cfg(test)]
mod nan_input_propagation_tests {
    use modelenv_core::indicators::momentum::{cci, rsi};
    use modelenv_core::indicators::support::fibonacci_retracements;
    use modelenv_core::indicators::trend::{adx, ichimoku, macd, moving_average, MovingAverageKind};
    use modelenv_core::indicators::volatility::bollinger_bands;
    use proptest::prelude::*;

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /// Insert NaN values at random positions in a vector
    fn insert_nans(data: &mut [f64], nan_positions: &[usize]) {
        for &pos in nan_positions {
            if pos < data.len() {
                data[pos] = f64::NAN;
            }
        }
    }

    /// Check that NaN at position `nan_pos` propagates to output indices
    /// within the window `[nan_pos - window_size + 1, nan_pos]` (for backward-looking windows)
    fn verify_nan_propagation_backward(
        output: &[f64],
        nan_pos: usize,
        window_size: usize,
        indicator_name: &str,
    ) -> Result<(), String> {
        // For a backward-looking window of size `window_size`, a NaN at position `nan_pos`
        // should affect output indices from `nan_pos` to `nan_pos + window_size - 1`
        // (since those windows include the NaN position)
        let start = nan_pos;
        let end = (nan_pos + window_size).min(output.len());

        for i in start..end {
            if !output[i].is_nan() {
                return Err(format!(
                    "{}: Expected NaN at output index {} due to NaN input at position {}, \
                     but got finite value {}. Window size: {}",
                    indicator_name, i, nan_pos, output[i], window_size
                ));
            }
        }
        Ok(())
    }

    // =========================================================================
    // RSI NaN Input Propagation
    // =========================================================================
    //
    // RSI uses a period-based calculation. NaN inputs should not cause panics
    // and should propagate to affected output indices.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn rsi_nan_input_no_panic(
            // Generate base close prices
            close in prop::collection::vec(50.0f64..150.0f64, 20..100),
            // Generate random positions for NaN insertion
            nan_positions in prop::collection::vec(0usize..100, 1..5),
            period in 5usize..20
        ) {
            let mut close_with_nan = close.clone();
            insert_nans(&mut close_with_nan, &nan_positions);

            // This should not panic
            let result = rsi(&close_with_nan, period);

            // Output length should match input length
            prop_assert_eq!(
                result.len(),
                close_with_nan.len(),
                "RSI output length should match input length even with NaN inputs"
            );
        }
    }

    // =========================================================================
    // CCI NaN Input Propagation
    // =========================================================================
    //
    // CCI uses high, low, close inputs. NaN in any input should propagate.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn cci_nan_input_no_panic(
            // Generate base OHLC prices
            base_price in prop::collection::vec(50.0f64..150.0f64, 20..100),
            // Generate random positions for NaN insertion
            nan_positions in prop::collection::vec(0usize..100, 1..5),
            // Which input to insert NaN into (0=high, 1=low, 2=close)
            nan_target in 0usize..3,
            period in 5usize..20
        ) {
            let n = base_price.len();
            let mut high: Vec<f64> = base_price.iter().map(|&p| p + 5.0).collect();
            let mut low: Vec<f64> = base_price.iter().map(|&p| p - 5.0).collect();
            let mut close = base_price.clone();

            // Insert NaN into the selected input
            match nan_target {
                0 => insert_nans(&mut high, &nan_positions),
                1 => insert_nans(&mut low, &nan_positions),
                _ => insert_nans(&mut close, &nan_positions),
            }

            // This should not panic
            let result = cci(&high, &low, &close, period);

            // Should return Ok (not an error due to NaN)
            prop_assert!(
                result.is_ok(),
                "CCI should not return error for NaN inputs, got {:?}",
                result
            );

            let output = result.unwrap();

            // Output length should match input length
            prop_assert_eq!(
                output.len(),
                n,
                "CCI output length should match input length even with NaN inputs"
            );
        }
    }

    // =========================================================================
    // ADX NaN Input Propagation
    // =========================================================================
    //
    // ADX uses high, low, close inputs. NaN in any input should propagate.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn adx_nan_input_no_panic(
            // Generate base OHLC prices
            base_price in prop::collection::vec(50.0f64..150.0f64, 30..100),
            // Generate random positions for NaN insertion
            nan_positions in prop::collection::vec(0usize..100, 1..5),
            // Which input to insert NaN into (0=high, 1=low, 2=close)
            nan_target in 0usize..3,
            period in 5usize..14
        ) {
            let n = base_price.len();
            let mut high: Vec<f64> = base_price.iter().map(|&p| p + 5.0).collect();
            let mut low: Vec<f64> = base_price.iter().map(|&p| p - 5.0).collect();
            let mut close = base_price.clone();

            // Insert NaN into the selected input
            match nan_target {
                0 => insert_nans(&mut high, &nan_positions),
                1 => insert_nans(&mut low, &nan_positions),
                _ => insert_nans(&mut close, &nan_positions),
            }

            // This should not panic
            let result = adx(&high, &low, &close, period);

            // Should return Ok (not an error due to NaN)
            prop_assert!(
                result.is_ok(),
                "ADX should not return error for NaN inputs, got {:?}",
                result
            );

            let output = result.unwrap();

            // Output length should match input length
            prop_assert_eq!(
                output.len(),
                n,
                "ADX output length should match input length even with NaN inputs"
            );
        }
    }

    // =========================================================================
    // MACD NaN Input Propagation
    // =========================================================================
    //
    // MACD uses close prices. NaN inputs should propagate to all three outputs.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn macd_nan_input_no_panic(
            // Generate base close prices
            close in prop::collection::vec(50.0f64..150.0f64, 40..100),
            // Generate random positions for NaN insertion
            nan_positions in prop::collection::vec(0usize..100, 1..5),
            // MACD parameters (fast < slow)
            fast in 5usize..12,
            slow in 13usize..26,
            signal in 5usize..9
        ) {
            let mut close_with_nan = close.clone();
            insert_nans(&mut close_with_nan, &nan_positions);

            // This should not panic
            let result = macd(&close_with_nan, fast, slow, signal);

            // Should return Ok (not an error due to NaN)
            prop_assert!(
                result.is_ok(),
                "MACD should not return error for NaN inputs, got {:?}",
                result
            );

            let output = result.unwrap();

            // All three output vectors should have the same length as input
            prop_assert_eq!(
                output.macd.len(),
                close_with_nan.len(),
                "MACD line output length should match input length"
            );
            prop_assert_eq!(
                output.signal.len(),
                close_with_nan.len(),
                "MACD signal output length should match input length"
            );
            prop_assert_eq!(
                output.hist.len(),
                close_with_nan.len(),
                "MACD histogram output length should match input length"
            );
        }
    }

    // =========================================================================
    // Moving Average NaN Input Propagation
    // =========================================================================
    //
    // Moving averages should propagate NaN inputs to affected output windows.
    // **Validates: Requirement 6.8**

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn moving_average_nan_input_no_panic(
            // Generate base close prices
            close in prop::collection::vec(50.0f64..150.0f64, 20..100),
            // Generate random positions for NaN insertion
            nan_positions in prop::collection::vec(0usize..100, 1..5),
            // Test all MA kinds
            kind_idx in 0usize..7,
            period in 5usize..20
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

            let mut close_with_nan = close.clone();
            insert_nans(&mut close_with_nan, &nan_positions);

            // This should not panic
            let result = moving_average(&close_with_nan, kind, period);

            // Should return Ok (not an error due to NaN)
            prop_assert!(
                result.is_ok(),
                "moving_average({:?}) should not return error for NaN inputs, got {:?}",
                kind, result
            );

            let output = result.unwrap();

            // Output length should match input length
            prop_assert_eq!(
                output.len(),
                close_with_nan.len(),
                "moving_average({:?}) output length should match input length",
                kind
            );
        }

        /// Test that NaN values propagate correctly to output windows for SMA
        /// **Validates: Requirement 6.8**
        #[test]
        fn sma_nan_propagation_correct(
            // Generate base close prices (enough for meaningful test)
            close in prop::collection::vec(50.0f64..150.0f64, 30..60),
            // Single NaN position in the middle of the data
            nan_pos in 10usize..20,
            period in 5usize..10
        ) {
            prop_assume!(nan_pos < close.len());

            let mut close_with_nan = close.clone();
            close_with_nan[nan_pos] = f64::NAN;

            let result = moving_average(&close_with_nan, MovingAverageKind::Sma, period).unwrap();

            // Verify NaN propagation: for SMA, a NaN at position `nan_pos` should
            // affect output indices from `nan_pos` to `nan_pos + period - 1`
            let propagation_result = verify_nan_propagation_backward(
                &result,
                nan_pos,
                period,
                "SMA"
            );

            prop_assert!(
                propagation_result.is_ok(),
                "{}",
                propagation_result.unwrap_err()
            );
        }
    }

    // =========================================================================
    // Bollinger Bands NaN Input Propagation
    // =========================================================================
    //
    // Bollinger Bands should propagate NaN inputs to all three output bands.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn bollinger_bands_nan_input_no_panic(
            // Generate base close prices
            close in prop::collection::vec(50.0f64..150.0f64, 25..100),
            // Generate random positions for NaN insertion
            nan_positions in prop::collection::vec(0usize..100, 1..5),
            period in 5usize..20,
            nbdev in 1.0f64..3.0f64
        ) {
            let mut close_with_nan = close.clone();
            insert_nans(&mut close_with_nan, &nan_positions);

            // This should not panic
            let result = bollinger_bands(&close_with_nan, period, nbdev);

            // Should return Ok (not an error due to NaN)
            prop_assert!(
                result.is_ok(),
                "bollinger_bands should not return error for NaN inputs, got {:?}",
                result
            );

            let output = result.unwrap();

            // All three output vectors should have the same length as input
            prop_assert_eq!(
                output.upper.len(),
                close_with_nan.len(),
                "Bollinger upper band output length should match input length"
            );
            prop_assert_eq!(
                output.middle.len(),
                close_with_nan.len(),
                "Bollinger middle band output length should match input length"
            );
            prop_assert_eq!(
                output.lower.len(),
                close_with_nan.len(),
                "Bollinger lower band output length should match input length"
            );
        }
    }

    // =========================================================================
    // Ichimoku Cloud NaN Input Propagation
    // =========================================================================
    //
    // Ichimoku uses high, low, close inputs. NaN in any input should propagate.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn ichimoku_nan_input_no_panic(
            // Generate base OHLC prices
            base_price in prop::collection::vec(50.0f64..150.0f64, 60..120),
            // Generate random positions for NaN insertion
            nan_positions in prop::collection::vec(0usize..120, 1..5),
            // Which input to insert NaN into (0=high, 1=low, 2=close)
            nan_target in 0usize..3,
            // Ichimoku parameters
            tenkan in 5usize..9,
            kijun in 15usize..26,
            senkou_b_period in 30usize..52
        ) {
            let n = base_price.len();
            let mut high: Vec<f64> = base_price.iter().map(|&p| p + 5.0).collect();
            let mut low: Vec<f64> = base_price.iter().map(|&p| p - 5.0).collect();
            let mut close = base_price.clone();

            // Insert NaN into the selected input
            match nan_target {
                0 => insert_nans(&mut high, &nan_positions),
                1 => insert_nans(&mut low, &nan_positions),
                _ => insert_nans(&mut close, &nan_positions),
            }

            // This should not panic
            let result = ichimoku(&high, &low, &close, tenkan, kijun, senkou_b_period);

            // Should return Ok (not an error due to NaN)
            prop_assert!(
                result.is_ok(),
                "ichimoku should not return error for NaN inputs, got {:?}",
                result
            );

            let output = result.unwrap();

            // All five output vectors should have the same length as input
            prop_assert_eq!(
                output.tenkan.len(),
                n,
                "Ichimoku tenkan output length should match input length"
            );
            prop_assert_eq!(
                output.kijun.len(),
                n,
                "Ichimoku kijun output length should match input length"
            );
            prop_assert_eq!(
                output.senkou_a.len(),
                n,
                "Ichimoku senkou_a output length should match input length"
            );
            prop_assert_eq!(
                output.senkou_b.len(),
                n,
                "Ichimoku senkou_b output length should match input length"
            );
            prop_assert_eq!(
                output.chikou.len(),
                n,
                "Ichimoku chikou output length should match input length"
            );
        }
    }

    // =========================================================================
    // Fibonacci Retracements NaN Input Propagation
    // =========================================================================
    //
    // Fibonacci retracements use high and low inputs. NaN should propagate.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn fibonacci_nan_input_no_panic(
            // Generate base OHLC prices
            base_price in prop::collection::vec(50.0f64..150.0f64, 20..100),
            // Generate random positions for NaN insertion
            nan_positions in prop::collection::vec(0usize..100, 1..5),
            // Which input to insert NaN into (0=high, 1=low)
            nan_target in 0usize..2,
            window in 5usize..20
        ) {
            let n = base_price.len();
            let mut high: Vec<f64> = base_price.iter().map(|&p| p + 5.0).collect();
            let mut low: Vec<f64> = base_price.iter().map(|&p| p - 5.0).collect();

            // Insert NaN into the selected input
            match nan_target {
                0 => insert_nans(&mut high, &nan_positions),
                _ => insert_nans(&mut low, &nan_positions),
            }

            // This should not panic
            let result = fibonacci_retracements(&high, &low, window);

            // Should return Ok (not an error due to NaN)
            prop_assert!(
                result.is_ok(),
                "fibonacci_retracements should not return error for NaN inputs, got {:?}",
                result
            );

            let output = result.unwrap();

            // All seven output vectors should have the same length as input
            prop_assert_eq!(output.fr_000.len(), n, "fr_000 length mismatch");
            prop_assert_eq!(output.fr_236.len(), n, "fr_236 length mismatch");
            prop_assert_eq!(output.fr_382.len(), n, "fr_382 length mismatch");
            prop_assert_eq!(output.fr_500.len(), n, "fr_500 length mismatch");
            prop_assert_eq!(output.fr_618.len(), n, "fr_618 length mismatch");
            prop_assert_eq!(output.fr_786.len(), n, "fr_786 length mismatch");
            prop_assert_eq!(output.fr_1000.len(), n, "fr_1000 length mismatch");
        }

        /// Test that NaN values propagate correctly to output windows for Fibonacci
        #[test]
        fn fibonacci_nan_propagation_correct(
            // Generate base prices (enough for meaningful test)
            base_price in prop::collection::vec(50.0f64..150.0f64, 30..60),
            // Single NaN position in the middle of the data
            nan_pos in 10usize..20,
            window in 5usize..10
        ) {
            prop_assume!(nan_pos < base_price.len());

            let mut high: Vec<f64> = base_price.iter().map(|&p| p + 5.0).collect();
            let low: Vec<f64> = base_price.iter().map(|&p| p - 5.0).collect();

            high[nan_pos] = f64::NAN;

            let result = fibonacci_retracements(&high, &low, window).unwrap();

            // Verify NaN propagation for fr_000 (representative of all levels)
            let propagation_result = verify_nan_propagation_backward(
                &result.fr_000,
                nan_pos,
                window,
                "Fibonacci fr_000"
            );

            prop_assert!(
                propagation_result.is_ok(),
                "{}",
                propagation_result.unwrap_err()
            );
        }
    }

    // =========================================================================
    // Unit Tests for Specific NaN Propagation Scenarios
    // =========================================================================

    #[test]
    fn test_rsi_single_nan_in_middle() {
        let mut close: Vec<f64> = (1..=30).map(|x| 100.0 + x as f64).collect();
        close[15] = f64::NAN;

        let result = rsi(&close, 14);

        assert_eq!(result.len(), close.len());
        // Should not panic - that's the main assertion
    }

    #[test]
    fn test_cci_nan_in_high() {
        let n = 30;
        let mut high: Vec<f64> = (0..n).map(|i| 110.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 90.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        high[15] = f64::NAN;

        let result = cci(&high, &low, &close, 14);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), n);
    }

    #[test]
    fn test_adx_nan_in_low() {
        let n = 40;
        let high: Vec<f64> = (0..n).map(|i| 110.0 + i as f64).collect();
        let mut low: Vec<f64> = (0..n).map(|i| 90.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        low[20] = f64::NAN;

        let result = adx(&high, &low, &close, 14);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), n);
    }

    #[test]
    fn test_macd_multiple_nans() {
        let mut close: Vec<f64> = (1..=50).map(|x| 100.0 + x as f64).collect();
        close[10] = f64::NAN;
        close[25] = f64::NAN;
        close[40] = f64::NAN;

        let result = macd(&close, 12, 26, 9);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.macd.len(), close.len());
        assert_eq!(output.signal.len(), close.len());
        assert_eq!(output.hist.len(), close.len());
    }

    #[test]
    fn test_bollinger_bands_nan_at_start() {
        let mut close: Vec<f64> = (1..=30).map(|x| 100.0 + x as f64).collect();
        close[0] = f64::NAN;
        close[1] = f64::NAN;

        let result = bollinger_bands(&close, 20, 2.0);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.upper.len(), close.len());
        assert_eq!(output.middle.len(), close.len());
        assert_eq!(output.lower.len(), close.len());
    }

    #[test]
    fn test_ichimoku_nan_in_close() {
        let n = 80;
        let high: Vec<f64> = (0..n).map(|i| 110.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 90.0 + i as f64).collect();
        let mut close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        close[40] = f64::NAN;

        let result = ichimoku(&high, &low, &close, 9, 26, 52);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.tenkan.len(), n);
        assert_eq!(output.kijun.len(), n);
        assert_eq!(output.senkou_a.len(), n);
        assert_eq!(output.senkou_b.len(), n);
        assert_eq!(output.chikou.len(), n);
    }

    #[test]
    fn test_fibonacci_nan_in_both_inputs() {
        let n = 30;
        let mut high: Vec<f64> = (0..n).map(|i| 110.0 + i as f64).collect();
        let mut low: Vec<f64> = (0..n).map(|i| 90.0 + i as f64).collect();

        high[10] = f64::NAN;
        low[20] = f64::NAN;

        let result = fibonacci_retracements(&high, &low, 10);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.fr_000.len(), n);
        assert_eq!(output.fr_500.len(), n);
        assert_eq!(output.fr_1000.len(), n);
    }

    #[test]
    fn test_all_nan_input() {
        // Edge case: all inputs are NaN
        let close: Vec<f64> = vec![f64::NAN; 30];

        // RSI should handle all-NaN input
        let rsi_result = rsi(&close, 14);
        assert_eq!(rsi_result.len(), 30);
        assert!(rsi_result.iter().all(|&v| v.is_nan()));

        // Moving average should handle all-NaN input
        let ma_result = moving_average(&close, MovingAverageKind::Sma, 10);
        assert!(ma_result.is_ok());
        let ma_output = ma_result.unwrap();
        assert_eq!(ma_output.len(), 30);
        assert!(ma_output.iter().all(|&v| v.is_nan()));
    }

    #[test]
    fn test_alternating_nan_values() {
        // Edge case: alternating NaN and finite values
        let close: Vec<f64> = (0..30)
            .map(|i| if i % 2 == 0 { f64::NAN } else { 100.0 + i as f64 })
            .collect();

        // RSI should handle alternating NaN input
        let rsi_result = rsi(&close, 14);
        assert_eq!(rsi_result.len(), 30);

        // Moving average should handle alternating NaN input
        let ma_result = moving_average(&close, MovingAverageKind::Sma, 5);
        assert!(ma_result.is_ok());
        assert_eq!(ma_result.unwrap().len(), 30);
    }
}


// =============================================================================
// Metamorphic Property Tests (Properties 17, 18, 19)
// =============================================================================
//
// Metamorphic testing verifies mathematical invariants of indicators by checking
// that certain transformations of the input produce predictable transformations
// of the output.
//
// **Validates: Metamorphic properties**

#[cfg(test)]
mod metamorphic_property_tests {
    use modelenv_core::indicators::momentum::rsi;
    use modelenv_core::indicators::trend::{adx, macd};
    use proptest::prelude::*;

    /// Tolerance for metamorphic property comparisons.
    /// Using 1e-9 as specified in the design document.
    const METAMORPHIC_TOLERANCE: f64 = 1e-9;

    // =========================================================================
    // Property 17: RSI Scale Invariance
    // =========================================================================
    //
    // **Property 17: RSI Scale Invariance**
    //
    // For any close price slice `c` and positive constant `k`, `rsi(c, period)`
    // SHALL equal `rsi(c * k, period)` within `1e-9` tolerance at every finite index.
    //
    // **Validates: Metamorphic property for RSI**
    //
    // Mathematical basis: RSI is computed from the ratio of average gains to
    // average losses. When all prices are scaled by a positive constant k,
    // both gains and losses scale by k, but their ratio remains unchanged.
    // Therefore, RSI(c * k) = RSI(c).

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn rsi_scale_invariance(
            // Generate a vector of positive close prices (no NaN values)
            // Using positive values to simulate realistic price data
            close in prop::collection::vec(1.0f64..1000.0f64, 20..200),
            // Period must be >= 1 and less than close length
            period in 1usize..20,
            // Positive scaling constant (avoiding very small values to prevent precision issues)
            scale in 0.1f64..100.0f64
        ) {
            // Ensure we have enough data for the period
            prop_assume!(close.len() > period);
            prop_assume!(scale > 0.0);

            // Compute RSI on original prices
            let rsi_original = rsi(&close, period);

            // Scale all prices by the constant k
            let close_scaled: Vec<f64> = close.iter().map(|&p| p * scale).collect();

            // Compute RSI on scaled prices
            let rsi_scaled = rsi(&close_scaled, period);

            // Verify output lengths match
            prop_assert_eq!(
                rsi_original.len(),
                rsi_scaled.len(),
                "RSI output lengths should match: original={}, scaled={}",
                rsi_original.len(),
                rsi_scaled.len()
            );

            // Compare RSI values at every finite index
            for (i, (&orig, &scaled)) in rsi_original.iter().zip(rsi_scaled.iter()).enumerate() {
                // Both should be NaN at the same indices
                if orig.is_nan() {
                    prop_assert!(
                        scaled.is_nan(),
                        "RSI scale invariance: index {} - original is NaN but scaled is {}",
                        i, scaled
                    );
                } else if scaled.is_nan() {
                    prop_assert!(
                        orig.is_nan(),
                        "RSI scale invariance: index {} - scaled is NaN but original is {}",
                        i, orig
                    );
                } else {
                    // Both are finite - they should be equal within tolerance
                    let diff = (orig - scaled).abs();
                    prop_assert!(
                        diff <= METAMORPHIC_TOLERANCE,
                        "RSI scale invariance violated at index {}: \
                         original={}, scaled={}, diff={}, scale factor={}\n\
                         Expected RSI to be unchanged when prices are scaled.",
                        i, orig, scaled, diff, scale
                    );
                }
            }
        }
    }

    // =========================================================================
    // Property 18: ADX Scale Invariance
    // =========================================================================
    //
    // **Property 18: ADX Scale Invariance**
    //
    // For any OHLC input and positive constant `k`, scaling all prices by `k`
    // SHALL leave ADX output unchanged within `1e-9` tolerance.
    //
    // **Validates: Metamorphic property for ADX**
    //
    // Mathematical basis: ADX measures trend strength using the ratio of
    // directional movement to true range. When all prices (high, low, close)
    // are scaled by a positive constant k, both directional movement and
    // true range scale by k, but their ratio remains unchanged.
    // Therefore, ADX(OHLC * k) = ADX(OHLC).

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn adx_scale_invariance(
            // Generate base prices for OHLC data
            base_price in prop::collection::vec(50.0f64..150.0f64, 30..150),
            // Period must be >= 1
            period in 2usize..15,
            // Positive scaling constant
            scale in 0.1f64..100.0f64
        ) {
            let n = base_price.len();
            // Ensure we have enough data for ADX (needs 2*period - 1 bars)
            prop_assume!(n >= 2 * period);
            prop_assume!(scale > 0.0);

            // Generate realistic OHLC data from base prices
            // high = base + random offset, low = base - random offset, close = base
            let high: Vec<f64> = base_price.iter().map(|&p| p + 5.0).collect();
            let low: Vec<f64> = base_price.iter().map(|&p| p - 5.0).collect();
            let close: Vec<f64> = base_price.clone();

            // Compute ADX on original prices
            let adx_original = adx(&high, &low, &close, period)
                .expect("ADX computation should succeed on valid input");

            // Scale all prices by the constant k
            let high_scaled: Vec<f64> = high.iter().map(|&p| p * scale).collect();
            let low_scaled: Vec<f64> = low.iter().map(|&p| p * scale).collect();
            let close_scaled: Vec<f64> = close.iter().map(|&p| p * scale).collect();

            // Compute ADX on scaled prices
            let adx_scaled = adx(&high_scaled, &low_scaled, &close_scaled, period)
                .expect("ADX computation should succeed on scaled input");

            // Verify output lengths match
            prop_assert_eq!(
                adx_original.len(),
                adx_scaled.len(),
                "ADX output lengths should match: original={}, scaled={}",
                adx_original.len(),
                adx_scaled.len()
            );

            // Compare ADX values at every finite index
            for (i, (&orig, &scaled)) in adx_original.iter().zip(adx_scaled.iter()).enumerate() {
                // Both should be NaN at the same indices
                if orig.is_nan() {
                    prop_assert!(
                        scaled.is_nan(),
                        "ADX scale invariance: index {} - original is NaN but scaled is {}",
                        i, scaled
                    );
                } else if scaled.is_nan() {
                    prop_assert!(
                        orig.is_nan(),
                        "ADX scale invariance: index {} - scaled is NaN but original is {}",
                        i, orig
                    );
                } else {
                    // Both are finite - they should be equal within tolerance
                    let diff = (orig - scaled).abs();
                    prop_assert!(
                        diff <= METAMORPHIC_TOLERANCE,
                        "ADX scale invariance violated at index {}: \
                         original={}, scaled={}, diff={}, scale factor={}\n\
                         Expected ADX to be unchanged when all prices are scaled.",
                        i, orig, scaled, diff, scale
                    );
                }
            }
        }
    }

    // =========================================================================
    // Property 19: MACD Translation Invariance
    // =========================================================================
    //
    // **Property 19: MACD Translation Invariance**
    //
    // For any close price slice `c` and constant `k`, adding `k` to every close
    // price SHALL leave the MACD histogram unchanged within `1e-9` tolerance.
    //
    // **Validates: Metamorphic property for MACD**
    //
    // Mathematical basis: MACD is computed as the difference between fast and
    // slow EMAs. The histogram is MACD - signal. When a constant k is added to
    // all prices:
    // - EMA(c + k) = EMA(c) + k (EMA is linear with respect to translation)
    // - MACD(c + k) = fast_EMA(c + k) - slow_EMA(c + k)
    //               = (fast_EMA(c) + k) - (slow_EMA(c) + k)
    //               = fast_EMA(c) - slow_EMA(c)
    //               = MACD(c)
    // - Similarly, signal(c + k) = signal(c) + k, but since MACD is unchanged,
    //   the signal line computation on MACD values is also unchanged.
    // - hist(c + k) = MACD(c + k) - signal(c + k) = MACD(c) - signal(c) = hist(c)
    //
    // Therefore, the histogram should be invariant under translation.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn macd_translation_invariance(
            // Generate a vector of positive close prices
            close in prop::collection::vec(50.0f64..150.0f64, 40..200),
            // MACD parameters: fast < slow
            fast in 5usize..12,
            slow in 13usize..30,
            signal in 5usize..12,
            // Translation constant (can be positive or negative)
            translation in -1000.0f64..1000.0f64
        ) {
            // Ensure we have enough data for MACD
            prop_assume!(close.len() >= slow + signal);
            prop_assume!(fast < slow);

            // Compute MACD on original prices
            let macd_original = macd(&close, fast, slow, signal)
                .expect("MACD computation should succeed on valid input");

            // Translate all prices by the constant k
            let close_translated: Vec<f64> = close.iter().map(|&p| p + translation).collect();

            // Compute MACD on translated prices
            let macd_translated = macd(&close_translated, fast, slow, signal)
                .expect("MACD computation should succeed on translated input");

            // Verify output lengths match
            prop_assert_eq!(
                macd_original.hist.len(),
                macd_translated.hist.len(),
                "MACD histogram lengths should match: original={}, translated={}",
                macd_original.hist.len(),
                macd_translated.hist.len()
            );

            // Compare histogram values at every finite index
            // The histogram should be unchanged by translation
            for (i, (&orig, &translated)) in macd_original.hist.iter()
                .zip(macd_translated.hist.iter())
                .enumerate()
            {
                // Both should be NaN at the same indices
                if orig.is_nan() {
                    prop_assert!(
                        translated.is_nan(),
                        "MACD histogram translation invariance: index {} - original is NaN but translated is {}",
                        i, translated
                    );
                } else if translated.is_nan() {
                    prop_assert!(
                        orig.is_nan(),
                        "MACD histogram translation invariance: index {} - translated is NaN but original is {}",
                        i, orig
                    );
                } else {
                    // Both are finite - they should be equal within tolerance
                    let diff = (orig - translated).abs();
                    prop_assert!(
                        diff <= METAMORPHIC_TOLERANCE,
                        "MACD histogram translation invariance violated at index {}: \
                         original={}, translated={}, diff={}, translation={}\n\
                         Expected MACD histogram to be unchanged when prices are translated.",
                        i, orig, translated, diff, translation
                    );
                }
            }

            // Additionally verify that MACD line itself is unchanged
            // (since MACD = fast_EMA - slow_EMA, and both EMAs shift by the same amount)
            for (i, (&orig, &translated)) in macd_original.macd.iter()
                .zip(macd_translated.macd.iter())
                .enumerate()
            {
                if !orig.is_nan() && !translated.is_nan() {
                    let diff = (orig - translated).abs();
                    prop_assert!(
                        diff <= METAMORPHIC_TOLERANCE,
                        "MACD line translation invariance violated at index {}: \
                         original={}, translated={}, diff={}, translation={}\n\
                         Expected MACD line to be unchanged when prices are translated.",
                        i, orig, translated, diff, translation
                    );
                }
            }

            // And verify signal line is unchanged
            for (i, (&orig, &translated)) in macd_original.signal.iter()
                .zip(macd_translated.signal.iter())
                .enumerate()
            {
                if !orig.is_nan() && !translated.is_nan() {
                    let diff = (orig - translated).abs();
                    prop_assert!(
                        diff <= METAMORPHIC_TOLERANCE,
                        "MACD signal translation invariance violated at index {}: \
                         original={}, translated={}, diff={}, translation={}\n\
                         Expected MACD signal to be unchanged when prices are translated.",
                        i, orig, translated, diff, translation
                    );
                }
            }
        }
    }

    // =========================================================================
    // Unit Tests for Metamorphic Properties
    // =========================================================================
    //
    // These unit tests provide specific examples to verify the metamorphic
    // properties work correctly with known inputs.

    #[test]
    fn test_rsi_scale_invariance_specific() {
        // Test with a specific known input
        let close: Vec<f64> = vec![
            44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5, 44.0, 44.5,
            45.0, 44.75, 45.25, 45.0, 44.5, 44.0, 44.25, 44.75, 45.0, 45.5,
        ];
        let period = 5;
        let scale = 2.0;

        let rsi_original = rsi(&close, period);
        let close_scaled: Vec<f64> = close.iter().map(|&p| p * scale).collect();
        let rsi_scaled = rsi(&close_scaled, period);

        assert_eq!(rsi_original.len(), rsi_scaled.len());

        for (i, (&orig, &scaled)) in rsi_original.iter().zip(rsi_scaled.iter()).enumerate() {
            if !orig.is_nan() && !scaled.is_nan() {
                let diff = (orig - scaled).abs();
                assert!(
                    diff <= METAMORPHIC_TOLERANCE,
                    "RSI scale invariance failed at index {}: orig={}, scaled={}, diff={}",
                    i, orig, scaled, diff
                );
            }
        }
    }

    #[test]
    fn test_adx_scale_invariance_specific() {
        // Test with a specific known input
        let high: Vec<f64> = vec![
            45.0, 45.5, 46.0, 45.5, 46.0, 45.5, 45.0, 44.5, 45.0, 45.5,
            46.0, 45.5, 46.0, 45.5, 45.0, 44.5, 45.0, 45.5, 46.0, 45.5,
            46.0, 45.5, 46.0, 45.5, 45.0, 44.5, 45.0, 45.5, 46.0, 45.5,
        ];
        let low: Vec<f64> = vec![
            44.0, 44.5, 45.0, 44.5, 45.0, 44.5, 44.0, 43.5, 44.0, 44.5,
            45.0, 44.5, 45.0, 44.5, 44.0, 43.5, 44.0, 44.5, 45.0, 44.5,
            45.0, 44.5, 45.0, 44.5, 44.0, 43.5, 44.0, 44.5, 45.0, 44.5,
        ];
        let close: Vec<f64> = vec![
            44.5, 45.0, 45.5, 45.0, 45.5, 45.0, 44.5, 44.0, 44.5, 45.0,
            45.5, 45.0, 45.5, 45.0, 44.5, 44.0, 44.5, 45.0, 45.5, 45.0,
            45.5, 45.0, 45.5, 45.0, 44.5, 44.0, 44.5, 45.0, 45.5, 45.0,
        ];
        let period = 5;
        let scale = 10.0;

        let adx_original = adx(&high, &low, &close, period).unwrap();
        let high_scaled: Vec<f64> = high.iter().map(|&p| p * scale).collect();
        let low_scaled: Vec<f64> = low.iter().map(|&p| p * scale).collect();
        let close_scaled: Vec<f64> = close.iter().map(|&p| p * scale).collect();
        let adx_scaled = adx(&high_scaled, &low_scaled, &close_scaled, period).unwrap();

        assert_eq!(adx_original.len(), adx_scaled.len());

        for (i, (&orig, &scaled)) in adx_original.iter().zip(adx_scaled.iter()).enumerate() {
            if !orig.is_nan() && !scaled.is_nan() {
                let diff = (orig - scaled).abs();
                assert!(
                    diff <= METAMORPHIC_TOLERANCE,
                    "ADX scale invariance failed at index {}: orig={}, scaled={}, diff={}",
                    i, orig, scaled, diff
                );
            }
        }
    }

    #[test]
    fn test_macd_translation_invariance_specific() {
        // Test with a specific known input
        let close: Vec<f64> = (1..=50).map(|x| 100.0 + x as f64).collect();
        let fast = 12;
        let slow = 26;
        let signal = 9;
        let translation = 500.0;

        let macd_original = macd(&close, fast, slow, signal).unwrap();
        let close_translated: Vec<f64> = close.iter().map(|&p| p + translation).collect();
        let macd_translated = macd(&close_translated, fast, slow, signal).unwrap();

        assert_eq!(macd_original.hist.len(), macd_translated.hist.len());

        // Check histogram invariance
        for (i, (&orig, &translated)) in macd_original.hist.iter()
            .zip(macd_translated.hist.iter())
            .enumerate()
        {
            if !orig.is_nan() && !translated.is_nan() {
                let diff = (orig - translated).abs();
                assert!(
                    diff <= METAMORPHIC_TOLERANCE,
                    "MACD histogram translation invariance failed at index {}: orig={}, translated={}, diff={}",
                    i, orig, translated, diff
                );
            }
        }

        // Check MACD line invariance
        for (i, (&orig, &translated)) in macd_original.macd.iter()
            .zip(macd_translated.macd.iter())
            .enumerate()
        {
            if !orig.is_nan() && !translated.is_nan() {
                let diff = (orig - translated).abs();
                assert!(
                    diff <= METAMORPHIC_TOLERANCE,
                    "MACD line translation invariance failed at index {}: orig={}, translated={}, diff={}",
                    i, orig, translated, diff
                );
            }
        }
    }

    #[test]
    fn test_rsi_scale_invariance_with_small_scale() {
        // Test with a very small scale factor
        let close: Vec<f64> = (1..=30).map(|x| 100.0 + x as f64).collect();
        let period = 14;
        let scale = 0.001;

        let rsi_original = rsi(&close, period);
        let close_scaled: Vec<f64> = close.iter().map(|&p| p * scale).collect();
        let rsi_scaled = rsi(&close_scaled, period);

        for (i, (&orig, &scaled)) in rsi_original.iter().zip(rsi_scaled.iter()).enumerate() {
            if !orig.is_nan() && !scaled.is_nan() {
                let diff = (orig - scaled).abs();
                assert!(
                    diff <= METAMORPHIC_TOLERANCE,
                    "RSI scale invariance with small scale failed at index {}: orig={}, scaled={}, diff={}",
                    i, orig, scaled, diff
                );
            }
        }
    }

    #[test]
    fn test_macd_translation_invariance_with_negative_translation() {
        // Test with a negative translation
        let close: Vec<f64> = (1..=50).map(|x| 100.0 + x as f64).collect();
        let fast = 12;
        let slow = 26;
        let signal = 9;
        let translation = -50.0;

        let macd_original = macd(&close, fast, slow, signal).unwrap();
        let close_translated: Vec<f64> = close.iter().map(|&p| p + translation).collect();
        let macd_translated = macd(&close_translated, fast, slow, signal).unwrap();

        for (i, (&orig, &translated)) in macd_original.hist.iter()
            .zip(macd_translated.hist.iter())
            .enumerate()
        {
            if !orig.is_nan() && !translated.is_nan() {
                let diff = (orig - translated).abs();
                assert!(
                    diff <= METAMORPHIC_TOLERANCE,
                    "MACD histogram translation invariance with negative translation failed at index {}: orig={}, translated={}, diff={}",
                    i, orig, translated, diff
                );
            }
        }
    }
}


// =============================================================================
// Property 12: Pattern Detection Parity Tests
// =============================================================================
//
// These tests verify that Rust pattern detection implementations match Python
// `ta/` package outputs.
//
// **Validates: Requirements 10.12, 11.13**
//
// Pattern detection tests verify:
// - Same number of patterns in same order
// - Exact equality on integer fields (idx1, idx2, neckline_idx, width_bars)
// - Exact equality on boolean fields (confirmed)
// - Exact equality on timestamp fields (ts1, ts2, min_before_ts, max_before_ts, min_after_ts, max_after_ts)
// - 1e-9 tolerance on floating-point fields (low1/high1, low2/high2, neckline, depth_pct,
//   min_before_val, max_before_val, min_after_val, max_after_val)

#[cfg(test)]
mod pattern_parity_tests {
    use super::*;
    use modelenv_core::indicators::patterns::{detect_double_bottoms, detect_double_tops};

    // =========================================================================
    // Double Bottom Parity Test
    // =========================================================================
    //
    // **Property 12: Pattern Detection Parity - Double Bottom**
    // **Validates: Requirement 10.12**
    //
    // WHEN `bars` is fed into both the Rust `detect_double_bottoms` and the
    // Python `detect_double_bottoms` with identical `(low, high, close, timestamp)`
    // series and the same `window`, `tolerance_pct`, `min_width`, THE Indicators_Module
    // SHALL produce the same number of patterns in the same order, with exact equality
    // on `idx1`, `idx2`, `ts1`, `ts2`, `neckline_idx`, `width_bars`, and `confirmed`,
    // and `1e-9` absolute parity on `low1`, `low2`, `neckline`, `depth_pct`,
    // `min_before_val`, `max_before_val`, `min_after_val`, `max_after_val`.

    #[test]
    fn test_double_bottom_parity() {
        let fixture = load_pattern_fixture("double_bottom.json");
        let bars = fixture.input.to_bars();

        // Compute double bottom patterns using Rust implementation
        let actual = detect_double_bottoms(
            &bars,
            fixture.params.window,
            fixture.params.tolerance_pct,
            fixture.params.min_width,
        )
        .expect("Double bottom detection should succeed");

        // Verify same number of patterns
        assert_eq!(
            actual.patterns.len(),
            fixture.output.patterns.len(),
            "Double bottom pattern count mismatch: Rust found {} patterns, Python found {}",
            actual.patterns.len(),
            fixture.output.patterns.len()
        );

        // Compare each pattern
        for (i, (rust_pattern, py_pattern)) in actual
            .patterns
            .iter()
            .zip(fixture.output.patterns.iter())
            .enumerate()
        {
            // Exact equality on integer fields
            assert_eq!(
                rust_pattern.idx1, py_pattern.idx1,
                "Pattern {} idx1 mismatch: Rust={}, Python={}",
                i, rust_pattern.idx1, py_pattern.idx1
            );
            assert_eq!(
                rust_pattern.idx2, py_pattern.idx2,
                "Pattern {} idx2 mismatch: Rust={}, Python={}",
                i, rust_pattern.idx2, py_pattern.idx2
            );
            assert_eq!(
                rust_pattern.neckline_idx, py_pattern.neckline_idx,
                "Pattern {} neckline_idx mismatch: Rust={}, Python={}",
                i, rust_pattern.neckline_idx, py_pattern.neckline_idx
            );
            assert_eq!(
                rust_pattern.width_bars, py_pattern.width_bars,
                "Pattern {} width_bars mismatch: Rust={}, Python={}",
                i, rust_pattern.width_bars, py_pattern.width_bars
            );

            // Exact equality on boolean fields
            assert_eq!(
                rust_pattern.confirmed, py_pattern.confirmed,
                "Pattern {} confirmed mismatch: Rust={}, Python={}",
                i, rust_pattern.confirmed, py_pattern.confirmed
            );

            // Exact equality on timestamp fields
            assert_eq!(
                rust_pattern.ts1, py_pattern.ts1,
                "Pattern {} ts1 mismatch: Rust={}, Python={}",
                i, rust_pattern.ts1, py_pattern.ts1
            );
            assert_eq!(
                rust_pattern.ts2, py_pattern.ts2,
                "Pattern {} ts2 mismatch: Rust={}, Python={}",
                i, rust_pattern.ts2, py_pattern.ts2
            );

            // 1e-9 tolerance on floating-point fields
            let low1_expected = py_pattern.low1.expect("Double bottom should have low1");
            assert!(
                (rust_pattern.low1 - low1_expected).abs() <= PARITY_TOLERANCE,
                "Pattern {} low1 mismatch: Rust={}, Python={}, diff={}",
                i, rust_pattern.low1, low1_expected, (rust_pattern.low1 - low1_expected).abs()
            );

            let low2_expected = py_pattern.low2.expect("Double bottom should have low2");
            assert!(
                (rust_pattern.low2 - low2_expected).abs() <= PARITY_TOLERANCE,
                "Pattern {} low2 mismatch: Rust={}, Python={}, diff={}",
                i, rust_pattern.low2, low2_expected, (rust_pattern.low2 - low2_expected).abs()
            );

            assert!(
                (rust_pattern.neckline - py_pattern.neckline).abs() <= PARITY_TOLERANCE,
                "Pattern {} neckline mismatch: Rust={}, Python={}, diff={}",
                i, rust_pattern.neckline, py_pattern.neckline, (rust_pattern.neckline - py_pattern.neckline).abs()
            );

            assert!(
                (rust_pattern.depth_pct - py_pattern.depth_pct).abs() <= PARITY_TOLERANCE,
                "Pattern {} depth_pct mismatch: Rust={}, Python={}, diff={}",
                i, rust_pattern.depth_pct, py_pattern.depth_pct, (rust_pattern.depth_pct - py_pattern.depth_pct).abs()
            );

            // Compare optional floating-point fields (min_before_val, max_before_val, min_after_val, max_after_val)
            compare_optional_f64(
                rust_pattern.min_before_val,
                py_pattern.min_before_val,
                &format!("Pattern {} min_before_val", i),
            );
            compare_optional_f64(
                rust_pattern.max_before_val,
                py_pattern.max_before_val,
                &format!("Pattern {} max_before_val", i),
            );
            compare_optional_f64(
                rust_pattern.min_after_val,
                py_pattern.min_after_val,
                &format!("Pattern {} min_after_val", i),
            );
            compare_optional_f64(
                rust_pattern.max_after_val,
                py_pattern.max_after_val,
                &format!("Pattern {} max_after_val", i),
            );

            // Compare optional timestamp fields
            compare_optional_i64(
                rust_pattern.min_before_ts,
                py_pattern.min_before_ts,
                &format!("Pattern {} min_before_ts", i),
            );
            compare_optional_i64(
                rust_pattern.max_before_ts,
                py_pattern.max_before_ts,
                &format!("Pattern {} max_before_ts", i),
            );
            compare_optional_i64(
                rust_pattern.min_after_ts,
                py_pattern.min_after_ts,
                &format!("Pattern {} min_after_ts", i),
            );
            compare_optional_i64(
                rust_pattern.max_after_ts,
                py_pattern.max_after_ts,
                &format!("Pattern {} max_after_ts", i),
            );
        }

        // Compare latest_min and latest_max
        compare_optional_f64(
            actual.latest_min,
            fixture.output.latest_min,
            "latest_min",
        );
        compare_optional_f64(
            actual.latest_max,
            fixture.output.latest_max,
            "latest_max",
        );
    }

    // =========================================================================
    // Double Top Parity Test
    // =========================================================================
    //
    // **Property 12: Pattern Detection Parity - Double Top**
    // **Validates: Requirement 11.13**
    //
    // WHEN `bars` is fed into both the Rust `detect_double_tops` and the
    // Python `detect_double_tops` with identical `(low, high, close, timestamp)`
    // series and the same `window`, `tolerance_pct`, `min_width`, THE Indicators_Module
    // SHALL produce the same number of patterns in the same order, with exact equality
    // on `idx1`, `idx2`, `ts1`, `ts2`, `neckline_idx`, `width_bars`, and `confirmed`,
    // and `1e-9` absolute parity on `high1`, `high2`, `neckline`, `depth_pct`,
    // `min_before_val`, `max_before_val`, `min_after_val`, `max_after_val`.

    #[test]
    fn test_double_top_parity() {
        let fixture = load_pattern_fixture("double_top.json");
        let bars = fixture.input.to_bars();

        // Compute double top patterns using Rust implementation
        let actual = detect_double_tops(
            &bars,
            fixture.params.window,
            fixture.params.tolerance_pct,
            fixture.params.min_width,
        )
        .expect("Double top detection should succeed");

        // Verify same number of patterns
        assert_eq!(
            actual.patterns.len(),
            fixture.output.patterns.len(),
            "Double top pattern count mismatch: Rust found {} patterns, Python found {}",
            actual.patterns.len(),
            fixture.output.patterns.len()
        );

        // Compare each pattern
        for (i, (rust_pattern, py_pattern)) in actual
            .patterns
            .iter()
            .zip(fixture.output.patterns.iter())
            .enumerate()
        {
            // Exact equality on integer fields
            assert_eq!(
                rust_pattern.idx1, py_pattern.idx1,
                "Pattern {} idx1 mismatch: Rust={}, Python={}",
                i, rust_pattern.idx1, py_pattern.idx1
            );
            assert_eq!(
                rust_pattern.idx2, py_pattern.idx2,
                "Pattern {} idx2 mismatch: Rust={}, Python={}",
                i, rust_pattern.idx2, py_pattern.idx2
            );
            assert_eq!(
                rust_pattern.neckline_idx, py_pattern.neckline_idx,
                "Pattern {} neckline_idx mismatch: Rust={}, Python={}",
                i, rust_pattern.neckline_idx, py_pattern.neckline_idx
            );
            assert_eq!(
                rust_pattern.width_bars, py_pattern.width_bars,
                "Pattern {} width_bars mismatch: Rust={}, Python={}",
                i, rust_pattern.width_bars, py_pattern.width_bars
            );

            // Exact equality on boolean fields
            assert_eq!(
                rust_pattern.confirmed, py_pattern.confirmed,
                "Pattern {} confirmed mismatch: Rust={}, Python={}",
                i, rust_pattern.confirmed, py_pattern.confirmed
            );

            // Exact equality on timestamp fields
            assert_eq!(
                rust_pattern.ts1, py_pattern.ts1,
                "Pattern {} ts1 mismatch: Rust={}, Python={}",
                i, rust_pattern.ts1, py_pattern.ts1
            );
            assert_eq!(
                rust_pattern.ts2, py_pattern.ts2,
                "Pattern {} ts2 mismatch: Rust={}, Python={}",
                i, rust_pattern.ts2, py_pattern.ts2
            );

            // 1e-9 tolerance on floating-point fields
            let high1_expected = py_pattern.high1.expect("Double top should have high1");
            assert!(
                (rust_pattern.high1 - high1_expected).abs() <= PARITY_TOLERANCE,
                "Pattern {} high1 mismatch: Rust={}, Python={}, diff={}",
                i, rust_pattern.high1, high1_expected, (rust_pattern.high1 - high1_expected).abs()
            );

            let high2_expected = py_pattern.high2.expect("Double top should have high2");
            assert!(
                (rust_pattern.high2 - high2_expected).abs() <= PARITY_TOLERANCE,
                "Pattern {} high2 mismatch: Rust={}, Python={}, diff={}",
                i, rust_pattern.high2, high2_expected, (rust_pattern.high2 - high2_expected).abs()
            );

            assert!(
                (rust_pattern.neckline - py_pattern.neckline).abs() <= PARITY_TOLERANCE,
                "Pattern {} neckline mismatch: Rust={}, Python={}, diff={}",
                i, rust_pattern.neckline, py_pattern.neckline, (rust_pattern.neckline - py_pattern.neckline).abs()
            );

            assert!(
                (rust_pattern.depth_pct - py_pattern.depth_pct).abs() <= PARITY_TOLERANCE,
                "Pattern {} depth_pct mismatch: Rust={}, Python={}, diff={}",
                i, rust_pattern.depth_pct, py_pattern.depth_pct, (rust_pattern.depth_pct - py_pattern.depth_pct).abs()
            );

            // Compare optional floating-point fields (min_before_val, max_before_val, min_after_val, max_after_val)
            compare_optional_f64(
                rust_pattern.min_before_val,
                py_pattern.min_before_val,
                &format!("Pattern {} min_before_val", i),
            );
            compare_optional_f64(
                rust_pattern.max_before_val,
                py_pattern.max_before_val,
                &format!("Pattern {} max_before_val", i),
            );
            compare_optional_f64(
                rust_pattern.min_after_val,
                py_pattern.min_after_val,
                &format!("Pattern {} min_after_val", i),
            );
            compare_optional_f64(
                rust_pattern.max_after_val,
                py_pattern.max_after_val,
                &format!("Pattern {} max_after_val", i),
            );

            // Compare optional timestamp fields
            compare_optional_i64(
                rust_pattern.min_before_ts,
                py_pattern.min_before_ts,
                &format!("Pattern {} min_before_ts", i),
            );
            compare_optional_i64(
                rust_pattern.max_before_ts,
                py_pattern.max_before_ts,
                &format!("Pattern {} max_before_ts", i),
            );
            compare_optional_i64(
                rust_pattern.min_after_ts,
                py_pattern.min_after_ts,
                &format!("Pattern {} min_after_ts", i),
            );
            compare_optional_i64(
                rust_pattern.max_after_ts,
                py_pattern.max_after_ts,
                &format!("Pattern {} max_after_ts", i),
            );
        }

        // Compare latest_min and latest_max
        compare_optional_f64(
            actual.latest_min,
            fixture.output.latest_min,
            "latest_min",
        );
        compare_optional_f64(
            actual.latest_max,
            fixture.output.latest_max,
            "latest_max",
        );
    }

    /// Helper function to compare optional f64 values with tolerance.
    fn compare_optional_f64(rust_val: Option<f64>, py_val: Option<f64>, field_name: &str) {
        match (rust_val, py_val) {
            (Some(r), Some(p)) => {
                assert!(
                    (r - p).abs() <= PARITY_TOLERANCE,
                    "{} mismatch: Rust={}, Python={}, diff={}",
                    field_name, r, p, (r - p).abs()
                );
            }
            (None, None) => {
                // Both None - OK
            }
            (Some(r), None) => {
                panic!("{} mismatch: Rust=Some({}), Python=None", field_name, r);
            }
            (None, Some(p)) => {
                panic!("{} mismatch: Rust=None, Python=Some({})", field_name, p);
            }
        }
    }

    /// Helper function to compare optional i64 values (exact equality).
    fn compare_optional_i64(rust_val: Option<i64>, py_val: Option<i64>, field_name: &str) {
        match (rust_val, py_val) {
            (Some(r), Some(p)) => {
                assert_eq!(
                    r, p,
                    "{} mismatch: Rust={}, Python={}",
                    field_name, r, p
                );
            }
            (None, None) => {
                // Both None - OK
            }
            (Some(r), None) => {
                panic!("{} mismatch: Rust=Some({}), Python=None", field_name, r);
            }
            (None, Some(p)) => {
                panic!("{} mismatch: Rust=None, Python=Some({})", field_name, p);
            }
        }
    }
}
