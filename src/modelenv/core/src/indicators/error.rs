/// Unified error type for all indicator functions.
#[derive(Debug, Clone, PartialEq)]
pub enum IndicatorError {
    /// Input slices have different lengths.
    LengthMismatch {
        expected: usize,
        actual: usize,
        param_name: &'static str,
    },
    /// Period or window parameter is invalid (e.g., zero).
    InvalidPeriod {
        param_name: &'static str,
        value: usize,
        reason: &'static str,
    },
    /// Input slice is empty where non-empty is required.
    EmptyInput,
    /// TA-Lib returned a non-zero error code.
    TalibFailure { code: i32 },
}

impl std::fmt::Display for IndicatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndicatorError::LengthMismatch {
                expected,
                actual,
                param_name,
            } => write!(
                f,
                "length mismatch for '{}': expected {}, got {}",
                param_name, expected, actual
            ),
            IndicatorError::InvalidPeriod {
                param_name,
                value,
                reason,
            } => write!(
                f,
                "invalid period for '{}': {} ({})",
                param_name, value, reason
            ),
            IndicatorError::EmptyInput => write!(f, "input slice is empty"),
            IndicatorError::TalibFailure { code } => {
                write!(f, "TA-Lib returned error code {}", code)
            }
        }
    }
}

impl std::error::Error for IndicatorError {}
