pub mod live;
pub mod sim;
pub mod training;

use crate::proto;
use anyhow::Result;
use async_trait::async_trait;

pub struct StepOutcome {
    pub observation: proto::Observation,
    pub reward: f64,
    pub done: bool,
    pub info: String,
}

#[async_trait]
pub trait Environment {
    async fn reset(&mut self, req: proto::ResetRequest) -> Result<proto::Observation>;
    async fn step(&mut self, action: proto::Action) -> Result<StepOutcome>;
    async fn observe(&self) -> Result<proto::Observation>;
}

/// USD/JPY quotes have 3 decimal places. Other majors have 5.
/// Rounding is required to avoid emitting invented precision beyond what the
/// venue actually quotes.
pub fn round_price(symbol: &str, price: f64) -> f64 {
    let decimals = if symbol.eq_ignore_ascii_case("USDJPY") || symbol.contains("JPY") {
        3
    } else {
        5
    };
    let scale = 10f64.powi(decimals);
    (price * scale).round() / scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usdjpy_rounds_to_three_decimals() {
        assert!((round_price("USDJPY", 150.123456) - 150.123).abs() < 1e-9);
        assert!((round_price("usdjpy", 150.1239) - 150.124).abs() < 1e-9);
    }

    #[test]
    fn non_jpy_rounds_to_five_decimals() {
        assert!((round_price("EURUSD", 1.0987654) - 1.09877).abs() < 1e-9);
    }
}
