// Position management module
use serde::{Deserialize, Serialize};

/// Number of nanoseconds in a day (86,400,000,000,000)
pub const NANOS_PER_DAY: i64 = 86_400_000_000_000;

/// Number of days in 12 months (approximately 365.25 * 12)
pub const DAYS_IN_12_MONTHS: i64 = 365 * 12 + 3; // Account for leap years

/// Position side (BUY or SELL)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

/// Represents a trading position with P/L tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub position_id: String,
    pub entry_price: f64,
    pub unrealised_pnl: f64,
    pub swap: f64,
    pub open_timestamp_ns: i64,
    pub volume: f64,
    pub side: Side,
    pub last_swap_accrual_ns: i64,
}

/// Represents a closed position with realised P/L
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosedPosition {
    pub position_id: String,
    pub entry_price: f64,
    pub close_price: f64,
    pub volume: f64,
    pub side: Side,
    pub realised_pnl: f64,
    pub swap: f64,
    pub open_timestamp_ns: i64,
    pub close_timestamp_ns: i64,
}

impl Position {
    /// Create a new position with entry price calculated from mid_price + half spread
    pub fn new(
        position_id: String,
        mid_price: f64,
        spread: f64,
        volume: f64,
        side: Side,
        open_timestamp_ns: i64,
    ) -> Self {
        // Entry price = mid_price + half the spread
        let entry_price = mid_price + (spread / 2.0);
        
        Position {
            position_id,
            entry_price,
            unrealised_pnl: 0.0,
            swap: 0.0,
            open_timestamp_ns,
            volume,
            side,
            last_swap_accrual_ns: open_timestamp_ns,
        }
    }

    /// Calculate unrealised P/L based on current mid price
    /// P/L = (current_mid_price - entry_price) * volume
    /// For BUY: positive P/L when price increases
    /// For SELL: positive P/L when price decreases
    pub fn calculate_unrealised_pnl(&self, current_mid_price: f64) -> f64 {
        let price_diff = match self.side {
            Side::Buy => current_mid_price - self.entry_price,
            Side::Sell => self.entry_price - current_mid_price,
        };
        price_diff * self.volume
    }

    /// Calculate realised P/L when closing a position
    /// P/L = (close_price - entry_price) * volume - transaction_cost
    /// For BUY: close_price > entry_price = profit
    /// For SELL: close_price < entry_price = profit
    pub fn calculate_realised_pnl(&self, close_price: f64, transaction_cost: f64) -> f64 {
        let gross_pnl = match self.side {
            Side::Buy => (close_price - self.entry_price) * self.volume,
            Side::Sell => (self.entry_price - close_price) * self.volume,
        };
        gross_pnl - transaction_cost
    }

    /// Accrue swap for the position based on elapsed time and daily swap rate
    /// Swap = swap_rate * days_elapsed * volume
    /// Only accrues when a day boundary is crossed
    pub fn accrue_swap(&mut self, current_timestamp_ns: i64, daily_swap_rate: f64) -> bool {
        let elapsed_days = (current_timestamp_ns - self.last_swap_accrual_ns) as f64 / NANOS_PER_DAY as f64;
        
        if elapsed_days >= 1.0 {
            // Only accrue if at least one day has elapsed
            // Use floor to get the number of full days elapsed
            let full_days = elapsed_days.floor();
            self.swap += daily_swap_rate * full_days * self.volume;
            self.last_swap_accrual_ns = current_timestamp_ns;
            true
        } else {
            false
        }
    }

    /// Convert to proto Position
    pub fn to_proto(&self) -> modelenv_proto::Position {
        modelenv_proto::Position {
            position_id: self.position_id.clone(),
            entry_price: self.entry_price,
            unrealised_pnl: self.unrealised_pnl,
            swap: self.swap,
            open_timestamp_ns: self.open_timestamp_ns,
        }
    }

    /// Convert position to closed position record
    pub fn to_closed_position(
        &self,
        close_price: f64,
        close_timestamp_ns: i64,
        transaction_cost: f64,
    ) -> ClosedPosition {
        let realised_pnl = self.calculate_realised_pnl(close_price, transaction_cost);
        
        ClosedPosition {
            position_id: self.position_id.clone(),
            entry_price: self.entry_price,
            close_price,
            volume: self.volume,
            side: self.side,
            realised_pnl,
            swap: self.swap,
            open_timestamp_ns: self.open_timestamp_ns,
            close_timestamp_ns,
        }
    }
}

impl ClosedPosition {
    /// Calculate total P/L including swap
    pub fn total_pnl(&self) -> f64 {
        self.realised_pnl + self.swap
    }

    /// Check if the closed position is within the 12-month window
    pub fn is_within_12_months(&self, current_timestamp_ns: i64) -> bool {
        let age_days = (current_timestamp_ns - self.close_timestamp_ns) as f64 / NANOS_PER_DAY as f64;
        age_days <= DAYS_IN_12_MONTHS as f64
    }
}

/// Manages a rolling 12-month window of closed positions
#[derive(Debug, Clone, Default)]
pub struct ClosedPositionWindow {
    closed_positions: Vec<ClosedPosition>,
}

impl ClosedPositionWindow {
    /// Create a new empty window
    pub fn new() -> Self {
        ClosedPositionWindow {
            closed_positions: Vec::new(),
        }
    }

    /// Add a closed position to the window
    pub fn add_closed_position(&mut self, closed_position: ClosedPosition) {
        self.closed_positions.push(closed_position);
    }

    /// Calculate total realised P/L (including swap) within the 12-month window
    pub fn total_realised_pnl_12m(&self, current_timestamp_ns: i64) -> f64 {
        self.closed_positions
            .iter()
            .filter(|p| p.is_within_12_months(current_timestamp_ns))
            .map(|p| p.total_pnl())
            .sum()
    }

    /// Remove closed positions older than 12 months
    pub fn prune_old_positions(&mut self, current_timestamp_ns: i64) {
        self.closed_positions.retain(|p| p.is_within_12_months(current_timestamp_ns));
    }

    /// Get the number of closed positions in the window
    pub fn len(&self) -> usize {
        self.closed_positions.len()
    }

    /// Check if the window is empty
    pub fn is_empty(&self) -> bool {
        self.closed_positions.is_empty()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_creation() {
        let position = Position::new(
            "pos_1".to_string(),
            150.0, // mid_price
            0.0001, // spread
            1.0, // volume
            Side::Buy,
            1000000000000,
        );

        // Entry price should be mid_price + half spread
        assert_eq!(position.entry_price, 150.0 + 0.00005);
        assert_eq!(position.volume, 1.0);
        assert_eq!(position.side, Side::Buy);
    }

    #[test]
    fn test_unrealised_pnl_buy() {
        let position = Position::new(
            "pos_1".to_string(),
            150.0, // mid_price
            0.0001, // spread
            1.0, // volume
            Side::Buy,
            1000000000000,
        );

        // Entry price = 150.0 + 0.00005 = 150.00005
        // Price increased to 151.0
        let pnl = position.calculate_unrealised_pnl(151.0);
        // (151.0 - 150.00005) * 1.0 = 0.99995
        assert!((pnl - 0.99995).abs() < 0.0001);
    }

    #[test]
    fn test_unrealised_pnl_sell() {
        let position = Position::new(
            "pos_1".to_string(),
            150.0, // mid_price
            0.0001, // spread
            1.0, // volume
            Side::Sell,
            1000000000000,
        );

        // Entry price = 150.0 + 0.00005 = 150.00005
        // Price decreased to 149.0
        let pnl = position.calculate_unrealised_pnl(149.0);
        // (150.00005 - 149.0) * 1.0 = 1.00005
        assert!((pnl - 1.00005).abs() < 0.0001);
    }

    #[test]
    fn test_realised_pnl() {
        let position = Position::new(
            "pos_1".to_string(),
            150.0, // mid_price
            0.0001, // spread
            1.0, // volume
            Side::Buy,
            1000000000000,
        );

        // Entry price = 150.0 + 0.00005 = 150.00005
        // Close at 151.0 with no transaction cost
        let pnl = position.calculate_realised_pnl(151.0, 0.0);
        // (151.0 - 150.00005) * 1.0 = 0.99995
        assert!((pnl - 0.99995).abs() < 0.0001);
    }

    #[test]
    fn test_realised_pnl_with_transaction_cost() {
        let position = Position::new(
            "pos_1".to_string(),
            150.0, // mid_price
            0.0001, // spread
            1.0, // volume
            Side::Buy,
            1000000000000,
        );

        // Entry price = 150.0 + 0.00005 = 150.00005
        // Close at 151.0 with transaction cost
        let pnl = position.calculate_realised_pnl(151.0, 0.0001);
        // (151.0 - 150.00005) * 1.0 - 0.0001 = 0.99985
        assert!((pnl - 0.99985).abs() < 0.0001);
    }

    #[test]
    fn test_swap_accrual() {
        let mut position = Position::new(
            "pos_1".to_string(),
            150.0,
            0.0001,
            1.0,
            Side::Buy,
            1000000000000,
        );

        // Accrue swap for 1 day at rate 0.01
        let current_timestamp = position.open_timestamp_ns + NANOS_PER_DAY;
        let swap_accrued = position.accrue_swap(current_timestamp, 0.01);

        // Swap should be 0.01 * 1.0 * 1.0 = 0.01
        assert_eq!(position.swap, 0.01);
        assert!(swap_accrued);
    }

    #[test]
    fn test_swap_accrual_no_day_boundary() {
        let mut position = Position::new(
            "pos_1".to_string(),
            150.0,
            0.0001,
            1.0,
            Side::Buy,
            1000000000000,
        );

        // Try to accrue swap for less than 1 day (e.g., 12 hours = 43,200,000,000,000 ns)
        let current_timestamp = position.open_timestamp_ns + NANOS_PER_DAY / 2;
        let swap_accrued = position.accrue_swap(current_timestamp, 0.01);

        // Swap should not be accrued since less than 1 day has elapsed
        assert_eq!(position.swap, 0.0);
        assert!(!swap_accrued);
    }

    #[test]
    fn test_swap_accrual_multiple_days() {
        let mut position = Position::new(
            "pos_1".to_string(),
            150.0,
            0.0001,
            1.0,
            Side::Buy,
            1000000000000,
        );

        // Accrue swap for 3.5 days at rate 0.01
        // Should only accrue for 3 full days
        let current_timestamp = position.open_timestamp_ns + NANOS_PER_DAY * 3 + NANOS_PER_DAY / 2;
        let swap_accrued = position.accrue_swap(current_timestamp, 0.01);

        // Swap should be 0.01 * 3.0 * 1.0 = 0.03 (only full days)
        assert_eq!(position.swap, 0.03);
        assert!(swap_accrued);
    }

    #[test]
    fn test_closed_position_window() {
        let mut window = ClosedPositionWindow::new();

        // Add a closed position
        let closed_position = ClosedPosition {
            position_id: "pos_1".to_string(),
            entry_price: 150.0,
            close_price: 151.0,
            volume: 1.0,
            side: Side::Buy,
            realised_pnl: 1.0,
            swap: 0.01,
            open_timestamp_ns: 1000000000000,
            close_timestamp_ns: 2000000000000,
        };

        window.add_closed_position(closed_position);

        // Total P/L should be 1.0 + 0.01 = 1.01
        let current_timestamp = 3000000000000;
        assert_eq!(window.total_realised_pnl_12m(current_timestamp), 1.01);
    }
}
