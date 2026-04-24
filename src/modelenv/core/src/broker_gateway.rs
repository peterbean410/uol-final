// Broker Gateway module for Production Mode operation
use anyhow::Result;

use modelenv_proto::{Action, Bar, Fill, Position};

/// Broker Gateway trait for Production Mode
///
/// This trait defines the interface for connecting to external broker/exchange systems
/// in Production Mode. Implementations of this trait can connect to different brokers
/// such as cTrader API, MetaTrader API, or Interactive Brokers API.
#[async_trait::async_trait]
pub trait BrokerGateway {
    /// Synchronise internal state with the broker's current positions
    ///
    /// This method is called during `Reset()` in Production Mode to ensure the
    /// environment's position state matches the broker's current state.
    ///
    /// # Arguments
    /// * `symbol` - The trading symbol to sync positions for
    ///
    /// # Returns
    /// A vector of Position messages representing the broker's current positions
    async fn sync_positions(&self, symbol: &str) -> Result<Vec<Position>>;

    /// Get the broker's current price bar for the symbol
    ///
    /// This method returns the most recent price bar (typically M1 interval)
    /// from the broker, which is used for calculating current P/L and entry prices.
    ///
    /// # Arguments
    /// * `symbol` - The trading symbol to get the bar for
    ///
    /// # Returns
    /// The current Bar message with the latest OHLCV data
    async fn current_bar(&self, symbol: &str) -> Result<Bar>;

    /// Submit an action to the broker and return the execution fill
    ///
    /// This method submits a trading action to the broker and returns the
    /// execution fill record with details about the trade.
    ///
    /// # Arguments
    /// * `action` - The action to submit with client order ID
    ///
    /// # Returns
    /// The Fill message representing the execution result
    async fn submit(&self, action: &Action) -> Result<Fill>;
}
