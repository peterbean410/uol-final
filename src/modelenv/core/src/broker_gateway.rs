// Broker Gateway module for Production Mode operation
pub mod ctrader;

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

/// Factory function to create a broker gateway instance
///
/// # Arguments
/// * `broker_type` - The broker gateway type (e.g., "ctrader", "metatrader", "ib")
/// * `username` - Optional broker username
/// * `password` - Optional broker password
/// * `account` - Optional broker account
/// * `symbol` - Trading symbol
///
/// # Returns
/// A boxed BrokerGateway implementation
pub fn create_broker_gateway_instance(
    broker_type: &str,
    username: Option<String>,
    password: Option<String>,
    account: Option<String>,
    symbol: &str,
) -> Result<Box<dyn BrokerGateway + Send + Sync>> {
    match broker_type.to_lowercase().as_str() {
        "ctrader" => {
            let username = username.ok_or_else(|| anyhow::anyhow!("cTrader username required"))?;
            let password = password.ok_or_else(|| anyhow::anyhow!("cTrader password required"))?;
            let account = account.ok_or_else(|| anyhow::anyhow!("cTrader account required"))?;

            let client = ctrader::client::CtraderClient::new(
                username,
                password,
                account,
                symbol.to_string(),
            );
            let gateway = ctrader::gateway::CtraderBrokerGateway::new(client, symbol.to_string());
            Ok(Box::new(gateway))
        }
        _ => Err(anyhow::anyhow!(
            "Unknown broker type '{}'. Supported types: ctrader",
            broker_type
        )),
    }
}
