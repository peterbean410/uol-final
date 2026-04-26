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
/// * `app_client_id` - Optional cTrader Open API app client ID
/// * `app_client_secret` - Optional cTrader Open API app client secret
/// * `access_token` - Optional cTrader Open API access token
/// * `refresh_token` - Optional cTrader Open API refresh token
/// * `account_id` - Optional cTrader trader account ID
/// * `symbol` - Trading symbol
///
/// # Returns
/// A boxed BrokerGateway implementation
pub fn create_broker_gateway_instance(
    broker_type: &str,
    app_client_id: Option<String>,
    app_client_secret: Option<String>,
    access_token: Option<String>,
    refresh_token: Option<String>,
    account_id: Option<String>,
    symbol: &str,
) -> Result<Box<dyn BrokerGateway + Send + Sync>> {
    match broker_type.to_lowercase().as_str() {
        "ctrader" => {
            let app_client_id =
                app_client_id.ok_or_else(|| anyhow::anyhow!("cTrader app client ID required"))?;
            let app_client_secret = app_client_secret
                .ok_or_else(|| anyhow::anyhow!("cTrader app client secret required"))?;
            let access_token =
                access_token.ok_or_else(|| anyhow::anyhow!("cTrader access token required"))?;
            let account_id =
                account_id.ok_or_else(|| anyhow::anyhow!("cTrader account ID required"))?;

            let client = ctrader::client::CtraderClient::new(
                app_client_id,
                app_client_secret,
                access_token,
                refresh_token,
                account_id,
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
