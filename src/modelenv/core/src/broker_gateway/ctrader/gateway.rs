// BrokerGateway trait implementation for cTrader
use anyhow::{anyhow, Result};
use log::{debug, error, info, warn};
use modelenv_proto::ActionType;
use std::sync::Arc;

use crate::broker_gateway::ctrader::client::CtraderClient;
use crate::broker_gateway::BrokerGateway;

/// cTrader Broker Gateway implementation
///
/// This struct implements the BrokerGateway trait for the cTrader API.
/// It wraps the CtraderClient and provides the interface for the FX RL Model Environment.
#[derive(Debug)]
pub struct CtraderBrokerGateway {
    /// cTrader API client
    client: Arc<tokio::sync::Mutex<CtraderClient>>,
    /// Trading symbol
    symbol: String,
}

impl CtraderBrokerGateway {
    /// Create a new cTrader Broker Gateway
    ///
    /// # Arguments
    /// * `client` - CtraderClient instance wrapped in Arc<Mutex>
    /// * `symbol` - Trading symbol
    ///
    /// # Returns
    /// New CtraderBrokerGateway instance
    pub fn new(client: CtraderClient, symbol: String) -> Self {
        CtraderBrokerGateway {
            client: Arc::new(tokio::sync::Mutex::new(client)),
            symbol,
        }
    }

    /// Get the trading symbol
    ///
    /// # Returns
    /// Trading symbol string
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    fn resolve_symbol(&self, symbol: &str) -> Result<String> {
        let configured_symbol = CtraderClient::normalise_symbol(&self.symbol).map_err(|err| {
            anyhow!(
                "ValidationError {{ field: symbol, details: configured symbol '{}' is invalid: {} }}",
                self.symbol,
                err
            )
        })?;

        if symbol.trim().is_empty() {
            return Ok(configured_symbol);
        }

        let requested_symbol = CtraderClient::normalise_symbol(symbol).map_err(|err| {
            anyhow!(
                "ValidationError {{ field: symbol, details: requested symbol '{}' is invalid: {} }}",
                symbol,
                err
            )
        })?;

        if requested_symbol != configured_symbol {
            return Err(anyhow!(
                "ValidationError {{ field: symbol, details: requested symbol '{}' does not match configured symbol '{}' }}",
                requested_symbol,
                configured_symbol
            ));
        }

        Ok(configured_symbol)
    }

    async fn ensure_connected(client: &mut CtraderClient) -> Result<()> {
        if client.is_connected() {
            return Ok(());
        }

        client.connect().await
    }

    fn is_connection_error(err: &anyhow::Error) -> bool {
        let error = err.to_string();
        error.contains("Not connected") || error.contains("ConnectionError")
    }

    fn is_retryable_submission_error(err: &anyhow::Error) -> bool {
        let error = err.to_string();
        error.contains("OrderError")
            || error.contains("TimeoutError")
            || error.contains("ConnectionError")
    }
}

#[async_trait::async_trait]
impl BrokerGateway for CtraderBrokerGateway {
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
    async fn sync_positions(&self, symbol: &str) -> Result<Vec<modelenv_proto::Position>> {
        let symbol = self.resolve_symbol(symbol)?;

        debug!("Syncing positions for symbol: {}", symbol);

        let mut client = self.client.lock().await;

        Self::ensure_connected(&mut client).await.map_err(|err| {
            anyhow!(
                "ConnectionError {{ symbol: {}, broker_error: {} }}",
                symbol,
                err
            )
        })?;

        // Refresh swap rates before syncing positions
        match client.refresh_swap_rates().await {
            Ok(_) => {}
            Err(e) => {
                warn!("Failed to refresh swap rates before position sync: {}", e);
                // Continue anyway - swap rates are cached
            }
        }

        match client.sync_positions(&symbol).await {
            Ok(positions) => {
                info!(
                    "Synced {} broker positions for symbol {}",
                    positions.len(),
                    symbol
                );
                Ok(positions)
            }
            Err(err) if Self::is_connection_error(&err) => {
                warn!(
                    "Position sync lost broker connection for {}. Attempting reconnect: {}",
                    symbol, err
                );

                client.reconnect().await.map_err(|reconnect_err| {
                    anyhow!(
                        "PositionError {{ symbol: {}, broker_error: {} }}",
                        symbol,
                        reconnect_err
                    )
                })?;

                client.sync_positions(&symbol).await.map_err(|retry_err| {
                    anyhow!(
                        "PositionError {{ symbol: {}, broker_error: {} }}",
                        symbol,
                        retry_err
                    )
                })
            }
            Err(err) => Err(anyhow!(
                "PositionError {{ symbol: {}, broker_error: {} }}",
                symbol,
                err
            )),
        }
    }

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
    async fn current_bar(&self, symbol: &str) -> Result<modelenv_proto::Bar> {
        let symbol = self.resolve_symbol(symbol)?;

        debug!("Getting current bar for symbol: {}", symbol);

        let mut client = self.client.lock().await;

        Self::ensure_connected(&mut client).await.map_err(|err| {
            anyhow!(
                "ConnectionError {{ symbol: {}, broker_error: {} }}",
                symbol,
                err
            )
        })?;

        match client.current_bar(&symbol).await {
            Ok(bar) => Ok(bar),
            Err(err) if Self::is_connection_error(&err) => {
                warn!(
                    "Current bar request lost broker connection for {}. Attempting reconnect: {}",
                    symbol, err
                );

                client.reconnect().await.map_err(|reconnect_err| {
                    anyhow!(
                        "BarError {{ symbol: {}, broker_error: {} }}",
                        symbol,
                        reconnect_err
                    )
                })?;

                client.current_bar(&symbol).await.map_err(|retry_err| {
                    anyhow!(
                        "BarError {{ symbol: {}, broker_error: {} }}",
                        symbol,
                        retry_err
                    )
                })
            }
            Err(err) => Err(anyhow!(
                "BarError {{ symbol: {}, broker_error: {} }}",
                symbol,
                err
            )),
        }
    }

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
    async fn submit(&self, action: &modelenv_proto::Action) -> Result<modelenv_proto::Fill> {
        if action.client_order_id.trim().is_empty() {
            return Err(anyhow!(
                "ValidationError {{ field: client_order_id, details: order ID cannot be empty }}"
            ));
        }

        let action_type = ActionType::try_from(action.action).map_err(|_| {
            anyhow!(
                "ValidationError {{ field: action, details: unsupported action type {} }}",
                action.action
            )
        })?;

        if action_type == ActionType::ActionHold {
            return Err(anyhow!(
                "ValidationError {{ field: action, details: hold actions are handled locally and must not be submitted to the broker }}"
            ));
        }

        debug!(
            "Submitting action: type={}, client_order_id={}",
            action.action as i32, action.client_order_id
        );

        let mut client = self.client.lock().await;

        Self::ensure_connected(&mut client).await.map_err(|err| {
            anyhow!(
                "ConnectionError {{ client_order_id: {}, broker_error: {} }}",
                action.client_order_id,
                err
            )
        })?;

        // Try to submit the action
        match client.submit_order(action).await {
            Ok(fill) => {
                let processed = client.process_action_queue().await;
                if processed > 0 {
                    info!(
                        "Processed {} queued cTrader actions after successful submission",
                        processed
                    );
                }
                Ok(fill)
            }
            Err(e) => {
                error!("Action submission failed: {}", e);

                // Check if it's a connection error that might be recoverable
                if Self::is_connection_error(&e) {
                    // Attempt reconnection
                    if let Err(reconnect_err) = client.reconnect().await {
                        error!("Reconnection failed: {}", reconnect_err);
                        return Err(anyhow!(
                            "OrderError {{ client_order_id: {}, broker_error: {} }}",
                            action.client_order_id,
                            reconnect_err
                        ));
                    }

                    // Retry after reconnection
                    match client.submit_order(action).await {
                        Ok(fill) => Ok(fill),
                        Err(retry_err) => {
                            error!("Retry after reconnection failed: {}", retry_err);
                            Err(anyhow!(
                                "OrderError {{ client_order_id: {}, broker_error: {} }}",
                                action.client_order_id,
                                retry_err
                            ))
                        }
                    }
                } else {
                    // Queue action for retry if it's a temporary error
                    if Self::is_retryable_submission_error(&e) {
                        client.queue_action(action.clone()).await;
                        Err(anyhow!(
                            "OrderError {{ client_order_id: {}, broker_error: {} }}",
                            action.client_order_id,
                            e
                        ))
                    } else {
                        Err(anyhow!(
                            "OrderError {{ client_order_id: {}, broker_error: {} }}",
                            action.client_order_id,
                            e
                        ))
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modelenv_proto::Action;

    fn test_gateway(symbol: &str) -> CtraderBrokerGateway {
        CtraderBrokerGateway::new(
            CtraderClient::new(
                "app-client-id".to_string(),
                "app-client-secret".to_string(),
                "access-token".to_string(),
                Some("refresh-token".to_string()),
                "account".to_string(),
                symbol.to_string(),
            ),
            symbol.to_string(),
        )
    }

    #[tokio::test]
    async fn sync_positions_connects_client_lazily() {
        let gateway = test_gateway("USDJPY");

        let positions = gateway.sync_positions("USDJPY").await.unwrap();

        assert!(positions.is_empty());
        assert!(gateway.client.lock().await.is_connected());
    }

    #[tokio::test]
    async fn current_bar_rejects_mismatched_symbols() {
        let gateway = test_gateway("USDJPY");

        let err = gateway.current_bar("EURUSD").await.unwrap_err();

        assert!(err.to_string().contains("does not match configured symbol"));
    }

    #[tokio::test]
    async fn submit_connects_client_and_returns_fill() {
        let gateway = test_gateway("USDJPY");

        let fill = gateway
            .submit(&Action {
                action: ActionType::ActionOpenBuy as i32,
                client_order_id: "order-1".to_string(),
            })
            .await
            .unwrap();

        assert_eq!(fill.side, ActionType::ActionOpenBuy as i32);
        assert!(fill.price > 100.0);
        assert!(gateway.client.lock().await.is_connected());
    }

    #[tokio::test]
    async fn submit_rejects_hold_actions() {
        let gateway = test_gateway("USDJPY");

        let err = gateway
            .submit(&Action {
                action: ActionType::ActionHold as i32,
                client_order_id: "noop".to_string(),
            })
            .await
            .unwrap_err();

        assert!(err.to_string().contains("hold actions are handled locally"));
    }

    #[tokio::test]
    async fn current_bar_connects_client_lazily() {
        let gateway = test_gateway("USDJPY");

        let bar = gateway.current_bar("USDJPY").await.unwrap();

        assert!(bar.close > 100.0);
        assert_eq!(bar.volume, 100.0);
        assert!(gateway.client.lock().await.is_connected());
    }

    #[tokio::test]
    async fn sync_positions_rejects_invalid_symbol_format() {
        let gateway = test_gateway("USDJPY");

        let err = gateway.sync_positions("USD/JPY").await.unwrap_err();

        assert!(err.to_string().contains("requested symbol"));
        assert!(err.to_string().contains("invalid symbol format"));
    }

    #[tokio::test]
    async fn submit_rejects_empty_client_order_id() {
        let gateway = test_gateway("USDJPY");

        let err = gateway
            .submit(&Action {
                action: ActionType::ActionOpenBuy as i32,
                client_order_id: " ".to_string(),
            })
            .await
            .unwrap_err();

        assert!(err.to_string().contains("client_order_id"));
    }

    #[tokio::test]
    async fn submit_rejects_unsupported_action_types() {
        let gateway = test_gateway("USDJPY");

        let err = gateway
            .submit(&Action {
                action: 999,
                client_order_id: "bad-action".to_string(),
            })
            .await
            .unwrap_err();

        assert!(err.to_string().contains("unsupported action type"));
    }
}
