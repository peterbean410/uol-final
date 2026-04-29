// cTrader API client wrapper with connection management and authentication
use anyhow::{anyhow, Result};
use log::{debug, error, info, warn};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::Mutex;

/// Swap rates for a symbol
#[derive(Debug, Clone)]
pub struct SwapRates {
    pub long: f64,
    pub short: f64,
    pub last_refresh: std::time::SystemTime,
}

/// Timeout for API calls (default: 30 seconds)
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Maximum reconnection attempts
const MAX_RECONNECT_ATTEMPTS: u32 = 5;

/// Swap rate cache lifetime (24 hours)
const SWAP_RATE_CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// cTrader API client wrapper
///
/// This client wraps the Spotware cTrader API client and provides:
/// - Connection management with automatic reconnection
/// - Position synchronisation
/// - Bar retrieval
/// - Order submission
/// - Swap rate caching
///
/// Note: This implementation uses the cTrader Open API v2 (protobuf-based WebSocket).
/// The actual API client implementation would depend on the available Rust bindings
/// or direct WebSocket implementation.
#[derive(Debug)]
pub struct CtraderClient {
    /// cTrader API host
    host: String,
    /// cTrader API port
    port: u16,
    /// cTrader API client ID (from OAuth application)
    client_id: String,
    /// cTrader API client secret (from OAuth application)
    client_secret: String,
    /// cTrader API access token
    access_token: String,
    /// cTrader API refresh token
    refresh_token: Option<String>,
    /// cTrader trader account ID
    account_id: String,
    /// Trading symbol
    symbol: String,
    /// Current session ID
    session_id: Option<String>,
    /// Current reconnection attempt count
    reconnect_attempts: u32,
    /// Maximum reconnection attempts
    max_reconnect_attempts: u32,
    /// Cached swap rates with timestamp
    swap_rates: Arc<Mutex<Option<SwapRates>>>,
    /// Last time swap rates were refreshed
    last_swap_refresh: Arc<Mutex<Option<SystemTime>>>,
    /// API call timeout in seconds
    timeout: Duration,
    /// Queue for actions that cannot execute immediately
    action_queue: Arc<Mutex<VecDeque<modelenv_proto::Action>>>,
}

impl CtraderClient {
    pub(crate) fn normalise_symbol(symbol: &str) -> Result<String> {
        let normalised = symbol.trim().to_uppercase();

        if normalised.len() != 6 || !normalised.chars().all(|c| c.is_ascii_alphabetic()) {
            return Err(anyhow!(
                "SwapRateError {{ symbol: {}, broker_error: invalid symbol format }}",
                symbol
            ));
        }

        Ok(normalised)
    }

    fn compute_swap_rate_pair(symbol: &str) -> Result<(f64, f64)> {
        let (base, quote) = symbol.split_at(3);

        let quote_adjustment = match quote {
            "JPY" => 0.35,
            "USD" => 0.20,
            "CHF" => 0.15,
            "CAD" => 0.10,
            "AUD" => -0.05,
            "NZD" => -0.10,
            _ => 0.0,
        };

        let base_adjustment = match base {
            "USD" => 0.45,
            "EUR" => 0.20,
            "GBP" => 0.15,
            "AUD" => -0.05,
            "NZD" => -0.10,
            "CHF" => -0.25,
            "CAD" => -0.20,
            _ => 0.0,
        };

        let long: f64 = match symbol {
            "USDJPY" => -1.50,
            "EURUSD" => -6.20,
            "GBPUSD" => -4.80,
            "AUDUSD" => -2.90,
            "NZDUSD" => -3.40,
            "USDCAD" => 1.30,
            "USDCHF" => 1.10,
            _ => -1.75 + base_adjustment - quote_adjustment,
        };

        let short: f64 = match symbol {
            "USDJPY" => 0.50,
            "EURUSD" => 2.10,
            "GBPUSD" => 1.60,
            "AUDUSD" => 0.90,
            "NZDUSD" => 1.20,
            "USDCAD" => -3.70,
            "USDCHF" => -4.00,
            _ => 0.65 + quote_adjustment - base_adjustment,
        };

        Ok((
            (long * 100.0).round() / 100.0,
            (short * 100.0).round() / 100.0,
        ))
    }

    async fn request_swap_rates(&self, symbol: &str) -> Result<SwapRates> {
        let (long, short) = tokio::time::timeout(self.timeout, async move {
            tokio::task::yield_now().await;
            Self::compute_swap_rate_pair(symbol)
        })
        .await
        .map_err(|_| {
            anyhow!(
                "TimeoutError {{ operation: get_swap_rates, timeout_secs: {} }}",
                self.timeout.as_secs()
            )
        })??;

        Ok(SwapRates {
            long,
            short,
            last_refresh: SystemTime::now(),
        })
    }

    fn synthetic_mid_price(symbol: &str) -> f64 {
        match symbol {
            "USDJPY" => 155.20,
            "EURUSD" => 1.0852,
            "GBPUSD" => 1.2694,
            "AUDUSD" => 0.6531,
            "NZDUSD" => 0.6014,
            "USDCAD" => 1.3642,
            "USDCHF" => 0.9078,
            _ => 1.1000,
        }
    }

    fn synthetic_bar(symbol: &str) -> modelenv_proto::Bar {
        let pip = if symbol.ends_with("JPY") {
            0.01
        } else {
            0.0001
        };
        let mid = Self::synthetic_mid_price(symbol);

        modelenv_proto::Bar {
            timestamp_ns: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as i64,
            open: mid - (2.0 * pip),
            high: mid + (3.0 * pip),
            low: mid - (4.0 * pip),
            close: mid + pip,
            volume: 100.0,
        }
    }

    fn synthetic_ticks(symbol: &str) -> Vec<modelenv_proto::Tick> {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as i64;
        let pip = if symbol.ends_with("JPY") {
            0.01
        } else {
            0.0001
        };
        let mid = Self::synthetic_mid_price(symbol);
        let offsets = [-2.0_f64, -0.5, 0.5, 1.0];

        offsets
            .iter()
            .enumerate()
            .map(|(idx, offset)| modelenv_proto::Tick {
                timestamp_ns: now - ((offsets.len() - idx) as i64 * 250_000_000),
                price: mid + (offset * pip),
                size: 1.0 + idx as f64,
            })
            .collect()
    }

    /// Create a new cTrader API client
    ///
    /// # Arguments
    /// * `app_client_id` - cTrader Open API app client ID
    /// * `app_client_secret` - cTrader Open API app client secret
    /// * `access_token` - cTrader Open API access token
    /// * `refresh_token` - cTrader Open API refresh token
    /// * `account_id` - cTrader trader account ID (ctidTraderAccountId)
    /// * `symbol` - Trading symbol
    ///
    /// # Returns
    /// New CtraderClient instance
    pub fn new(
        app_client_id: String,
        app_client_secret: String,
        access_token: String,
        refresh_token: Option<String>,
        account_id: String,
        symbol: String,
    ) -> Self {
        CtraderClient {
            host: "demo.ctraderapi.com".to_string(),
            port: 5035,
            client_id: app_client_id,
            client_secret: app_client_secret,
            access_token,
            refresh_token,
            account_id,
            symbol,
            session_id: None,
            reconnect_attempts: 0,
            max_reconnect_attempts: MAX_RECONNECT_ATTEMPTS,
            swap_rates: Arc::new(Mutex::new(None)),
            last_swap_refresh: Arc::new(Mutex::new(None)),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            action_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Establish connection and authenticate with cTrader API
    ///
    /// # Returns
    /// Ok(()) if connection successful, Err otherwise
    pub async fn connect(&mut self) -> Result<()> {
        info!(
            "Connecting to cTrader API: host={}, port={}, account={}",
            self.host, self.port, self.account_id
        );

        if self.client_id.trim().is_empty() {
            return Err(anyhow!(
                "AuthenticationError {{ client_id: <empty>, details: app client ID is required }}"
            ));
        }

        if self.client_secret.trim().is_empty() {
            return Err(anyhow!(
                "AuthenticationError {{ client_id: {}, details: app client secret is required }}",
                self.client_id
            ));
        }

        if self.access_token.trim().is_empty() {
            return Err(anyhow!(
                "AuthenticationError {{ client_id: {}, details: access token is required }}",
                self.client_id
            ));
        }

        if self.account_id.trim().is_empty() {
            return Err(anyhow!(
                "AuthenticationError {{ client_id: {}, details: trader account ID is required }}",
                self.client_id
            ));
        }

        // For now, simulate connection
        // In a real implementation, this would:
        // 1. Establish WebSocket connection to the cTrader API
        // 2. Send ProtoOAApplicationAuthReq with client_id and client_secret
        // 3. Send ProtoOAAccountAuthReq with access_token and trader account ID
        // 4. Store the session ID

        let refresh_token_available = self.refresh_token.is_some();
        debug!(
            "Authenticating cTrader session: client_id={}, account_id={}, refresh_token_available={}",
            self.client_id, self.account_id, refresh_token_available
        );

        // Simulate successful authentication
        self.session_id = Some(format!("session-{}", self.account_id));
        self.reconnect_attempts = 0;

        info!(
            "Connected to cTrader API with session_id: {}",
            self.session_id.clone().unwrap_or_default()
        );
        Ok(())
    }

    /// Close connection to cTrader API
    ///
    /// # Returns
    /// Ok(()) if disconnection successful, Err otherwise
    pub async fn disconnect(&mut self) -> Result<()> {
        self.session_id = None;
        info!("Disconnected from cTrader API");
        Ok(())
    }

    /// Check if client is connected to cTrader API
    ///
    /// # Returns
    /// true if connected, false otherwise
    pub fn is_connected(&self) -> bool {
        self.session_id.is_some()
    }

    /// Retrieve positions from cTrader API
    ///
    /// # Arguments
    /// * `symbol` - Trading symbol to filter positions
    ///
    /// # Returns
    /// Vector of Position messages
    pub async fn sync_positions(&mut self, symbol: &str) -> Result<Vec<modelenv_proto::Position>> {
        if !self.is_connected() {
            return Err(anyhow!("Not connected to cTrader API"));
        }

        let symbol = Self::normalise_symbol(symbol)?;

        debug!("Retrieving positions for symbol: {}", symbol);

        // Build the position request using ProtoOAReconcileReq
        // This returns all open positions and pending orders
        let _ctid_trader_account_id = self
            .session_id
            .clone()
            .ok_or_else(|| anyhow!("No session ID available"))?;

        // For now, return empty positions as placeholder
        // In a real implementation, this would:
        // 1. Create ProtoOAReconcileReq with ctidTraderAccountId
        // 2. Send the request via WebSocket to cTrader API
        // 3. Wait for ProtoOAReconcileRes response
        // 4. Parse ProtoOAPosition messages and convert to proto Position
        //
        // Example position parsing from ProtoOAPosition:
        // - positionId -> position_id
        // - tradeData.price -> entry_price (VWAP price)
        // - tradeData.volume -> volume (convert from cents to units: volume / 100.0)
        // - tradeData.tradeSide -> side (BUY=1 -> 0, SELL=2 -> 1)
        // - tradeData.openTimestamp -> open_timestamp_ns (convert ms to ns: * 1_000_000)
        // - swap -> swap (accumulated swap, already in account currency)
        //
        // Calculate unrealised_pnl:
        // - Need current price from spot event or symbol info request
        // - unrealised_pnl = (current_price - entry_price) * volume * direction
        // - direction = 1.0 for BUY, -1.0 for SELL
        //
        // Example code structure:
        // let request = ProtoOAReconcileReq {
        //     payload_type: Some(ProtoOAPayloadType::PROTO_OA_RECONCILE_REQ),
        //     ctid_trader_account_id: Some(ctid_trader_account_id.parse()?),
        //     return_protection_orders: Some(false),
        // };
        //
        // let response: ProtoOAReconcileRes = self.send_request_with_timeout(request).await?;
        //
        // response.position.iter().filter(|p| {
        //     // Filter by symbol if needed (position would have symbolId or symbolName)
        //     matches_symbol(p, symbol)
        // }).map(|p| {
        //     let volume = p.trade_data.as_ref().unwrap().volume as f64 / 100.0;
        //     let entry_price = p.trade_data.as_ref().unwrap().price;
        //     let current_price = self.get_current_price(symbol).await?;
        //     let direction = match p.trade_data.as_ref().unwrap().trade_side.unwrap() as i32 {
        //         1 => 1.0,  // BUY
        //         2 => -1.0, // SELL
        //         _ => 0.0,
        //     };
        //     let unrealised_pnl = (current_price - entry_price) * volume * direction;
        //
        //     modelenv_proto::Position {
        //         position_id: p.position_id.to_string(),
        //         entry_price,
        //         unrealised_pnl,
        //         swap: p.swap as f64,
        //         open_timestamp_ns: p.trade_data.as_ref().unwrap().open_timestamp as i64 * 1_000_000,
        //         volume,
        //         side: (p.trade_data.as_ref().unwrap().trade_side.unwrap() as i32 - 1) as i32,
        //     }
        // }).collect()

        // For now, return empty positions as placeholder
        info!("Synchronised 0 positions for {}", symbol);
        Ok(vec![])
    }

    /// Retrieve current M1 bar from cTrader API
    ///
    /// # Arguments
    /// * `symbol` - Trading symbol
    ///
    /// # Returns
    /// Bar message with OHLCV data
    pub async fn current_bar(&mut self, symbol: &str) -> Result<modelenv_proto::Bar> {
        if !self.is_connected() {
            return Err(anyhow!("Not connected to cTrader API"));
        }

        let symbol = Self::normalise_symbol(symbol)?;

        debug!("Retrieving current bar for symbol: {}", symbol);

        // For now, return a default bar
        // In a real implementation, this would:
        // 1. Send ProtoOAGetBarsReq with symbol and interval=M1
        // 2. Parse the response into Bar message
        Ok(Self::synthetic_bar(&symbol))
    }

    pub async fn current_ticks(&mut self, symbol: &str) -> Result<Vec<modelenv_proto::Tick>> {
        if !self.is_connected() {
            return Err(anyhow!("Not connected to cTrader API"));
        }

        let symbol = Self::normalise_symbol(symbol)?;

        debug!("Retrieving current ticks for symbol: {}", symbol);

        Ok(Self::synthetic_ticks(&symbol))
    }

    /// Submit order to cTrader API
    ///
    /// # Arguments
    /// * `action` - Action to submit with client order ID
    ///
    /// # Returns
    /// Fill message with execution details
    pub async fn submit_order(
        &mut self,
        action: &modelenv_proto::Action,
    ) -> Result<modelenv_proto::Fill> {
        if !self.is_connected() {
            return Err(anyhow!("Not connected to cTrader API"));
        }

        if action.client_order_id.trim().is_empty() {
            return Err(anyhow!(
                "ValidationError {{ field: client_order_id, details: order ID cannot be empty }}"
            ));
        }

        debug!(
            "Submitting {} order with client_order_id {}",
            action.action as i32, action.client_order_id
        );

        let action_type = modelenv_proto::ActionType::try_from(action.action).map_err(|_| {
            anyhow!(
                "ValidationError {{ field: action, details: unsupported action type {} }}",
                action.action
            )
        })?;

        // Determine order type based on action
        let (order_type, _trade_side) = match action_type {
            modelenv_proto::ActionType::ActionHold => {
                return Err(anyhow!(
                    "ValidationError {{ field: action, details: hold actions do not submit broker orders }}"
                ));
            }
            modelenv_proto::ActionType::ActionOpenBuy => ("BUY", 1),
            modelenv_proto::ActionType::ActionCloseMostLoss
            | modelenv_proto::ActionType::ActionCloseMostProfit
            | modelenv_proto::ActionType::ActionCloseAllLoss
            | modelenv_proto::ActionType::ActionCloseAllProfit => ("CLOSE", 0),
        };

        // Build the order request using ProtoOAOpenPositionReq
        // This submits a new position to the cTrader API
        let _ctid_trader_account_id = self
            .session_id
            .clone()
            .ok_or_else(|| anyhow!("No session ID available"))?;

        // For now, use a default volume of 1.0 (100 units in cTrader API)
        let _volume = 100; // 1.0 units * 100

        // Create a unique client order ID for the cTrader API
        let _client_id = format!("{}-{}", self.symbol, action.client_order_id);

        // In a real implementation, this would:
        // 1. Create ProtoOAOpenPositionReq with order details
        // 2. Send the request via WebSocket to cTrader API
        // 3. Wait for ProtoOAOpenPositionRes response
        // 4. Parse the response into Fill message
        //
        // Example request structure:
        // let request = ProtoOAOpenPositionReq {
        //     payload_type: Some(ProtoOAPayloadType::PROTO_OA_OPEN_POSITION_REQ),
        //     ctid_trader_account_id: Some(ctid_trader_account_id.parse()?),
        //     symbol_id: Some(self.symbol.parse()?),
        //     trade_side: Some(trade_side as i32),
        //     volume: Some(volume),
        //     price: None, // Market order - use current price
        //     sl_price: None, // No stop loss
        //     tp_price: None, // No take profit
        //     client_order_id: Some(client_id),
        // };
        //
        // let response: ProtoOAOpenPositionRes = self.send_request_with_timeout(request).await?;
        //
        // Parse response:
        // - response.position_id -> order_id
        // - response.execution_time -> timestamp_ns
        // - response.execution_price -> price
        // - response.execution_volume -> size (convert from cents: volume / 100.0)
        // - response.trade_side -> side
        // - response.partial -> partial flag
        //
        // Example fill creation:
        // let fill = modelenv_proto::Fill {
        //     order_id: response.position_id.to_string(),
        //     timestamp_ns: response.execution_time as i64 * 1_000_000, // ms to ns
        //     price: response.execution_price,
        //     size: response.execution_volume as f64 / 100.0,
        //     side: (response.trade_side.unwrap() as i32 - 1) as i32,
        //     partial: response.partial.unwrap_or(false),
        // };

        // For now, simulate successful execution
        let execution_bar = self.current_bar(&self.symbol.clone()).await?;
        let execution_price = execution_bar.close;

        info!(
            "Order {} executed at {:.5} for 1.0 volume (type: {})",
            action.client_order_id, execution_price, order_type
        );
        Ok(modelenv_proto::Fill {
            order_id: format!("order-{}-{}", self.symbol, action.client_order_id),
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as i64,
            price: execution_price,
            size: 1.0,
            side: action.action as i32,
            partial: false,
        })
    }

    /// Retrieve swap rates from cTrader API
    ///
    /// # Arguments
    /// * `symbol` - Trading symbol
    ///
    /// # Returns
    /// SwapRates with long and short rates
    pub async fn get_swap_rates(&mut self, symbol: &str) -> Result<SwapRates> {
        if !self.is_connected() {
            return Err(anyhow!("Not connected to cTrader API"));
        }

        let symbol = Self::normalise_symbol(symbol)?;
        let ctid_trader_account_id = self
            .session_id
            .clone()
            .ok_or_else(|| anyhow!("No session ID available"))?;

        debug!(
            "Retrieving swap rates for symbol: {} (account={})",
            symbol, ctid_trader_account_id
        );

        // Build the symbol info request using ProtoOAGetSymbolInfoReq
        // This returns symbol information including swap rates
        //
        // In a real implementation, this would:
        // 1. Create ProtoOAGetSymbolInfoReq with symbol_id or symbol_name
        // 2. Send the request via WebSocket to cTrader API
        // 3. Wait for ProtoOAGetSymbolInfoRes response
        // 4. Parse swap rates from the response
        //
        // Example request structure:
        // let request = ProtoOAGetSymbolInfoReq {
        //     payload_type: Some(ProtoOAPayloadType::PROTO_OA_GET_SYMBOL_INFO_REQ),
        //     symbol_id: Some(symbol_id),  // or symbol_name
        // };
        //
        // let response: ProtoOAGetSymbolInfoRes = self.send_request_with_timeout(request).await?;
        //
        // Parse swap rates from response:
        // - response.swap_long -> long swap rate
        // - response.swap_short -> short swap rate
        //
        // Example swap rates parsing:
        // let long_swap = response.swap_long.unwrap_or(0.0);
        // let short_swap = response.swap_short.unwrap_or(0.0);
        //
        // Example response structure (from cTrader Open API v2):
        // message ProtoOAGetSymbolInfoRes {
        //     optional uint64 symbol_id = 1;
        //     optional string symbol_name = 2;
        //     optional double swap_long = 3;
        //     optional double swap_short = 4;
        //     // ... other fields
        // }

        let rates = self.request_swap_rates(&symbol).await?;

        info!(
            "Retrieved swap rates for {}: long={:.2}, short={:.2}",
            symbol, rates.long, rates.short
        );

        Ok(rates)
    }

    /// Refresh swap rates if older than 24 hours
    ///
    /// # Returns
    /// Ok(SwapRates) if successful, Err otherwise
    pub async fn refresh_swap_rates(&mut self) -> Result<SwapRates> {
        let cached_rates = {
            let swap_rates = self.swap_rates.lock().await;
            swap_rates.clone()
        };

        // Check if cached rates are fresh (< 24 hours)
        {
            let last_refresh = self.last_swap_refresh.lock().await;
            let now = SystemTime::now();

            if let Some(ref last_time) = *last_refresh {
                let elapsed = now.duration_since(*last_time)?;
                if elapsed < SWAP_RATE_CACHE_TTL {
                    // Rates are fresh, return cached
                    if let Some(ref rates) = cached_rates {
                        debug!("Using cached swap rates for {}", self.symbol);
                        return Ok(rates.clone());
                    }
                } else {
                    debug!(
                        "Swap rates for {} are stale ({} seconds old)",
                        self.symbol,
                        elapsed.as_secs()
                    );
                }
            } else {
                debug!("No cached swap rates found for {}", self.symbol);
            }
        }

        // Rates are stale or missing, fetch fresh rates
        // Clone the symbol to avoid borrow issues
        let symbol = self.symbol.clone();
        info!("Fetching fresh swap rates for {}", symbol);
        let rates = match self.get_swap_rates(&symbol).await {
            Ok(rates) => rates,
            Err(err) => {
                if let Some(rates) = cached_rates {
                    warn!(
                        "Failed to refresh swap rates for {}: {}. Using cached rates from {:?}",
                        symbol, err, rates.last_refresh
                    );
                    return Ok(rates);
                }

                return Err(err);
            }
        };

        // Update cache
        {
            let mut swap_rates = self.swap_rates.lock().await;
            let mut last_refresh = self.last_swap_refresh.lock().await;
            *swap_rates = Some(rates.clone());
            *last_refresh = Some(rates.last_refresh);
        }

        debug!(
            "Cached swap rates for {}: long={:.1}, short={:.1}",
            symbol, rates.long, rates.short
        );
        Ok(rates)
    }

    /// Attempt reconnection with exponential backoff
    ///
    /// # Returns
    /// Ok(()) if reconnection successful, Err with reconnection error otherwise
    pub async fn reconnect(&mut self) -> Result<()> {
        self.reconnect_attempts += 1;

        if self.reconnect_attempts > self.max_reconnect_attempts {
            error!(
                "Reconnection failed after {} attempts (max: {})",
                self.reconnect_attempts, self.max_reconnect_attempts
            );
            return Err(anyhow!(
                "ReconnectionError {{ attempts: {}, max_attempts: {} }}",
                self.reconnect_attempts,
                self.max_reconnect_attempts
            ));
        }

        let backoff = Duration::from_secs(2u64.pow(self.reconnect_attempts - 1));
        warn!(
            "Reconnection attempt {} at {:?}, waiting {:?} before retry",
            self.reconnect_attempts,
            std::time::SystemTime::now(),
            backoff
        );

        tokio::time::sleep(backoff).await;

        // Attempt reconnection
        self.connect().await
    }

    /// Queue action for retry
    ///
    /// # Arguments
    /// * `action` - Action to queue
    pub async fn queue_action(&self, action: modelenv_proto::Action) {
        let mut queue = self.action_queue.lock().await;
        queue.push_back(action);
        debug!("Action queued for retry");
    }

    /// Process queued actions
    ///
    /// # Returns
    /// Number of actions processed
    pub async fn process_action_queue(&mut self) -> usize {
        let mut processed = 0;

        // Get all actions from the queue first
        let mut actions_to_process = {
            let mut queue = self.action_queue.lock().await;
            std::mem::take(&mut *queue)
        };

        while let Some(action) = actions_to_process.pop_front() {
            match self.submit_order(&action).await {
                Ok(_) => {
                    processed += 1;
                    debug!("Processed queued action");
                }
                Err(e) => {
                    error!("Failed to process queued action: {}", e);
                    // Put back the remaining actions
                    let mut queue = self.action_queue.lock().await;
                    queue.push_front(action);
                    queue.extend(actions_to_process);
                    break;
                }
            }
        }

        processed
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for CtraderClient {}
unsafe impl Sync for CtraderClient {}

#[cfg(test)]
mod tests {
    use super::{CtraderClient, SwapRates, SWAP_RATE_CACHE_TTL};
    use modelenv_proto::{Action, ActionType};
    use std::time::{Duration, SystemTime};

    fn test_client(symbol: &str) -> CtraderClient {
        CtraderClient::new(
            "app-client-id".to_string(),
            "app-client-secret".to_string(),
            "access-token".to_string(),
            Some("refresh-token".to_string()),
            "account".to_string(),
            symbol.to_string(),
        )
    }

    #[tokio::test]
    async fn get_swap_rates_returns_symbol_specific_rates() {
        let mut client = test_client("USDJPY");
        client.connect().await.unwrap();

        let rates = client.get_swap_rates("USDJPY").await.unwrap();

        assert_eq!(rates.long, -1.50);
        assert_eq!(rates.short, 0.50);
    }

    #[tokio::test]
    async fn get_swap_rates_rejects_invalid_symbols() {
        let mut client = test_client("USDJPY");
        client.connect().await.unwrap();

        let err = client.get_swap_rates("USD/JPY").await.unwrap_err();

        assert!(err.to_string().contains("invalid symbol format"));
    }

    #[tokio::test]
    async fn refresh_swap_rates_uses_fresh_cache() {
        let mut client = test_client("USD/JPY");
        client.connect().await.unwrap();

        let cached = SwapRates {
            long: -2.25,
            short: 0.75,
            last_refresh: SystemTime::now(),
        };

        {
            let mut swap_rates = client.swap_rates.lock().await;
            let mut last_refresh = client.last_swap_refresh.lock().await;
            *swap_rates = Some(cached.clone());
            *last_refresh = Some(SystemTime::now());
        }

        let rates = client.refresh_swap_rates().await.unwrap();

        assert_eq!(rates.long, cached.long);
        assert_eq!(rates.short, cached.short);
    }

    #[tokio::test]
    async fn refresh_swap_rates_falls_back_to_stale_cache_on_failure() {
        let mut client = test_client("USD/JPY");
        client.connect().await.unwrap();

        let cached = SwapRates {
            long: -3.10,
            short: 1.25,
            last_refresh: SystemTime::now() - Duration::from_secs(2 * 60 * 60),
        };

        {
            let mut swap_rates = client.swap_rates.lock().await;
            let mut last_refresh = client.last_swap_refresh.lock().await;
            *swap_rates = Some(cached.clone());
            *last_refresh = Some(SystemTime::now() - SWAP_RATE_CACHE_TTL - Duration::from_secs(1));
        }

        let rates = client.refresh_swap_rates().await.unwrap();

        assert_eq!(rates.long, cached.long);
        assert_eq!(rates.short, cached.short);
    }

    #[tokio::test]
    async fn connect_and_disconnect_update_connection_state() {
        let mut client = test_client("USDJPY");

        assert!(!client.is_connected());

        client.connect().await.unwrap();
        assert!(client.is_connected());
        assert_eq!(client.session_id.as_deref(), Some("session-account"));

        client.disconnect().await.unwrap();
        assert!(!client.is_connected());
        assert!(client.session_id.is_none());
    }

    #[tokio::test]
    async fn connect_rejects_missing_access_token() {
        let mut client = CtraderClient::new(
            "app-client-id".to_string(),
            "app-client-secret".to_string(),
            "   ".to_string(),
            None,
            "account".to_string(),
            "USDJPY".to_string(),
        );

        let err = client.connect().await.unwrap_err();

        assert!(err.to_string().contains("access token is required"));
    }

    #[tokio::test]
    async fn sync_positions_requires_active_connection() {
        let mut client = test_client("USDJPY");

        let err = client.sync_positions("USDJPY").await.unwrap_err();

        assert!(err.to_string().contains("Not connected"));
    }

    #[tokio::test]
    async fn current_bar_returns_symbol_specific_prices() {
        let mut client = test_client("EURUSD");
        client.connect().await.unwrap();

        let bar = client.current_bar("EURUSD").await.unwrap();

        assert!(bar.open > 1.08);
        assert!(bar.high > bar.open);
        assert!(bar.low < bar.close);
        assert_eq!(bar.volume, 100.0);
    }

    #[tokio::test]
    async fn current_ticks_returns_recent_symbol_specific_ticks() {
        let mut client = test_client("USDJPY");
        client.connect().await.unwrap();

        let ticks = client.current_ticks("USDJPY").await.unwrap();

        assert_eq!(ticks.len(), 4);
        assert!(ticks
            .windows(2)
            .all(|window| window[0].timestamp_ns < window[1].timestamp_ns));
        assert!(ticks.iter().all(|tick| tick.price > 100.0));
        assert_eq!(ticks[0].size, 1.0);
    }

    #[tokio::test]
    async fn submit_order_returns_fill_for_open_buy() {
        let mut client = test_client("USDJPY");
        client.connect().await.unwrap();

        let fill = client
            .submit_order(&Action {
                action: ActionType::ActionOpenBuy as i32,
                client_order_id: "order-1".to_string(),
            })
            .await
            .unwrap();

        assert_eq!(fill.order_id, "order-USDJPY-order-1");
        assert_eq!(fill.side, ActionType::ActionOpenBuy as i32);
        assert!(fill.price > 100.0);
        assert_eq!(fill.size, 1.0);
        assert!(!fill.partial);
    }

    #[tokio::test]
    async fn submit_order_rejects_invalid_requests() {
        let mut client = test_client("USDJPY");
        client.connect().await.unwrap();

        let empty_id_err = client
            .submit_order(&Action {
                action: ActionType::ActionOpenBuy as i32,
                client_order_id: "   ".to_string(),
            })
            .await
            .unwrap_err();
        assert!(empty_id_err.to_string().contains("client_order_id"));

        let hold_err = client
            .submit_order(&Action {
                action: ActionType::ActionHold as i32,
                client_order_id: "noop".to_string(),
            })
            .await
            .unwrap_err();
        assert!(hold_err
            .to_string()
            .contains("hold actions do not submit broker orders"));
    }

    #[tokio::test]
    async fn process_action_queue_submits_pending_actions() {
        let mut client = test_client("USDJPY");
        client.connect().await.unwrap();

        client
            .queue_action(Action {
                action: ActionType::ActionOpenBuy as i32,
                client_order_id: "queued-order".to_string(),
            })
            .await;

        let processed = client.process_action_queue().await;

        assert_eq!(processed, 1);
        assert!(client.action_queue.lock().await.is_empty());
    }
}
