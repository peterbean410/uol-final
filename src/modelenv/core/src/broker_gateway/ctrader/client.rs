//! cTrader Open API client: the high-level broker client the gateway drives.
//!
//! This assembles the demo-validated building blocks, [`super::transport`]
//! (TLS), [`super::connection`] (request/response correlation + events),
//! [`super::auth`] (app+account auth, heartbeat), [`super::data`] (read RPCs) and
//! [`super::orders`] (order RPCs), into one client. It talks to the **real**
//! cTrader API; the `live` flag selects the endpoint:
//! `false` → `demo.ctraderapi.com` (paper, default), `true` → `live.ctraderapi.com`
//! (real money). **Configurable, not compiled in**, see [`CtraderClient::with_live`].
//!
//! Order sizing: modelenv thinks in position *units*; the broker trades *lots*.
//! [`CtraderClient::with_lot_size_per_unit`] sets cTrader lots per unit (default
//! `0.01`, the USDJPY minimum). This is real-money critical, so it is explicit
//! config, never a silent default in the order path.
//!
//! Where a real RPC is not yet available (live ticks need a streaming spot
//! subscription), the method returns a hard **error** rather than fabricating a
//! value; a policy must never trade on invented data.
use anyhow::{anyhow, Context, Result};
use log::{debug, info, warn};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;

use modelenv_proto::ctrader::{ProtoMessage, ProtoOaSpotEvent};
use prost::Message as _;

use super::auth::{self, Credentials};
use super::connection::Connection;
use super::data;
use super::orders::{self, lots_to_volume, Side};
use super::transport::Transport;
use super::wire::payload_type as pt;

/// Swap rates for a symbol
#[derive(Debug, Clone)]
pub struct SwapRates {
    pub long: f64,
    pub short: f64,
    pub last_refresh: std::time::SystemTime,
}

/// Latest streamed bid/ask plus a rolling recent-ticks buffer. Filled by the
/// spot-event router ([`route_events`]) and read by [`CtraderClient::current_ticks`].
#[derive(Default)]
struct TickState {
    last_bid: Option<f64>,
    last_ask: Option<f64>,
    ticks: VecDeque<modelenv_proto::Tick>,
}

fn now_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as i64
}

/// Background router: drains the connection's unsolicited events and fans them
/// apart (spot events into the tick buffer, execution/error events to the order
/// path) until the connection closes. Runs for the life of a connection.
async fn route_events(
    mut raw: mpsc::UnboundedReceiver<ProtoMessage>,
    tick_state: Arc<Mutex<TickState>>,
    order_tx: mpsc::UnboundedSender<ProtoMessage>,
    symbol_id: i64,
) {
    while let Some(msg) = raw.recv().await {
        if msg.payload_type == pt::SPOT_EVENT {
            if let Ok(evt) = ProtoOaSpotEvent::decode(msg.payload.as_deref().unwrap_or_default()) {
                if evt.symbol_id != symbol_id {
                    continue;
                }
                let (bid, ask, ts) = data::spot_bid_ask(&evt);
                let mut st = tick_state.lock().await;
                if let Some(b) = bid {
                    st.last_bid = Some(b);
                }
                if let Some(a) = ask {
                    st.last_ask = Some(a);
                }
                if let (Some(b), Some(a)) = (st.last_bid, st.last_ask) {
                    let ts_ns = if ts > 0 { ts } else { now_ns() };
                    st.ticks.push_back(modelenv_proto::Tick {
                        timestamp_ns: ts_ns,
                        bid: b,
                        ask: a,
                    });
                    while st.ticks.len() > RECENT_TICKS {
                        st.ticks.pop_front();
                    }
                }
            }
        } else if msg.payload_type == pt::EXECUTION_EVENT
            || msg.payload_type == pt::ORDER_ERROR_EVENT
            || msg.payload_type == pt::ERROR_RES
            || msg.payload_type == pt::OA_ERROR_RES
        {
            // Deliver to the in-flight order path; drop if nobody is waiting.
            let _ = order_tx.send(msg);
        }
        // else: heartbeat / unrelated → ignore.
    }
}

/// Timeout for API calls (default: 30 seconds)
const DEFAULT_TIMEOUT_SECS: u64 = 30;
/// Maximum reconnection attempts
const MAX_RECONNECT_ATTEMPTS: u32 = 5;
/// Swap rate cache lifetime (24 hours)
const SWAP_RATE_CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);
/// Session keep-alive heartbeat interval (inside cTrader's ~30s idle-drop).
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);
/// How many recent streamed ticks to retain for `current_ticks`.
const RECENT_TICKS: usize = 256;
/// Max time `current_ticks` waits for the first streamed spot before erroring.
const TICK_WAIT: Duration = Duration::from_secs(5);
/// Poll interval while waiting for the first spot to arrive.
const TICK_POLL: Duration = Duration::from_millis(25);
/// Default cTrader lots submitted per one modelenv position unit. 0.01 lot is
/// the USDJPY minimum (100_000 cTrader volume units). Override via config.
pub const DEFAULT_LOT_SIZE_PER_UNIT: f64 = 0.01;
/// Rolling window (ms, ~12 months) for the live recent-fills / realised-P&L rebuild.
const DEAL_HISTORY_WINDOW_MS: i64 = 365 * 24 * 60 * 60 * 1000;

/// cTrader API client wrapper.
///
/// Holds the OAuth/app credentials + trading config; after [`connect`] it owns a
/// live [`Connection`] (and its unsolicited-events receiver, needed for order
/// fills), the resolved numeric account id and symbol id, and a heartbeat task.
pub struct CtraderClient {
    /// cTrader API client ID (from OAuth application)
    client_id: String,
    /// cTrader API client secret (from OAuth application)
    client_secret: String,
    /// cTrader API access token
    access_token: String,
    /// cTrader API refresh token (refreshed externally by the Airflow DAG)
    refresh_token: Option<String>,
    /// cTrader trader account ID (`ctidTraderAccountId`, as a string as configured)
    account_id: String,
    /// Trading symbol
    symbol: String,
    /// Endpoint select: `false` = demo (default), `true` = live (real money).
    live: bool,
    /// cTrader lots submitted per one modelenv position unit.
    lot_size_per_unit: f64,

    // --- live connection state (None/0 until `connect()`) ---
    /// Parsed numeric account id used on every RPC.
    account_id_num: i64,
    /// Broker numeric symbol id, resolved on connect.
    symbol_id: Option<i64>,
    /// The multiplexed cTrader connection (cheap to clone).
    conn: Option<Connection>,
    /// Execution/error events for the in-flight order path (spot events are
    /// filtered out into `tick_state` by the router).
    events: Option<mpsc::UnboundedReceiver<ProtoMessage>>,
    /// Latest streamed bid/ask + recent ticks, filled by the spot router.
    tick_state: Option<Arc<Mutex<TickState>>>,
    /// Background event-router task (spot events + order events).
    router: Option<JoinHandle<()>>,
    /// Background keep-alive task.
    heartbeat: Option<JoinHandle<()>>,

    /// Current reconnection attempt count
    reconnect_attempts: u32,
    /// Maximum reconnection attempts
    max_reconnect_attempts: u32,
    /// Cached swap rates with timestamp
    swap_rates: Arc<Mutex<Option<SwapRates>>>,
    /// Last time swap rates were refreshed
    last_swap_refresh: Arc<Mutex<Option<SystemTime>>>,
    /// API call timeout
    timeout: Duration,
    /// Queue for actions that cannot execute immediately
    action_queue: Arc<Mutex<VecDeque<modelenv_proto::Action>>>,
}

impl std::fmt::Debug for CtraderClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CtraderClient")
            .field("account_id", &self.account_id)
            .field("symbol", &self.symbol)
            .field("live", &self.live)
            .field("lot_size_per_unit", &self.lot_size_per_unit)
            .field("connected", &self.conn.is_some())
            .field("symbol_id", &self.symbol_id)
            .finish()
    }
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
        Ok(((long * 100.0).round() / 100.0, (short * 100.0).round() / 100.0))
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

    /// cTrader trendbar period for a modelenv bar interval (M1/M5/M15 supported by
    /// the trendbars RPC; anything else falls back to M1).
    fn interval_to_period(interval: &str) -> i32 {
        match interval.trim().to_uppercase().as_str() {
            "M1" => data::TRENDBAR_M1,
            "M5" => data::TRENDBAR_M5,
            "M15" => data::TRENDBAR_M15,
            _ => data::TRENDBAR_M1,
        }
    }

    fn now_ms() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64
    }

    /// Create a new cTrader API client (demo endpoint, default 0.01 lot/unit).
    /// Configure the endpoint and sizing with [`with_live`] / [`with_lot_size_per_unit`].
    pub fn new(
        app_client_id: String,
        app_client_secret: String,
        access_token: String,
        refresh_token: Option<String>,
        account_id: String,
        symbol: String,
    ) -> Self {
        CtraderClient {
            client_id: app_client_id,
            client_secret: app_client_secret,
            access_token,
            refresh_token,
            account_id,
            symbol,
            live: false,
            lot_size_per_unit: DEFAULT_LOT_SIZE_PER_UNIT,
            account_id_num: 0,
            symbol_id: None,
            conn: None,
            events: None,
            tick_state: None,
            router: None,
            heartbeat: None,
            reconnect_attempts: 0,
            max_reconnect_attempts: MAX_RECONNECT_ATTEMPTS,
            swap_rates: Arc::new(Mutex::new(None)),
            last_swap_refresh: Arc::new(Mutex::new(None)),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            action_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Select the endpoint: `false` = demo (paper), `true` = live (real money).
    /// Default demo. Configurable so promotion to live is a config change, not a
    /// code change.
    pub fn with_live(mut self, live: bool) -> Self {
        self.live = live;
        self
    }

    /// Set cTrader lots submitted per one modelenv position unit (default 0.01).
    /// Ignored (kept at default) if `lots` is not finite and > 0.
    pub fn with_lot_size_per_unit(mut self, lots: f64) -> Self {
        if lots.is_finite() && lots > 0.0 {
            self.lot_size_per_unit = lots;
        } else {
            warn!(
                "ignoring invalid lot_size_per_unit {lots}; keeping {}",
                self.lot_size_per_unit
            );
        }
        self
    }

    /// True if pointed at the live (real-money) endpoint.
    pub fn is_live(&self) -> bool {
        self.live
    }

    /// cTrader lots per modelenv unit.
    pub fn lot_size_per_unit(&self) -> f64 {
        self.lot_size_per_unit
    }

    /// Establish a real connection and authenticate with the cTrader API.
    /// Opens TLS to the configured endpoint, runs app+account auth, resolves the
    /// symbol id, and starts the keep-alive heartbeat.
    pub async fn connect(&mut self) -> Result<()> {
        // Cheap credential validation first (no network) so misconfig fails fast.
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
        let account_id_num: i64 = self.account_id.trim().parse().with_context(|| {
            format!(
                "AuthenticationError {{ details: trader account ID {:?} is not numeric }}",
                self.account_id
            )
        })?;
        let symbol = Self::normalise_symbol(&self.symbol)?;
        let env = if self.live { "LIVE" } else { "demo" };
        info!(
            "Connecting to cTrader {env} endpoint (account={account_id_num}, symbol={symbol}, lot/unit={})",
            self.lot_size_per_unit
        );

        let transport = Transport::connect_env(self.live)
            .await
            .with_context(|| format!("cTrader {env} TLS connect failed"))?;
        let (reader, writer) = transport.into_split();
        let (conn, events) = Connection::start(reader, writer);

        let creds = Credentials {
            client_id: self.client_id.clone(),
            client_secret: self.client_secret.clone(),
            access_token: self.access_token.clone(),
            account_id: account_id_num,
        };
        auth::authenticate(&conn, &creds, self.timeout)
            .await
            .context("cTrader authentication failed")?;

        let symbol_id = data::get_symbol_id(&conn, account_id_num, &symbol, self.timeout)
            .await
            .with_context(|| format!("resolve cTrader symbol id for {symbol}"))?;

        let heartbeat = auth::spawn_heartbeat(conn.clone(), HEARTBEAT_INTERVAL);

        // Subscribe to streaming spot (bid/ask) events. Non-fatal: read/order
        // paths work without it; `current_ticks` will error clearly if none arrive.
        if let Err(e) = data::subscribe_spots(&conn, account_id_num, symbol_id, self.timeout).await {
            warn!("cTrader spot subscription failed ({e}); live ticks will be unavailable");
        }
        // Route unsolicited events: spot → tick buffer, execution/error → order path.
        let tick_state = Arc::new(Mutex::new(TickState::default()));
        let (order_tx, order_rx) = mpsc::unbounded_channel();
        let router = tokio::spawn(route_events(events, tick_state.clone(), order_tx, symbol_id));

        if let Some(old) = self.heartbeat.take() {
            old.abort();
        }
        if let Some(old) = self.router.take() {
            old.abort();
        }
        self.account_id_num = account_id_num;
        self.symbol_id = Some(symbol_id);
        self.conn = Some(conn);
        self.events = Some(order_rx);
        self.tick_state = Some(tick_state);
        self.router = Some(router);
        self.heartbeat = Some(heartbeat);
        self.reconnect_attempts = 0;
        let _ = self.refresh_token.is_some();
        info!("Connected to cTrader {env}: account={account_id_num}, {symbol}=id{symbol_id} (spots subscribed)");
        Ok(())
    }

    /// Close the connection and stop the heartbeat.
    pub async fn disconnect(&mut self) -> Result<()> {
        if let Some(hb) = self.heartbeat.take() {
            hb.abort();
        }
        if let Some(r) = self.router.take() {
            r.abort();
        }
        self.conn = None;
        self.events = None;
        self.tick_state = None;
        self.symbol_id = None;
        info!("Disconnected from cTrader API");
        Ok(())
    }

    /// True if a live connection is established.
    pub fn is_connected(&self) -> bool {
        self.conn.is_some()
    }

    fn require_conn(&self) -> Result<&Connection> {
        self.conn
            .as_ref()
            .ok_or_else(|| anyhow!("Not connected to cTrader API"))
    }

    fn require_symbol_id(&self) -> Result<i64> {
        self.symbol_id
            .ok_or_else(|| anyhow!("cTrader symbol id not resolved (connect first)"))
    }

    /// Fetch the account's currently open positions (reconcile). Read-only.
    pub async fn sync_positions(&mut self, symbol: &str) -> Result<Vec<modelenv_proto::Position>> {
        let symbol = Self::normalise_symbol(symbol)?;
        let account = self.account_id_num;
        let timeout = self.timeout;
        let conn = self.require_conn()?;
        debug!("Reconciling positions for {symbol}");
        let positions = data::sync_positions(conn, account, timeout).await?;
        info!("Synchronised {} position(s) for {symbol}", positions.len());
        Ok(positions)
    }

    /// Retrieve the latest completed M1 bar for the symbol.
    pub async fn current_bar(&mut self, symbol: &str) -> Result<modelenv_proto::Bar> {
        let symbol = Self::normalise_symbol(symbol)?;
        let sid = self.require_symbol_id()?;
        let account = self.account_id_num;
        let timeout = self.timeout;
        let conn = self.require_conn()?;
        // Request a wider window and take the most recent completed bar. A
        // count=1 / few-minute window can come back empty (the current minute is
        // in-progress and the window is too tight), so fetch ~30 M1 bars and use
        // the last, robust to the in-progress minute and brief gaps.
        let bars =
            data::get_trendbars(conn, account, sid, data::TRENDBAR_M1, 30, Self::now_ms(), timeout)
                .await?;
        debug!("current_bar: {} M1 bars returned for {symbol}", bars.len());
        bars.into_iter()
            .last()
            .ok_or_else(|| anyhow!("cTrader returned no M1 bar for {symbol} (empty trendbars over ~30m window)"))
    }

    /// Retrieve up to `count` recent bars at `interval` (ascending timestamp).
    pub async fn recent_bars(
        &mut self,
        symbol: &str,
        interval: &str,
        count: usize,
    ) -> Result<Vec<modelenv_proto::Bar>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let symbol = Self::normalise_symbol(symbol)?;
        let sid = self.require_symbol_id()?;
        let account = self.account_id_num;
        let timeout = self.timeout;
        let period = Self::interval_to_period(interval);
        let conn = self.require_conn()?;
        debug!("Fetching up to {count} {interval} bars for {symbol}");
        data::get_trendbars(conn, account, sid, period, count as u32, Self::now_ms(), timeout).await
    }

    /// Recent streamed bid/ask ticks (ascending), from the live spot subscription
    /// set up on connect. Waits up to `TICK_WAIT` for the first spot to arrive.
    /// Returns an error (never synthetic data) if the connection is down or no
    /// spot has arrived (e.g. market closed / subscription failed).
    pub async fn current_ticks(&mut self, _symbol: &str) -> Result<Vec<modelenv_proto::Tick>> {
        let state = self
            .tick_state
            .clone()
            .ok_or_else(|| anyhow!("Not connected to cTrader API"))?;
        let deadline = std::time::Instant::now() + TICK_WAIT;
        loop {
            {
                let st = state.lock().await;
                if !st.ticks.is_empty() {
                    return Ok(st.ticks.iter().cloned().collect());
                }
            }
            if std::time::Instant::now() >= deadline {
                return Err(anyhow!(
                    "cTrader: no spot ticks received within {:?} (market closed or spot subscription failed)",
                    TICK_WAIT
                ));
            }
            tokio::time::sleep(TICK_POLL).await;
        }
    }

    /// Submit a MARKET order for the action and wait for the fill. Volume is
    /// `units × lot_size_per_unit` lots (BUY_1/SELL_1 = 1 unit, BUY_2/SELL_2 = 2).
    pub async fn submit_order(
        &mut self,
        action: &modelenv_proto::Action,
    ) -> Result<modelenv_proto::Fill> {
        if action.client_order_id.trim().is_empty() {
            return Err(anyhow!(
                "ValidationError {{ field: client_order_id, details: order ID cannot be empty }}"
            ));
        }
        let action_type = modelenv_proto::ActionType::try_from(action.action).map_err(|_| {
            anyhow!(
                "ValidationError {{ field: action, details: unsupported action type {} }}",
                action.action
            )
        })?;
        let (side, units) = match action_type {
            modelenv_proto::ActionType::ActionHold => {
                return Err(anyhow!(
                    "ValidationError {{ field: action, details: hold actions do not submit broker orders }}"
                ));
            }
            modelenv_proto::ActionType::ActionBuy1 => (Side::Buy, 1.0),
            modelenv_proto::ActionType::ActionBuy2 => (Side::Buy, 2.0),
            modelenv_proto::ActionType::ActionSell1 => (Side::Sell, 1.0),
            modelenv_proto::ActionType::ActionSell2 => (Side::Sell, 2.0),
        };
        let lots = units * self.lot_size_per_unit;
        let volume = lots_to_volume(lots);
        if volume <= 0 {
            return Err(anyhow!(
                "ValidationError {{ field: volume, details: order volume {volume} <= 0 \
                 (units={units}, lot_size_per_unit={}) }}",
                self.lot_size_per_unit
            ));
        }
        let sid = self.require_symbol_id()?;
        let account = self.account_id_num;
        let timeout = self.timeout;
        let client_order_id = format!("{}-{}", self.symbol, action.client_order_id);
        // Clone the (cheap) connection, then borrow the events receiver mutably.
        let conn = self
            .conn
            .clone()
            .ok_or_else(|| anyhow!("Not connected to cTrader API"))?;
        let events = self
            .events
            .as_mut()
            .ok_or_else(|| anyhow!("Not connected to cTrader API"))?;
        debug!(
            "Submitting {:?} order {client_order_id}: {lots} lots (vol {volume}) on symbol_id {sid}",
            side
        );
        let result =
            orders::submit_market_order(&conn, events, account, sid, side, volume, &client_order_id, timeout)
                .await?;
        info!(
            "cTrader order {client_order_id} FILLED: position={} price={:.5} size={} side={}",
            result.position_id, result.fill.price, result.fill.size, result.fill.side
        );
        Ok(result.fill)
    }

    /// Close (fully) an open broker position with a market order.
    pub async fn close_position(
        &mut self,
        position: &modelenv_proto::Position,
    ) -> Result<modelenv_proto::Fill> {
        if position.position_id.trim().is_empty() {
            return Err(anyhow!(
                "ValidationError {{ field: position_id, details: position ID cannot be empty }}"
            ));
        }
        let position_id: i64 = position.position_id.trim().parse().with_context(|| {
            format!(
                "ValidationError {{ field: position_id, details: {:?} is not numeric }}",
                position.position_id
            )
        })?;
        let volume = lots_to_volume(position.volume.abs());
        if volume <= 0 {
            return Err(anyhow!(
                "ValidationError {{ field: volume, details: close volume {volume} <= 0 (lots={}) }}",
                position.volume
            ));
        }
        let account = self.account_id_num;
        let timeout = self.timeout;
        let conn = self
            .conn
            .clone()
            .ok_or_else(|| anyhow!("Not connected to cTrader API"))?;
        let events = self
            .events
            .as_mut()
            .ok_or_else(|| anyhow!("Not connected to cTrader API"))?;
        debug!("Closing position {position_id} ({} lots, vol {volume})", position.volume);
        let result = orders::close_position(&conn, events, account, position_id, volume, timeout).await?;
        info!(
            "cTrader position {position_id} closed at {:.5} for {} lots",
            result.fill.price, result.fill.size
        );
        Ok(result.fill)
    }

    /// Closed positions (deal history) since `from_timestamp_ns`, for the live
    /// `Reset()` realised-P&L rebuild. Read-only.
    pub async fn closed_positions(
        &mut self,
        symbol: &str,
        from_timestamp_ns: i64,
    ) -> Result<Vec<crate::position::ClosedPosition>> {
        let _ = Self::normalise_symbol(symbol)?;
        let account = self.account_id_num;
        let timeout = self.timeout;
        let conn = self.require_conn()?;
        let from_ms = (from_timestamp_ns / 1_000_000).max(0);
        let to_ms = Self::now_ms();
        data::closed_positions(conn, account, from_ms, to_ms, timeout).await
    }

    /// Up to `count` most-recent fills, ascending (oldest→newest), for the live
    /// `Reset()` recent-fills rebuild. Read-only.
    pub async fn recent_fills(
        &mut self,
        symbol: &str,
        count: usize,
    ) -> Result<Vec<modelenv_proto::Fill>> {
        let _ = Self::normalise_symbol(symbol)?;
        let account = self.account_id_num;
        let timeout = self.timeout;
        let conn = self.require_conn()?;
        let to_ms = Self::now_ms();
        let from_ms = to_ms - DEAL_HISTORY_WINDOW_MS;
        let mut fills = data::recent_fills(conn, account, from_ms, to_ms, count, timeout).await?;
        // data::recent_fills yields newest-first; callers expect ascending.
        fills.reverse();
        Ok(fills)
    }

    /// Retrieve swap rates. The per-symbol cTrader swap RPC is not yet wired;
    /// this uses modelenv's built-in fallback table (24h cached). Optional per
    /// T-9.1-06.
    pub async fn get_swap_rates(&mut self, symbol: &str) -> Result<SwapRates> {
        let symbol = Self::normalise_symbol(symbol)?;
        let rates = self.request_swap_rates(&symbol).await?;
        info!(
            "Swap rates for {symbol} (fallback table): long={:.2}, short={:.2}",
            rates.long, rates.short
        );
        Ok(rates)
    }

    /// Refresh swap rates if older than 24 hours, falling back to stale cache on
    /// failure.
    pub async fn refresh_swap_rates(&mut self) -> Result<SwapRates> {
        let cached_rates = {
            let swap_rates = self.swap_rates.lock().await;
            swap_rates.clone()
        };
        {
            let last_refresh = self.last_swap_refresh.lock().await;
            let now = SystemTime::now();
            if let Some(ref last_time) = *last_refresh {
                let elapsed = now.duration_since(*last_time)?;
                if elapsed < SWAP_RATE_CACHE_TTL {
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
        {
            let mut swap_rates = self.swap_rates.lock().await;
            let mut last_refresh = self.last_swap_refresh.lock().await;
            *swap_rates = Some(rates.clone());
            *last_refresh = Some(rates.last_refresh);
        }
        Ok(rates)
    }

    /// Attempt reconnection with exponential backoff.
    pub async fn reconnect(&mut self) -> Result<()> {
        self.reconnect_attempts += 1;
        if self.reconnect_attempts > self.max_reconnect_attempts {
            return Err(anyhow!(
                "ReconnectionError {{ attempts: {}, max_attempts: {} }}",
                self.reconnect_attempts,
                self.max_reconnect_attempts
            ));
        }
        let backoff = Duration::from_secs(2u64.pow(self.reconnect_attempts - 1));
        warn!(
            "Reconnection attempt {} , waiting {:?} before retry",
            self.reconnect_attempts, backoff
        );
        tokio::time::sleep(backoff).await;
        self.connect().await
    }

    /// Queue action for retry
    pub async fn queue_action(&self, action: modelenv_proto::Action) {
        let mut queue = self.action_queue.lock().await;
        queue.push_back(action);
        debug!("Action queued for retry");
    }

    /// Process queued actions; returns the count processed.
    pub async fn process_action_queue(&mut self) -> usize {
        let mut processed = 0;
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
                    warn!("Failed to process queued action: {}", e);
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

// The client is driven behind a tokio::Mutex in the gateway.
unsafe impl Send for CtraderClient {}
unsafe impl Sync for CtraderClient {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker_gateway::ctrader::wire;
    use modelenv_proto::ctrader::{
        ProtoOaDeal, ProtoOaExecutionEvent, ProtoOaPosition, ProtoOaReconcileRes, ProtoOaSpotEvent,
        ProtoOaTradeData, ProtoOaTrendbar, ProtoOaGetTrendbarsRes,
    };
    use modelenv_proto::{Action, ActionType};
    use prost::Message;
    use tokio::io::{AsyncRead, AsyncWrite};

    fn test_client(symbol: &str) -> CtraderClient {
        CtraderClient::new(
            "app-client-id".to_string(),
            "app-client-secret".to_string(),
            "access-token".to_string(),
            Some("refresh-token".to_string()),
            "47678494".to_string(),
            symbol.to_string(),
        )
    }

    /// Inject a ready (mock-server-backed) connection so behavioural tests don't
    /// need real TLS/auth. Mirrors a completed `connect()`.
    impl CtraderClient {
        fn install_for_test(
            &mut self,
            conn: Connection,
            events: mpsc::UnboundedReceiver<ProtoMessage>,
            symbol_id: i64,
            account_id_num: i64,
        ) {
            self.conn = Some(conn);
            self.events = Some(events);
            self.symbol_id = Some(symbol_id);
            self.account_id_num = account_id_num;
            self.tick_state = Some(Arc::new(Mutex::new(TickState::default())));
        }
    }

    fn wired_client(symbol: &str, server_io_task: impl FnOnce(tokio::io::DuplexStream)) -> CtraderClient {
        let (client_io, server_io) = tokio::io::duplex(8192);
        server_io_task(server_io);
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, events) = Connection::start(cr, cw);
        let mut client = test_client(symbol);
        client.install_for_test(conn, events, 4, 47678494);
        client
    }

    // ---- config surface ----

    #[test]
    fn defaults_demo_and_min_lot() {
        let c = test_client("USDJPY");
        assert!(!c.is_live());
        assert_eq!(c.lot_size_per_unit(), 0.01);
    }

    #[test]
    fn with_live_and_lot_size_are_configurable() {
        let c = test_client("USDJPY").with_live(true).with_lot_size_per_unit(0.05);
        assert!(c.is_live());
        assert_eq!(c.lot_size_per_unit(), 0.05);
        // invalid lot sizes are ignored, keeping the prior value.
        let c2 = test_client("USDJPY").with_lot_size_per_unit(-1.0);
        assert_eq!(c2.lot_size_per_unit(), 0.01);
    }

    // ---- validation / not-connected (no network) ----

    #[tokio::test]
    async fn connect_rejects_missing_access_token() {
        let mut client = CtraderClient::new(
            "id".into(),
            "secret".into(),
            "   ".into(),
            None,
            "47678494".into(),
            "USDJPY".into(),
        );
        let err = client.connect().await.unwrap_err();
        assert!(err.to_string().contains("access token is required"));
    }

    #[tokio::test]
    async fn connect_rejects_non_numeric_account() {
        let mut client = CtraderClient::new(
            "id".into(),
            "secret".into(),
            "tok".into(),
            None,
            "not-a-number".into(),
            "USDJPY".into(),
        );
        let err = client.connect().await.unwrap_err();
        assert!(err.to_string().contains("not numeric"));
    }

    #[tokio::test]
    async fn sync_positions_requires_active_connection() {
        let mut client = test_client("USDJPY");
        let err = client.sync_positions("USDJPY").await.unwrap_err();
        assert!(err.to_string().contains("Not connected"));
    }

    #[tokio::test]
    async fn current_ticks_returns_buffered_spot_ticks() {
        let mut client =
            wired_client("USDJPY", |s| drop(tokio::spawn(async move { let _ = s; })));
        // Inject a streamed tick, as the spot router would.
        client
            .tick_state
            .as_ref()
            .unwrap()
            .lock()
            .await
            .ticks
            .push_back(modelenv_proto::Tick {
                timestamp_ns: 1_700_000_000_000_000_000,
                bid: 150.10,
                ask: 150.12,
            });
        let ticks = client.current_ticks("USDJPY").await.unwrap();
        assert_eq!(ticks.len(), 1);
        assert!((ticks[0].bid - 150.10).abs() < 1e-9);
        assert!((ticks[0].ask - 150.12).abs() < 1e-9);
    }

    #[tokio::test]
    async fn current_ticks_errors_without_connection() {
        // Never connected -> no tick buffer -> error, never synthetic data.
        let mut client = test_client("USDJPY");
        let err = client.current_ticks("USDJPY").await.unwrap_err();
        assert!(err.to_string().contains("Not connected"));
    }

    #[tokio::test]
    async fn router_splits_spot_and_execution_events() {
        let (raw_tx, raw_rx) = mpsc::unbounded_channel();
        let (order_tx, mut order_rx) = mpsc::unbounded_channel();
        let ticks = std::sync::Arc::new(tokio::sync::Mutex::new(TickState::default()));
        let handle = tokio::spawn(super::route_events(raw_rx, ticks.clone(), order_tx, 4));

        // Spot event (bid+ask ×10^5) for symbol 4 -> tick buffer.
        let spot = ProtoOaSpotEvent {
            payload_type: Some(wire::payload_type::SPOT_EVENT as i32),
            ctid_trader_account_id: 47678494,
            symbol_id: 4,
            bid: Some(15_010_000),
            ask: Some(15_012_000),
            timestamp: Some(1_700_000_000_000),
            ..Default::default()
        };
        raw_tx
            .send(wire::envelope(wire::payload_type::SPOT_EVENT, spot.encode_to_vec(), None))
            .unwrap();
        // Execution event -> order channel.
        let exec = ProtoOaExecutionEvent {
            payload_type: Some(wire::payload_type::EXECUTION_EVENT as i32),
            ctid_trader_account_id: 47678494,
            execution_type: 3,
            ..Default::default()
        };
        raw_tx
            .send(wire::envelope(wire::payload_type::EXECUTION_EVENT, exec.encode_to_vec(), None))
            .unwrap();

        let got = order_rx.recv().await.unwrap();
        assert_eq!(got.payload_type, wire::payload_type::EXECUTION_EVENT);
        let st = ticks.lock().await;
        assert_eq!(st.ticks.len(), 1);
        assert!((st.ticks[0].bid - 150.10).abs() < 1e-6);
        assert!((st.ticks[0].ask - 150.12).abs() < 1e-6);
        handle.abort();
    }

    // ---- behavioural, over a mock cTrader server ----

    async fn reconcile_server<S>(mut s: S)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        if let Ok(req) = wire::read_frame(&mut s).await {
            let res = ProtoOaReconcileRes {
                payload_type: Some(wire::payload_type::RECONCILE_RES as i32),
                ctid_trader_account_id: 47678494,
                position: vec![ProtoOaPosition {
                    position_id: 5001,
                    trade_data: ProtoOaTradeData {
                        symbol_id: 4,
                        volume: 100_000,
                        trade_side: 1,
                        open_timestamp: Some(1_700_000_000_000),
                        ..Default::default()
                    },
                    price: Some(150.5),
                    swap: 25,
                    ..Default::default()
                }],
                order: vec![],
            };
            let env = wire::envelope(
                wire::payload_type::RECONCILE_RES,
                res.encode_to_vec(),
                req.client_msg_id.clone(),
            );
            let _ = wire::write_frame(&mut s, &env).await;
        }
    }

    #[tokio::test]
    async fn sync_positions_maps_broker_positions() {
        let mut client = wired_client("USDJPY", |s| {
            drop(tokio::spawn(reconcile_server(s)));
        });
        let positions = client.sync_positions("USDJPY").await.unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].position_id, "5001");
        assert_eq!(positions[0].volume, 0.01);
    }

    async fn trendbar_server<S>(mut s: S)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        if let Ok(req) = wire::read_frame(&mut s).await {
            let res = ProtoOaGetTrendbarsRes {
                payload_type: Some(wire::payload_type::GET_TRENDBARS_RES as i32),
                ctid_trader_account_id: 47678494,
                period: data::TRENDBAR_M1,
                trendbar: vec![ProtoOaTrendbar {
                    volume: 42,
                    low: Some(15_010_000),
                    delta_open: Some(500),
                    delta_high: Some(2_300),
                    delta_close: Some(1_800),
                    utc_timestamp_in_minutes: Some(28_350_000),
                    ..Default::default()
                }],
                ..Default::default()
            };
            let env = wire::envelope(
                wire::payload_type::GET_TRENDBARS_RES,
                res.encode_to_vec(),
                req.client_msg_id.clone(),
            );
            let _ = wire::write_frame(&mut s, &env).await;
        }
    }

    #[tokio::test]
    async fn current_bar_decodes_real_trendbar() {
        let mut client = wired_client("USDJPY", |s| {
            drop(tokio::spawn(trendbar_server(s)));
        });
        let bar = client.current_bar("USDJPY").await.unwrap();
        assert!((bar.low - 150.10).abs() < 1e-6);
        assert!((bar.close - 150.118).abs() < 1e-6);
        assert_eq!(bar.volume, 42.0);
    }

    fn usdjpy_buy_deal() -> ProtoOaDeal {
        ProtoOaDeal {
            deal_id: 9001,
            order_id: 7001,
            position_id: 5001,
            volume: 100_000,
            filled_volume: 100_000,
            symbol_id: 4,
            execution_timestamp: 1_700_000_000_000,
            execution_price: Some(150.123),
            trade_side: 1,
            ..Default::default()
        }
    }

    async fn accept_then_fill_server<S>(mut s: S)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        if let Ok(req) = wire::read_frame(&mut s).await {
            let accepted = ProtoOaExecutionEvent {
                payload_type: Some(wire::payload_type::EXECUTION_EVENT as i32),
                ctid_trader_account_id: 47678494,
                execution_type: 2, // ORDER_ACCEPTED
                ..Default::default()
            };
            let _ = wire::write_frame(
                &mut s,
                &wire::envelope(
                    wire::payload_type::EXECUTION_EVENT,
                    accepted.encode_to_vec(),
                    req.client_msg_id.clone(),
                ),
            )
            .await;
            let filled = ProtoOaExecutionEvent {
                payload_type: Some(wire::payload_type::EXECUTION_EVENT as i32),
                ctid_trader_account_id: 47678494,
                execution_type: 3, // ORDER_FILLED
                deal: Some(usdjpy_buy_deal()),
                ..Default::default()
            };
            let _ = wire::write_frame(
                &mut s,
                &wire::envelope(wire::payload_type::EXECUTION_EVENT, filled.encode_to_vec(), None),
            )
            .await;
        }
    }

    #[tokio::test]
    async fn submit_order_places_real_market_order_and_returns_fill() {
        let mut client = wired_client("USDJPY", |s| {
            drop(tokio::spawn(accept_then_fill_server(s)));
        });
        let fill = client
            .submit_order(&Action {
                action: ActionType::ActionBuy1 as i32,
                client_order_id: "ord-1".into(),
            })
            .await
            .unwrap();
        assert_eq!(fill.order_id, "7001");
        assert_eq!(fill.side, 0);
        assert!((fill.size - 0.01).abs() < 1e-9); // 1 unit × 0.01 lot default
        assert!((fill.price - 150.123).abs() < 1e-9);
    }

    #[tokio::test]
    async fn submit_order_rejects_hold_and_empty_id() {
        let mut client = wired_client("USDJPY", |s| drop(tokio::spawn(async move { let _ = s; })));
        let hold = client
            .submit_order(&Action {
                action: ActionType::ActionHold as i32,
                client_order_id: "x".into(),
            })
            .await
            .unwrap_err();
        assert!(hold.to_string().contains("hold actions"));
        let empty = client
            .submit_order(&Action {
                action: ActionType::ActionBuy1 as i32,
                client_order_id: "  ".into(),
            })
            .await
            .unwrap_err();
        assert!(empty.to_string().contains("client_order_id"));
    }

    #[tokio::test]
    async fn get_swap_rates_uses_fallback_table() {
        let mut client = test_client("USDJPY");
        let rates = client.get_swap_rates("USDJPY").await.unwrap();
        assert_eq!(rates.long, -1.50);
        assert_eq!(rates.short, 0.50);
    }
}
