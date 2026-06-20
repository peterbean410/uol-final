// Environment module
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

use log::info;
use modelenv_proto::{
    Action, ActionType, Bar, BarList, FillSide, ObserveRequest, RecentBarsRequest,
    RecentBarsResponse, Reference, ResetRequest, StepResponse, Tick,
};

use crate::broker_gateway::BrokerGateway;
use crate::config::Mode;
use crate::data_loader::{now_ns, DEFAULT_LOCAL_CACHE_DIR, TIME_INTERVALS};
use crate::episode::{initialize_episode, preload_training_market_data, Episode, RECENT_WINDOW};
use crate::live_data::LiveData;
use crate::indicators::{
    compute_interval_indicators, compute_m15_double_bottom_high, compute_m15_double_bottom_low,
    compute_m15_double_top_high, compute_m15_double_top_low, compute_time_features,
    detect_all_patterns, state_columns,
};
use crate::normalisation::Normaliser;
use crate::market_data_cache::MarketDataCache;
use crate::position::{ClosedPositionWindow, Position, Side};
use crate::reconciliation::reconcile_positions;
use crate::trade_log::{FillEvent, TradeLogger};

const DEFAULT_STEP_SIZE_NS: i64 = 5_000_000_000;

/// Convert a Unix-seconds timestamp from a Reset/proto request to nanoseconds
/// for internal use. Passes through 0 unchanged ("no constraint" sentinel).
fn seconds_to_ns(ts_seconds: i64, field_name: &str) -> Result<i64> {
    if ts_seconds == 0 {
        return Ok(0);
    }
    ts_seconds.checked_mul(1_000_000_000).ok_or_else(|| {
        anyhow::anyhow!("{} ({}) is too large to convert to nanoseconds", field_name, ts_seconds)
    })
}

/// Relative BB-width boundaries for low/mid and mid/high volatility regime bins.
const DEFAULT_VOL_LOW_THRESHOLD: f64 = 0.0008; // 0.08% of price
const DEFAULT_VOL_HIGH_THRESHOLD: f64 = 0.0020; // 0.20% of price

/// Per-bin running reward statistics for regime-conditional z-score normalisation.
#[derive(Clone, Default)]
struct BinStats {
    count: u64,
    sum: f64,
    sum_sq: f64,
}

impl BinStats {
    fn update(&mut self, reward: f64) {
        self.count += 1;
        self.sum += reward;
        self.sum_sq += reward * reward;
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f64
    }

    fn std_dev(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance = (self.sum_sq / self.count as f64) - (mean * mean);
        variance.max(0.0).sqrt()
    }
}

/// Three-bin regime-conditional reward normalisation.
///
/// Bin 0 = low vol, Bin 1 = mid vol, Bin 2 = high vol.
/// Stats persist across episodes so the z-score scale is mature from step one
/// of every episode. Only `prev_total_equity` resets per episode.
#[derive(Clone)]
struct RegimeRewardStats {
    bins: [BinStats; 3],
    vol_low_threshold: f64,
    vol_high_threshold: f64,
}

impl RegimeRewardStats {
    fn new(vol_low_threshold: f64, vol_high_threshold: f64) -> Self {
        RegimeRewardStats {
            bins: [BinStats::default(), BinStats::default(), BinStats::default()],
            vol_low_threshold,
            vol_high_threshold,
        }
    }

    fn regime_bin(&self, bb_width_relative: f64) -> usize {
        if bb_width_relative <= 0.0 {
            // No data yet, default to mid-vol
            return 1;
        }
        if bb_width_relative < self.vol_low_threshold {
            0
        } else if bb_width_relative < self.vol_high_threshold {
            1
        } else {
            2
        }
    }

    fn update(&mut self, bin: usize, reward: f64) {
        self.bins[bin].update(reward);
    }

    fn normalise(&self, bin: usize, reward: f64) -> f64 {
        let stats = &self.bins[bin];
        let std_dev = stats.std_dev();
        if std_dev > 1e-8 {
            (reward - stats.mean()) / std_dev
        } else {
            // Fall back to a different bin's stats if this bin is mature enough,
            // otherwise pass reward through raw (first few steps globally).
            reward
        }
    }
}

/// The main environment struct
#[derive(Clone)]
pub struct Environment {
    mode: Mode,
    symbol: String,
    s3_prefix: String,
    local_cache_dir: String,
    price_snapshot_ts: Option<i64>,
    /// Optional (start_ns, end_ns) constraining the training-mode startup
    /// preload to a narrow tick range. ``None`` falls back to the full M1
    /// reference span. Reset() still loads outside-the-window data lazily.
    training_tick_window: Option<(i64, i64)>,
    market_data_cache: MarketDataCache,
    step_size_ns: i64,
    episode: Option<Episode>,
    positions: Vec<Position>,
    closed_position_window: ClosedPositionWindow,
    recent_fills: Vec<Fill>,
    last_action: Option<ActionType>,
    // Configuration for P/L calculations
    transaction_cost: f64,
    // Per-symbol (long, short) daily swap rates: the per-day, per-unit-volume
    // financing applied to open positions when a day boundary is crossed.
    // Defaults to (0, 0) for an unconfigured symbol, i.e. no overnight financing.
    daily_swap_rates: HashMap<String, (f64, f64)>,
    // Track the last timestamp when swap was accrued for day boundary detection
    last_swap_accrual_timestamp: i64,
    // Trading-session start hour (UTC, mod 24; None = 0h). Anchors the
    // `session_realised_pnl` observation feature: positions closed since the
    // most recent session start count toward it. Does not gate trading.
    trading_session_start_hour: Option<u32>,
    // Trading-session end hour (UTC, mod 24). When set, the environment
    // liquidates at each session end: close all shorts + losing longs (net of
    // swap), keep winning/break-even longs. Training/backtest closes against
    // the episode book; Live places close orders at the broker. None disables
    // session liquidation.
    trading_session_end_hour: Option<u32>,
    // Broker gateway for Production Mode (Arc for cloneability)
    broker_gateway: Option<Arc<dyn BrokerGateway + Send + Sync>>,
    // Reward function configuration
    reward_lambda: f64,
    reward_action_penalty: f64,
    reward_holding_penalty: f64,
    // Fixed scale for the reward denominator. delta_v_t and penalties are divided
    // by this constant so different currency pairs train on comparable reward
    // magnitudes. For USDJPY one pip = 0.01 price units, so 0.01 is a natural
    // default (a 1-pip gain → reward ≈ 1). Unlike the old per-regime z-score
    // normalisation, this preserves the relative scale of large vs small moves
    // so the agent can distinguish a 50-pip loss from a 1-pip loss.
    reward_scale: f64,
    // Symmetric clamp on the final (post-scale) per-step reward. A defensive
    // rail against data artifacts (synthetic M1-derived ticks with bid=low /
    // ask=high, or price gaps) producing an oversized single-step mark; it
    // does NOT reshape normal signal. <= 0 disables the clamp.
    reward_clip: f64,
    disable_hedging: bool,
    // Leverage ratio for margin calculation: max_total_margin = total_notional / leverage.
    leverage: f64,
    // Track previous step's total equity for delta_V_t calculation
    prev_total_equity: Option<f64>,
    // Raw (un-normalised) equity change from the most recent calculate_reward
    // call, surfaced on the Observation as raw_pnl_delta for monetary reporting.
    last_raw_pnl_delta: f64,
    // Peak total margin required in the current episode (sum of volume *
    // entry_price across open positions, divided by leverage).  Resets to 0.0 on
    // reset(), monotonically increases during the episode.  Units are
    // quote-currency notional / leverage (e.g. JPY / leverage for USDJPY).
    max_total_margin: f64,
    // Regime-conditional reward normalisation (3 volatility bins).
    // Stats persist across episodes; only prev_total_equity resets.
    regime_reward_stats: RegimeRewardStats,
    current_regime_bin: usize,
    last_observation_timestamp_ns: Option<i64>,
    // State feature normaliser (rolling z-scores, volume log-transform, etc.).
    normaliser: Normaliser,
    // In-memory bar storage shared by training and live modes.
    // Training: populated from Episode after parquet load.
    // Live: updated from broker gateway on each observation.
    bars: HashMap<String, Vec<Bar>>,
    // Optional durable JSONL log of fills and position closes. None disables it.
    // Purely a side-channel for offline review/debugging; never read back.
    trade_logger: Option<Arc<TradeLogger>>,
}

/// Represents a trade execution record
#[derive(Debug, Clone)]
pub struct Fill {
    pub order_id: String,
    pub timestamp_ns: i64,
    pub price: f64,
    pub size: f64,
    pub side: FillSide,
    pub partial: bool,
}

/// Built-in daily swap (overnight financing) rates for Training/backtest mode,
/// keyed by symbol -> (long, short) in price units per unit volume per day
/// (1 pip = 0.01, volume 1.0 = 1 standard lot). A negative value is a cost.
///
/// These are used ONLY in Training mode (training + backtest). In Live mode the
/// table is left empty and real swap is synced from the broker on position sync.
pub fn default_daily_swap_rates() -> HashMap<String, (f64, f64)> {
    // Pepperstone Standard USD/JPY, per 1 lot ($100,000): long earns ~+11.21
    // pips/day, short pays ~-24.24 pips/day. modelenv works in price units
    // (1 pip = 0.01), so multiply the pip figures by 0.01.
    HashMap::from([("USDJPY".to_string(), (0.1121, -0.2424))])
}

/// The built-in (long, short) Training-mode swap default for `symbol`, or
/// (0.0, 0.0) if the symbol has no entry in [`default_daily_swap_rates`].
pub fn default_swap_rate_for(symbol: &str) -> (f64, f64) {
    default_daily_swap_rates()
        .get(symbol)
        .copied()
        .unwrap_or((0.0, 0.0))
}

impl Environment {
    /// Create a new environment instance
    pub fn new(mode: Mode, symbol: String, s3_prefix: String) -> Self {
        // Seed the built-in swap-rate table only in Training/backtest mode; Live
        // mode pulls real swap from the broker, so it starts empty.
        let daily_swap_rates = if matches!(mode, Mode::Training) {
            default_daily_swap_rates()
        } else {
            HashMap::new()
        };
        Environment {
            mode,
            symbol,
            s3_prefix,
            local_cache_dir: DEFAULT_LOCAL_CACHE_DIR.to_string(),
            price_snapshot_ts: None,
            training_tick_window: None,
            market_data_cache: MarketDataCache::new(),
            step_size_ns: DEFAULT_STEP_SIZE_NS,
            episode: None,
            positions: Vec::new(),
            closed_position_window: ClosedPositionWindow::new(),
            recent_fills: Vec::new(),
            last_action: None,
            transaction_cost: 0.0, // Default no transaction cost
            daily_swap_rates,
            last_swap_accrual_timestamp: 0, // Will be set on reset
            trading_session_start_hour: None,
            trading_session_end_hour: None,
            broker_gateway: None,
            reward_lambda: 0.5, // Linear downside weight: losses weigh (1+λ)=1.5x gains
            reward_action_penalty: 0.001, // Default action penalty (scaled to USD/JPY spread)
            reward_holding_penalty: 1e-6, // Default holding penalty (orders of magnitude smaller)
            reward_scale: 0.01, // 1 pip = 0.01 price units for USDJPY; tune for other CCYs
            reward_clip: 150.0, // Artifact rail; bounds a 1-yen (100-pip) step at λ=0.5
            disable_hedging: true,
            leverage: 200.0,
            prev_total_equity: None,
            last_raw_pnl_delta: 0.0,
            max_total_margin: 0.0,
            regime_reward_stats: RegimeRewardStats::new(
                DEFAULT_VOL_LOW_THRESHOLD,
                DEFAULT_VOL_HIGH_THRESHOLD,
            ),
            current_regime_bin: 1, // Default to mid-vol until first observation
            last_observation_timestamp_ns: None,
            normaliser: Normaliser::new(&state_columns()),
            bars: HashMap::new(),
            trade_logger: None,
        }
    }

    /// Set the transaction cost per trade
    pub fn with_transaction_cost(mut self, cost: f64) -> Self {
        self.transaction_cost = cost;
        self
    }

    /// Set the (long, short) daily swap rates for a symbol.
    ///
    /// Each is the per-day, per-unit-volume financing accrued on an open
    /// position of that side when a day boundary is crossed (long rate for BUY
    /// positions, short rate for SELL). A negative rate is a cost; positive is a
    /// credit. Leaving this unset (the default) means no overnight financing.
    pub fn with_daily_swap_rate(mut self, symbol: String, long_rate: f64, short_rate: f64) -> Self {
        self.daily_swap_rates.insert(symbol, (long_rate, short_rate));
        self
    }

    /// Enable end-of-session liquidation at `session_end_hour` (UTC, mod 24).
    ///
    /// When set, each time a step crosses the session-end instant, all short
    /// positions and all losing long positions (net of accrued swap) are
    /// closed, and winning/break-even long positions carry into the next
    /// session. Training/backtest closes against the episode book; Live
    /// detects the crossing on wall-clock step timestamps and places close
    /// orders at the broker. `None` disables it.
    /// Set the trading-session start hour (UTC, mod 24; `None` = 0h UTC).
    ///
    /// Anchors the session-scoped realised P&L observation feature (proto
    /// field `session_realised_pnl`): positions closed since the most recent
    /// session-start instant count toward it. Does not gate trading.
    pub fn with_trading_session_start_hour(mut self, session_start_hour: Option<u32>) -> Self {
        self.trading_session_start_hour = session_start_hour;
        self
    }

    pub fn with_trading_session_end_hour(mut self, session_end_hour: Option<u32>) -> Self {
        self.trading_session_end_hour = session_end_hour;
        self
    }

    pub fn with_price_snapshot_ts(mut self, price_snapshot_ts: i64) -> Self {
        self.price_snapshot_ts = Some(price_snapshot_ts);
        self
    }

    /// Scope the training-mode preload to a specific (start_ns, end_ns) tick
    /// window. See `Config::training_tick_window` for the computation.
    pub fn with_training_tick_window(
        mut self,
        start_ns: i64,
        end_ns: i64,
    ) -> Self {
        self.training_tick_window = Some((start_ns, end_ns));
        self
    }

    pub fn with_local_cache_dir(mut self, local_cache_dir: String) -> Self {
        self.local_cache_dir = local_cache_dir;
        self
    }

    pub fn with_reward_lambda(mut self, reward_lambda: f64) -> Self {
        self.reward_lambda = reward_lambda;
        self
    }

    pub fn with_reward_action_penalty(mut self, reward_action_penalty: f64) -> Self {
        self.reward_action_penalty = reward_action_penalty;
        self
    }

    pub fn with_reward_holding_penalty(mut self, reward_holding_penalty: f64) -> Self {
        self.reward_holding_penalty = reward_holding_penalty;
        self
    }

    pub fn with_reward_scale(mut self, reward_scale: f64) -> Self {
        self.reward_scale = reward_scale;
        self
    }

    pub fn with_reward_clip(mut self, reward_clip: f64) -> Self {
        self.reward_clip = reward_clip;
        self
    }

    pub fn with_reward_vol_thresholds(
        mut self,
        vol_low: f64,
        vol_high: f64,
    ) -> Self {
        self.regime_reward_stats = RegimeRewardStats::new(vol_low, vol_high);
        self
    }

    pub fn with_disable_hedging(mut self, disable_hedging: bool) -> Self {
        self.disable_hedging = disable_hedging;
        self
    }

    pub fn with_leverage(mut self, leverage: f64) -> Self {
        self.leverage = leverage;
        self
    }

    /// Enable the durable JSONL trade log at `path`. A `None` path leaves it
    /// disabled. If the file cannot be opened the error is logged and the
    /// environment continues without a trade log, logging must never block
    /// trading or training startup.
    pub fn with_trade_log(mut self, path: Option<String>) -> Self {
        if let Some(path) = path {
            match TradeLogger::open(&path) {
                Ok(logger) => {
                    info!("Trade log enabled at {path}");
                    self.trade_logger = Some(Arc::new(logger));
                }
                Err(err) => {
                    log::error!("Failed to open trade log {path}, continuing without it: {err:#}");
                }
            }
        }
        self
    }

    /// Update the episode peak margin from the currently open positions.
    /// Margin is approximated as total notional (Σ volume × entry_price)
    /// divided by leverage. The value resets to 0 on reset() and monotonically
    /// tracks the episode high, so closing positions never lowers it. Called
    /// after apply_action() in both training and live step paths.
    fn update_max_total_margin(&mut self) {
        let total_notional: f64 = self
            .positions
            .iter()
            .map(|p| p.volume * p.entry_price)
            .sum();
        self.max_total_margin = self.max_total_margin.max(total_notional / self.leverage);
    }

    /// Record a fill: append it to the in-episode buffer and, if enabled, to the
    /// durable trade log. Centralises both fill push sites (open and close).
    fn record_fill(&mut self, event: FillEvent, fill: Fill) {
        if let Some(logger) = self.trade_logger.as_ref() {
            logger.log_fill(&self.symbol, event, &fill);
        }
        self.recent_fills.push(fill);
    }

    pub fn reward_parameters(&self) -> (f64, f64, f64, f64) {
        (
            self.reward_lambda,
            self.reward_action_penalty,
            self.reward_holding_penalty,
            self.reward_scale,
        )
    }

    pub async fn preload_training_data(&self) -> Result<()> {
        if self.mode != Mode::Training {
            return Ok(());
        }

        info!(
            "Preloading training market data for {} from {}",
            self.symbol, self.s3_prefix
        );
        preload_training_market_data(
            &self.symbol,
            &self.s3_prefix,
            &self.local_cache_dir,
            self.price_snapshot_ts,
            self.training_tick_window,
            &self.market_data_cache,
        )
        .await
    }

    /// Set the broker gateway for Production Mode
    pub fn with_broker_gateway(
        mut self,
        broker_gateway: Arc<dyn BrokerGateway + Send + Sync>,
    ) -> Self {
        self.broker_gateway = Some(broker_gateway);
        self
    }

    /// Get the current mode
    pub fn mode(&self) -> &Mode {
        &self.mode
    }

    /// Get the (long, short) daily swap rates for the symbol; (0, 0) if unset.
    fn get_swap_rates(&self) -> (f64, f64) {
        *self.daily_swap_rates.get(&self.symbol).unwrap_or(&(0.0, 0.0))
    }

    /// Check if broker gateway is configured
    pub fn has_broker_gateway(&self) -> bool {
        self.broker_gateway.is_some()
    }

    /// Get the broker gateway (returns error if not configured in Production Mode)
    fn get_broker_gateway(&self) -> Result<&(dyn BrokerGateway + Send + Sync)> {
        self.broker_gateway
            .as_ref()
            .map(|bg| bg.as_ref())
            .ok_or_else(|| anyhow::anyhow!("Broker gateway not configured"))
    }

    fn reset_episode_state(&mut self) {
        self.positions.clear();
        self.closed_position_window = ClosedPositionWindow::new();
        self.recent_fills.clear();
        self.last_action = None;
        self.last_swap_accrual_timestamp = 0;
        self.last_observation_timestamp_ns = None;
        self.max_total_margin = 0.0;
    }

    fn reset_reward_state(&mut self) {
        self.prev_total_equity = None;
        // Regime stats persist across episodes so the z-score scale is stable
        // from step one of every episode.
        self.current_regime_bin = 1;
    }

    /// Reset the environment and initialize a new episode
    pub async fn reset(&mut self, req: ResetRequest) -> Result<modelenv_proto::Observation> {
        self.reset_episode_state();

        match self.mode {
            Mode::Training => {
                self.reset_reward_state();

                // Validate episode timestamps
                if req.episode_end_ts > 0 && req.episode_start_ts > req.episode_end_ts {
                    return Err(anyhow::anyhow!(
                        "episode_start_ts ({}) must be <= episode_end_ts ({})",
                        req.episode_start_ts,
                        req.episode_end_ts
                    ));
                }
                if req.step_size_seconds < 0 {
                    return Err(anyhow::anyhow!(
                        "step_size_seconds ({}) must be >= 0",
                        req.step_size_seconds
                    ));
                }

                self.step_size_ns = if req.step_size_seconds > 0 {
                    req.step_size_seconds
                        .checked_mul(1_000_000_000)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "step_size_seconds ({}) is too large",
                                req.step_size_seconds
                            )
                        })?
                } else {
                    DEFAULT_STEP_SIZE_NS
                };

                // Convert episode_*_ts from Unix seconds (the proto contract,
                // per every Python client docstring) to nanoseconds for the
                // internal loader/cursor code, mirroring how step_size_seconds
                // is converted above.
                let episode_start_ts_ns = seconds_to_ns(
                    req.episode_start_ts,
                    "episode_start_ts",
                )?;
                let episode_end_ts_ns = seconds_to_ns(
                    req.episode_end_ts,
                    "episode_end_ts",
                )?;

                // Initialize episode with S3 parquet loading
                let market_data_cache = self.market_data_cache.clone();
                let mut episode = initialize_episode(
                    &req.symbol,
                    &self.s3_prefix,
                    &self.local_cache_dir,
                    self.price_snapshot_ts,
                    episode_start_ts_ns,
                    episode_end_ts_ns,
                    self.step_size_ns,
                    &market_data_cache,
                )
                .await?;
                // Advance the cursor by one step so the first observation has a
                // populated live tick window (cursor lands at first_bar_ts + step_size).
                episode.advance(self.step_size_ns);
                self.bars = episode.bars.clone();
                self.episode = Some(episode);

                // Get initial observation
                self.observe(ObserveRequest { symbol: req.symbol }).await
            }
            Mode::Live => {
                // Production mode - synchronise with broker's current positions
                info!("Production mode: synchronising with broker positions");

                // Get broker gateway reference
                let broker = self.get_broker_gateway()?;

                // Sync positions with broker
                let broker_positions = broker.sync_positions(&req.symbol).await?;

                // Clear all existing internal positions, unrealised P/L, and accumulated swap
                self.positions.clear();

                // Load synchronised positions into environment state
                for p in &broker_positions {
                    let position = Position::from_proto(p);
                    info!(
                        "Synchronised position: id={}, entry_price={}, unrealised_pnl={}, swap={}",
                        position.position_id,
                        position.entry_price,
                        position.unrealised_pnl,
                        position.swap
                    );
                    self.positions.push(position);
                }

                // Log synchronisation summary
                info!(
                    "Synchronised {} positions from broker",
                    self.positions.len()
                );

                // Rebuild account-history observation features from broker
                // deal history. reset_episode_state() cleared the realised-P&L
                // window and recent fills; training observations carry both as
                // features, so leaving them zeroed after a restart would feed
                // the policy a falsified account history until it rebuilt
                // organically. The cutoff is the most recent session start,
                // matching training, where the window resets at each
                // per-session episode's Reset(), so the restored value has the
                // same "realised P&L this session" semantics the policy
                // learned.
                let now_ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as i64;
                let from_ts = self.session_start_cutoff(now_ns);
                let (closed_positions, broker_fills) = {
                    let broker = self.get_broker_gateway()?;
                    let closed = broker.closed_positions(&req.symbol, from_ts).await?;
                    let fills = broker.recent_fills(&req.symbol, RECENT_WINDOW).await?;
                    (closed, fills)
                };
                info!(
                    "Synchronised {} closed positions and {} recent fills from broker",
                    closed_positions.len(),
                    broker_fills.len()
                );
                for closed_position in closed_positions {
                    self.closed_position_window
                        .add_closed_position(closed_position);
                }
                // Restore directly rather than via record_fill: these fills
                // are history, and re-logging them into the durable trade log
                // on every reset would duplicate entries.
                self.recent_fills = broker_fills
                    .iter()
                    .map(|fill| Fill {
                        order_id: fill.order_id.clone(),
                        timestamp_ns: fill.timestamp_ns,
                        price: fill.price,
                        size: fill.size,
                        side: FillSide::try_from(fill.side).unwrap_or(FillSide::Buy),
                        partial: fill.partial,
                    })
                    .collect();

                // Get current bar from broker
                let broker = self.get_broker_gateway()?;
                let current_bar = broker.current_bar(&req.symbol).await?;

                // Log the current bar
                info!(
                    "Fetched current bar: timestamp={}, open={}, high={}, low={}, close={}, volume={}",
                    current_bar.timestamp_ns,
                    current_bar.open,
                    current_bar.high,
                    current_bar.low,
                    current_bar.close,
                    current_bar.volume
                );

                // Get initial observation
                self.observe(ObserveRequest { symbol: req.symbol }).await
            }
        }
    }

    fn positions_for_observation(&self) -> Vec<modelenv_proto::Position> {
        let mut proto_positions: Vec<modelenv_proto::Position> =
            self.positions.iter().map(|p| p.to_proto()).collect();
        proto_positions.sort_by(|a, b| b.open_timestamp_ns.cmp(&a.open_timestamp_ns));
        proto_positions
    }

    fn recent_fills_for_observation(&self) -> Vec<modelenv_proto::Fill> {
        self.recent_fills
            .iter()
            .rev()
            .take(RECENT_WINDOW)
            .map(|f| modelenv_proto::Fill {
                order_id: f.order_id.clone(),
                timestamp_ns: f.timestamp_ns,
                price: f.price,
                size: f.size,
                side: f.side as i32,
                partial: f.partial,
            })
            .collect()
    }

    fn live_observation_timestamp(current_bar: &Bar, recent_ticks: &[Tick]) -> i64 {
        recent_ticks
            .last()
            .map(|tick| tick.timestamp_ns)
            .unwrap_or(current_bar.timestamp_ns)
    }

    fn live_ticks_since(recent_ticks: &[Tick], lower_bound: i64) -> Vec<Tick> {
        recent_ticks
            .iter()
            .filter(|tick| tick.timestamp_ns > lower_bound)
            .cloned()
            .collect()
    }

    async fn build_live_observation(&mut self, symbol: String) -> Result<LiveData> {
        let mut live_bars: HashMap<String, Bar> = HashMap::new();
        let mut fallback_current_bar: Option<Bar> = None;
        let mut ta: Vec<modelenv_proto::IntervalIndicators> =
            Vec::with_capacity(TIME_INTERVALS.len());
        let mut double_bottoms: Vec<modelenv_proto::DoubleBottomPattern> = Vec::new();
        let mut double_tops: Vec<modelenv_proto::DoubleTopPattern> = Vec::new();
        let recent_ticks_raw: Vec<Tick> = if let Some(broker) = &self.broker_gateway {
            // Fetch bars from broker and store in the shared in-memory dataframe.
            for interval in TIME_INTERVALS {
                let broker_bars =
                    broker.recent_bars(&symbol, interval, RECENT_WINDOW).await?;
                self.bars.insert(interval.to_string(), broker_bars);
            }
            // Compute indicators and patterns from in-memory bars (same source as training).
            for interval in TIME_INTERVALS {
                if let Some(bars) = self.bars.get(*interval) {
                    let interval_ta = compute_interval_indicators(bars);
                    if *interval == "M15" {
                        let (mut dbs, mut dts) = detect_all_patterns(bars);
                        dbs.reverse();
                        dbs.truncate(12);
                        dts.reverse();
                        dts.truncate(12);
                        double_bottoms = dbs;
                        double_tops = dts;
                    }
                    ta.push(interval_ta);
                    if let Some(latest) = bars.last().cloned() {
                        live_bars.insert(interval.to_string(), latest.clone());
                        if *interval == "M1" {
                            fallback_current_bar = Some(latest);
                        }
                    }
                } else {
                    ta.push(modelenv_proto::IntervalIndicators::default());
                }
            }
            broker.current_ticks(&symbol).await?
        } else {
            Vec::new()
        };

        let current_bar = fallback_current_bar
            .ok_or_else(|| anyhow::anyhow!("No M1 bar available for live observation"))?;
        if !live_bars.contains_key("M1") {
            live_bars.insert("M1".to_string(), current_bar.clone());
        }

        let current_timestamp = Self::live_observation_timestamp(&current_bar, &recent_ticks_raw);
        let live_lower = self
            .last_observation_timestamp_ns
            .map(|prev| prev.max(current_timestamp - crate::episode::LIVE_TICK_WINDOW_NS))
            .unwrap_or_else(|| current_timestamp - crate::episode::LIVE_TICK_WINDOW_NS);

        let mut live_ticks = Self::live_ticks_since(recent_ticks_raw.as_slice(), live_lower);
        live_ticks.reverse();

        let recent_fills = self.recent_fills_for_observation();

        let proto_positions = self.positions_for_observation();

        let m15_double_bottom_low =
            compute_m15_double_bottom_low(&double_bottoms, &live_ticks);
        let m15_double_bottom_high =
            compute_m15_double_bottom_high(&double_bottoms, &live_ticks, m15_double_bottom_low);
        let m15_double_top_high =
            compute_m15_double_top_high(&double_tops, &live_ticks);
        let m15_double_top_low =
            compute_m15_double_top_low(&double_tops, &live_ticks, m15_double_top_high);
        let (sin_hour, cos_hour) = compute_time_features(current_timestamp);

        let observation = LiveData {
            timestamp_ns: current_timestamp,
            symbol,
            live_bars,
            positions: proto_positions,
            session_realised_pnl: self
                .closed_position_window
                .total_realised_pnl_since(self.session_start_cutoff(current_timestamp)),
            recent_fills,
            ta,
            double_bottoms,
            double_tops,
            live_ticks,
            done: false,
            reward: 0.0,
            raw_pnl_delta: 0.0,
            max_total_margin: 0.0,
            m15_double_bottom_low,
            m15_double_bottom_high,
            m15_double_top_high,
            m15_double_top_low,
            sin_hour,
            cos_hour,
        };
        self.last_observation_timestamp_ns = Some(observation.timestamp_ns);
        Ok(observation)
    }

    /// Take a step in the environment
    pub async fn step(&mut self, action: Action) -> Result<StepResponse> {
        match self.mode {
            Mode::Training => {
                let (prev_timestamp, current_timestamp, still_running) = {
                    let episode = self.episode.as_mut().ok_or_else(|| {
                        anyhow::anyhow!("Episode not initialized. Call reset() first.")
                    })?;
                    let prev_timestamp = episode.get_cursor_timestamp();
                    let still_running = episode.advance(self.step_size_ns);
                    let current_timestamp = episode.get_cursor_timestamp();
                    (prev_timestamp, current_timestamp, still_running)
                };

                if !self.positions.is_empty() {
                    let episode = self.episode.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("Episode not initialized. Call reset() first.")
                    })?;
                    if episode.has_day_boundary_crossed(prev_timestamp, current_timestamp) {
                        self.accrue_swap_on_positions()?;
                    }
                }

                self.mark_positions_to_market()?;

                // End-of-session liquidation: if this step crossed the configured
                // trading-session end hour, close all shorts + losing longs (net
                // of swap) and keep winning/break-even longs, before the new
                // action is applied. mark_positions_to_market above keeps
                // unrealised_pnl current for the win/lose decision.
                if let Some(end_hour) = self.trading_session_end_hour {
                    if !self.positions.is_empty()
                        && crate::episode::has_session_end_crossed(
                            prev_timestamp,
                            current_timestamp,
                            end_hour,
                        )
                    {
                        self.close_session_positions()?;
                    }
                }

                self.apply_action(&action)?;

                self.update_max_total_margin();

                let session_realised_pnl = self
                    .closed_position_window
                    .total_realised_pnl_since(self.session_start_cutoff(current_timestamp));
                let proto_positions = self.positions_for_observation();
                let recent_fills = self.recent_fills_for_observation();
                let mut observation = self
                    .episode
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Episode not initialized. Call reset() first."))?
                    .get_observation(
                        proto_positions.as_slice(),
                        session_realised_pnl,
                        Some(prev_timestamp),
                    );
                observation.recent_fills = recent_fills;
                self.last_observation_timestamp_ns = Some(observation.timestamp_ns);

                // Determine volatility regime from M5 BB width before reward
                // calculation so the reward is normalised against the right bin.
                if let Some(bin) =
                    Self::extract_regime_bin(&observation.ta, &self.regime_reward_stats)
                {
                    self.current_regime_bin = bin;
                }

                // Calculate reward based on the previous step's action state.
                let reward = self.calculate_reward(&action)?;

                // Update last_action after calculating reward so action-switch penalties
                // compare the current action against the previous step's action.
                self.last_action = match ActionType::try_from(action.action) {
                    Ok(action_type) => Some(action_type),
                    Err(_) => None,
                };

                observation.reward = reward;
                observation.raw_pnl_delta = self.last_raw_pnl_delta;
                observation.max_total_margin = self.max_total_margin;
                observation.done = !still_running;
                Ok(StepResponse {
                    data: Some(self.normalise_observation(observation.into_observation())),
                    info: "".to_string(),
                })
            }
            Mode::Live => {
                // Production mode - submit action to broker and return execution results
                info!("Production mode: submitting action to broker");

                // End-of-session liquidation, mirroring Training: when the
                // wall clock crossed the session-end hour since the previous
                // step's observation, close all shorts + losing longs (net of
                // swap) at the broker before the new action is submitted.
                if let Some(end_hour) = self.trading_session_end_hour {
                    let now_ns = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as i64;
                    if let Some(prev_ns) = self.last_observation_timestamp_ns {
                        if crate::episode::has_session_end_crossed(prev_ns, now_ns, end_hour) {
                            self.close_session_positions_live().await?;
                        }
                    }
                }

                let action_type = ActionType::try_from(action.action).map_err(|_| {
                    anyhow::anyhow!("Unsupported action type {} in live mode", action.action)
                })?;

                if action_type != ActionType::ActionHold {
                    // Get broker gateway reference
                    let broker = self.get_broker_gateway()?;

                    // Submit action to broker
                    let fill = broker.submit(&action).await?;

                    // Record the fill with all required fields including partial flag
                    let order_id = fill.order_id.clone();
                    let price = fill.price;
                    let size = fill.size;
                    let partial = fill.partial;

                    self.record_fill(
                        FillEvent::Open,
                        Fill {
                            order_id: order_id.clone(),
                            timestamp_ns: fill.timestamp_ns,
                            price,
                            size,
                            side: match FillSide::try_from(fill.side) {
                                Ok(s) => s,
                                Err(_) => FillSide::Buy,
                            },
                            partial,
                        },
                    );

                    info!(
                        "Recorded fill: order_id={}, price={}, size={}, partial={}",
                        order_id, price, size, partial
                    );
                }

                // Sync positions from broker and reconcile before splitting
                let broker_positions = {
                    let broker = self.get_broker_gateway()?;
                    broker.sync_positions(&self.symbol).await?
                };

                // Reconcile internal positions against broker positions
                let broker_realised_pnl = 0.0;
                let session_realised_pnl = self.session_realised_pnl();
                reconcile_positions(
                    &self
                        .positions
                        .iter()
                        .map(|p| p.to_proto())
                        .collect::<Vec<_>>(),
                    &broker_positions,
                    session_realised_pnl,
                    broker_realised_pnl,
                );

                // Replace internal positions with broker state, split to
                // 1-unit chunks when hedging is disabled so reduce_side can
                // net individual units (matching training-mode behaviour).
                if self.disable_hedging {
                    let mut split = Vec::new();
                    for bp in &broker_positions {
                        let side = match bp.side {
                            0 => Side::Buy,
                            _ => Side::Sell,
                        };
                        let count = bp.volume as usize;
                        for i in 0..count {
                            split.push(Position::new(
                                format!("{}_{}", bp.position_id, i),
                                bp.entry_price,
                                0.0, // spread already baked into broker entry price
                                1.0,
                                side,
                                bp.open_timestamp_ns,
                            ));
                        }
                    }
                    self.positions = split;
                } else {
                    self.positions = broker_positions
                        .iter()
                        .map(Position::from_proto)
                        .collect();
                }

                // Update unrealised P/L based on current broker price
                let broker = self.get_broker_gateway()?;
                let current_bar = broker.current_bar(&self.symbol).await?;
                let current_mid_price = (current_bar.open + current_bar.close) / 2.0;
                for position in &mut self.positions {
                    position.unrealised_pnl = position.calculate_unrealised_pnl(current_mid_price);
                }

                let mut observation = self.build_live_observation(self.symbol.clone()).await?;

                self.update_max_total_margin();

                // Determine volatility regime from M5 BB width.
                if let Some(bin) =
                    Self::extract_regime_bin(&observation.ta, &self.regime_reward_stats)
                {
                    self.current_regime_bin = bin;
                }

                // Calculate reward
                let reward = self.calculate_reward(&action)?;

                // Update last_action after calculating reward
                self.last_action = match ActionType::try_from(action.action) {
                    Ok(action_type) => Some(action_type),
                    Err(_) => None,
                };

                observation.reward = reward;
                observation.raw_pnl_delta = self.last_raw_pnl_delta;
                observation.max_total_margin = self.max_total_margin;
                Ok(StepResponse {
                    data: Some(self.normalise_observation(observation.into_observation())),
                    info: "".to_string(),
                })
            }
        }
    }

    /// Get current observation without advancing
    /// Return the raw structured observation (for debugging / inspection).
    pub async fn reference_data(&mut self, _req: ObserveRequest) -> Result<Reference> {
        match self.mode {
            Mode::Training => {
                let episode = self.episode.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Episode not initialized. Call reset() first.")
                })?;
                let proto_positions = self.positions_for_observation();
                let recent_fills = self.recent_fills_for_observation();
                let realised_pnl = self.session_realised_pnl();
                let mut live = episode.get_observation(
                    proto_positions.as_slice(),
                    realised_pnl,
                    self.last_observation_timestamp_ns,
                );
                live.recent_fills = recent_fills;
                self.last_observation_timestamp_ns = Some(live.timestamp_ns);
                let mut reference = live.into_reference();
                reference.session_start_ts = self.session_start_cutoff(reference.timestamp_ns);
                Ok(reference)
            }
            Mode::Live => {
                let live = self
                    .build_live_observation(self.symbol.clone())
                    .await?;
                self.last_observation_timestamp_ns = Some(live.timestamp_ns);
                let mut reference = live.into_reference();
                reference.session_start_ts = self.session_start_cutoff(reference.timestamp_ns);
                Ok(reference)
            }
        }
    }

    pub async fn observe(&mut self, req: ObserveRequest) -> Result<modelenv_proto::Observation> {
        match self.mode {
            Mode::Training => {
                let episode = self.episode.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Episode not initialized. Call reset() first.")
                })?;

                let proto_positions = self.positions_for_observation();

                let recent_fills = self.recent_fills_for_observation();
                let mut observation = episode.get_observation(
                    proto_positions.as_slice(),
                    self.session_realised_pnl(),
                    self.last_observation_timestamp_ns,
                );
                observation.recent_fills = recent_fills;
                self.last_observation_timestamp_ns = Some(observation.timestamp_ns);
                Ok(self.normalise_observation(observation.into_observation()))
            }
            Mode::Live => {
                self.build_live_observation(req.symbol)
                    .await
                    .map(|obs| self.normalise_observation(obs.into_observation()))
            }
        }
    }

    /// Update running normalisation statistics from raw values and return the
    /// normalised feature vector for the observation.
    fn normalise_observation(&mut self, mut obs: modelenv_proto::Observation) -> modelenv_proto::Observation {
        if let Some(row) = obs.state_data.first() {
            self.normaliser.update(&row.values);
            let normalised = self.normaliser.normalise_all(&row.values);
            obs.state_data = vec![modelenv_proto::StateRow { values: normalised }];
        }
        obs
    }

    /// Return recent bars from the current timestamp cursor for all intervals.
    pub async fn recent_bars(
        &self,
        req: RecentBarsRequest,
    ) -> Result<RecentBarsResponse> {
        use crate::episode::RECENT_WINDOW;
        // 0 (unset) falls back to RECENT_WINDOW. A larger count lets deep-history
        // consumers (e.g. the forecaster's 1440-bar feature window) pull more.
        let window = if req.count == 0 {
            RECENT_WINDOW
        } else {
            req.count as usize
        };
        let mut bars: HashMap<String, BarList> = HashMap::new();
        match self.mode {
            Mode::Training => {
                let episode = self.episode.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Episode not initialized. Call reset() first.")
                })?;
                let cursor = episode.get_cursor_timestamp();
                for interval in crate::data_loader::TIME_INTERVALS {
                    if let Some(all_bars) = self.bars.get(*interval) {
                        if let Some(idx) =
                            episode.interval_cursor_at_or_before(interval, cursor)
                        {
                            let start = idx.saturating_sub(window);
                            let recent: Vec<modelenv_proto::Bar> = all_bars
                                .get(start..idx)
                                .map(|s| s.to_vec())
                                .unwrap_or_default()
                                .into_iter()
                                .rev()
                                .collect();
                            bars.insert(interval.to_string(), BarList {
                                bars: recent,
                            });
                        }
                    }
                }
            }
            Mode::Live => {
                let broker = self.get_broker_gateway()?;
                for interval in crate::data_loader::TIME_INTERVALS {
                    let recent = broker
                        .recent_bars(&req.symbol, interval, window)
                        .await?;
                    bars.insert(interval.to_string(), BarList {
                        bars: recent.into_iter().rev().collect(),
                    });
                }
            }
        }
        Ok(RecentBarsResponse { bars })
    }

    /// Return ticks in the 60-second window before live_ticks.
    pub async fn recent_ticks(
        &self,
        req: modelenv_proto::RecentTicksRequest,
    ) -> Result<modelenv_proto::RecentTicksResponse> {
        let ticks = match self.mode {
            Mode::Training => {
                let episode = self.episode.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Episode not initialized. Call reset() first.")
                })?;
                let cursor = episode.get_cursor_timestamp();
                let live_lower = self
                    .last_observation_timestamp_ns
                    .map(|prev| {
                        prev.max(cursor - crate::episode::LIVE_TICK_WINDOW_NS)
                    })
                    .unwrap_or_else(|| cursor - crate::episode::LIVE_TICK_WINDOW_NS);
                let recent_lower = live_lower - crate::episode::RECENT_TICK_WINDOW_NS;
                episode.ticks_in_range(recent_lower, live_lower)
            }
            Mode::Live => {
                let broker = self.get_broker_gateway()?;
                broker.current_ticks(&req.symbol).await?
            }
        };
        Ok(modelenv_proto::RecentTicksResponse {
            ticks: ticks.into_iter().rev().collect(),
        })
    }

    /// Return recent news from the current cursor, capped at RECENT_WINDOW.
    pub async fn recent_news(
        &self,
        _req: modelenv_proto::RecentNewsRequest,
    ) -> Result<modelenv_proto::RecentNewsResponse> {
        let news = match self.mode {
            Mode::Training => {
                let episode = self.episode.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Episode not initialized. Call reset() first.")
                })?;
                episode.recent_news(episode.get_cursor_timestamp())
            }
            Mode::Live => vec![],
        };
        Ok(modelenv_proto::RecentNewsResponse { news })
    }

    /// The session-start cutoff for the realised P&L feature: the most recent
    /// instant at or before `now_ns` where the UTC hour equals the configured
    /// `trading_session_start_hour` (0h UTC when unset).
    fn session_start_cutoff(&self, now_ns: i64) -> i64 {
        crate::episode::most_recent_session_start(
            now_ns,
            self.trading_session_start_hour.unwrap_or(0),
        )
    }

    /// Realised P/L (incl. swap) of positions closed since the most recent
    /// session start. Carried on the observation in the `session_realised_pnl`
    /// proto field and state column.
    fn session_realised_pnl(&self) -> f64 {
        let current_timestamp = self.current_timestamp();
        self.closed_position_window
            .total_realised_pnl_since(self.session_start_cutoff(current_timestamp))
    }

    /// Get the current timestamp from the episode
    fn current_timestamp(&self) -> i64 {
        if let Some(episode) = &self.episode {
            episode.get_cursor_timestamp()
        } else {
            now_ns()
        }
    }

    /// Extract the M5 relative Bollinger Band width from TA indicators and
    /// discretize into a regime bin (0 = low vol, 1 = mid vol, 2 = high vol).
    ///
    /// Returns `None` when M5 volatility data hasn't been computed yet.
    fn extract_regime_bin(
        ta: &[modelenv_proto::IntervalIndicators],
        stats: &RegimeRewardStats,
    ) -> Option<usize> {
        let m5_idx = TIME_INTERVALS.iter().position(|&iv| iv == "M5")?;
        let vol = ta.get(m5_idx)?.volatility.as_ref()?;
        if vol.bb_middle <= 0.0 {
            return None;
        }
        let bb_width = (vol.bb_upper - vol.bb_lower) / vol.bb_middle;
        Some(stats.regime_bin(bb_width))
    }

    /// Calculate reward based on action.
    ///
    /// `reward = clamp((ΔV_t − asymmetric_penalty − action_penalty −
    /// holding_penalty) / reward_scale, ±reward_clip)`, where the asymmetric
    /// penalty is `reward_lambda · |ΔV_t|` on losing steps (linear loss
    /// aversion: a loss weighs (1+λ)× an equal gain). Dividing by
    /// ``reward_scale`` (default 0.01 = 1 pip for USDJPY) expresses the reward
    /// in pips and keeps pairs comparable; the clamp is a defensive rail
    /// against data artifacts.
    fn calculate_reward(&mut self, action: &Action) -> Result<f64> {
        // Get current timestamp
        let current_timestamp = self.current_timestamp();

        // Calculate current total equity (unrealised + cumulative realised).
        // Deliberately NOT the session-scoped sum the observation carries: the
        // reward differences equity between steps, and a session-scoped sum
        // would drop at each session-start crossing, injecting a spurious
        // negative reward with no economic loss behind it.
        let current_unrealised_pnl: f64 = self.positions.iter().map(|p| p.unrealised_pnl).sum();
        let current_realised_pnl = self.closed_position_window.total_realised_pnl();
        let current_total_equity = current_unrealised_pnl + current_realised_pnl;

        // Calculate delta_V_t (change in total equity)
        let delta_v_t = if let Some(prev_equity) = self.prev_total_equity {
            current_total_equity - prev_equity
        } else {
            // First step - no previous equity to compare
            0.0
        };

        // Update previous total equity for next step
        self.prev_total_equity = Some(current_total_equity);

        // Surface the raw (pre-penalty, pre-scaling) equity change so the
        // observation can report monetary PnL alongside the scaled reward.
        self.last_raw_pnl_delta = delta_v_t;

        // Asymmetric (loss-averse) penalty, LINEAR in the loss size, applied
        // only when delta_V_t is negative. A losing step nets
        // `(1 + reward_lambda) * delta_v_t`, i.e. losses weigh (1+λ)x an equal
        // gain; a fixed, bounded loss-aversion ratio.
        //
        // This was `reward_lambda * delta_v_t²` (squared). That made the
        // aversion ratio grow without bound with the move size (51x at a 50-unit
        // step) and mixed yen with yen², so per-step reward spanned millions and
        // the value-maximising policy collapsed to "never trade". Linear keeps
        // loss aversion, stays dimensionally consistent, and is Q-fittable.
        let asymmetric_penalty = if delta_v_t < 0.0 {
            self.reward_lambda * delta_v_t.abs()
        } else {
            0.0
        };

        // Calculate action penalty (c_a) for action toggling
        let current_action = match ActionType::try_from(action.action) {
            Ok(action_type) => action_type,
            Err(_) => return Err(anyhow::anyhow!("Invalid action type")),
        };
        let action_penalty = if let Some(last_action) = self.last_action {
            if last_action != current_action {
                self.reward_action_penalty
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Calculate holding penalty (c_h) for position duration in days
        let holding_penalty = if !self.positions.is_empty() {
            let total_duration_ns: i64 = self
                .positions
                .iter()
                .map(|p| current_timestamp - p.open_timestamp_ns)
                .sum();
            let total_days = total_duration_ns as f64 / crate::position::NANOS_PER_DAY as f64;
            self.reward_holding_penalty * total_days
        } else {
            0.0
        };

        // Calculate final reward = money PnL divided by a fixed per-symbol
        // scale, so 1 pip ≈ 1 reward unit (for USDJPY at reward_scale=0.01).
        // Penalties share the scale; they are measured in the same units.
        let raw = delta_v_t - asymmetric_penalty - action_penalty - holding_penalty;
        let reward = raw / self.reward_scale;

        // Defensive clamp on the final per-step reward (artifact rail; <= 0
        // disables). Loose enough not to touch realistic moves, so the
        // loss-aversion asymmetry is preserved across the normal range.
        if self.reward_clip > 0.0 {
            Ok(reward.clamp(-self.reward_clip, self.reward_clip))
        } else {
            Ok(reward)
        }
    }

    /// Apply an action to the environment
    fn apply_action(&mut self, action: &Action) -> Result<()> {
        match action.action() {
            ActionType::ActionHold => {
                self.accrue_swap_on_positions()?;
            }
            ActionType::ActionBuy1 => self.net_open(1.0, Side::Buy)?,
            ActionType::ActionBuy2 => self.net_open(2.0, Side::Buy)?,
            ActionType::ActionSell1 => self.net_open(1.0, Side::Sell)?,
            ActionType::ActionSell2 => self.net_open(2.0, Side::Sell)?,
        }
        Ok(())
    }

    /// Open `volume` on `side`. When hedging is disabled, first reduce
    /// opposite-side positions by `volume`, then open any remainder on the
    /// desired side.  Reducing sells uses FIFO (oldest first); reducing
    /// buys uses LIFO (newest first).
    fn net_open(&mut self, volume: f64, side: Side) -> Result<()> {
        if self.disable_hedging {
            let opposite = match side {
                Side::Buy => Side::Sell,
                Side::Sell => Side::Buy,
            };
            // Buy action → FIFO on existing sells; Sell action → LIFO on existing buys.
            let newest_first = matches!(side, Side::Sell);
            let remaining = self.reduce_side(opposite, volume, newest_first)?;
            if remaining > 0.0 {
                self.open_position(remaining, side)?;
            }
        } else {
            self.open_position(volume, side)?;
        }
        Ok(())
    }

    /// Reduce positions of `side` by up to `volume` units. Returns the volume
    /// that could NOT be covered, zero means existing positions fully absorbed
    /// the request.
    ///
    /// When `newest_first` is true, close the newest matching positions first
    /// (LIFO).  When false, close the oldest first (FIFO).
    fn reduce_side(&mut self, side: Side, volume: f64, newest_first: bool) -> Result<f64> {
        let mut remaining = volume as usize;
        if remaining == 0 {
            return Ok(0.0);
        }

        // Collect position IDs to close in the requested order
        let mut to_close: Vec<String> = Vec::new();
        let iter: Box<dyn Iterator<Item = &Position>> = if newest_first {
            Box::new(self.positions.iter().rev())
        } else {
            Box::new(self.positions.iter())
        };
        for pos in iter {
            if remaining == 0 {
                break;
            }
            if pos.side == side {
                to_close.push(pos.position_id.clone());
                remaining -= 1;
            }
        }

        for id in to_close {
            self.close_position(&id)?;
        }

        Ok(remaining as f64)
    }

    /// Close a specific position by id (full close)
    fn close_position(&mut self, position_id: &str) -> Result<()> {
        let position = self
            .positions
            .iter()
            .find(|p| p.position_id == position_id)
            .cloned();
        let position = match position {
            Some(p) => p,
            None => return Ok(()),
        };

        let current_timestamp = self.current_timestamp();
        // Closing a BUY → SELL, closing a SELL → BUY
        let close_side = match position.side {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        };
        let close_price = self.fill_price(close_side)?;

        let closed_position =
            position.to_closed_position(close_price, current_timestamp, self.transaction_cost);

        if let Some(logger) = self.trade_logger.as_ref() {
            logger.log_close(&self.symbol, &closed_position);
        }

        self.closed_position_window
            .add_closed_position(closed_position);

        self.positions
            .retain(|p| p.position_id != position_id);

        self.record_fill(
            FillEvent::Close,
            Fill {
                order_id: format!("fill_{}", current_timestamp),
                timestamp_ns: current_timestamp,
                price: close_price,
                size: position.volume,
                side: match position.side {
                    Side::Buy => FillSide::Buy,
                    Side::Sell => FillSide::Sell,
                },
                partial: false,
            },
        );

        Ok(())
    }

    /// End-of-session liquidation.
    ///
    /// Close all short positions and all *losing* long positions, where a long
    /// is "losing" when its net P&L (mark-to-market `unrealised_pnl` + accrued
    /// `swap`) is strictly negative. Winning and break-even long positions
    /// (net P&L >= 0) are kept and carry into the next session. Callers must
    /// mark positions to market first so `unrealised_pnl` is current. Closing
    /// reuses `close_position`, so realised P&L, the closed-position window, the
    /// trade log, and fills are all recorded normally.
    fn close_session_positions(&mut self) -> Result<()> {
        let to_close: Vec<String> = self
            .positions
            .iter()
            .filter(|p| matches!(p.side, Side::Sell) || (p.unrealised_pnl + p.swap) < 0.0)
            .map(|p| p.position_id.clone())
            .collect();
        if to_close.is_empty() {
            return Ok(());
        }
        let kept = self.positions.len() - to_close.len();
        for id in &to_close {
            self.close_position(id)?;
        }
        log::info!(
            "session-end liquidation: symbol={} closed={} (shorts + losing longs), kept={} winning longs",
            self.symbol,
            to_close.len(),
            kept,
        );
        Ok(())
    }

    /// End-of-session liquidation against the broker (Live mode).
    ///
    /// Live counterpart of `close_session_positions`: sync the current broker
    /// positions, then close every short and every losing long (net P&L
    /// (mark-to-market at the broker mid + accrued swap) strictly negative)
    /// with broker close orders. Winning/break-even longs are left open.
    /// Close fills are recorded so they appear in `recent_fills`; the normal
    /// `sync_positions` later in the step refreshes the internal book.
    async fn close_session_positions_live(&mut self) -> Result<()> {
        let symbol = self.symbol.clone();
        let (to_close, kept) = {
            let broker = self.get_broker_gateway()?;
            let broker_positions = broker.sync_positions(&symbol).await?;
            if broker_positions.is_empty() {
                return Ok(());
            }
            let current_bar = broker.current_bar(&symbol).await?;
            let current_mid_price = (current_bar.open + current_bar.close) / 2.0;
            let mut to_close = Vec::new();
            let mut kept = 0usize;
            for proto in &broker_positions {
                let position = Position::from_proto(proto);
                let net_pnl =
                    position.calculate_unrealised_pnl(current_mid_price) + position.swap;
                if matches!(position.side, Side::Sell) || net_pnl < 0.0 {
                    to_close.push(proto.clone());
                } else {
                    kept += 1;
                }
            }
            (to_close, kept)
        };
        if to_close.is_empty() {
            return Ok(());
        }

        let closed = to_close.len();
        for proto in to_close {
            let fill = {
                let broker = self.get_broker_gateway()?;
                broker.close_position(&proto).await?
            };
            self.record_fill(
                FillEvent::Close,
                Fill {
                    order_id: fill.order_id.clone(),
                    timestamp_ns: fill.timestamp_ns,
                    price: fill.price,
                    size: fill.size,
                    side: match FillSide::try_from(fill.side) {
                        Ok(s) => s,
                        Err(_) => FillSide::Buy,
                    },
                    partial: fill.partial,
                },
            );
        }
        log::info!(
            "session-end liquidation (live): symbol={} closed={} (shorts + losing longs), kept={} winning longs",
            symbol,
            closed,
            kept,
        );
        Ok(())
    }

    /// Accrue swap on all open positions
    /// Returns true if swap was accrued for any position
    fn accrue_swap_on_positions(&mut self) -> Result<bool> {
        let current_timestamp = self.current_timestamp();
        let (long_rate, short_rate) = self.get_swap_rates();

        let mut swap_accrued = false;
        for position in &mut self.positions {
            // Long positions carry the long rate, short positions the short rate
            // (USDJPY overnight carry is asymmetric).
            let rate = match position.side {
                Side::Buy => long_rate,
                Side::Sell => short_rate,
            };
            if position.accrue_swap(current_timestamp, rate) {
                swap_accrued = true;
            }
        }

        // Update the last swap accrual timestamp only if swap was actually accrued
        if swap_accrued {
            self.last_swap_accrual_timestamp = current_timestamp;
        }

        Ok(swap_accrued)
    }

    fn mark_positions_to_market(&mut self) -> Result<()> {
        if self.positions.is_empty() {
            return Ok(());
        }

        let current_mid_price = self.get_current_mid_price()?;
        for position in &mut self.positions {
            position.unrealised_pnl = position.calculate_unrealised_pnl(current_mid_price);
        }

        Ok(())
    }

    /// Return the worst-case fill price for `side` in the current step window.
    ///
    /// BUY  → highest `tick.ask` in `(last_obs_ts, current_ts]`.
    /// SELL → lowest  `tick.bid` in `(last_obs_ts, current_ts]`.
    /// Falls back to the bar mid ± half spread when no ticks are available.
    fn fill_price(&self, side: Side) -> Result<f64> {
        let to_ns = self.current_timestamp();
        let from_ns = self.last_observation_timestamp_ns.unwrap_or(to_ns);

        let episode = self
            .episode
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Episode not initialized"))?;
        let ticks = episode.ticks_in_range(from_ns + 1, to_ns + 1);

        match side {
            Side::Buy => {
                if let Some(max_ask) = ticks.iter().map(|t| t.ask).fold(None, |acc, a| {
                    Some(acc.map_or(a, |prev: f64| prev.max(a)))
                }) {
                    return Ok(max_ask);
                }
            }
            Side::Sell => {
                if let Some(min_bid) = ticks.iter().map(|t| t.bid).fold(None, |acc, b| {
                    Some(acc.map_or(b, |prev: f64| prev.min(b)))
                }) {
                    return Ok(min_bid);
                }
            }
        }

        // Fallback to bar-based pricing
        let mid = self.get_current_mid_price()?;
        let spread = self.get_current_spread()?;
        match side {
            Side::Buy => Ok(mid + spread / 2.0),
            Side::Sell => Ok(mid - spread / 2.0),
        }
    }

    /// Open `volume` 1-unit positions on `side`.
    fn open_position(&mut self, volume: f64, side: Side) -> Result<()> {
        let current_timestamp = self.current_timestamp();
        let fill = self.fill_price(side)?;

        let count = volume as usize;
        for i in 0..count {
            let position = Position::new(
                format!("pos_{}_{}", current_timestamp, i),
                fill,   // entry_price = fill (pass spread=0 so no adjustment)
                0.0,
                1.0,
                side,
                current_timestamp,
            );
            self.positions.push(position);
        }
        Ok(())
    }

    /// Get the current mid price from the live bar
    fn get_current_mid_price(&self) -> Result<f64> {
        let episode = self
            .episode
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Episode not initialized"))?;

        if let Some(bar) = episode.current_bar("M1") {
            return Ok((bar.open + bar.close) / 2.0);
        }

        Err(anyhow::anyhow!("No current price available"))
    }

    /// Get the current spread (for entry price calculation)
    /// Default to 0.0001 (1 pip for most FX pairs)
    fn get_current_spread(&self) -> Result<f64> {
        // For USD/JPY, default spread might be different
        if self.symbol.contains("JPY") {
            Ok(0.05) // 5 pips for USD/JPY
        } else {
            Ok(0.0001) // 1 pip for other pairs
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::{ClosedPosition, Position, NANOS_PER_DAY};
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::path::PathBuf;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tempfile::tempdir;

    /// Helper: find the value at a named column in the first row of an Observation.
    fn obs_value(obs: &modelenv_proto::Observation, column: &str) -> f64 {
        let idx = obs
            .state_columns
            .iter()
            .position(|c| c == column)
            .unwrap_or_else(|| panic!("column not found: {}", column));
        obs.state_data[0].values[idx]
    }

    #[test]
    fn update_max_total_margin_tracks_monotonic_episode_peak() {
        let mut env =
            Environment::new(Mode::Training, "USDJPY".to_string(), "s3://unused".to_string())
                .with_leverage(200.0);
        assert_eq!(env.max_total_margin, 0.0, "starts at zero");

        // One position: entry 150.0, volume 2 → notional 300 → /200 = 1.5.
        env.positions
            .push(Position::new("p1".to_string(), 150.0, 0.0, 2.0, Side::Buy, 0));
        env.update_max_total_margin();
        assert!(
            (env.max_total_margin - 1.5).abs() < 1e-9,
            "single open position must set the peak, got {}",
            env.max_total_margin
        );

        // Add a second position → notional 450 → /200 = 2.25, peak rises.
        env.positions
            .push(Position::new("p2".to_string(), 150.0, 0.0, 1.0, Side::Sell, 0));
        env.update_max_total_margin();
        assert!(
            (env.max_total_margin - 2.25).abs() < 1e-9,
            "added exposure must raise the peak, got {}",
            env.max_total_margin
        );

        // Closing every position must NOT lower the recorded peak (monotonic).
        env.positions.clear();
        env.update_max_total_margin();
        assert!(
            (env.max_total_margin - 2.25).abs() < 1e-9,
            "flat book must preserve the episode peak, got {}",
            env.max_total_margin
        );
    }

    #[derive(Default)]
    struct MockBrokerGateway {
        submit_calls: Arc<AtomicUsize>,
        // Broker book returned by sync_positions; close_position records the
        // closed ids here so tests can assert which positions were liquidated.
        positions: Vec<modelenv_proto::Position>,
        close_calls: Arc<std::sync::Mutex<Vec<String>>>,
        // Deal history returned by closed_positions / recent_fills, for
        // live-Reset state-rebuild tests.
        broker_closed_positions: Vec<ClosedPosition>,
        broker_recent_fills: Vec<modelenv_proto::Fill>,
    }

    fn write_test_parquet(path: &PathBuf, timestamps: &[i64], opens: &[f64]) -> Result<()> {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("timestamp", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("open", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("high", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("low", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("close", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("volume", arrow::datatypes::DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(timestamps.to_vec())),
                Arc::new(Float64Array::from(opens.to_vec())),
                Arc::new(Float64Array::from(
                    opens.iter().map(|value| value + 0.5).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    opens.iter().map(|value| value - 0.5).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    opens.iter().map(|value| value + 0.25).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(vec![1.0; timestamps.len()])),
            ],
        )?;

        let file = std::fs::File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    fn write_news_parquet(path: &PathBuf, timestamps: &[i64]) -> Result<()> {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("timestamp_ns", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("headline", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("source", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("sentiment_score", arrow::datatypes::DataType::Float64, false),
        ]));
        let n = timestamps.len();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(timestamps.to_vec())),
                Arc::new(StringArray::from(vec!["test"; n])),
                Arc::new(StringArray::from(vec!["source"; n])),
                Arc::new(Float64Array::from(vec![0.0; n])),
            ],
        )?;
        let file = std::fs::File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    fn write_tick_parquet(path: &PathBuf, timestamps: &[i64]) -> Result<()> {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("timestamp_ns", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("bid", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("ask", arrow::datatypes::DataType::Float64, false),
        ]));
        let n = timestamps.len();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(timestamps.to_vec())),
                Arc::new(Float64Array::from(vec![100.0; n])),
                Arc::new(Float64Array::from(vec![100.01; n])),
            ],
        )?;
        let file = std::fs::File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    #[async_trait::async_trait]
    impl BrokerGateway for MockBrokerGateway {
        async fn sync_positions(&self, _symbol: &str) -> Result<Vec<modelenv_proto::Position>> {
            let closed = self.close_calls.lock().unwrap().clone();
            Ok(self
                .positions
                .iter()
                .filter(|p| !closed.contains(&p.position_id))
                .cloned()
                .collect())
        }

        async fn current_bar(&self, _symbol: &str) -> Result<Bar> {
            Ok(Bar {
                timestamp_ns: 1,
                open: 155.20,
                high: 155.23,
                low: 155.18,
                close: 155.21,
                volume: 100.0,
            })
        }

        async fn current_ticks(&self, _symbol: &str) -> Result<Vec<Tick>> {
            Ok(vec![
                Tick {
                    timestamp_ns: 1,
                    bid: 155.20,
                    ask: 155.21,
                },
                Tick {
                    timestamp_ns: 2,
                    bid: 155.21,
                    ask: 155.22,
                },
            ])
        }

        async fn recent_bars(
            &self,
            _symbol: &str,
            _interval: &str,
            count: usize,
        ) -> Result<Vec<Bar>> {
            if count == 0 {
                return Ok(Vec::new());
            }
            Ok(vec![Bar {
                timestamp_ns: 1,
                open: 155.20,
                high: 155.23,
                low: 155.18,
                close: 155.21,
                volume: 100.0,
            }])
        }

        async fn submit(&self, _action: &Action) -> Result<modelenv_proto::Fill> {
            self.submit_calls.fetch_add(1, Ordering::SeqCst);
            Ok(modelenv_proto::Fill {
                order_id: "mock-order".to_string(),
                timestamp_ns: 1,
                price: 155.21,
                size: 1.0,
                side: FillSide::Buy as i32,
                partial: false,
            })
        }

        async fn close_position(
            &self,
            position: &modelenv_proto::Position,
        ) -> Result<modelenv_proto::Fill> {
            self.close_calls
                .lock()
                .unwrap()
                .push(position.position_id.clone());
            Ok(modelenv_proto::Fill {
                order_id: format!("close-{}", position.position_id),
                timestamp_ns: 2,
                price: 155.21,
                size: position.volume,
                side: match position.side {
                    0 => FillSide::Sell as i32,
                    _ => FillSide::Buy as i32,
                },
                partial: false,
            })
        }

        async fn closed_positions(
            &self,
            _symbol: &str,
            from_timestamp_ns: i64,
        ) -> Result<Vec<ClosedPosition>> {
            Ok(self
                .broker_closed_positions
                .iter()
                .filter(|p| p.close_timestamp_ns >= from_timestamp_ns)
                .cloned()
                .collect())
        }

        async fn recent_fills(
            &self,
            _symbol: &str,
            count: usize,
        ) -> Result<Vec<modelenv_proto::Fill>> {
            let skip = self.broker_recent_fills.len().saturating_sub(count);
            Ok(self.broker_recent_fills[skip..].to_vec())
        }
    }

    #[test]
    fn test_position_creation() {
        let position = Position::new(
            "pos_1".to_string(),
            150.0,  // mid_price
            0.0001, // spread
            1.0,    // volume
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
            150.0,  // mid_price
            0.0001, // spread
            1.0,    // volume
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
            150.0,  // mid_price
            0.0001, // spread
            1.0,    // volume
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
            150.0,  // mid_price
            0.0001, // spread
            1.0,    // volume
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
            150.0,  // mid_price
            0.0001, // spread
            1.0,    // volume
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
        position.accrue_swap(current_timestamp, 0.01);

        // Swap should be 0.01 * 1.0 * 1.0 = 0.01
        assert_eq!(position.swap, 0.01);
    }

    #[test]
    fn training_mode_seeds_builtin_swap_defaults() {
        // Training mode: USDJPY gets the built-in Pepperstone default table value.
        let env =
            Environment::new(Mode::Training, "USDJPY".to_string(), "s3://x".to_string());
        assert_eq!(env.get_swap_rates(), (0.1121, -0.2424));

        // A symbol with no table entry falls back to no financing.
        let other =
            Environment::new(Mode::Training, "EURUSD".to_string(), "s3://x".to_string());
        assert_eq!(other.get_swap_rates(), (0.0, 0.0));
    }

    #[test]
    fn live_mode_does_not_seed_builtin_swap_defaults() {
        // Live mode: table is empty; real swap comes from the broker.
        let env = Environment::new(Mode::Live, "USDJPY".to_string(), "s3://x".to_string());
        assert_eq!(env.get_swap_rates(), (0.0, 0.0));
    }

    #[test]
    fn with_daily_swap_rate_overrides_builtin_default() {
        let env = Environment::new(Mode::Training, "USDJPY".to_string(), "s3://x".to_string())
            .with_daily_swap_rate("USDJPY".to_string(), -0.5, 0.25);
        assert_eq!(env.get_swap_rates(), (-0.5, 0.25));
    }

    #[test]
    fn default_swap_rate_for_known_and_unknown_symbol() {
        assert_eq!(default_swap_rate_for("USDJPY"), (0.1121, -0.2424));
        assert_eq!(default_swap_rate_for("EURUSD"), (0.0, 0.0));
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
        assert_eq!(window.total_realised_pnl(), 1.01);
        assert_eq!(window.total_realised_pnl_since(1500000000000), 1.01);
        assert_eq!(window.total_realised_pnl_since(2000000000001), 0.0);
    }

    #[tokio::test]
    async fn test_live_hold_action_skips_broker_submission() {
        let submit_calls = Arc::new(AtomicUsize::new(0));
        let broker_gateway = Arc::new(MockBrokerGateway {
            submit_calls: Arc::clone(&submit_calls),
            ..Default::default()
        });

        let mut environment =
            Environment::new(Mode::Live, "USDJPY".to_string(), "s3://unused".to_string())
                .with_broker_gateway(broker_gateway);

        let response = environment
            .step(Action {
                action: ActionType::ActionHold as i32,
                client_order_id: "hold-1".to_string(),
            })
            .await
            .unwrap();

        assert_eq!(submit_calls.load(Ordering::SeqCst), 0);
        assert!(response.data.is_some());
        assert!(environment.recent_fills.is_empty());
        // tick_count normalised as Count { cap: 100 } → 2 / 100
        assert!((obs_value(response.data.as_ref().unwrap(), "tick_count") - 0.02).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_training_reset_uses_custom_step_size_seconds() {
        let mut environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        );

        environment
            .reset(ResetRequest {
                symbol: "USDJPY".to_string(),
                episode_start_ts: 0,
                episode_end_ts: 0,
                seed: 0,
                step_size_seconds: 7,
            })
            .await
            .unwrap_err();

        assert_eq!(environment.step_size_ns, 7_000_000_000);
    }

    #[tokio::test]
    async fn test_training_reset_clears_reward_state() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let ts = &[1_325_484_000_000_000_000, 1_325_484_060_000_000_000];
        let prices = &[102.0, 103.0];
        write_test_parquet_for_interval(root, "marketdata/eoh-snapshot", "M1", "year=2012/month=01/day=02/hour=06", "20120102T060000Z.parquet", ts, prices);
        write_test_parquet_for_interval(root, "marketdata/eoh-snapshot", "M5", "year=2012/month=01/day=02/hour=06", "20120102T060000Z.parquet", ts, prices);
        write_test_parquet_for_interval(root, "marketdata/eoh-snapshot", "M15", "year=2012/month=01/day=02/hour=06", "20120102T060000Z.parquet", ts, prices);
        write_test_parquet_for_interval(root, "marketdata/eod-snapshot", "H1", "year=2012/month=01/day=02", "20120102T000000Z.parquet", ts, prices);
        write_test_parquet_for_interval(root, "marketdata/eod-snapshot", "H4", "year=2012/month=01/day=02", "20120102T000000Z.parquet", ts, prices);
        write_test_parquet_for_interval(root, "marketdata/eod-snapshot", "D1", "year=2012/month=01/day=02", "20120102T000000Z.parquet", ts, prices);
        write_test_parquet_for_interval(root, "marketdata/eow-snapshot", "W1", "year=2012/month=01", "20120102T000000Z.parquet", ts, prices);
        write_test_parquet_for_interval(root, "marketdata/eom-snapshot", "MN1", "year=2012/month=01", "20120101T000000Z.parquet", ts, prices);
        let news_dir = root.join("marketdata/eod-news-snapshot/symbol=USD-JPY/year=2012/month=01/day=01");
        std::fs::create_dir_all(&news_dir).unwrap();
        write_news_parquet(&news_dir.join("20120101T000000Z.parquet"), ts).unwrap();
        let tick_dir = root.join("marketdata/interval-price/symbol=USDJPY/interval=ticks/year=2012/month=01/day=01/hour=00");
        std::fs::create_dir_all(&tick_dir).unwrap();
        write_tick_parquet(&tick_dir.join("20120101T000000Z.parquet"), ts).unwrap();

        let mut environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            dir.path().to_string_lossy().to_string(),
        );

        environment.preload_training_data().await.unwrap();
        environment.prev_total_equity = Some(-0.25);
        // Pre-populate regime stats to verify they survive reset.
        environment.regime_reward_stats.update(1, 1.0);
        environment.regime_reward_stats.update(1, -0.5);
        environment.current_regime_bin = 0;

        environment
            .reset(ResetRequest {
                symbol: "USDJPY".to_string(),
                episode_start_ts: 0,
                episode_end_ts: 0,
                seed: 0,
                step_size_seconds: 0,
            })
            .await
            .unwrap();

        // prev_total_equity must be cleared; the regime stats must NOT be.
        assert_eq!(environment.prev_total_equity, None);
        assert_eq!(environment.current_regime_bin, 1);
        let mid_bin = &environment.regime_reward_stats.bins[1];
        assert_eq!(mid_bin.count, 2);
        assert!((mid_bin.sum - 0.5).abs() < 1e-9);

        let response = environment
            .step(Action {
                action: ActionType::ActionHold as i32,
                client_order_id: "hold-after-reset".to_string(),
            })
            .await
            .unwrap();

        assert_eq!(response.data.as_ref().unwrap().reward, 0.0);
    }

    #[tokio::test]
    async fn test_training_step_response_done_matches_episode_done() {
        let bars = vec![Bar {
            timestamp_ns: 0,
            open: 100.0,
            high: 101.0,
            low: 99.0,
            close: 100.5,
            volume: 1000.0,
        }];

        let episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            1_000_000_000,
        );

        let mut environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        );
        environment.step_size_ns = 2_000_000_000;
        environment.episode = Some(episode);

        let response = environment
            .step(Action {
                action: ActionType::ActionHold as i32,
                client_order_id: "done-test".to_string(),
            })
            .await
            .unwrap();

        assert!(response.data.as_ref().unwrap().done);
    }

    #[tokio::test]
    async fn test_training_step_advances_timestamp_and_returns_opened_position() {
        let m1_bars = (0..6)
            .map(|i| Bar {
                timestamp_ns: i * 60_000_000_000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
            })
            .collect::<Vec<_>>();
        let m5_bars = vec![
            Bar {
                timestamp_ns: 0,
                open: 200.0,
                high: 201.0,
                low: 199.0,
                close: 200.5,
                volume: 5000.0,
            },
            Bar {
                timestamp_ns: 300_000_000_000,
                open: 205.0,
                high: 206.0,
                low: 204.0,
                close: 205.5,
                volume: 5000.0,
            },
        ];

        let episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), m1_bars), ("M5".to_string(), m5_bars)]
                .into_iter()
                .collect(),
            0,
            300_000_000_000,
        )
        .with_ticks(vec![
            Tick {
                timestamp_ns: 61_000_000_000,
                bid: 101.10,
                ask: 101.11,
            },
            Tick {
                timestamp_ns: 119_000_000_000,
                bid: 102.20,
                ask: 102.21,
            },
        ]);

        let mut environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        );
        environment.step_size_ns = 60_000_000_000;
        environment.episode = Some(episode);

        let first = environment
            .step(Action {
                action: ActionType::ActionHold as i32,
                client_order_id: "hold-1".to_string(),
            })
            .await
            .unwrap();
        assert!(first.data.is_some());

        let second = environment
            .step(Action {
                action: ActionType::ActionBuy1 as i32,
                client_order_id: "buy-1".to_string(),
            })
            .await
            .unwrap();

        let observation = second.data.unwrap();
        // num_positions_buy normalised as Count { cap: 5 } → 1 / 5
        assert!((obs_value(&observation, "num_positions_buy") - 0.2).abs() < 1e-9);
        // tick_count normalised as Count { cap: 100 } → 1 / 100
        assert!((obs_value(&observation, "tick_count") - 0.01).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_training_step_surfaces_max_total_margin_after_buy() {
        // End-to-end guard for the full Step RPC path: a buy must surface a
        // non-zero max_total_margin on the returned Observation, matching the
        // open-position book / leverage. Regression cover for a server that
        // silently returns 0 (the field never being populated).
        let m1_bars = (0..6)
            .map(|i| Bar {
                timestamp_ns: i * 60_000_000_000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
            })
            .collect::<Vec<_>>();
        let m5_bars = vec![
            Bar {
                timestamp_ns: 0,
                open: 200.0,
                high: 201.0,
                low: 199.0,
                close: 200.5,
                volume: 5000.0,
            },
            Bar {
                timestamp_ns: 300_000_000_000,
                open: 205.0,
                high: 206.0,
                low: 204.0,
                close: 205.5,
                volume: 5000.0,
            },
        ];

        let episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), m1_bars), ("M5".to_string(), m5_bars)]
                .into_iter()
                .collect(),
            0,
            300_000_000_000,
        )
        .with_ticks(vec![
            Tick {
                timestamp_ns: 61_000_000_000,
                bid: 101.10,
                ask: 101.11,
            },
            Tick {
                timestamp_ns: 119_000_000_000,
                bid: 102.20,
                ask: 102.21,
            },
        ]);

        let leverage = 200.0;
        let mut environment =
            Environment::new(Mode::Training, "USDJPY".to_string(), "s3://unused".to_string())
                .with_leverage(leverage);
        environment.step_size_ns = 60_000_000_000;
        environment.episode = Some(episode);

        // Before any position is opened the peak is zero.
        let hold = environment
            .step(Action {
                action: ActionType::ActionHold as i32,
                client_order_id: "hold-1".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(
            hold.data.unwrap().max_total_margin,
            0.0,
            "flat book before any trade must report zero margin"
        );

        // A buy opens a position; the Observation must surface a positive peak.
        let buy = environment
            .step(Action {
                action: ActionType::ActionBuy1 as i32,
                client_order_id: "buy-1".to_string(),
            })
            .await
            .unwrap();
        let observation = buy.data.unwrap();

        let expected = environment
            .positions
            .iter()
            .map(|p| p.volume * p.entry_price)
            .sum::<f64>()
            / leverage;
        assert!(expected > 0.0, "buy step must leave an open position");
        assert!(
            observation.max_total_margin > 0.0,
            "Observation.max_total_margin must be non-zero after a buy, got {}",
            observation.max_total_margin
        );
        assert!(
            (observation.max_total_margin - expected).abs() < 1e-9,
            "surfaced margin {} must equal notional/leverage {}",
            observation.max_total_margin,
            expected
        );
    }

    #[tokio::test]
    async fn test_live_observation_ta_has_expected_columns() {
        let submit_calls = Arc::new(AtomicUsize::new(0));
        let broker_gateway = Arc::new(MockBrokerGateway {
            submit_calls: Arc::clone(&submit_calls),
            ..Default::default()
        });

        let mut environment =
            Environment::new(Mode::Live, "USDJPY".to_string(), "s3://unused".to_string())
                .with_broker_gateway(broker_gateway);

        let response = environment
            .step(Action {
                action: ActionType::ActionHold as i32,
                client_order_id: "ind-len".to_string(),
            })
            .await
            .unwrap();
        let observation = response.data.unwrap();
        // STATE_INTERVALS all have ta columns present
        for iv in &["M5", "M15", "H1", "W1"] {
            assert!(
                observation.state_columns.contains(&format!("{}_bar_close", iv)),
                "missing bar_close for {iv}"
            );
        }
        // Live bars are empty so ta values should be 0.0 (no warmup)
        assert_eq!(obs_value(&observation, "M5_ta_rsi_14"), 0.0);
        assert_eq!(obs_value(&observation, "M15_ta_adx_14"), 0.0);
    }

    #[tokio::test]
    async fn test_training_observation_ta_has_expected_columns() {
        let bars = (0..3)
            .map(|i| Bar {
                timestamp_ns: i * 60_000_000_000,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.0,
                volume: 1000.0,
            })
            .collect::<Vec<_>>();

        let episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            120_000_000_000,
        );

        let mut environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        );
        environment.step_size_ns = 60_000_000_000;
        environment.episode = Some(episode);

        let response = environment
            .step(Action {
                action: ActionType::ActionHold as i32,
                client_order_id: "ind-len-training".to_string(),
            })
            .await
            .unwrap();
        let observation = response.data.unwrap();
        // STATE_INTERVALS columns present
        for iv in &["M5", "M15", "H1", "W1"] {
            assert!(
                observation.state_columns.contains(&format!("{}_bar_close", iv)),
                "missing bar_close for {iv}"
            );
        }
    }

    #[tokio::test]
    async fn test_positions_observation_sorted_latest_first() {
        use crate::position::Side;

        let mut environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        );
        environment.positions.push(Position::new(
            "old".to_string(),
            150.0,
            0.0001,
            1.0,
            Side::Buy,
            1_000,
        ));
        environment.positions.push(Position::new(
            "new".to_string(),
            150.0,
            0.0001,
            1.0,
            Side::Buy,
            5_000,
        ));
        environment.positions.push(Position::new(
            "mid".to_string(),
            150.0,
            0.0001,
            1.0,
            Side::Buy,
            3_000,
        ));

        let positions = environment.positions_for_observation();
        assert_eq!(positions.len(), 3);
        assert_eq!(positions[0].open_timestamp_ns, 5_000);
        assert_eq!(positions[1].open_timestamp_ns, 3_000);
        assert_eq!(positions[2].open_timestamp_ns, 1_000);
    }

    #[tokio::test]
    async fn test_live_observation_populates_live_bars_and_reverses_live_ticks() {
        let submit_calls = Arc::new(AtomicUsize::new(0));
        let broker_gateway = Arc::new(MockBrokerGateway {
            submit_calls: Arc::clone(&submit_calls),
            ..Default::default()
        });

        let mut environment =
            Environment::new(Mode::Live, "USDJPY".to_string(), "s3://unused".to_string())
                .with_broker_gateway(broker_gateway);

        let response = environment
            .step(Action {
                action: ActionType::ActionHold as i32,
                client_order_id: "hold-1".to_string(),
            })
            .await
            .unwrap();
        let observation = response.data.unwrap();

        // All columns present (M5_bar_close, tick_ask are z-scored → 0.0 before warmup).
        assert!(observation.state_columns.contains(&"M5_bar_close".to_string()));
        assert!((obs_value(&observation, "tick_count") - 0.02).abs() < 1e-9);
        assert!(observation.state_columns.contains(&"tick_ask".to_string()));
    }

    #[tokio::test]
    async fn test_recent_fills_capped_and_sorted_latest_first() {
        let mut environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        );
        for i in 0..(crate::episode::RECENT_WINDOW + 5) {
            environment.recent_fills.push(Fill {
                order_id: format!("fill_{}", i),
                timestamp_ns: i as i64,
                price: 100.0 + i as f64,
                size: 1.0,
                side: FillSide::Buy,
                partial: false,
            });
        }

        let fills = environment.recent_fills_for_observation();
        assert_eq!(fills.len(), crate::episode::RECENT_WINDOW);
        assert_eq!(
            fills[0].timestamp_ns,
            (crate::episode::RECENT_WINDOW + 4) as i64
        );
        assert!(fills[0].timestamp_ns > fills[1].timestamp_ns);
    }

    #[tokio::test]
    async fn test_training_action_switch_applies_reward_penalty() {
        let bars = (0..3)
            .map(|i| Bar {
                timestamp_ns: i * 60_000_000_000,
                open: 100.0,
                high: 100.0,
                low: 100.0,
                close: 100.0,
                volume: 1000.0,
            })
            .collect::<Vec<_>>();

        let episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            120_000_000_000,
        );

        let mut environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        );
        environment.step_size_ns = 60_000_000_000;
        environment.reward_lambda = 0.0;
        environment.reward_holding_penalty = 0.0;
        environment.reward_action_penalty = 0.25;
        environment.episode = Some(episode);

        let first = environment
            .step(Action {
                action: ActionType::ActionHold as i32,
                client_order_id: "hold-1".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(first.data.as_ref().unwrap().reward, 0.0);

        let second = environment
            .step(Action {
                action: ActionType::ActionBuy1 as i32,
                client_order_id: "buy-1".to_string(),
            })
            .await
            .unwrap();

        // Old regime-zscore+clip path would have clamped this at -1.0. With the
        // fixed reward_scale=0.01, -0.25 / 0.01 = -25.0, so the penalty is
        // both visible and proportional to price units.
        assert!((second.data.as_ref().unwrap().reward - (-25.0)).abs() < 1e-9);
        assert_eq!(environment.last_action, Some(ActionType::ActionBuy1));
    }

    #[test]
    fn test_server_price_snapshot_ts_is_stored_on_environment() {
        let environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        )
        .with_price_snapshot_ts(123);

        assert_eq!(environment.price_snapshot_ts, Some(123));
    }

    #[test]
    fn test_local_cache_dir_is_stored_on_environment() {
        let environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        )
        .with_local_cache_dir("/cache/modelenv".to_string());

        assert_eq!(environment.local_cache_dir, "/cache/modelenv");
    }

    #[test]
    fn test_reward_parameter_setters_are_stored_on_environment() {
        let environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        )
        .with_reward_lambda(2.5)
        .with_reward_action_penalty(0.05)
        .with_reward_holding_penalty(0.0002);

        assert_eq!(environment.reward_parameters(), (2.5, 0.05, 0.0002, 0.01));
    }

    /// Reward is LINEAR in the loss (not squared) and applies the clamp.
    #[test]
    fn test_reward_linear_asymmetric_penalty_and_clip() {
        use crate::position::Side;

        let make = |unrealised: f64, clip: f64| {
            let mut env = Environment::new(
                Mode::Training,
                "USDJPY".to_string(),
                "s3://unused".to_string(),
            );
            // Isolate the asymmetric term: zero the other penalties, scale 0.01,
            // lambda 0.5 (losses weigh 1.5x), prev equity 0 so delta == unrealised.
            env.reward_lambda = 0.5;
            env.reward_action_penalty = 0.0;
            env.reward_holding_penalty = 0.0;
            env.reward_scale = 0.01;
            env.reward_clip = clip;
            env.prev_total_equity = Some(0.0);
            let mut pos = Position::new("p".to_string(), 150.0, 0.0, 1.0, Side::Buy, 0);
            pos.unrealised_pnl = unrealised;
            env.positions.push(pos);
            env
        };
        let hold = Action {
            action: ActionType::ActionHold as i32,
            client_order_id: "h".to_string(),
        };

        // Gain: delta=+1.0 → no penalty → +1.0/0.01 = +100.
        let mut env = make(1.0, 0.0);
        assert!((env.calculate_reward(&hold).unwrap() - 100.0).abs() < 1e-9);

        // Loss: delta=-1.0 → (-1.0 - 0.5*1.0)/0.01 = -150 (LINEAR; the old
        // squared form would have given (-1 - 1)/0.01 = -200).
        let mut env = make(-1.0, 0.0);
        assert!((env.calculate_reward(&hold).unwrap() - (-150.0)).abs() < 1e-9);

        // Larger loss stays proportional: delta=-2.0 → (-2 - 1)/0.01 = -300
        // (linear); squared would have been (-2 - 4)/0.01 = -600.
        let mut env = make(-2.0, 0.0);
        assert!((env.calculate_reward(&hold).unwrap() - (-300.0)).abs() < 1e-9);

        // Clip: same -300 reward clamped to the ±150 rail.
        let mut env = make(-2.0, 150.0);
        assert!((env.calculate_reward(&hold).unwrap() - (-150.0)).abs() < 1e-9);
    }

    fn write_test_parquet_for_interval(
        root: &std::path::Path,
        branch: &str,
        interval: &str,
        date_parts: &str,
        filename: &str,
        timestamps: &[i64],
        opens: &[f64],
    ) {
        let dir = root.join(format!(
            "{}/symbol=USDJPY/interval={}/{}",
            branch, interval, date_parts
        ));
        std::fs::create_dir_all(&dir).unwrap();
        write_test_parquet(&dir.join(filename), timestamps, opens).unwrap();
    }

    #[tokio::test]
    async fn test_training_preload_warms_first_reset() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let ts = &[1_325_484_000_000_000_000, 1_325_484_060_000_000_000];
        let prices = &[102.0, 103.0];

        // Write one parquet file per interval so preload doesn't fail
        write_test_parquet_for_interval(root, "marketdata/eoh-snapshot", "M1", "year=2012/month=01/day=02/hour=06", "20120102T060000Z.parquet", ts, prices);
        write_test_parquet_for_interval(root, "marketdata/eoh-snapshot", "M5", "year=2012/month=01/day=02/hour=06", "20120102T060000Z.parquet", ts, prices);
        write_test_parquet_for_interval(root, "marketdata/eoh-snapshot", "M15", "year=2012/month=01/day=02/hour=06", "20120102T060000Z.parquet", ts, prices);
        write_test_parquet_for_interval(root, "marketdata/eod-snapshot", "H1", "year=2012/month=01/day=02", "20120102T000000Z.parquet", ts, prices);
        write_test_parquet_for_interval(root, "marketdata/eod-snapshot", "H4", "year=2012/month=01/day=02", "20120102T000000Z.parquet", ts, prices);
        write_test_parquet_for_interval(root, "marketdata/eod-snapshot", "D1", "year=2012/month=01/day=02", "20120102T000000Z.parquet", ts, prices);
        write_test_parquet_for_interval(root, "marketdata/eow-snapshot", "W1", "year=2012/month=01", "20120102T000000Z.parquet", ts, prices);
        write_test_parquet_for_interval(root, "marketdata/eom-snapshot", "MN1", "year=2012/month=01", "20120101T000000Z.parquet", ts, prices);
        let news_dir = root.join("marketdata/eod-news-snapshot/symbol=USD-JPY/year=2012/month=01/day=01");
        std::fs::create_dir_all(&news_dir).unwrap();
        write_news_parquet(&news_dir.join("20120101T000000Z.parquet"), ts).unwrap();
        let tick_dir = root.join("marketdata/interval-price/symbol=USDJPY/interval=ticks/year=2012/month=01/day=01/hour=00");
        std::fs::create_dir_all(&tick_dir).unwrap();
        write_tick_parquet(&tick_dir.join("20120101T000000Z.parquet"), ts).unwrap();

        let mut environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            dir.path().to_string_lossy().to_string(),
        );

        environment.preload_training_data().await.unwrap();

        let observation = environment
            .reset(ResetRequest {
                symbol: "USDJPY".to_string(),
                episode_start_ts: 0,
                episode_end_ts: 0,
                seed: 0,
                step_size_seconds: 0,
            })
            .await
            .unwrap();

        // Training data loaded: M5 columns are present (z-scored → 0.0 before warmup).
        assert!(observation.state_columns.contains(&"M5_bar_close".to_string()));
        // sin_hour is passthrough, so raw value is present regardless of warmup.
        assert!(obs_value(&observation, "sin_hour").abs() <= 1.0);
    }

    #[tokio::test]
    async fn test_disable_hedging_nets_out_equal_volume_fully() {
        let m1_bars = (0..10)
            .map(|i| Bar {
                timestamp_ns: i * 60_000_000_000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
            })
            .collect::<Vec<_>>();

        let episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), m1_bars)].into_iter().collect(),
            0,
            600_000_000_000,
        );

        let mut environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        )
        .with_disable_hedging(true);
        environment.step_size_ns = 60_000_000_000;
        environment.episode = Some(episode);

        // BUY_1: open 1-unit long
        environment
            .step(Action {
                action: ActionType::ActionBuy1 as i32,
                client_order_id: "buy-1".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(environment.positions.len(), 1);
        assert_eq!(environment.positions[0].side, Side::Buy);

        // SELL_1: nets out exactly, positions go flat
        environment
            .step(Action {
                action: ActionType::ActionSell1 as i32,
                client_order_id: "sell-1".to_string(),
            })
            .await
            .unwrap();

        assert_eq!(environment.positions.len(), 0);
    }

    #[tokio::test]
    async fn test_disable_hedging_partially_reduces_larger_position() {
        let m1_bars = (0..10)
            .map(|i| Bar {
                timestamp_ns: i * 60_000_000_000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
            })
            .collect::<Vec<_>>();

        let episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), m1_bars)].into_iter().collect(),
            0,
            600_000_000_000,
        );

        let mut environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        )
        .with_disable_hedging(true);
        environment.step_size_ns = 60_000_000_000;
        environment.episode = Some(episode);

        // BUY_2: open two 1-unit long positions
        environment
            .step(Action {
                action: ActionType::ActionBuy2 as i32,
                client_order_id: "buy-2".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(environment.positions.len(), 2);
        assert!(environment.positions.iter().all(|p| p.volume == 1.0));

        // SELL_1: closes one, 1-unit long remains
        environment
            .step(Action {
                action: ActionType::ActionSell1 as i32,
                client_order_id: "sell-1".to_string(),
            })
            .await
            .unwrap();

        assert_eq!(environment.positions.len(), 1);
        assert_eq!(environment.positions[0].side, Side::Buy);
        assert_eq!(environment.positions[0].volume, 1.0);
    }

    #[tokio::test]
    async fn test_disable_hedging_excess_opens_remainder() {
        let m1_bars = (0..10)
            .map(|i| Bar {
                timestamp_ns: i * 60_000_000_000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
            })
            .collect::<Vec<_>>();

        let episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), m1_bars)].into_iter().collect(),
            0,
            600_000_000_000,
        );

        let mut environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        )
        .with_disable_hedging(true);
        environment.step_size_ns = 60_000_000_000;
        environment.episode = Some(episode);

        // BUY_1: open 1-unit long
        environment
            .step(Action {
                action: ActionType::ActionBuy1 as i32,
                client_order_id: "buy-1".to_string(),
            })
            .await
            .unwrap();

        // SELL_2: closes the 1-unit buy, opens remaining 1-unit sell
        environment
            .step(Action {
                action: ActionType::ActionSell2 as i32,
                client_order_id: "sell-2".to_string(),
            })
            .await
            .unwrap();

        assert_eq!(environment.positions.len(), 1);
        assert_eq!(environment.positions[0].side, Side::Sell);
        assert_eq!(environment.positions[0].volume, 1.0);
    }

    #[tokio::test]
    async fn test_hedging_enabled_allows_both_sides() {
        let m1_bars = (0..10)
            .map(|i| Bar {
                timestamp_ns: i * 60_000_000_000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
            })
            .collect::<Vec<_>>();

        let episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), m1_bars)].into_iter().collect(),
            0,
            600_000_000_000,
        );

        let mut environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        )
        .with_disable_hedging(false);
        environment.step_size_ns = 60_000_000_000;
        environment.episode = Some(episode);

        environment
            .step(Action {
                action: ActionType::ActionBuy1 as i32,
                client_order_id: "buy-1".to_string(),
            })
            .await
            .unwrap();

        environment
            .step(Action {
                action: ActionType::ActionSell1 as i32,
                client_order_id: "sell-1".to_string(),
            })
            .await
            .unwrap();

        // Both positions coexist (hedging allowed)
        assert_eq!(environment.positions.len(), 2);
    }

    #[tokio::test]
    async fn test_disable_hedging_fifo_sells_lifo_buys() {
        // Buy action reduces sells FIFO (oldest sell first).
        // Sell action reduces buys LIFO (newest buy first).
        let m1_bars = (0..10)
            .map(|i| Bar {
                timestamp_ns: i * 60_000_000_000,
                open: 145.0 + i as f64,
                high: 146.0 + i as f64,
                low: 144.0 + i as f64,
                close: 145.5 + i as f64,
                volume: 1000.0,
            })
            .collect::<Vec<_>>();

        // --- Buy action (reduce sells) → FIFO ---
        {
            let episode = Episode::new(
                "USDJPY".to_string(),
                [("M1".to_string(), m1_bars.clone())]
                    .into_iter()
                    .collect(),
                0,
                600_000_000_000,
            );

            let mut env = Environment::new(
                Mode::Training,
                "USDJPY".to_string(),
                "s3://unused".to_string(),
            )
            .with_disable_hedging(true);
            env.step_size_ns = 60_000_000_000;
            env.episode = Some(episode);

            // Push three sells with distinct timestamps.
            env.positions.push(Position::new(
                "sell_old".to_string(), 145.0, 0.0, 1.0, Side::Sell, 1_000,
            ));
            env.positions.push(Position::new(
                "sell_mid".to_string(), 145.0, 0.0, 1.0, Side::Sell, 2_000,
            ));
            env.positions.push(Position::new(
                "sell_new".to_string(), 145.0, 0.0, 1.0, Side::Sell, 3_000,
            ));

            // Buy1 → should close oldest sell (sell_old, t=1000).
            env.step(Action {
                action: ActionType::ActionBuy1 as i32,
                client_order_id: "buy-reduce-sells".to_string(),
            })
            .await
            .unwrap();

            assert_eq!(env.positions.len(), 2);
            assert!(env.positions.iter().all(|p| p.side == Side::Sell));
            let ids: Vec<&str> = env.positions.iter().map(|p| p.position_id.as_str()).collect();
            assert!(ids.contains(&"sell_mid"));
            assert!(ids.contains(&"sell_new"));
        }

        // --- Sell action (reduce buys) → LIFO ---
        {
            let episode = Episode::new(
                "USDJPY".to_string(),
                [("M1".to_string(), m1_bars.clone())]
                    .into_iter()
                    .collect(),
                0,
                600_000_000_000,
            );

            let mut env = Environment::new(
                Mode::Training,
                "USDJPY".to_string(),
                "s3://unused".to_string(),
            )
            .with_disable_hedging(true);
            env.step_size_ns = 60_000_000_000;
            env.episode = Some(episode);

            // Push three buys with distinct timestamps.
            env.positions.push(Position::new(
                "buy_old".to_string(), 145.0, 0.0, 1.0, Side::Buy, 1_000,
            ));
            env.positions.push(Position::new(
                "buy_mid".to_string(), 145.0, 0.0, 1.0, Side::Buy, 2_000,
            ));
            env.positions.push(Position::new(
                "buy_new".to_string(), 145.0, 0.0, 1.0, Side::Buy, 3_000,
            ));

            // Sell1 → should close newest buy (buy_new, t=3000).
            env.step(Action {
                action: ActionType::ActionSell1 as i32,
                client_order_id: "sell-reduce-buys".to_string(),
            })
            .await
            .unwrap();

            assert_eq!(env.positions.len(), 2);
            assert!(env.positions.iter().all(|p| p.side == Side::Buy));
            let ids: Vec<&str> = env.positions.iter().map(|p| p.position_id.as_str()).collect();
            assert!(ids.contains(&"buy_old"));
            assert!(ids.contains(&"buy_mid"));
        }
    }

    #[tokio::test]
    async fn test_session_liquidation_closes_shorts_and_losing_longs_keeps_winners() {
        use crate::position::Side;

        // Minimal episode so close_position -> fill_price can price the close.
        let bars = vec![Bar {
            timestamp_ns: 0,
            open: 100.0,
            high: 101.0,
            low: 99.0,
            close: 100.5,
            volume: 1000.0,
        }];
        let episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            1_000_000_000,
        );
        let mut env = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        );
        env.episode = Some(episode);

        // Net P&L = unrealised_pnl + swap. The caller marks to market first; we
        // set the fields directly to exercise the selection predicate.
        let mut win_long = Position::new("win_long".to_string(), 100.0, 0.0, 1.0, Side::Buy, 0);
        win_long.unrealised_pnl = 5.0; // net +5 -> kept
        let mut even_long = Position::new("even_long".to_string(), 100.0, 0.0, 1.0, Side::Buy, 0);
        even_long.unrealised_pnl = 1.0;
        even_long.swap = -1.0; // net 0 (break-even) -> kept
        let mut lose_long = Position::new("lose_long".to_string(), 100.0, 0.0, 1.0, Side::Buy, 0);
        lose_long.unrealised_pnl = 0.5;
        lose_long.swap = -2.0; // net -1.5 -> closed
        let mut short_win = Position::new("short_win".to_string(), 100.0, 0.0, 1.0, Side::Sell, 0);
        short_win.unrealised_pnl = 10.0; // profitable short is still closed

        env.positions.push(win_long);
        env.positions.push(even_long);
        env.positions.push(lose_long);
        env.positions.push(short_win);

        env.close_session_positions().unwrap();

        let ids: Vec<&str> = env.positions.iter().map(|p| p.position_id.as_str()).collect();
        assert_eq!(env.positions.len(), 2, "kept positions: {:?}", ids);
        assert!(ids.contains(&"win_long"));
        assert!(ids.contains(&"even_long"));
        assert!(!ids.contains(&"lose_long"));
        assert!(!ids.contains(&"short_win"));
    }

    #[tokio::test]
    async fn test_session_liquidation_no_positions_is_noop() {
        let mut env = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        );
        env.close_session_positions().unwrap();
        assert!(env.positions.is_empty());
    }

    #[tokio::test]
    async fn test_live_session_liquidation_closes_at_broker_keeps_winning_longs() {
        // Mock broker bar: open 155.20 / close 155.21 -> mid 155.205. The
        // win/lose decision recomputes unrealised P&L from entry vs. this mid
        // (plus the broker-reported swap), not from the proto's stale field.
        let proto_position = |id: &str, entry: f64, swap: f64, side: i32| modelenv_proto::Position {
            position_id: id.to_string(),
            entry_price: entry,
            unrealised_pnl: 0.0,
            swap,
            open_timestamp_ns: 0,
            volume: 1.0,
            side,
        };
        let close_calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let broker_gateway = Arc::new(MockBrokerGateway {
            positions: vec![
                proto_position("win_long", 150.0, 0.0, 0), // net +5.205 -> kept
                proto_position("lose_long", 160.0, 0.0, 0), // net -4.795 -> closed
                proto_position("swap_lose_long", 155.0, -10.0, 0), // net -9.795 -> closed
                proto_position("short_win", 160.0, 0.0, 1), // shorts always closed
            ],
            close_calls: Arc::clone(&close_calls),
            ..Default::default()
        });

        let mut env = Environment::new(
            Mode::Live,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        )
        .with_broker_gateway(broker_gateway);

        env.close_session_positions_live().await.unwrap();

        let closed = close_calls.lock().unwrap().clone();
        assert_eq!(closed.len(), 3, "closed: {:?}", closed);
        assert!(closed.contains(&"lose_long".to_string()));
        assert!(closed.contains(&"swap_lose_long".to_string()));
        assert!(closed.contains(&"short_win".to_string()));
        assert!(!closed.contains(&"win_long".to_string()));
        // Close fills are recorded so they surface in recent_fills.
        assert_eq!(env.recent_fills.len(), 3);
    }

    #[tokio::test]
    async fn test_live_reset_rebuilds_session_realised_pnl_and_recent_fills() {
        use crate::position::Side;

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;
        let closed = |id: &str, close_ts: i64| ClosedPosition {
            position_id: id.to_string(),
            entry_price: 150.0,
            close_price: 151.0,
            volume: 1.0,
            side: Side::Buy,
            realised_pnl: 5.0,
            swap: -1.0,
            open_timestamp_ns: close_ts - NANOS_PER_DAY,
            close_timestamp_ns: close_ts,
        };
        let fill = |id: &str, ts: i64| modelenv_proto::Fill {
            order_id: id.to_string(),
            timestamp_ns: ts,
            price: 151.0,
            size: 1.0,
            side: FillSide::Sell as i32,
            partial: false,
        };
        let broker_gateway = Arc::new(MockBrokerGateway {
            broker_closed_positions: vec![
                // Closed this session (1ns ago; the default session start is
                // 0h UTC today, which is always <= now - 1).
                closed("this_session_deal", now_ns - 1),
                // Closed in a previous session: outside the cutoff, must not
                // be restored.
                closed("previous_session_deal", now_ns - 2 * NANOS_PER_DAY),
            ],
            broker_recent_fills: vec![
                fill("deal-1", now_ns - 2),
                fill("deal-2", now_ns - 1),
            ],
            ..Default::default()
        });

        let mut env = Environment::new(
            Mode::Live,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        )
        .with_broker_gateway(broker_gateway);

        env.reset(ResetRequest {
            symbol: "USDJPY".to_string(),
            episode_start_ts: 0,
            episode_end_ts: 0,
            seed: 0,
            step_size_seconds: 0,
        })
        .await
        .unwrap();

        // Session-scoped realised P&L restored from broker deal history
        // (5.0 - 1.0 swap), previous sessions excluded.
        assert_eq!(env.closed_position_window.len(), 1);
        assert!(
            (env.closed_position_window.total_realised_pnl() - 4.0).abs() < 1e-9,
            "expected restored net realised P&L of 4.0, got {}",
            env.closed_position_window.total_realised_pnl()
        );
        // Recent fills restored in ascending order (not re-logged).
        assert_eq!(env.recent_fills.len(), 2);
        assert_eq!(env.recent_fills[0].order_id, "deal-1");
        assert_eq!(env.recent_fills[1].order_id, "deal-2");
    }

    #[tokio::test]
    async fn test_live_session_liquidation_empty_broker_book_is_noop() {
        let close_calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let broker_gateway = Arc::new(MockBrokerGateway {
            close_calls: Arc::clone(&close_calls),
            ..Default::default()
        });
        let mut env = Environment::new(
            Mode::Live,
            "USDJPY".to_string(),
            "s3://unused".to_string(),
        )
        .with_broker_gateway(broker_gateway);

        env.close_session_positions_live().await.unwrap();

        assert!(close_calls.lock().unwrap().is_empty());
        assert!(env.recent_fills.is_empty());
    }
}
