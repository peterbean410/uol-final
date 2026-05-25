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
    daily_swap_rates: HashMap<String, f64>,
    // Track the last timestamp when swap was accrued for day boundary detection
    last_swap_accrual_timestamp: i64,
    // Broker gateway for Production Mode (Arc for cloneability)
    broker_gateway: Option<Arc<dyn BrokerGateway + Send + Sync>>,
    // Reward function configuration
    reward_lambda: f64,
    reward_action_penalty: f64,
    reward_holding_penalty: f64,
    disable_hedging: bool,
    // Track previous step's total equity for delta_V_t calculation
    prev_total_equity: Option<f64>,
    // Running statistics for reward normalisation
    reward_running_sum: f64,
    reward_running_sum_sq: f64,
    reward_count: u64,
    last_observation_timestamp_ns: Option<i64>,
    // State feature normaliser (rolling z-scores, volume log-transform, etc.).
    normaliser: Normaliser,
    // In-memory bar storage shared by training and live modes.
    // Training: populated from Episode after parquet load.
    // Live: updated from broker gateway on each observation.
    bars: HashMap<String, Vec<Bar>>,
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

impl Environment {
    /// Create a new environment instance
    pub fn new(mode: Mode, symbol: String, s3_prefix: String) -> Self {
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
            daily_swap_rates: HashMap::new(),
            last_swap_accrual_timestamp: 0, // Will be set on reset
            broker_gateway: None,
            reward_lambda: 1.0, // Default asymmetric drawdown penalty coefficient
            reward_action_penalty: 0.001, // Default action penalty (scaled to USD/JPY spread)
            reward_holding_penalty: 1e-6, // Default holding penalty (orders of magnitude smaller)
            disable_hedging: true,
            prev_total_equity: None,
            reward_running_sum: 0.0,
            reward_running_sum_sq: 0.0,
            reward_count: 0,
            last_observation_timestamp_ns: None,
            normaliser: Normaliser::new(&state_columns()),
            bars: HashMap::new(),
        }
    }

    /// Set the transaction cost per trade
    pub fn with_transaction_cost(mut self, cost: f64) -> Self {
        self.transaction_cost = cost;
        self
    }

    /// Set the daily swap rate for a symbol
    pub fn with_daily_swap_rate(mut self, symbol: String, rate: f64) -> Self {
        self.daily_swap_rates.insert(symbol, rate);
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

    pub fn with_disable_hedging(mut self, disable_hedging: bool) -> Self {
        self.disable_hedging = disable_hedging;
        self
    }

    pub fn reward_parameters(&self) -> (f64, f64, f64) {
        (
            self.reward_lambda,
            self.reward_action_penalty,
            self.reward_holding_penalty,
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

    /// Get the current swap rate for the symbol
    fn get_swap_rate(&self) -> f64 {
        *self.daily_swap_rates.get(&self.symbol).unwrap_or(&0.0)
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
    }

    fn reset_reward_state(&mut self) {
        self.prev_total_equity = None;
        self.reward_running_sum = 0.0;
        self.reward_running_sum_sq = 0.0;
        self.reward_count = 0;
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
            realised_pnl_12m: self
                .closed_position_window
                .total_realised_pnl_12m(current_timestamp),
            recent_fills,
            ta,
            double_bottoms,
            double_tops,
            live_ticks,
            done: false,
            reward: 0.0,
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
                self.apply_action(&action)?;

                let realised_pnl_12m = self
                    .closed_position_window
                    .total_realised_pnl_12m(current_timestamp);
                let proto_positions = self.positions_for_observation();
                let recent_fills = self.recent_fills_for_observation();
                let mut observation = self
                    .episode
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Episode not initialized. Call reset() first."))?
                    .get_observation(
                        proto_positions.as_slice(),
                        realised_pnl_12m,
                        Some(prev_timestamp),
                    );
                observation.recent_fills = recent_fills;
                self.last_observation_timestamp_ns = Some(observation.timestamp_ns);

                // Calculate reward based on the previous step's action state.
                let reward = self.calculate_reward(&action)?;

                // Update last_action after calculating reward so action-switch penalties
                // compare the current action against the previous step's action.
                self.last_action = match ActionType::try_from(action.action) {
                    Ok(action_type) => Some(action_type),
                    Err(_) => None,
                };

                observation.reward = reward;
                observation.done = !still_running;
                Ok(StepResponse {
                    data: Some(self.normalise_observation(observation.into_observation())),
                    info: "".to_string(),
                })
            }
            Mode::Live => {
                // Production mode - submit action to broker and return execution results
                info!("Production mode: submitting action to broker");

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

                    self.recent_fills.push(Fill {
                        order_id: order_id.clone(),
                        timestamp_ns: fill.timestamp_ns,
                        price,
                        size,
                        side: match FillSide::try_from(fill.side) {
                            Ok(s) => s,
                            Err(_) => FillSide::Buy,
                        },
                        partial,
                    });

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
                let realised_pnl_12m = self.realised_pnl_12m();
                reconcile_positions(
                    &self
                        .positions
                        .iter()
                        .map(|p| p.to_proto())
                        .collect::<Vec<_>>(),
                    &broker_positions,
                    realised_pnl_12m,
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

                // Calculate reward
                let reward = self.calculate_reward(&action)?;

                // Update last_action after calculating reward
                self.last_action = match ActionType::try_from(action.action) {
                    Ok(action_type) => Some(action_type),
                    Err(_) => None,
                };

                observation.reward = reward;
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
                let realised_pnl = self.realised_pnl_12m();
                let mut live = episode.get_observation(
                    proto_positions.as_slice(),
                    realised_pnl,
                    self.last_observation_timestamp_ns,
                );
                live.recent_fills = recent_fills;
                self.last_observation_timestamp_ns = Some(live.timestamp_ns);
                Ok(live.into_reference())
            }
            Mode::Live => {
                let live = self
                    .build_live_observation(self.symbol.clone())
                    .await?;
                self.last_observation_timestamp_ns = Some(live.timestamp_ns);
                Ok(live.into_reference())
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
                    self.realised_pnl_12m(),
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
                            let start = idx.saturating_sub(RECENT_WINDOW);
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
                        .recent_bars(&req.symbol, interval, RECENT_WINDOW)
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

    /// Calculate the rolling 12-month realised P/L
    fn realised_pnl_12m(&self) -> f64 {
        let current_timestamp = self.current_timestamp();
        self.closed_position_window
            .total_realised_pnl_12m(current_timestamp)
    }

    /// Get the current timestamp from the episode
    fn current_timestamp(&self) -> i64 {
        if let Some(episode) = &self.episode {
            episode.get_cursor_timestamp()
        } else {
            now_ns()
        }
    }

    /// Calculate reward based on action
    fn calculate_reward(&mut self, action: &Action) -> Result<f64> {
        // Get current timestamp
        let current_timestamp = self.current_timestamp();

        // Calculate current total equity (unrealised_pnl + realised_pnl_12m)
        let current_unrealised_pnl: f64 = self.positions.iter().map(|p| p.unrealised_pnl).sum();
        let current_realised_pnl_12m = self
            .closed_position_window
            .total_realised_pnl_12m(current_timestamp);
        let current_total_equity = current_unrealised_pnl + current_realised_pnl_12m;

        // Calculate delta_V_t (change in total equity)
        let delta_v_t = if let Some(prev_equity) = self.prev_total_equity {
            current_total_equity - prev_equity
        } else {
            // First step - no previous equity to compare
            0.0
        };

        // Update previous total equity for next step
        self.prev_total_equity = Some(current_total_equity);

        // Calculate asymmetric drawdown penalty
        // Only apply penalty when delta_V_t is negative
        let asymmetric_penalty = if delta_v_t < 0.0 {
            self.reward_lambda * (delta_v_t.abs()).powf(2.0)
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

        // Calculate final reward
        let reward = delta_v_t - asymmetric_penalty - action_penalty - holding_penalty;

        // Update running statistics for reward normalisation
        self.reward_running_sum += reward;
        self.reward_running_sum_sq += reward * reward;
        self.reward_count += 1;

        // Calculate running mean and standard deviation
        let mean = self.reward_running_sum / self.reward_count as f64;
        let variance = (self.reward_running_sum_sq / self.reward_count as f64) - (mean * mean);
        let std_dev = variance.max(0.0).sqrt();

        // Normalise reward using running statistics
        // Keep signal between -1.0 and 1.0
        let normalised_reward = if std_dev > 1e-8 {
            (reward - mean) / std_dev
        } else {
            reward
        };

        // Clip to [-1.0, 1.0]
        let clipped_reward = normalised_reward.clamp(-1.0, 1.0);

        Ok(clipped_reward)
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

        self.closed_position_window
            .add_closed_position(closed_position);

        self.positions
            .retain(|p| p.position_id != position_id);

        self.recent_fills.push(Fill {
            order_id: format!("fill_{}", current_timestamp),
            timestamp_ns: current_timestamp,
            price: close_price,
            size: position.volume,
            side: match position.side {
                Side::Buy => FillSide::Buy,
                Side::Sell => FillSide::Sell,
            },
            partial: false,
        });

        Ok(())
    }

    /// Accrue swap on all open positions
    /// Returns true if swap was accrued for any position
    fn accrue_swap_on_positions(&mut self) -> Result<bool> {
        let current_timestamp = self.current_timestamp();
        let swap_rate = self.get_swap_rate();

        let mut swap_accrued = false;
        for position in &mut self.positions {
            if position.accrue_swap(current_timestamp, swap_rate) {
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

    #[test]
    fn test_seconds_to_ns_passes_zero_through() {
        assert_eq!(seconds_to_ns(0, "episode_start_ts").unwrap(), 0);
    }

    #[test]
    fn test_seconds_to_ns_converts_2012_episode_boundary() {
        // 2012-01-02T23:00:00 UTC in Unix seconds
        let ts_s = 1_325_545_200_i64;
        let expected_ns = 1_325_545_200_000_000_000_i64;
        assert_eq!(
            seconds_to_ns(ts_s, "episode_end_ts").unwrap(),
            expected_ns
        );
    }

    #[test]
    fn test_seconds_to_ns_rejects_overflow() {
        // Any value > i64::MAX / 1e9 (~ year 2262) overflows on multiply.
        let huge = i64::MAX / 1_000_000_000 + 1;
        let err = seconds_to_ns(huge, "episode_end_ts").unwrap_err();
        assert!(err.to_string().contains("too large"));
    }

    /// Helper: find the value at a named column in the first row of an Observation.
    fn obs_value(obs: &modelenv_proto::Observation, column: &str) -> f64 {
        let idx = obs
            .state_columns
            .iter()
            .position(|c| c == column)
            .unwrap_or_else(|| panic!("column not found: {}", column));
        obs.state_data[0].values[idx]
    }

    struct MockBrokerGateway {
        submit_calls: Arc<AtomicUsize>,
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
            Ok(vec![])
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

    #[tokio::test]
    async fn test_live_hold_action_skips_broker_submission() {
        let submit_calls = Arc::new(AtomicUsize::new(0));
        let broker_gateway = Arc::new(MockBrokerGateway {
            submit_calls: Arc::clone(&submit_calls),
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
        environment.reward_running_sum = 5.0;
        environment.reward_running_sum_sq = 25.0;
        environment.reward_count = 4;

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

        assert_eq!(environment.prev_total_equity, None);
        assert_eq!(environment.reward_running_sum, 0.0);
        assert_eq!(environment.reward_running_sum_sq, 0.0);
        assert_eq!(environment.reward_count, 0);

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
    async fn test_live_observation_ta_has_expected_columns() {
        let submit_calls = Arc::new(AtomicUsize::new(0));
        let broker_gateway = Arc::new(MockBrokerGateway {
            submit_calls: Arc::clone(&submit_calls),
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

        assert!(second.data.as_ref().unwrap().reward < -0.9);
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

        assert_eq!(environment.reward_parameters(), (2.5, 0.05, 0.0002));
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
}
