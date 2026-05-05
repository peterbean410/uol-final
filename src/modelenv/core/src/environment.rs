// Environment module
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

use log::info;
use modelenv_proto::{
    Action, ActionType, Bar, BarList, Observation, ObserveRequest, ResetRequest, StepResponse,
    Tick,
};

use crate::broker_gateway::BrokerGateway;
use crate::config::Mode;
use crate::data_loader::{now_ns, DEFAULT_LOCAL_CACHE_DIR, TIME_INTERVALS};
use crate::episode::{initialize_episode, preload_training_market_data, Episode, RECENT_WINDOW};
use crate::indicators::{compute_interval_indicators, INDICATORS_PER_INTERVAL};
use crate::market_data_cache::MarketDataCache;
use crate::position::{ClosedPositionWindow, Position, Side};
use crate::reconciliation::reconcile_positions;

const DEFAULT_STEP_SIZE_NS: i64 = 5_000_000_000;

/// The main environment struct
#[derive(Clone)]
pub struct Environment {
    mode: Mode,
    symbol: String,
    s3_prefix: String,
    local_cache_dir: String,
    price_snapshot_ts: Option<i64>,
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
    // Track previous step's total equity for delta_V_t calculation
    prev_total_equity: Option<f64>,
    // Running statistics for reward normalisation
    reward_running_sum: f64,
    reward_running_sum_sq: f64,
    reward_count: u64,
    last_observation_timestamp_ns: Option<i64>,
}

/// Represents a trade execution record
#[derive(Debug, Clone)]
pub struct Fill {
    pub order_id: String,
    pub timestamp_ns: i64,
    pub price: f64,
    pub size: f64,
    pub side: ActionType,
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
            prev_total_equity: None,
            reward_running_sum: 0.0,
            reward_running_sum_sq: 0.0,
            reward_count: 0,
            last_observation_timestamp_ns: None,
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
    pub async fn reset(&mut self, req: ResetRequest) -> Result<Observation> {
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

                // Initialize episode with S3 parquet loading
                let market_data_cache = self.market_data_cache.clone();
                let mut episode = initialize_episode(
                    &req.symbol,
                    &self.s3_prefix,
                    &self.local_cache_dir,
                    self.price_snapshot_ts,
                    req.episode_start_ts,
                    req.episode_end_ts,
                    &market_data_cache,
                )
                .await?;
                // Advance the cursor by one step so the first observation has a
                // populated live tick window (cursor lands at first_bar_ts + step_size).
                episode.advance(self.step_size_ns);
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

    async fn build_live_observation(&mut self, symbol: String) -> Result<Observation> {
        let mut live_bars: HashMap<String, Bar> = HashMap::new();
        let mut recent_bars: HashMap<String, BarList> = HashMap::new();
        let mut fallback_current_bar: Option<Bar> = None;
        let mut indicators: Vec<f64> =
            Vec::with_capacity(TIME_INTERVALS.len() * INDICATORS_PER_INTERVAL);
        let recent_ticks_raw: Vec<Tick> = if let Some(broker) = &self.broker_gateway {
            for interval in TIME_INTERVALS {
                let bars = broker.recent_bars(&symbol, interval, RECENT_WINDOW).await?;
                let interval_block = compute_interval_indicators(&bars);
                indicators.extend_from_slice(&interval_block);
                if let Some(latest) = bars.last().cloned() {
                    live_bars.insert(interval.to_string(), latest.clone());
                    if *interval == "M1" {
                        fallback_current_bar = Some(latest);
                    }
                    let recent: Vec<Bar> = if bars.len() > 1 {
                        bars[..bars.len() - 1]
                            .iter()
                            .rev()
                            .take(RECENT_WINDOW)
                            .cloned()
                            .collect()
                    } else {
                        Vec::new()
                    };
                    recent_bars.insert(interval.to_string(), BarList { bars: recent });
                }
            }
            broker.current_ticks(&symbol).await?
        } else {
            indicators.resize(TIME_INTERVALS.len() * INDICATORS_PER_INTERVAL, 0.0);
            Vec::new()
        };

        let current_bar = fallback_current_bar.unwrap_or(Bar {
            timestamp_ns: now_ns(),
            open: 0.0,
            high: 0.0,
            low: 0.0,
            close: 0.0,
            volume: 0.0,
        });
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

        let recent_lower = live_lower - crate::episode::RECENT_TICK_WINDOW_NS;
        let recent_ticks: Vec<Tick> = recent_ticks_raw
            .iter()
            .filter(|t| t.timestamp_ns >= recent_lower && t.timestamp_ns < live_lower)
            .rev()
            .cloned()
            .collect();

        let proto_positions = self.positions_for_observation();

        let observation = Observation {
            timestamp_ns: current_timestamp,
            symbol,
            live_bars,
            recent_bars,
            positions: proto_positions,
            realised_pnl_12m: self
                .closed_position_window
                .total_realised_pnl_12m(current_timestamp),
            recent_fills,
            indicators,
            recent_ticks,
            live_ticks,
            recent_news: vec![],
            done: false,
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

                Ok(StepResponse {
                    observation: Some(observation),
                    reward,
                    done: !still_running,
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
                        side: match ActionType::try_from(fill.side) {
                            Ok(action_type) => action_type,
                            Err(_) => ActionType::ActionHold,
                        },
                        partial,
                    });

                    info!(
                        "Recorded fill: order_id={}, price={}, size={}, partial={}",
                        order_id, price, size, partial
                    );
                }

                // Update positions based on broker response
                // Get current bar to calculate P/L
                let broker = self.get_broker_gateway()?;
                let current_bar = broker.current_bar(&self.symbol).await?;

                // Update unrealised P/L for all positions based on current bar
                let current_mid_price = (current_bar.open + current_bar.close) / 2.0;
                for position in &mut self.positions {
                    position.unrealised_pnl = position.calculate_unrealised_pnl(current_mid_price);
                }

                // Reconcile with broker positions
                // Get broker positions for reconciliation
                let broker = self.get_broker_gateway()?;
                let broker_positions = broker.sync_positions(&self.symbol).await?;

                // Get broker's reported realised P/L
                // Note: The broker gateway doesn't currently provide realised P/L
                // For now, we'll use 0.0 as a placeholder and log if reconciliation shows discrepancy
                let broker_realised_pnl = 0.0; // TODO: Add method to get broker realised P/L
                let realised_pnl_12m = self.realised_pnl_12m();

                // Perform reconciliation - this logs warnings for discrepancies
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

                let observation = self.build_live_observation(self.symbol.clone()).await?;

                // Calculate reward
                let reward = self.calculate_reward(&action)?;

                // Update last_action after calculating reward
                self.last_action = match ActionType::try_from(action.action) {
                    Ok(action_type) => Some(action_type),
                    Err(_) => None,
                };

                Ok(StepResponse {
                    observation: Some(observation),
                    reward,
                    done: false,
                    info: "".to_string(),
                })
            }
        }
    }

    /// Get current observation without advancing
    pub async fn observe(&mut self, req: ObserveRequest) -> Result<Observation> {
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
                Ok(observation)
            }
            Mode::Live => self.build_live_observation(req.symbol).await,
        }
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

        // Calculate holding penalty (c_h) for position duration
        // Sum the duration of each open position
        let holding_penalty = if !self.positions.is_empty() {
            let total_duration_ns: i64 = self
                .positions
                .iter()
                .map(|p| current_timestamp - p.open_timestamp_ns)
                .sum();
            // Convert nanoseconds to a reasonable time unit and apply penalty
            // Using 1e-6 as the holding penalty coefficient
            self.reward_holding_penalty * (total_duration_ns as f64)
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
                // Hold - take no action
                // But we need to accrue swap on open positions
                self.accrue_swap_on_positions()?;
            }
            ActionType::ActionOpenBuy => {
                // Open a new buy position
                self.open_position(1.0, Side::Buy)?; // Default volume
            }
            ActionType::ActionCloseMostLoss => {
                // Close the position with the largest unrealised loss
                self.close_most_loss()?;
            }
            ActionType::ActionCloseMostProfit => {
                // Close the position with the largest unrealised profit
                self.close_most_profit()?;
            }
            ActionType::ActionCloseAllLoss => {
                // Close all positions at a loss
                self.close_all_loss()?;
            }
            ActionType::ActionCloseAllProfit => {
                // Close all positions that are profitable
                self.close_all_profit()?;
            }
        }
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

    /// Open a new position
    fn open_position(&mut self, volume: f64, side: Side) -> Result<()> {
        let current_timestamp = self.current_timestamp();

        // Get the current bar to calculate mid_price and spread
        let mid_price = self.get_current_mid_price()?;
        let spread = self.get_current_spread()?;

        let position = Position::new(
            format!("pos_{}", current_timestamp),
            mid_price,
            spread,
            volume,
            side,
            current_timestamp,
        );

        self.positions.push(position);
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

    /// Close the position with the largest unrealised loss
    fn close_most_loss(&mut self) -> Result<()> {
        if self.positions.is_empty() {
            return Ok(());
        }

        // Find the position with the largest unrealised loss
        let positions_to_close: Vec<Position> = self
            .positions
            .iter()
            .filter(|p| p.unrealised_pnl < 0.0)
            .cloned()
            .collect();

        if positions_to_close.is_empty() {
            return Ok(()); // No positions at a loss
        }

        // Find the minimum unrealised P/L
        let min_pnl = positions_to_close
            .iter()
            .map(|p| p.unrealised_pnl)
            .fold(f64::INFINITY, f64::min);

        // Close all positions with the minimum P/L
        let positions_to_close: Vec<Position> = self
            .positions
            .iter()
            .filter(|p| p.unrealised_pnl == min_pnl)
            .cloned()
            .collect();

        for position in positions_to_close {
            self.close_position(&position)?;
        }

        Ok(())
    }

    /// Close the position with the largest unrealised profit
    fn close_most_profit(&mut self) -> Result<()> {
        if self.positions.is_empty() {
            return Ok(());
        }

        // Find the position with the largest unrealised profit
        let positions_to_close: Vec<Position> = self
            .positions
            .iter()
            .filter(|p| p.unrealised_pnl > 0.0)
            .cloned()
            .collect();

        if positions_to_close.is_empty() {
            return Ok(()); // No positions at a profit
        }

        // Find the maximum unrealised P/L
        let max_pnl = positions_to_close
            .iter()
            .map(|p| p.unrealised_pnl)
            .fold(f64::NEG_INFINITY, f64::max);

        // Close all positions with the maximum P/L
        let positions_to_close: Vec<Position> = self
            .positions
            .iter()
            .filter(|p| p.unrealised_pnl == max_pnl)
            .cloned()
            .collect();

        for position in positions_to_close {
            self.close_position(&position)?;
        }

        Ok(())
    }

    /// Close all positions at a loss
    fn close_all_loss(&mut self) -> Result<()> {
        let positions_to_close: Vec<Position> = self
            .positions
            .iter()
            .filter(|p| p.unrealised_pnl < 0.0)
            .cloned()
            .collect();

        for position in positions_to_close {
            self.close_position(&position)?;
        }

        Ok(())
    }

    /// Close all positions that are profitable
    fn close_all_profit(&mut self) -> Result<()> {
        let positions_to_close: Vec<Position> = self
            .positions
            .iter()
            .filter(|p| p.unrealised_pnl > 0.0)
            .cloned()
            .collect();

        for position in positions_to_close {
            self.close_position(&position)?;
        }

        Ok(())
    }

    /// Close a specific position
    fn close_position(&mut self, position: &Position) -> Result<()> {
        let current_timestamp = self.current_timestamp();
        let close_price = self.get_current_mid_price()?;

        // Calculate realised P/L
        let _realised_pnl = position.calculate_realised_pnl(close_price, self.transaction_cost);

        // Create closed position record
        let closed_position =
            position.to_closed_position(close_price, current_timestamp, self.transaction_cost);

        // Add to closed position window
        self.closed_position_window
            .add_closed_position(closed_position);

        // Remove from open positions
        self.positions
            .retain(|p| p.position_id != position.position_id);

        // Record the fill
        self.recent_fills.push(Fill {
            order_id: format!("fill_{}", current_timestamp),
            timestamp_ns: current_timestamp,
            price: close_price,
            size: position.volume,
            side: match position.side {
                Side::Buy => ActionType::ActionCloseMostLoss, // Placeholder for closing
                Side::Sell => ActionType::ActionCloseMostProfit, // Placeholder for closing
            },
            partial: false,
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::{ClosedPosition, Position, NANOS_PER_DAY};
    use arrow::array::{Float64Array, Int64Array};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::path::PathBuf;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tempfile::tempdir;

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
                side: ActionType::ActionOpenBuy as i32,
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
        assert!(response.observation.is_some());
        assert!(environment.recent_fills.is_empty());
        assert_eq!(response.observation.as_ref().unwrap().live_ticks.len(), 2);
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
        let base = dir.path().join(
            "marketdata/eoh-snapshot/symbol=USDJPY/interval=M1/year=2012/month=01/day=02/hour=06",
        );
        std::fs::create_dir_all(&base).unwrap();

        let parquet_path = base.join("20120102T060000Z.parquet");
        write_test_parquet(
            &parquet_path,
            &[1_325_484_000_000_000_000, 1_325_484_060_000_000_000],
            &[102.0, 103.0],
        )
        .unwrap();

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

        assert_eq!(response.reward, 0.0);
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

        assert!(response.done);
        assert!(response.observation.unwrap().done);
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
        assert_eq!(
            first.observation.as_ref().unwrap().timestamp_ns,
            60_000_000_000
        );

        let second = environment
            .step(Action {
                action: ActionType::ActionOpenBuy as i32,
                client_order_id: "buy-1".to_string(),
            })
            .await
            .unwrap();

        let observation = second.observation.unwrap();
        assert_eq!(observation.timestamp_ns, 120_000_000_000);
        assert_eq!(observation.live_bars["M5"].timestamp_ns, 0);
        assert_eq!(observation.positions.len(), 1);
        assert_eq!(observation.positions[0].open_timestamp_ns, 120_000_000_000);
        assert_eq!(observation.live_ticks.len(), 1);
        assert_eq!(observation.recent_ticks.len(), 1);
    }

    #[tokio::test]
    async fn test_live_observation_indicators_length_matches_intervals() {
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
        let observation = response.observation.unwrap();
        assert_eq!(
            observation.indicators.len(),
            TIME_INTERVALS.len() * INDICATORS_PER_INTERVAL
        );
        assert!(observation.indicators.iter().all(|v| *v == 0.0));
    }

    #[tokio::test]
    async fn test_training_observation_indicators_length_matches_intervals() {
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
        let observation = response.observation.unwrap();
        assert_eq!(
            observation.indicators.len(),
            TIME_INTERVALS.len() * INDICATORS_PER_INTERVAL
        );
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
    async fn test_live_observation_populates_recent_bars_and_reverses_live_ticks() {
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
        let observation = response.observation.unwrap();

        assert!(observation.recent_bars.contains_key("M1"));
        assert_eq!(observation.recent_bars["M1"].bars.len(), 0);
        assert_eq!(observation.live_bars["M1"].close, 155.21);

        assert_eq!(observation.live_ticks.len(), 2);
        assert_eq!(observation.live_ticks[0].timestamp_ns, 2);
        assert_eq!(observation.live_ticks[1].timestamp_ns, 1);

        assert_eq!(observation.recent_ticks.len(), 0);
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
                side: ActionType::ActionOpenBuy,
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
        assert_eq!(first.reward, 0.0);

        let second = environment
            .step(Action {
                action: ActionType::ActionOpenBuy as i32,
                client_order_id: "buy-1".to_string(),
            })
            .await
            .unwrap();

        assert!(second.reward < -0.9);
        assert_eq!(environment.last_action, Some(ActionType::ActionOpenBuy));
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

    #[tokio::test]
    async fn test_training_preload_warms_first_reset() {
        let dir = tempdir().unwrap();
        let base = dir.path().join(
            "marketdata/eoh-snapshot/symbol=USDJPY/interval=M1/year=2012/month=01/day=02/hour=06",
        );
        std::fs::create_dir_all(&base).unwrap();

        let parquet_path = base.join("20120102T060000Z.parquet");
        write_test_parquet(
            &parquet_path,
            &[1_325_484_000_000_000_000, 1_325_484_060_000_000_000],
            &[102.0, 103.0],
        )
        .unwrap();

        let mut environment = Environment::new(
            Mode::Training,
            "USDJPY".to_string(),
            dir.path().to_string_lossy().to_string(),
        );

        environment.preload_training_data().await.unwrap();
        std::fs::remove_file(&parquet_path).unwrap();

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

        assert_eq!(observation.live_bars["M1"].open, 102.0);
    }
}
