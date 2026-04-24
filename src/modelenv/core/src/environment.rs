// Environment module
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

use log::info;
use modelenv_proto::{Action, ObserveRequest, Observation, ResetRequest, StepResponse, ActionType};

use crate::broker_gateway::BrokerGateway;
use crate::config::Mode;
use crate::data_loader::now_ns;
use crate::episode::{initialize_episode, Episode};
use crate::position::{Position, ClosedPositionWindow, Side};

/// The main environment struct
#[derive(Clone)]
pub struct Environment {
    mode: Mode,
    symbol: String,
    s3_prefix: String,
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
            episode: None,
            positions: Vec::new(),
            closed_position_window: ClosedPositionWindow::new(),
            recent_fills: Vec::new(),
            last_action: None,
            transaction_cost: 0.0, // Default no transaction cost
            daily_swap_rates: HashMap::new(),
            last_swap_accrual_timestamp: 0, // Will be set on reset
            broker_gateway: None,
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

    /// Set the broker gateway for Production Mode
    pub fn with_broker_gateway(mut self, broker_gateway: Arc<dyn BrokerGateway + Send + Sync>) -> Self {
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

    /// Reset the environment and initialize a new episode
    pub async fn reset(&mut self, req: ResetRequest) -> Result<Observation> {
        // Clear positions and state
        self.positions.clear();
        self.closed_position_window = ClosedPositionWindow::new();
        self.recent_fills.clear();
        self.last_action = None;
        self.last_swap_accrual_timestamp = 0; // Will be set when we get the first timestamp

        match self.mode {
            Mode::Training => {
                // Validate episode timestamps
                if req.episode_end_ts > 0 && req.episode_start_ts > req.episode_end_ts {
                    return Err(anyhow::anyhow!(
                        "episode_start_ts ({}) must be <= episode_end_ts ({})",
                        req.episode_start_ts,
                        req.episode_end_ts
                    ));
                }

                // Initialize episode with S3 parquet loading
                self.episode = Some(
                    initialize_episode(
                        &req.symbol,
                        &self.s3_prefix,
                        req.episode_start_ts,
                        req.episode_end_ts,
                    )
                    .await?,
                );

                // Get initial observation
                self.observe(ObserveRequest {
                    symbol: req.symbol,
                })
                .await
            }
            Mode::Live => {
                // Production mode - sync with broker
                // First, get the broker gateway reference
                let broker = self.get_broker_gateway()?;
                
                // Sync positions with broker
                let broker_positions = broker.sync_positions(&req.symbol).await?;
                
                // Clear all existing internal positions, unrealised P/L, and accumulated swap
                self.positions.clear();
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
                
                // Get current bar from broker
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
                self.observe(ObserveRequest {
                    symbol: req.symbol,
                })
                .await
            }
        }
    }

    /// Take a step in the environment
    pub async fn step(&mut self, action: Action) -> Result<StepResponse> {
        // Get current timestamp before mutable borrow
        let current_timestamp = self.current_timestamp();
        
        match self.mode {
            Mode::Training => {
                // Training mode - use episode
                let episode = self
                    .episode
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("Episode not initialized. Call reset() first."))?;

                // Get the timestamp at the current cursor before advancing
                let prev_timestamp = episode.get_cursor_timestamp();
                
                // Advance the episode first to release the mutable borrow
                let done = episode.advance(5_000_000_000); // 5 seconds in nanoseconds

                // Calculate realised P/L before getting observation
                let realised_pnl_12m = self.closed_position_window.total_realised_pnl_12m(current_timestamp);

                // Convert positions to proto format
                let proto_positions: Vec<modelenv_proto::Position> = self.positions.iter()
                    .map(|p| p.to_proto())
                    .collect();

                // Get observation
                let observation = episode.get_observation(proto_positions.as_slice(), realised_pnl_12m);

                // Accrue swap if day boundary was crossed during advancement
                // This must be done before apply_action to avoid borrow conflicts
                if !self.positions.is_empty() {
                    if episode.has_day_boundary_crossed(prev_timestamp, current_timestamp) {
                        self.accrue_swap_on_positions()?;
                    }
                }

                // Apply the action
                self.apply_action(&action)?;

                // Calculate reward based on action (after releasing episode borrow)
                let reward = self.calculate_reward(&action)?;

                Ok(StepResponse {
                    observation: Some(observation),
                    reward,
                    done,
                    info: "".to_string(),
                })
            }
            Mode::Live => {
                // Production mode - submit action to broker
                let broker = self.get_broker_gateway()?;
                
                // Submit action to broker
                let fill = broker.submit(&action).await?;
                
                // Record the fill
                self.recent_fills.push(Fill {
                    order_id: fill.order_id,
                    timestamp_ns: fill.timestamp_ns,
                    price: fill.price,
                    size: fill.size,
                    side: match ActionType::try_from(fill.side) {
                        Ok(action) => action,
                        Err(_) => ActionType::ActionHold,
                    },
                    partial: fill.partial,
                });
                
                // TODO: Update positions based on broker response
                // For now, return an error as we need to implement position updates from broker
                Err(anyhow::anyhow!("Production mode step not fully implemented"))
            }
        }
    }

    /// Get current observation without advancing
    pub async fn observe(&self, _req: ObserveRequest) -> Result<Observation> {
        let episode = self
            .episode
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Episode not initialized. Call reset() first."))?;

        // Convert positions to proto format
        let proto_positions: Vec<modelenv_proto::Position> = self.positions.iter()
            .map(|p| p.to_proto())
            .collect();

        Ok(episode.get_observation(proto_positions.as_slice(), self.realised_pnl_12m()))
    }

    /// Calculate the rolling 12-month realised P/L
    fn realised_pnl_12m(&self) -> f64 {
        let current_timestamp = self.current_timestamp();
        self.closed_position_window.total_realised_pnl_12m(current_timestamp)
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
    fn calculate_reward(&self, _action: &Action) -> Result<f64> {
        // Placeholder reward calculation
        // TODO: Implement proper reward calculation
        Ok(0.0)
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
        
        // Get the current bar for M1 interval (or any available)
        if let Some(bars) = episode.bars.get("M1") {
            if let Some(bar) = bars.get(episode.cursor) {
                // Mid price = (open + close) / 2 for simplicity
                // Could also use (high + low) / 2 or just close
                return Ok((bar.open + bar.close) / 2.0);
            }
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
        let positions_to_close: Vec<Position> = self.positions.iter()
            .filter(|p| p.unrealised_pnl < 0.0)
            .cloned()
            .collect();
        
        if positions_to_close.is_empty() {
            return Ok(()); // No positions at a loss
        }

        // Find the minimum unrealised P/L
        let min_pnl = positions_to_close.iter()
            .map(|p| p.unrealised_pnl)
            .fold(f64::INFINITY, f64::min);
        
        // Close all positions with the minimum P/L
        let positions_to_close: Vec<Position> = self.positions.iter()
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
        let positions_to_close: Vec<Position> = self.positions.iter()
            .filter(|p| p.unrealised_pnl > 0.0)
            .cloned()
            .collect();
        
        if positions_to_close.is_empty() {
            return Ok(()); // No positions at a profit
        }

        // Find the maximum unrealised P/L
        let max_pnl = positions_to_close.iter()
            .map(|p| p.unrealised_pnl)
            .fold(f64::NEG_INFINITY, f64::max);
        
        // Close all positions with the maximum P/L
        let positions_to_close: Vec<Position> = self.positions.iter()
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
        let positions_to_close: Vec<Position> = self.positions.iter()
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
        let positions_to_close: Vec<Position> = self.positions.iter()
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
        let closed_position = position.to_closed_position(
            close_price,
            current_timestamp,
            self.transaction_cost,
        );
        
        // Add to closed position window
        self.closed_position_window.add_closed_position(closed_position);
        
        // Remove from open positions
        self.positions.retain(|p| p.position_id != position.position_id);
        
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
    use crate::position::{Position, ClosedPosition, NANOS_PER_DAY};

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
}
