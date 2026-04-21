use super::{Environment, StepOutcome};
use crate::proto;
use anyhow::{anyhow, Result};
use async_trait::async_trait;

/// Live broker adapter. Production implementations wire this up to cTrader /
/// FIX / broker SDKs. Actions in live mode have real financial consequences;
/// every branch here must either forward to the broker or refuse with an error.
#[async_trait]
pub trait BrokerGateway: Send + Sync {
    async fn sync_positions(&self, symbol: &str) -> Result<Vec<proto::Position>>;
    async fn current_bar(&self, symbol: &str) -> Result<proto::Bar>;
    async fn submit(&self, symbol: &str, action: &proto::Action) -> Result<proto::Fill>;
}

pub struct LiveBackend {
    pub symbol: String,
    pub gateway: Option<Box<dyn BrokerGateway>>,
    pub positions: Vec<proto::Position>,
    pub recent_fills: Vec<proto::Fill>,
    pub last_bar: Option<proto::Bar>,
    pub realised_pnl_12m: f64,
}

impl LiveBackend {
    pub fn new(symbol: String) -> Result<Self> {
        Ok(Self {
            symbol,
            gateway: None,
            positions: Vec::new(),
            recent_fills: Vec::new(),
            last_bar: None,
            realised_pnl_12m: 0.0,
        })
    }

    pub fn with_gateway(mut self, gw: Box<dyn BrokerGateway>) -> Self {
        self.gateway = Some(gw);
        self
    }

    fn require_gateway(&self) -> Result<&dyn BrokerGateway> {
        self.gateway
            .as_deref()
            .ok_or_else(|| anyhow!("live mode: broker gateway not configured, refusing to act"))
    }

    fn build_observation(&self) -> proto::Observation {
        proto::Observation {
            timestamp_ns: self.last_bar.as_ref().map(|b| b.timestamp_ns).unwrap_or(0),
            symbol: self.symbol.clone(),
            current_bar: self.last_bar.clone(),
            recent_bars: Vec::new(),
            positions: self.positions.clone(),
            realised_pnl_12m: self.realised_pnl_12m,
            recent_fills: self.recent_fills.clone(),
            indicators: Vec::new(),
            done: false,
        }
    }

    fn validate(&self, action: &proto::Action) -> Result<()> {
        let _ = proto::ActionType::try_from(action.action)
            .map_err(|_| anyhow!("unknown action type {}", action.action))?;
        Ok(())
    }

    /// Compare internal state against broker-reported positions and log discrepancies.
    fn reconcile(&self, broker_positions: &[proto::Position]) {
        // Position count mismatch.
        if self.positions.len() != broker_positions.len() {
            tracing::warn!(
                internal = self.positions.len(),
                broker = broker_positions.len(),
                "reconciliation: position count mismatch"
            );
        }

        // Per-position checks (match by position_id).
        for internal in &self.positions {
            if let Some(broker) = broker_positions.iter().find(|b| b.position_id == internal.position_id) {
                let price_diff = (internal.entry_price - broker.entry_price).abs();
                if price_diff > 1e-6 {
                    tracing::warn!(
                        position_id = %internal.position_id,
                        internal_price = internal.entry_price,
                        broker_price = broker.entry_price,
                        "reconciliation: entry price drift"
                    );
                }

                let pnl_diff = (internal.unrealised_pnl - broker.unrealised_pnl).abs();
                if pnl_diff > 0.01 {
                    tracing::warn!(
                        position_id = %internal.position_id,
                        internal_pnl = internal.unrealised_pnl,
                        broker_pnl = broker.unrealised_pnl,
                        "reconciliation: unrealised P/L divergence"
                    );
                }

                let swap_diff = (internal.swap - broker.swap).abs();
                if swap_diff > 0.01 {
                    tracing::warn!(
                        position_id = %internal.position_id,
                        internal_swap = internal.swap,
                        broker_swap = broker.swap,
                        "reconciliation: swap cost divergence"
                    );
                }
            } else {
                tracing::warn!(
                    position_id = %internal.position_id,
                    "reconciliation: internal position not found on broker"
                );
            }
        }

        // Check for broker positions we don't know about.
        for broker in broker_positions {
            if !self.positions.iter().any(|p| p.position_id == broker.position_id) {
                tracing::warn!(
                    position_id = %broker.position_id,
                    "reconciliation: unexpected position on broker"
                );
            }
        }
    }
}

#[async_trait]
impl Environment for LiveBackend {
    async fn reset(&mut self, req: proto::ResetRequest) -> Result<proto::Observation> {
        let symbol = if req.symbol.is_empty() { self.symbol.clone() } else { req.symbol };
        self.symbol = symbol.clone();
        let (positions, bar) = {
            let gw = self.require_gateway()?;
            let positions = gw.sync_positions(&self.symbol).await?;
            let bar = gw.current_bar(&self.symbol).await?;
            (positions, bar)
        };
        self.positions = positions;
        self.last_bar = Some(bar);
        self.recent_fills.clear();
        Ok(self.build_observation())
    }

    async fn step(&mut self, action: proto::Action) -> Result<StepOutcome> {
        self.validate(&action)?;
        let (fill, broker_positions, bar) = {
            let gw = self.require_gateway()?;
            let fill = gw.submit(&self.symbol, &action).await?;
            let positions = gw.sync_positions(&self.symbol).await?;
            let bar = gw.current_bar(&self.symbol).await?;
            (fill, positions, bar)
        };

        self.recent_fills.push(fill);
        if self.recent_fills.len() > 64 {
            let overflow = self.recent_fills.len() - 64;
            self.recent_fills.drain(0..overflow);
        }

        // Reconcile before updating internal state.
        self.reconcile(&broker_positions);

        // Broker is source of truth, adopt its state.
        self.positions = broker_positions;
        self.last_bar = Some(bar);

        let obs = self.build_observation();
        Ok(StepOutcome { observation: obs, reward: 0.0, done: false, info: String::from("live") })
    }

    async fn observe(&self) -> Result<proto::Observation> {
        Ok(self.build_observation())
    }
}
