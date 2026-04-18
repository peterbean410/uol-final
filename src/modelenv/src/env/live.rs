use super::{sim::SimEngine, Environment, StepOutcome};
use crate::proto;
use anyhow::{anyhow, Result};
use async_trait::async_trait;

/// Live broker adapter. Production implementations wire this up to cTrader /
/// FIX / broker SDKs. The skeleton validates actions and keeps the same
/// observation shape as the training backend so the model cannot tell them
/// apart. Actions in live mode have real financial consequences; every
/// branch here must either forward to the broker or refuse with an error.
#[async_trait]
pub trait BrokerGateway: Send + Sync {
    async fn sync_state(&self, symbol: &str) -> Result<proto::Position>;
    async fn current_bar(&self, symbol: &str) -> Result<proto::Bar>;
    async fn submit(&self, symbol: &str, action: &proto::Action) -> Result<proto::Fill>;
}

pub struct LiveBackend {
    pub symbol: String,
    pub gateway: Option<Box<dyn BrokerGateway>>,
    pub sim: SimEngine,
    pub last_bar: Option<proto::Bar>,
}

impl LiveBackend {
    pub fn new(symbol: String) -> Result<Self> {
        Ok(Self {
            symbol: symbol.clone(),
            gateway: None,
            sim: SimEngine::new(symbol, 0.01, 0.0),
            last_bar: None,
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
            position: Some(self.sim.position.clone()),
            recent_fills: self.sim.recent_fills.clone(),
            indicators: Vec::new(),
            done: false,
        }
    }

    fn validate(&self, action: &proto::Action) -> Result<()> {
        if !action.size.is_finite() || action.size < 0.0 {
            return Err(anyhow!("invalid action size: {}", action.size));
        }
        let _ = proto::ActionType::try_from(action.action)
            .map_err(|_| anyhow!("unknown action type {}", action.action))?;
        Ok(())
    }
}

#[async_trait]
impl Environment for LiveBackend {
    async fn reset(&mut self, req: proto::ResetRequest) -> Result<proto::Observation> {
        let symbol = if req.symbol.is_empty() { self.symbol.clone() } else { req.symbol };
        self.symbol = symbol.clone();
        let (position, bar) = {
            let gw = self.require_gateway()?;
            let position = gw.sync_state(&self.symbol).await?;
            let bar = gw.current_bar(&self.symbol).await?;
            (position, bar)
        };
        self.sim.position = position;
        self.last_bar = Some(bar);
        Ok(self.build_observation())
    }

    async fn step(&mut self, action: proto::Action) -> Result<StepOutcome> {
        self.validate(&action)?;
        let (fill, position, bar) = {
            let gw = self.require_gateway()?;
            let fill = gw.submit(&self.symbol, &action).await?;
            // Resync position from broker rather than inferring it; the broker
            // is the source of truth once the order leaves the gateway.
            let position = gw.sync_state(&self.symbol).await?;
            let bar = gw.current_bar(&self.symbol).await?;
            (fill, position, bar)
        };
        self.sim.recent_fills.push(fill);
        self.sim.position = position;
        self.last_bar = Some(bar);
        let obs = self.build_observation();
        Ok(StepOutcome { observation: obs, reward: 0.0, done: false, info: String::from("live") })
    }

    async fn observe(&self) -> Result<proto::Observation> {
        Ok(self.build_observation())
    }
}
