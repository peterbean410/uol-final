use super::{sim::SimEngine, Environment, StepOutcome};
use crate::data::ParquetBarLoader;
use crate::proto;
use anyhow::{anyhow, Result};
use async_trait::async_trait;

/// Training backend. Replays historical bars loaded from parquet on S3 and uses
/// the in-memory `SimEngine` to bookkeep fills/PnL. One bar = one step.
pub struct TrainingBackend {
    pub symbol: String,
    pub loader: ParquetBarLoader,
    pub bars: Vec<proto::Bar>,
    pub cursor: usize,
    pub sim: SimEngine,
    pub lookback: usize,
}

impl TrainingBackend {
    pub async fn new(s3_prefix: String, symbol: String) -> Result<Self> {
        let loader = ParquetBarLoader::from_s3_prefix(&s3_prefix)?;
        Ok(Self {
            symbol: symbol.clone(),
            loader,
            bars: Vec::new(),
            cursor: 0,
            sim: SimEngine::new(symbol, 0.01, 0.0),
            lookback: 32,
        })
    }

    fn current_bar(&self) -> Option<&proto::Bar> {
        self.bars.get(self.cursor)
    }

    fn build_observation(&self, done: bool) -> proto::Observation {
        let current = self.current_bar().cloned();
        let start = self.cursor.saturating_sub(self.lookback);
        let recent = self.bars.get(start..self.cursor).unwrap_or(&[]).to_vec();
        proto::Observation {
            timestamp_ns: current.as_ref().map(|b| b.timestamp_ns).unwrap_or(0),
            symbol: self.symbol.clone(),
            current_bar: current,
            recent_bars: recent,
            position: Some(self.sim.position.clone()),
            recent_fills: self.sim.recent_fills.clone(),
            indicators: Vec::new(),
            done,
        }
    }
}

#[async_trait]
impl Environment for TrainingBackend {
    async fn reset(&mut self, req: proto::ResetRequest) -> Result<proto::Observation> {
        let symbol = if req.symbol.is_empty() { self.symbol.clone() } else { req.symbol };
        self.symbol = symbol.clone();
        self.sim = SimEngine::new(symbol.clone(), 0.01, 0.0);

        // Convention: episode parquet lives at `<symbol>/M1/<episode_start_ts>.parquet`.
        // Callers that want a different layout can extend ResetRequest.
        let rel = if req.episode_start_ts.is_empty() {
            format!("{}/M1/latest.parquet", symbol)
        } else {
            format!("{}/M1/{}.parquet", symbol, req.episode_start_ts)
        };
        self.bars = self.loader.load_bars(&rel).await?;
        if self.bars.is_empty() {
            return Err(anyhow!("no bars loaded for {rel}"));
        }
        self.cursor = 0;
        Ok(self.build_observation(false))
    }

    async fn step(&mut self, action: proto::Action) -> Result<StepOutcome> {
        if self.bars.is_empty() {
            return Err(anyhow!("reset() must be called before step()"));
        }
        let bar = self.bars[self.cursor].clone();
        let action_type = proto::ActionType::try_from(action.action)
            .unwrap_or(proto::ActionType::ActionHold);

        let (realised_delta, _fill) = self.sim.apply(
            action_type,
            action.size,
            bar.close,
            bar.timestamp_ns,
            action.client_order_id.clone(),
        );

        let prev_unrealised = self.sim.position.unrealised_pnl;
        self.cursor += 1;
        let done = self.cursor >= self.bars.len();
        let next_mid = if done { bar.close } else { self.bars[self.cursor].close };
        self.sim.mark_to_market(next_mid);
        let unrealised_delta = self.sim.position.unrealised_pnl - prev_unrealised;

        let reward = realised_delta + unrealised_delta;
        let obs = self.build_observation(done);
        Ok(StepOutcome { observation: obs, reward, done, info: String::new() })
    }

    async fn observe(&self) -> Result<proto::Observation> {
        Ok(self.build_observation(self.cursor >= self.bars.len()))
    }
}
