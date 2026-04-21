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

/// Nanoseconds in one day, used to detect rollover boundaries for swap accrual.
const NS_PER_DAY: i64 = 86_400_000_000_000;

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
            positions: self.sim.positions.clone(),
            realised_pnl_12m: self.sim.realised_pnl_12m(),
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

        // Compute total unrealised P/L before the action.
        let prev_total_unrealised: f64 = self.sim.positions.iter().map(|p| p.unrealised_pnl).sum();

        let realised_delta = self.sim.apply(
            action_type,
            bar.close,
            bar.timestamp_ns,
            action.client_order_id.clone(),
        );

        // Advance cursor.
        let prev_ts = bar.timestamp_ns;
        self.cursor += 1;
        let done = self.cursor >= self.bars.len();
        let next_mid = if done { bar.close } else { self.bars[self.cursor].close };

        // Accrue swap if we crossed a day boundary.
        if !done {
            let next_ts = self.bars[self.cursor].timestamp_ns;
            let days_elapsed = (next_ts / NS_PER_DAY) - (prev_ts / NS_PER_DAY);
            if days_elapsed > 0 {
                self.sim.accrue_swap(days_elapsed as f64);
            }
        }

        self.sim.mark_to_market(next_mid);

        let new_total_unrealised: f64 = self.sim.positions.iter().map(|p| p.unrealised_pnl).sum();
        let unrealised_delta = new_total_unrealised - prev_total_unrealised;

        let reward = realised_delta + unrealised_delta;
        let obs = self.build_observation(done);
        Ok(StepOutcome { observation: obs, reward, done, info: String::new() })
    }

    async fn observe(&self) -> Result<proto::Observation> {
        Ok(self.build_observation(self.cursor >= self.bars.len()))
    }
}
