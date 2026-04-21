use crate::proto;
use super::round_price;
use std::collections::VecDeque;

/// A closed position record for rolling realised P/L tracking.
#[derive(Clone, Debug)]
struct ClosedPosition {
    pnl: f64,
    swap: f64,
    closed_timestamp_ns: i64,
}

/// In-memory simulator tracking multiple buy-only positions, fills, swap costs,
/// and rolling 12-month realised P/L for a single symbol.
pub struct SimEngine {
    pub symbol: String,
    pub positions: Vec<proto::Position>,
    pub recent_fills: Vec<proto::Fill>,
    /// Rolling window of closed position P/L (kept for 12 months).
    closed_positions: VecDeque<ClosedPosition>,
    /// Spread in price units (not pips). For USD/JPY 0.01 == 1 pip.
    pub spread: f64,
    /// Per-unit transaction cost applied on every fill, in quote currency.
    pub cost_per_unit: f64,
    /// Swap rate per day for long positions (in price units).
    pub swap_long_per_day: f64,
    pub order_seq: u64,
    /// Current environment timestamp (nanoseconds) for swap calculation.
    pub current_ts_ns: i64,
    /// Fixed volume per position.
    pub default_volume: f64,
}

/// Nanoseconds in one day.
const NS_PER_DAY: i64 = 86_400_000_000_000;

/// 12 months in nanoseconds (approximate: 365 days).
const TWELVE_MONTHS_NS: i64 = 365 * NS_PER_DAY;

impl SimEngine {
    pub fn new(symbol: String, spread: f64, cost_per_unit: f64) -> Self {
        Self {
            symbol,
            positions: Vec::new(),
            recent_fills: Vec::new(),
            closed_positions: VecDeque::new(),
            spread,
            cost_per_unit,
            swap_long_per_day: 0.0,
            order_seq: 0,
            current_ts_ns: 0,
            default_volume: 1.0,
        }
    }

    pub fn reset(&mut self) {
        self.positions.clear();
        self.recent_fills.clear();
        self.closed_positions.clear();
        self.order_seq = 0;
        self.current_ts_ns = 0;
    }

    /// Update unrealised P/L on all open positions against the current mid price.
    pub fn mark_to_market(&mut self, mid_price: f64) {
        for pos in &mut self.positions {
            pos.unrealised_pnl = (mid_price - pos.entry_price) * self.default_volume;
        }
    }

    /// Accrue daily swap on all open positions. Call once per simulated day
    /// (when the bar crosses a rollover boundary).
    pub fn accrue_swap(&mut self, days: f64) {
        for pos in &mut self.positions {
            pos.swap += self.swap_long_per_day * days * self.default_volume;
        }
    }

    /// Realised P/L over the trailing 12 months, inclusive of swap costs.
    pub fn realised_pnl_12m(&self) -> f64 {
        let cutoff = self.current_ts_ns - TWELVE_MONTHS_NS;
        self.closed_positions
            .iter()
            .filter(|c| c.closed_timestamp_ns >= cutoff)
            .map(|c| c.pnl + c.swap)
            .sum()
    }

    /// Prune closed positions older than 12 months.
    fn prune_closed(&mut self) {
        let cutoff = self.current_ts_ns - TWELVE_MONTHS_NS;
        while self.closed_positions.front().is_some_and(|c| c.closed_timestamp_ns < cutoff) {
            self.closed_positions.pop_front();
        }
    }

    /// Apply a discrete action at the given mid price and timestamp.
    /// Returns the realised P/L delta from this step.
    pub fn apply(
        &mut self,
        action: proto::ActionType,
        mid_price: f64,
        ts_ns: i64,
        client_order_id: String,
    ) -> f64 {
        self.current_ts_ns = ts_ns;
        let half_spread = self.spread / 2.0;

        match action {
            proto::ActionType::ActionHold => 0.0,

            proto::ActionType::ActionOpenBuy => {
                let fill_price = round_price(&self.symbol, mid_price + half_spread);
                self.order_seq += 1;
                let pos_id = format!("sim-pos-{}", self.order_seq);

                self.positions.push(proto::Position {
                    position_id: pos_id.clone(),
                    entry_price: fill_price,
                    unrealised_pnl: 0.0,
                    swap: 0.0,
                    open_timestamp_ns: ts_ns,
                });

                let cost = self.default_volume * self.cost_per_unit;
                self.push_fill(fill_price, ts_ns, proto::ActionType::ActionOpenBuy, client_order_id);
                -cost
            }

            proto::ActionType::ActionCloseMostLoss => {
                if self.positions.is_empty() {
                    return 0.0;
                }
                let idx = self.positions
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.unrealised_pnl.partial_cmp(&b.unrealised_pnl).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                self.close_position(idx, mid_price - half_spread, ts_ns, client_order_id)
            }

            proto::ActionType::ActionCloseMostProfit => {
                if self.positions.is_empty() {
                    return 0.0;
                }
                let idx = self.positions
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.unrealised_pnl.partial_cmp(&b.unrealised_pnl).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                self.close_position(idx, mid_price - half_spread, ts_ns, client_order_id)
            }

            proto::ActionType::ActionCloseAllLoss => {
                let indices: Vec<usize> = self.positions
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.unrealised_pnl < 0.0)
                    .map(|(i, _)| i)
                    .collect();
                self.close_positions_by_indices(indices, mid_price - half_spread, ts_ns, client_order_id)
            }

            proto::ActionType::ActionCloseAllProfit => {
                let indices: Vec<usize> = self.positions
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.unrealised_pnl > 0.0)
                    .map(|(i, _)| i)
                    .collect();
                self.close_positions_by_indices(indices, mid_price - half_spread, ts_ns, client_order_id)
            }
        }
    }

    fn close_position(
        &mut self,
        idx: usize,
        fill_price: f64,
        ts_ns: i64,
        client_order_id: String,
    ) -> f64 {
        let pos = self.positions.remove(idx);
        let fill_price = round_price(&self.symbol, fill_price);
        let raw_pnl = (fill_price - pos.entry_price) * self.default_volume;
        let cost = self.default_volume * self.cost_per_unit;
        let realised = raw_pnl - cost;

        self.closed_positions.push_back(ClosedPosition {
            pnl: realised,
            swap: pos.swap,
            closed_timestamp_ns: ts_ns,
        });
        self.prune_closed();

        self.push_fill(fill_price, ts_ns, proto::ActionType::ActionCloseMostLoss, client_order_id);
        realised
    }

    fn close_positions_by_indices(
        &mut self,
        mut indices: Vec<usize>,
        fill_price: f64,
        ts_ns: i64,
        client_order_id: String,
    ) -> f64 {
        // Remove from highest index first to avoid shifting.
        indices.sort_unstable_by(|a, b| b.cmp(a));
        let mut total = 0.0;
        for (i, idx) in indices.into_iter().enumerate() {
            let oid = if i == 0 {
                client_order_id.clone()
            } else {
                format!("{}-{}", client_order_id, i)
            };
            total += self.close_position(idx, fill_price, ts_ns, oid);
        }
        total
    }

    fn push_fill(
        &mut self,
        price: f64,
        ts_ns: i64,
        side: proto::ActionType,
        client_order_id: String,
    ) {
        self.order_seq += 1;
        let order_id = if client_order_id.is_empty() {
            format!("sim-{}", self.order_seq)
        } else {
            client_order_id
        };
        let fill = proto::Fill {
            order_id,
            timestamp_ns: ts_ns,
            price,
            size: self.default_volume,
            side: side as i32,
            partial: false,
        };
        self.recent_fills.push(fill);
        if self.recent_fills.len() > 64 {
            let overflow = self.recent_fills.len() - 64;
            self.recent_fills.drain(0..overflow);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> SimEngine {
        SimEngine::new("USDJPY".to_string(), 0.01, 0.0)
    }

    #[test]
    fn open_buy_creates_position() {
        let mut e = engine();
        e.apply(proto::ActionType::ActionOpenBuy, 150.000, 1000, String::new());
        assert_eq!(e.positions.len(), 1);
        // buy at mid + half_spread = 150.005
        assert!((e.positions[0].entry_price - 150.005).abs() < 1e-6);
    }

    #[test]
    fn close_most_loss_removes_worst() {
        let mut e = engine();
        e.apply(proto::ActionType::ActionOpenBuy, 150.000, 1000, "a".into());
        e.apply(proto::ActionType::ActionOpenBuy, 151.000, 2000, "b".into());
        // Mark to market at 150.500, first position up, second down
        e.mark_to_market(150.500);
        assert!(e.positions[0].unrealised_pnl > 0.0);
        assert!(e.positions[1].unrealised_pnl < 0.0);

        e.apply(proto::ActionType::ActionCloseMostLoss, 150.500, 3000, "c".into());
        assert_eq!(e.positions.len(), 1);
        // The remaining position should be the profitable one (entered at 150.005)
        assert!((e.positions[0].entry_price - 150.005).abs() < 1e-6);
    }

    #[test]
    fn close_all_loss_removes_all_losing() {
        let mut e = engine();
        e.apply(proto::ActionType::ActionOpenBuy, 150.000, 1000, "a".into());
        e.apply(proto::ActionType::ActionOpenBuy, 151.000, 2000, "b".into());
        e.apply(proto::ActionType::ActionOpenBuy, 152.000, 3000, "c".into());
        e.mark_to_market(150.800);
        // First position profitable, second and third at a loss
        e.apply(proto::ActionType::ActionCloseAllLoss, 150.800, 4000, "d".into());
        assert_eq!(e.positions.len(), 1);
        assert!((e.positions[0].entry_price - 150.005).abs() < 1e-6);
    }

    #[test]
    fn hold_does_nothing() {
        let mut e = engine();
        e.apply(proto::ActionType::ActionOpenBuy, 150.000, 1000, "a".into());
        let delta = e.apply(proto::ActionType::ActionHold, 150.500, 2000, String::new());
        assert_eq!(delta, 0.0);
        assert_eq!(e.positions.len(), 1);
    }

    #[test]
    fn swap_accrues_on_open_positions() {
        let mut e = engine();
        e.swap_long_per_day = -0.05; // negative swap (cost)
        e.apply(proto::ActionType::ActionOpenBuy, 150.000, 1000, "a".into());
        e.accrue_swap(1.0);
        assert!((e.positions[0].swap - (-0.05)).abs() < 1e-9);
        e.accrue_swap(1.0);
        assert!((e.positions[0].swap - (-0.10)).abs() < 1e-9);
    }

    #[test]
    fn realised_pnl_12m_includes_swap() {
        let mut e = engine();
        e.swap_long_per_day = -0.05;
        e.apply(proto::ActionType::ActionOpenBuy, 150.000, 1000, "a".into());
        e.accrue_swap(1.0);
        e.mark_to_market(150.200);
        e.apply(proto::ActionType::ActionCloseMostProfit, 150.200, 2000, "b".into());

        let pnl_with_swap = e.realised_pnl_12m();
        // Raw close P/L: (150.195 - 150.005) * 1.0 = 0.190
        // Swap cost: 0.05 (absolute value, subtracted)
        // Net: 0.190 - 0.05 = 0.14
        assert!((pnl_with_swap - 0.14).abs() < 1e-6, "pnl_with_swap={pnl_with_swap}");
    }
}
