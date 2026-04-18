use crate::proto;
use super::round_price;

/// In-memory simulator tracking position, fills, and PnL for a single symbol.
/// Used by the training backend; the live backend delegates fills to the broker
/// but reuses the same PnL bookkeeping for the observation payload.
pub struct SimEngine {
    pub symbol: String,
    pub position: proto::Position,
    pub recent_fills: Vec<proto::Fill>,
    /// Spread in price units (not pips). For USD/JPY 0.01 == 1 pip.
    pub spread: f64,
    /// Per-unit transaction cost applied on every fill, in quote currency.
    pub cost_per_unit: f64,
    pub order_seq: u64,
}

impl SimEngine {
    pub fn new(symbol: String, spread: f64, cost_per_unit: f64) -> Self {
        Self {
            symbol,
            position: proto::Position {
                side: proto::PositionSide::PositionFlat as i32,
                size: 0.0,
                entry_price: 0.0,
                unrealised_pnl: 0.0,
                realised_pnl: 0.0,
            },
            recent_fills: Vec::new(),
            spread,
            cost_per_unit,
            order_seq: 0,
        }
    }

    pub fn reset(&mut self) {
        self.position = proto::Position {
            side: proto::PositionSide::PositionFlat as i32,
            size: 0.0,
            entry_price: 0.0,
            unrealised_pnl: 0.0,
            realised_pnl: 0.0,
        };
        self.recent_fills.clear();
        self.order_seq = 0;
    }

    pub fn mark_to_market(&mut self, mid_price: f64) {
        let side = proto::PositionSide::try_from(self.position.side).unwrap_or(proto::PositionSide::PositionFlat);
        self.position.unrealised_pnl = match side {
            proto::PositionSide::PositionFlat => 0.0,
            proto::PositionSide::PositionLong => (mid_price - self.position.entry_price) * self.position.size,
            proto::PositionSide::PositionShort => (self.position.entry_price - mid_price) * self.position.size,
        };
    }

    /// Apply an action at the given mid price. Returns realised PnL delta and the fill, if any.
    pub fn apply(
        &mut self,
        action: proto::ActionType,
        size: f64,
        mid_price: f64,
        ts_ns: i64,
        client_order_id: String,
    ) -> (f64, Option<proto::Fill>) {
        let half_spread = self.spread / 2.0;
        let size = size.max(0.0);

        let (fill_price, fill_side) = match action {
            proto::ActionType::ActionHold => return (0.0, None),
            proto::ActionType::ActionBuy => (round_price(&self.symbol, mid_price + half_spread), proto::ActionType::ActionBuy),
            proto::ActionType::ActionSell => (round_price(&self.symbol, mid_price - half_spread), proto::ActionType::ActionSell),
            proto::ActionType::ActionClose => {
                let side = proto::PositionSide::try_from(self.position.side).unwrap_or(proto::PositionSide::PositionFlat);
                match side {
                    proto::PositionSide::PositionLong => (round_price(&self.symbol, mid_price - half_spread), proto::ActionType::ActionSell),
                    proto::PositionSide::PositionShort => (round_price(&self.symbol, mid_price + half_spread), proto::ActionType::ActionBuy),
                    proto::PositionSide::PositionFlat => return (0.0, None),
                }
            }
        };

        let close_size = match (action, proto::PositionSide::try_from(self.position.side)) {
            (proto::ActionType::ActionClose, _) => self.position.size,
            (proto::ActionType::ActionBuy, Ok(proto::PositionSide::PositionShort)) => size.min(self.position.size),
            (proto::ActionType::ActionSell, Ok(proto::PositionSide::PositionLong)) => size.min(self.position.size),
            _ => 0.0,
        };

        let mut realised_delta = 0.0;
        if close_size > 0.0 {
            let current_side = proto::PositionSide::try_from(self.position.side).unwrap_or(proto::PositionSide::PositionFlat);
            realised_delta = match current_side {
                proto::PositionSide::PositionLong => (fill_price - self.position.entry_price) * close_size,
                proto::PositionSide::PositionShort => (self.position.entry_price - fill_price) * close_size,
                proto::PositionSide::PositionFlat => 0.0,
            };
            self.position.realised_pnl += realised_delta;
            self.position.size -= close_size;
            if self.position.size.abs() < f64::EPSILON {
                self.position.size = 0.0;
                self.position.side = proto::PositionSide::PositionFlat as i32;
                self.position.entry_price = 0.0;
            }
        }

        let open_size = match action {
            proto::ActionType::ActionClose => 0.0,
            _ => (size - close_size).max(0.0),
        };
        if open_size > 0.0 {
            let new_side = match fill_side {
                proto::ActionType::ActionBuy => proto::PositionSide::PositionLong,
                proto::ActionType::ActionSell => proto::PositionSide::PositionShort,
                _ => proto::PositionSide::PositionFlat,
            };
            if self.position.size == 0.0 {
                self.position.side = new_side as i32;
                self.position.entry_price = fill_price;
                self.position.size = open_size;
            } else {
                let existing = self.position.size;
                let blended = (self.position.entry_price * existing + fill_price * open_size)
                    / (existing + open_size);
                self.position.entry_price = round_price(&self.symbol, blended);
                self.position.size += open_size;
            }
        }

        let filled_size = close_size + open_size;
        let costs = filled_size * self.cost_per_unit;
        self.position.realised_pnl -= costs;
        realised_delta -= costs;

        self.order_seq += 1;
        let order_id = if client_order_id.is_empty() {
            format!("sim-{}", self.order_seq)
        } else {
            client_order_id
        };

        let fill = proto::Fill {
            order_id,
            timestamp_ns: ts_ns,
            price: fill_price,
            size: filled_size,
            side: fill_side as i32,
            partial: false,
        };
        self.recent_fills.push(fill.clone());
        if self.recent_fills.len() > 64 {
            let overflow = self.recent_fills.len() - 64;
            self.recent_fills.drain(0..overflow);
        }

        (realised_delta, Some(fill))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> SimEngine {
        SimEngine::new("USDJPY".to_string(), 0.01, 0.0)
    }

    #[test]
    fn buy_then_close_realises_pnl_with_spread() {
        let mut e = engine();
        let (_, fill) = e.apply(proto::ActionType::ActionBuy, 1.0, 150.000, 0, String::new());
        assert!(fill.is_some());
        assert_eq!(e.position.side, proto::PositionSide::PositionLong as i32);
        // buy at mid+half_spread = 150.005
        assert!((e.position.entry_price - 150.005).abs() < 1e-6);

        let (delta, _) = e.apply(proto::ActionType::ActionClose, 0.0, 150.200, 1, String::new());
        // close at 150.200 - 0.005 = 150.195 → pnl = (150.195 - 150.005) * 1 = 0.190
        assert!((delta - 0.190).abs() < 1e-6);
        assert_eq!(e.position.side, proto::PositionSide::PositionFlat as i32);
    }

    #[test]
    fn sell_updates_position_and_mark_to_market() {
        let mut e = engine();
        e.apply(proto::ActionType::ActionSell, 2.0, 150.000, 0, String::new());
        assert_eq!(e.position.side, proto::PositionSide::PositionShort as i32);
        e.mark_to_market(149.800);
        // short 2 @ 149.995, mtm at 149.800 → (149.995 - 149.800) * 2 = 0.390
        assert!((e.position.unrealised_pnl - 0.390).abs() < 1e-6);
    }
}
