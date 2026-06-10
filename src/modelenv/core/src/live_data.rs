use std::collections::HashMap;

use modelenv_proto::{
    Bar, DoubleBottomPattern, DoubleTopPattern, Fill, IntervalIndicators, Observation, Position,
    Reference, StateRow, Tick,
};

use crate::data_loader::TIME_INTERVALS;
use crate::indicators::state_columns;

/// All computed observation fields except `state_columns`.
///
/// Built by the training and live observation pipelines then converted
/// into a [`Reference`] via [`LiveData::into_reference`] or into an
/// [`Observation`] via [`LiveData::into_observation`].
pub struct LiveData {
    pub timestamp_ns: i64,
    pub symbol: String,
    pub live_bars: HashMap<String, Bar>,
    pub positions: Vec<Position>,
    pub session_realised_pnl: f64,
    pub recent_fills: Vec<Fill>,
    pub ta: Vec<IntervalIndicators>,
    pub double_bottoms: Vec<DoubleBottomPattern>,
    pub double_tops: Vec<DoubleTopPattern>,
    pub live_ticks: Vec<Tick>,
    pub done: bool,
    pub reward: f64,
    pub raw_pnl_delta: f64,
    pub max_total_margin: f64,
    pub m15_double_bottom_low: f64,
    pub m15_double_bottom_high: f64,
    pub m15_double_top_high: f64,
    pub m15_double_top_low: f64,
    pub sin_hour: f64,
    pub cos_hour: f64,
}

impl LiveData {
    pub fn into_reference(self) -> Reference {
        Reference {
            timestamp_ns: self.timestamp_ns,
            symbol: self.symbol,
            live_bars: self.live_bars,
            positions: self.positions,
            session_realised_pnl: self.session_realised_pnl,
            recent_fills: self.recent_fills,
            ta: self.ta,
            double_bottoms: self.double_bottoms,
            double_tops: self.double_tops,
            live_ticks: self.live_ticks,
            done: self.done,
            state_columns: state_columns(),
            m15_double_bottom_low: self.m15_double_bottom_low,
            m15_double_bottom_high: self.m15_double_bottom_high,
            m15_double_top_high: self.m15_double_top_high,
            m15_double_top_low: self.m15_double_top_low,
            sin_hour: self.sin_hour,
            cos_hour: self.cos_hour,
        }
    }

    /// Flatten this observation into an [`Observation`] proto.
    /// Values are **raw** (un-normalised); callers must apply
    /// [`Normaliser::normalise_all`] if normalisation is desired.
    pub fn into_observation(&self) -> Observation {
        let columns = state_columns();
        let values = self.flatten(&columns);
        Observation {
            state_columns: columns,
            state_data: vec![StateRow { values }],
            reward: self.reward,
            done: self.done,
            raw_pnl_delta: self.raw_pnl_delta,
            max_total_margin: self.max_total_margin,
        }
    }

    /// Build a flat `Vec<f64>` in the order specified by `columns`.
    fn flatten(&self, columns: &[String]) -> Vec<f64> {
        // Build lookup tables
        let ta_by_interval: HashMap<&str, &IntervalIndicators> = TIME_INTERVALS
            .iter()
            .zip(self.ta.iter())
            .map(|(iv, ta)| (*iv, ta))
            .collect();

        let mut vals = Vec::with_capacity(columns.len());
        for col in columns {
            vals.push(self.extract_column(col, &ta_by_interval));
        }
        vals
    }

    fn extract_column(
        &self,
        col: &str,
        ta_by_interval: &HashMap<&str, &IntervalIndicators>,
    ) -> f64 {
        // --- per-interval bar fields ---
        if col.ends_with("_bar_close") {
            let iv = col.strip_suffix("_bar_close").unwrap();
            return self.live_bars.get(iv).map(|b| b.close).unwrap_or(0.0);
        }
        if col.ends_with("_bar_volume") {
            let iv = col.strip_suffix("_bar_volume").unwrap();
            return self.live_bars.get(iv).map(|b| b.volume).unwrap_or(0.0);
        }

        // --- per-interval TA fields ---
        if let Some(_ta_name) = col.strip_prefix("M5_ta_")
            .or_else(|| col.strip_prefix("M15_ta_"))
            .or_else(|| col.strip_prefix("H1_ta_"))
            .or_else(|| col.strip_prefix("W1_ta_"))
        {
            // ta_name is e.g. "rsi_14" or "fr_618"; we need to figure out the interval
            for iv in TIME_INTERVALS {
                let prefix = format!("{}_ta_", iv);
                if col.starts_with(&prefix) {
                    let field = col.strip_prefix(&prefix).unwrap();
                    if let Some(indicators) = ta_by_interval.get(iv) {
                        return Self::extract_ta_field(indicators, field);
                    }
                    return 0.0;
                }
            }
        }

        // --- global scalars ---
        match col {
            "session_realised_pnl" => self.session_realised_pnl,
            "unrealised_pnl" => self
                .positions
                .iter()
                .map(|p| p.unrealised_pnl)
                .sum(),
            "num_positions_buy" => self
                .positions
                .iter()
                .filter(|p| p.side == 0) // FILL_SIDE_BUY
                .count() as f64,
            "num_positions_sell" => self
                .positions
                .iter()
                .filter(|p| p.side == 1) // FILL_SIDE_SELL
                .count() as f64,
            "swap_fees" => self.positions.iter().map(|p| p.swap).sum(),
            "tick_ask" => self.live_ticks.first().map(|t| t.ask).unwrap_or(0.0),
            "tick_spread" => self
                .live_ticks
                .first()
                .map(|t| t.ask - t.bid)
                .unwrap_or(0.0),
            "tick_count" => self.live_ticks.len() as f64,
            "M15_double_bottom_low" => self.m15_double_bottom_low,
            "M15_double_bottom_high" => self.m15_double_bottom_high,
            "M15_double_top_high" => self.m15_double_top_high,
            "M15_double_top_low" => self.m15_double_top_low,
            "sin_hour" => self.sin_hour,
            "cos_hour" => self.cos_hour,
            _ => 0.0,
        }
    }

    fn extract_ta_field(indicators: &IntervalIndicators, field: &str) -> f64 {
        let v = match field {
            // Momentum
            "rsi_14" => indicators.momentum.as_ref().map(|m| m.rsi_14).unwrap_or(0.0),
            "cci_14" => indicators.momentum.as_ref().map(|m| m.cci_14).unwrap_or(0.0),
            // Trend
            "adx_14" => indicators.trend.as_ref().map(|t| t.adx_14).unwrap_or(0.0),
            "macd" => indicators.trend.as_ref().map(|t| t.macd).unwrap_or(0.0),
            "macd_signal" => indicators.trend.as_ref().map(|t| t.macd_signal).unwrap_or(0.0),
            "macd_hist" => indicators.trend.as_ref().map(|t| t.macd_hist).unwrap_or(0.0),
            "sma_10" => indicators.trend.as_ref().map(|t| t.sma_10).unwrap_or(0.0),
            "sma_20" => indicators.trend.as_ref().map(|t| t.sma_20).unwrap_or(0.0),
            "sma_50" => indicators.trend.as_ref().map(|t| t.sma_50).unwrap_or(0.0),
            "ema_10" => indicators.trend.as_ref().map(|t| t.ema_10).unwrap_or(0.0),
            "ema_20" => indicators.trend.as_ref().map(|t| t.ema_20).unwrap_or(0.0),
            "ema_50" => indicators.trend.as_ref().map(|t| t.ema_50).unwrap_or(0.0),
            "ichimoku_tenkan" => indicators
                .trend
                .as_ref()
                .map(|t| t.ichimoku_tenkan)
                .unwrap_or(0.0),
            "ichimoku_kijun" => indicators
                .trend
                .as_ref()
                .map(|t| t.ichimoku_kijun)
                .unwrap_or(0.0),
            "ichimoku_senkou_a" => indicators
                .trend
                .as_ref()
                .map(|t| t.ichimoku_senkou_a)
                .unwrap_or(0.0),
            "ichimoku_senkou_b" => indicators
                .trend
                .as_ref()
                .map(|t| t.ichimoku_senkou_b)
                .unwrap_or(0.0),
            "ichimoku_chikou" => indicators
                .trend
                .as_ref()
                .map(|t| t.ichimoku_chikou)
                .unwrap_or(0.0),
            // Volatility
            "bb_upper" => indicators.volatility.as_ref().map(|v| v.bb_upper).unwrap_or(0.0),
            "bb_middle" => indicators.volatility.as_ref().map(|v| v.bb_middle).unwrap_or(0.0),
            "bb_lower" => indicators.volatility.as_ref().map(|v| v.bb_lower).unwrap_or(0.0),
            // Support
            "fr_000" => indicators.support.as_ref().map(|s| s.fr_000).unwrap_or(0.0),
            "fr_236" => indicators.support.as_ref().map(|s| s.fr_236).unwrap_or(0.0),
            "fr_382" => indicators.support.as_ref().map(|s| s.fr_382).unwrap_or(0.0),
            "fr_500" => indicators.support.as_ref().map(|s| s.fr_500).unwrap_or(0.0),
            "fr_618" => indicators.support.as_ref().map(|s| s.fr_618).unwrap_or(0.0),
            "fr_786" => indicators.support.as_ref().map(|s| s.fr_786).unwrap_or(0.0),
            "fr_1000" => indicators.support.as_ref().map(|s| s.fr_1000).unwrap_or(0.0),
            _ => 0.0,
        };
        if v.is_finite() { v } else { 0.0 }
    }
}
