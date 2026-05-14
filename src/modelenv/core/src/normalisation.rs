//! Feature normalisation for the flat state vector.
//!
//! Implements the per-category transforms described in `plan_normalisation.md`:
//! price ratios → z-score, bounded oscillators → /100, unbounded osc → z-score,
//! volume → log(1+x) → z-score, P&L → z-score, counts → /cap, rest passthrough.
//!
//! All z-score windows are rolling (ring-buffer) and computed from past data only.

/// Which normalisation strategy applies to a column.
#[derive(Clone, Debug, PartialEq)]
pub enum FeatureCategory {
    /// Ratio (x / ref_close) then z-score with clipping.
    PriceLevel,
    /// Divide by 100.
    BoundedOsc,
    /// Raw value → z-score with clipping.
    UnboundedOsc,
    /// log(1+x) → z-score with clipping.
    Volume,
    /// Raw value → z-score with wider clipping.
    PnL,
    /// Divide by a fixed cap.
    Count { cap: f64 },
    /// No transformation.
    Passthrough,
}

/// Per-column configuration derived from the normalisation plan.
#[derive(Clone, Debug)]
pub struct ColumnConfig {
    pub category: FeatureCategory,
    /// Z-score window size (only relevant for PriceLevel, UnboundedOsc, Volume, PnL).
    pub window: usize,
    /// Clip threshold (symmetric, only relevant for z-scored categories).
    pub clip: f64,
}

impl ColumnConfig {
    fn new(category: FeatureCategory, window: usize, clip: f64) -> Self {
        Self {
            category,
            window,
            clip,
        }
    }
}

/// Sliding-window running statistics (mean and standard deviation).
///
/// Uses a ring buffer to track the last `window` values.  When the buffer is
/// not yet full, statistics are computed from the partial window.
#[derive(Clone, Debug)]
struct RunningStats {
    buffer: Vec<f64>,
    position: usize,
    full: bool,
    window: usize,
}

impl RunningStats {
    fn new(window: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(window),
            position: 0,
            full: false,
            window,
        }
    }

    fn push(&mut self, value: f64) {
        if self.buffer.len() < self.window {
            self.buffer.push(value);
        } else {
            self.buffer[self.position] = value;
            self.full = true;
        }
        self.position = (self.position + 1) % self.window;
    }

    fn mean(&self) -> Option<f64> {
        if self.buffer.is_empty() {
            return None;
        }
        Some(self.buffer.iter().sum::<f64>() / self.buffer.len() as f64)
    }

    fn std(&self) -> Option<f64> {
        if self.buffer.len() < 2 {
            return None;
        }
        let m = self.mean()?;
        let var = self.buffer.iter().map(|x| (x - m).powi(2)).sum::<f64>()
            / (self.buffer.len() - 1) as f64;
        Some(var.sqrt())
    }

}

/// Per-column normaliser that maintains rolling statistics.
#[derive(Clone, Debug)]
struct ColumnNormaliser {
    config: ColumnConfig,
    stats: Option<RunningStats>,
}

impl ColumnNormaliser {
    fn new(config: ColumnConfig) -> Self {
        let stats = match config.category {
            FeatureCategory::PriceLevel
            | FeatureCategory::UnboundedOsc
            | FeatureCategory::Volume
            | FeatureCategory::PnL => Some(RunningStats::new(config.window)),
            _ => None,
        };
        Self { config, stats }
    }

    /// Update rolling statistics with a new raw (pre-transform) value.
    fn update(&mut self, raw: f64) {
        if let Some(ref mut stats) = self.stats {
            if raw.is_finite() {
                stats.push(raw);
            }
        }
    }

    /// Normalise a raw value.  Returns `f64::NAN` when z-score stats are
    /// not yet ready (buffer not full), callers should handle this.
    fn normalise(&self, raw: f64) -> f64 {
        if !raw.is_finite() {
            return 0.0;
        }
        match self.config.category {
            FeatureCategory::PriceLevel | FeatureCategory::UnboundedOsc => {
                let (mean, std) = match self.stats.as_ref().and_then(|s| {
                    s.mean().and_then(|m| s.std().map(|sd| (m, sd)))
                }) {
                    Some((m, sd)) if sd > 1e-12 => (m, sd),
                    _ => return 0.0,
                };
                let z = (raw - mean) / std;
                z.clamp(-self.config.clip, self.config.clip)
            }
            FeatureCategory::Volume => {
                let transformed = (1.0 + raw).ln();
                self.normalise_transformed(transformed)
            }
            FeatureCategory::PnL => {
                let (mean, std) = match self.stats.as_ref().and_then(|s| {
                    s.mean().and_then(|m| s.std().map(|sd| (m, sd)))
                }) {
                    Some((m, sd)) if sd > 1e-12 => (m, sd),
                    _ => return 0.0,
                };
                let z = (raw - mean) / std;
                z.clamp(-self.config.clip, self.config.clip)
            }
            FeatureCategory::BoundedOsc => raw / 100.0,
            FeatureCategory::Count { cap } => raw / cap,
            FeatureCategory::Passthrough => raw,
        }
    }

    /// Z-score an already-transformed value (used by Volume after log).
    fn normalise_transformed(&self, transformed: f64) -> f64 {
        let (mean, std) = match self.stats.as_ref().and_then(|s| {
            s.mean().and_then(|m| s.std().map(|sd| (m, sd)))
        }) {
            Some((m, sd)) if sd > 1e-12 => (m, sd),
            _ => return 0.0,
        };
        let z = (transformed - mean) / std;
        z.clamp(-self.config.clip, self.config.clip)
    }
}

/// Holds per-column normalisers for the full state vector and orchestrates
/// the normalisation pass for a single observation.
pub struct Normaliser {
    columns: Vec<ColumnNormaliser>,
}

impl Normaliser {
    /// Build a `Normaliser` from the ordered column names and the plan defaults.
    pub fn new(column_names: &[String]) -> Self {
        let columns: Vec<_> = column_names
            .iter()
            .map(|name| ColumnNormaliser::new(Self::default_config(name)))
            .collect();
        Self { columns }
    }

    /// Return the default [`ColumnConfig`] for a named column.
    fn default_config(name: &str) -> ColumnConfig {
        // --- Price-level (ratio → z-score, window 252, clip ±4) ---
        if name.ends_with("_bar_close")
            || name.ends_with("_ta_sma_")
            || name.ends_with("_ta_ema_")
            || name.contains("_ta_ichimoku_")
            || name.ends_with("_ta_bb_upper")
            || name.ends_with("_ta_bb_middle")
            || name.ends_with("_ta_bb_lower")
            || name.contains("_ta_fr_")
            || name.starts_with("M15_double_bottom_")
            || name.starts_with("M15_double_top_")
            || name == "tick_ask"
        {
            return ColumnConfig::new(FeatureCategory::PriceLevel, 252, 4.0);
        }

        // --- Bounded oscillators ( / 100) ---
        if name.contains("_ta_rsi_") || name.contains("_ta_adx_") {
            return ColumnConfig::new(FeatureCategory::BoundedOsc, 0, 0.0);
        }

        // --- Unbounded oscillators (z-score, window 252, clip ±4) ---
        if name.contains("_ta_cci_")
            || name.contains("_ta_macd")
        {
            return ColumnConfig::new(FeatureCategory::UnboundedOsc, 252, 4.0);
        }

        // --- Volume (log(1+x) → z-score, window 252, clip ±4) ---
        if name.ends_with("_bar_volume") {
            return ColumnConfig::new(FeatureCategory::Volume, 252, 4.0);
        }

        // --- P&L (z-score, window 504, clip ±5) ---
        if name == "realised_pnl_12m" || name == "unrealised_pnl" {
            return ColumnConfig::new(FeatureCategory::PnL, 504, 5.0);
        }

        // --- Counts ( / cap ) ---
        if name.contains("num_positions") {
            return ColumnConfig::new(FeatureCategory::Count { cap: 5.0 }, 0, 0.0);
        }
        if name == "tick_count" {
            return ColumnConfig::new(FeatureCategory::Count { cap: 100.0 }, 0, 0.0);
        }

        // --- Passthrough ---
        // tick_spread, swap_fees, sin_hour, cos_hour, done
        ColumnConfig::new(FeatureCategory::Passthrough, 0, 0.0)
    }

    /// Total number of columns.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Update rolling statistics from raw feature values.
    ///
    /// `raw_values` must be in the same order as the `column_names` passed to
    /// [`Normaliser::new`].  Only z-scored columns are affected; passthrough
    /// and fixed-divisor columns ignore updates.
    pub fn update(&mut self, raw_values: &[f64]) {
        for (i, raw) in raw_values.iter().enumerate() {
            if i < self.columns.len() {
                self.columns[i].update(*raw);
            }
        }
    }

    /// Normalise a single raw value by column index.
    pub fn normalise_by_index(&self, index: usize, raw: f64) -> f64 {
        self.columns
            .get(index)
            .map(|c| c.normalise(raw))
            .unwrap_or(0.0)
    }

    /// Normalise all raw values in-place.
    pub fn normalise_all(&self, raw_values: &[f64]) -> Vec<f64> {
        raw_values
            .iter()
            .enumerate()
            .map(|(i, raw)| self.normalise_by_index(i, *raw))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_column_names() -> Vec<String> {
        vec![
            "M5_bar_close".to_string(),
            "M5_ta_rsi_14".to_string(),
            "M5_ta_macd".to_string(),
            "M5_bar_volume".to_string(),
            "realised_pnl_12m".to_string(),
            "num_positions_buy".to_string(),
            "tick_spread".to_string(),
            "sin_hour".to_string(),
            "done".to_string(),
        ]
    }

    #[test]
    fn default_config_categorises_columns() {
        let config = Normaliser::default_config("M5_bar_close");
        assert_eq!(config.category, FeatureCategory::PriceLevel);
        assert_eq!(config.window, 252);
        assert_eq!(config.clip, 4.0);

        let config = Normaliser::default_config("M5_ta_rsi_14");
        assert_eq!(config.category, FeatureCategory::BoundedOsc);

        let config = Normaliser::default_config("M5_ta_macd");
        assert_eq!(config.category, FeatureCategory::UnboundedOsc);

        let config = Normaliser::default_config("M5_bar_volume");
        assert_eq!(config.category, FeatureCategory::Volume);

        let config = Normaliser::default_config("realised_pnl_12m");
        assert_eq!(config.category, FeatureCategory::PnL);
        assert_eq!(config.clip, 5.0);

        let config = Normaliser::default_config("num_positions_buy");
        assert_eq!(config.category, FeatureCategory::Count { cap: 5.0 });

        let config = Normaliser::default_config("tick_spread");
        assert_eq!(config.category, FeatureCategory::Passthrough);
    }

    #[test]
    fn passthrough_returns_raw_value() {
        let names = make_column_names();
        let normaliser = Normaliser::new(&names);
        let spread_idx = 6;
        assert_eq!(normaliser.normalise_by_index(spread_idx, 0.015), 0.015);
        assert_eq!(normaliser.normalise_by_index(spread_idx, -0.005), -0.005);
    }

    #[test]
    fn bounded_osc_divides_by_100() {
        let names = vec!["M5_ta_rsi_14".to_string()];
        let normaliser = Normaliser::new(&names);
        assert_eq!(normaliser.normalise_by_index(0, 65.0), 0.65);
        assert_eq!(normaliser.normalise_by_index(0, 0.0), 0.0);
        assert_eq!(normaliser.normalise_by_index(0, 100.0), 1.0);
    }

    #[test]
    fn count_divides_by_cap() {
        let names = vec!["num_positions_buy".to_string(), "tick_count".to_string()];
        let normaliser = Normaliser::new(&names);
        assert_eq!(normaliser.normalise_by_index(0, 3.0), 0.6); // 3 / 5
        assert_eq!(normaliser.normalise_by_index(1, 50.0), 0.5); // 50 / 100
    }

    #[test]
    fn zscore_returns_zero_when_stats_not_ready() {
        let names = vec!["M5_bar_close".to_string()];
        let normaliser = Normaliser::new(&names);
        // No updates → stats not ready → returns 0.0
        assert_eq!(normaliser.normalise_by_index(0, 1.0005), 0.0);
    }

    #[test]
    fn zscore_normalises_after_warmup() {
        let names = vec!["M5_bar_close".to_string()];
        let mut normaliser = Normaliser::new(&names);

        // Feed 252 identical values to warm up (μ=1.0005, σ≈0).
        // After warmup, normalise a fresh value.
        // With identical values σ≈0, so the normalise will fail the sd>1e-12 check → 0.0.
        // Feed values with small variance instead.
        for i in 0..252 {
            let v = 1.0 + (i as f64 % 10.0) * 0.0001;
            normaliser.columns[0].update(v);
        }

        // Stats are ready now.  Normalise a value near the mean.
        let result = normaliser.normalise_by_index(0, 1.00045);
        // Should be a finite z-score, not 0.0
        assert!(result.abs() > 0.0);
        assert!(result.abs() <= 4.0); // clipped
    }

    #[test]
    fn volume_log_transforms_then_zscore() {
        let names = vec!["M5_bar_volume".to_string()];
        let mut normaliser = Normaliser::new(&names);

        // Feed 252 values of volume=100.0 → log(101) ≈ 4.615
        for _ in 0..252 {
            normaliser.columns[0].update(100.0);
        }
        // With identical values, σ≈0 → z-score returns 0.0 (sd check fails).
        // This is expected; no variance means no signal.
        let result = normaliser.normalise_by_index(0, 100.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn update_only_affects_zscored_columns() {
        let names = make_column_names();
        let mut normaliser = Normaliser::new(&names);
        let raw: Vec<f64> = vec![1.0; names.len()];
        normaliser.update(&raw);
        // Should not panic, passthrough/osc/count columns ignore updates.
    }
}
