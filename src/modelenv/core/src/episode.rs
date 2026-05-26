// Episode management module
use anyhow::{Context, Result};
use log::{info, warn};
use std::collections::HashMap;

const ONE_MINUTE_NS: i64 = 60_000_000_000;

use modelenv_proto::{Bar, News, Tick};

use crate::live_data::LiveData;

use crate::data_loader::{
    build_interval_data_source, build_news_data_source, build_tick_data_source,
    load_bars_from_parquet_with_range_cached_from_local_cache_dir,
    load_news_from_parquet_with_range_cached_from_local_cache_dir,
    load_ticks_from_parquet_with_range_cached_from_local_cache_dir, DataLoaderError, TIME_INTERVALS,
};
use crate::indicators::{
    compute_interval_indicators, compute_m15_double_bottom_high, compute_m15_double_bottom_low,
    compute_m15_double_top_high, compute_m15_double_top_low, compute_time_features,
    detect_all_patterns,
};
use crate::market_data_cache::MarketDataCache;
use crate::position::NANOS_PER_DAY;

pub const RECENT_WINDOW: usize = 64;
pub const LIVE_TICK_WINDOW_NS: i64 = 5_000_000_000;
pub const RECENT_TICK_WINDOW_NS: i64 = 60_000_000_000;

/// Nanosecond duration of a single bar for the given interval.
pub fn interval_duration_ns(interval: &str) -> i64 {
    match interval {
        "M1" => 60_000_000_000,
        "M5" => 300_000_000_000,
        "M15" => 900_000_000_000,
        "H1" => 3_600_000_000_000,
        "H4" => 14_400_000_000_000,
        "D1" => 86_400_000_000_000,
        "W1" => 604_800_000_000_000,
        "MN" => 2_592_000_000_000_000,
        _ => 60_000_000_000,
    }
}

/// Represents a loaded episode with price bars for all time intervals
#[derive(Clone)]
pub struct Episode {
    pub symbol: String,
    pub bars: HashMap<String, Vec<Bar>>,
    pub ticks: Vec<Tick>,
    pub news: Vec<News>,
    pub cursor_timestamp: i64,
    pub episode_start_ts: i64,
    pub episode_end_ts: i64,
    pub done: bool,
}

impl Episode {
    /// Create a new episode with the given bars for each time interval
    pub fn new(
        symbol: String,
        bars: HashMap<String, Vec<Bar>>,
        episode_start_ts: i64,
        episode_end_ts: i64,
    ) -> Self {
        Episode {
            symbol,
            bars,
            ticks: Vec::new(),
            news: Vec::new(),
            cursor_timestamp: episode_start_ts,
            episode_start_ts,
            episode_end_ts,
            done: false,
        }
    }

    pub fn with_news(mut self, news: Vec<News>) -> Self {
        self.news = news;
        self
    }

    pub fn with_ticks(mut self, ticks: Vec<Tick>) -> Self {
        self.ticks = ticks;
        self
    }

    pub fn interval_cursor_at_or_before(
        &self,
        interval: &str,
        current_timestamp: i64,
    ) -> Option<usize> {
        let bars = self.bars.get(interval)?;
        if bars.is_empty() {
            return None;
        }

        let upper_bound = bars.partition_point(|bar| bar.timestamp_ns <= current_timestamp);
        Some(upper_bound.saturating_sub(1))
    }

    /// Synthesize the partial "live" bar for `interval` at the current cursor.
    /// `open` comes from the historical bar covering the current cursor (its
    /// timestamp is always <= current_timestamp). `high`/`low`/`close`/`volume`
    /// reflect ticks observed in `[period_start, current_timestamp]` so that
    /// the model sees a bar that is still forming, mirroring live mode.
    pub fn forming_bar(&self, interval: &str, current_timestamp: i64) -> Option<Bar> {
        let interval_cursor = self.interval_cursor_at_or_before(interval, current_timestamp)?;
        let bars = self.bars.get(interval)?;
        let historical = bars.get(interval_cursor)?;
        let period_start = historical.timestamp_ns;

        let start_idx = self
            .ticks
            .partition_point(|tick| tick.timestamp_ns < period_start);
        let end_idx = self
            .ticks
            .partition_point(|tick| tick.timestamp_ns <= current_timestamp);
        let window = self.ticks.get(start_idx..end_idx).unwrap_or(&[]);

        let mut high = historical.open;
        let mut low = historical.open;
        let mut close = historical.open;
        for tick in window {
            let mid = (tick.bid + tick.ask) / 2.0;
            if mid > high {
                high = mid;
            }
            if mid < low {
                low = mid;
            }
            close = mid;
        }

        Some(Bar {
            timestamp_ns: period_start,
            open: historical.open,
            high,
            low,
            close,
            volume: window.len() as f64,
        })
    }

    pub fn current_bar(&self, interval: &str) -> Option<Bar> {
        let current_timestamp = self.get_cursor_timestamp();
        self.forming_bar(interval, current_timestamp)
    }

    /// Return the last **completed** bar for `interval` at `current_timestamp`.
    ///
    /// A bar is complete when `bar.timestamp_ns + interval_duration_ns(interval) <= current_timestamp`,
    /// i.e. the full bar period has elapsed. Returns `None` when no bar has completed yet.
    pub fn completed_bar(&self, interval: &str, current_timestamp: i64) -> Option<Bar> {
        let duration = interval_duration_ns(interval);
        let cutoff = current_timestamp.saturating_sub(duration);
        // A negative cutoff means no bar can possibly have completed yet
        if cutoff < 0 {
            return None;
        }
        let idx = self.interval_cursor_at_or_before(interval, cutoff)?;
        self.bars.get(interval)?.get(idx).cloned()
    }

    /// Get the current observation for the episode
    pub fn get_observation(
        &self,
        positions: &[modelenv_proto::Position],
        realised_pnl_12m: f64,
        previous_timestamp_ns: Option<i64>,
    ) -> LiveData {
        let mut live_bars = HashMap::new();
        let mut ta: Vec<modelenv_proto::IntervalIndicators> =
            Vec::with_capacity(TIME_INTERVALS.len());
        let mut double_bottoms: Vec<modelenv_proto::DoubleBottomPattern> = Vec::new();
        let mut double_tops: Vec<modelenv_proto::DoubleTopPattern> = Vec::new();

        let current_timestamp = self.get_cursor_timestamp();

        for interval in TIME_INTERVALS {
            let mut interval_ta = modelenv_proto::IntervalIndicators::default();
            if let Some(bars) = self.bars.get(*interval) {
                if let Some(interval_cursor) =
                    self.interval_cursor_at_or_before(interval, current_timestamp)
                {
                    if let Some(completed) = self.completed_bar(interval, current_timestamp) {
                        live_bars.insert(interval.to_string(), completed);
                    }

                    let start_idx = interval_cursor.saturating_sub(RECENT_WINDOW);
                    // Exclude the forming bar at interval_cursor; TA/patterns must only
                    // see completed bars to avoid future-data leakage.
                    let end_idx = interval_cursor;
                    let recent: Vec<Bar> = bars
                        .get(start_idx..end_idx)
                        .map(|slice| slice.to_vec())
                        .unwrap_or_default();

                    interval_ta = compute_interval_indicators(&recent);

                    // Collect detected patterns from the primary interval (M1 only to
                    // avoid duplicates; M1 has the finest granularity).
                    // Use all bars up to the cursor, patterns can span much wider
                    // than the indicator lookback window.
                    if *interval == "M15" {
                        let all_bars: Vec<Bar> = bars
                            .get(0..end_idx)
                            .map(|slice| slice.to_vec())
                            .unwrap_or_default();
                        let (mut dbs, mut dts) = detect_all_patterns(&all_bars);
                        dbs.reverse();
                        dbs.truncate(12);
                        dts.reverse();
                        dts.truncate(12);
                        double_bottoms = dbs;
                        double_tops = dts;
                    }
                }
            }
            ta.push(interval_ta);
        }

        let live_lower = previous_timestamp_ns
            .map(|prev| prev.max(current_timestamp - LIVE_TICK_WINDOW_NS))
            .unwrap_or_else(|| current_timestamp - LIVE_TICK_WINDOW_NS);

        let live_ticks = self.live_ticks(live_lower, current_timestamp);

        let m15_double_bottom_low =
            compute_m15_double_bottom_low(&double_bottoms, &live_ticks);
        let m15_double_bottom_high =
            compute_m15_double_bottom_high(&double_bottoms, &live_ticks, m15_double_bottom_low);
        let m15_double_top_high =
            compute_m15_double_top_high(&double_tops, &live_ticks);
        let m15_double_top_low =
            compute_m15_double_top_low(&double_tops, &live_ticks, m15_double_top_high);
        let (sin_hour, cos_hour) = compute_time_features(current_timestamp);

        LiveData {
            timestamp_ns: current_timestamp,
            symbol: self.symbol.clone(),
            live_bars,
            positions: positions.to_vec(),
            realised_pnl_12m,
            recent_fills: Vec::new(),
            ta,
            double_bottoms,
            double_tops,
            live_ticks,
            done: self.done,
            reward: 0.0,
            m15_double_bottom_low,
            m15_double_bottom_high,
            m15_double_top_high,
            m15_double_top_low,
            sin_hour,
            cos_hour,
        }
    }

    /// All ticks in `[live_lower - RECENT_TICK_WINDOW_NS, live_lower)`; the
    /// 60-second window right before live_ticks starts.
    /// Return ticks with `lower <= timestamp_ns < upper`, sorted oldest first.
    pub fn ticks_in_range(&self, lower: i64, upper: i64) -> Vec<Tick> {
        if self.ticks.is_empty() {
            return Vec::new();
        }
        let end = self.ticks.partition_point(|t| t.timestamp_ns < upper);
        let start = self.ticks.partition_point(|t| t.timestamp_ns < lower);
        self.ticks.get(start..end).map(|s| s.to_vec()).unwrap_or_default()
    }

    fn live_ticks(&self, live_lower: i64, current_timestamp: i64) -> Vec<Tick> {
        if self.ticks.is_empty() {
            return Vec::new();
        }
        let end_idx = self
            .ticks
            .partition_point(|tick| tick.timestamp_ns <= current_timestamp);
        let start_idx = self
            .ticks
            .partition_point(|tick| tick.timestamp_ns <= live_lower);
        let mut window: Vec<Tick> = self
            .ticks
            .get(start_idx..end_idx)
            .map(|slice| slice.to_vec())
            .unwrap_or_default();
        window.reverse();
        window
    }

    pub fn recent_news(&self, current_timestamp: i64) -> Vec<News> {
        let mut recent: Vec<News> = self
            .news
            .iter()
            .filter(|item| item.timestamp_ns <= current_timestamp)
            .cloned()
            .collect();
        if recent.len() > RECENT_WINDOW {
            recent.drain(0..recent.len() - RECENT_WINDOW);
        }
        recent.reverse();
        recent
    }

    /// Advance the cursor by `step_size_ns`. The cursor is free-running and does
    /// not snap to bar boundaries. Returns true if the episode is still running.
    pub fn advance(&mut self, step_size_ns: i64) -> bool {
        if self.done {
            return false;
        }

        let target_timestamp = self.cursor_timestamp + step_size_ns;
        if target_timestamp > self.episode_end_ts {
            self.done = true;
            return false;
        }

        self.cursor_timestamp = target_timestamp;
        true
    }

    /// Check if a day boundary has been crossed between two timestamps
    pub fn has_day_boundary_crossed(&self, from_timestamp_ns: i64, to_timestamp_ns: i64) -> bool {
        let from_day = from_timestamp_ns / NANOS_PER_DAY;
        let to_day = to_timestamp_ns / NANOS_PER_DAY;
        from_day != to_day
    }

    pub fn get_cursor_timestamp(&self) -> i64 {
        self.cursor_timestamp
    }

    /// Check if the episode is done
    pub fn is_done(&self) -> bool {
        self.done
    }
}

fn resolve_episode_bounds(
    reference_bars: &[Bar],
    symbol: &str,
    episode_start_ts: i64,
    episode_end_ts: i64,
) -> Result<(i64, i64)> {
    let resolved_start_ts = if episode_start_ts == 0 {
        reference_bars
            .first()
            .ok_or_else(|| anyhow::anyhow!("No reference bars loaded for {}", symbol))?
            .timestamp_ns
    } else {
        episode_start_ts
    };
    let resolved_end_ts = if episode_end_ts == 0 {
        reference_bars
            .last()
            .ok_or_else(|| anyhow::anyhow!("No reference bars loaded for {}", symbol))?
            .timestamp_ns
    } else {
        episode_end_ts
    };

    Ok((resolved_start_ts, resolved_end_ts))
}

fn resolve_training_tick_query(
    price_snapshot_ts: Option<i64>,
    episode_start_ts: i64,
    episode_end_ts: i64,
    resolved_start_ts: i64,
    resolved_end_ts: i64,
) -> (Option<i64>, Option<i64>, Option<i64>) {
    // Tick parquet files are per-hour, not cumulative-from-inception like the
    // bar snapshots. We must always pass a range so the loader takes the
    // range-list path rather than the exact-key speculative path; the latter
    // would error on any range with no tick file at the snapshot hour (e.g.
    // pre-2021 USDJPY where ticks don't exist) instead of triggering the M1
    // fallback. Explicit episode_*_ts override the resolved range when set.
    let start = if episode_start_ts > 0 {
        episode_start_ts
    } else {
        resolved_start_ts
    };
    let end = if episode_end_ts > 0 {
        episode_end_ts
    } else {
        resolved_end_ts
    };
    (price_snapshot_ts, Some(start), Some(end))
}

#[cfg(test)]
mod tests {
    use super::*;
    use modelenv_proto::Bar;

    #[test]
    fn test_advance_5_seconds() {
        // Create bars with 1-second intervals (for testing purposes)
        let mut bars = Vec::new();
        for i in 0..10 {
            bars.push(Bar {
                timestamp_ns: i * 1_000_000_000, // 1 second intervals
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
            });
        }

        let mut episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            10_000_000_000,
        );

        // Initial cursor sits at episode_start_ts.
        assert_eq!(episode.get_cursor_timestamp(), 0);

        // Advance by 5 seconds (5_000_000_000 ns)
        let still_running = episode.advance(5_000_000_000);

        // Should still be running
        assert!(still_running);

        // Free-running cursor lands exactly at +5s.
        assert_eq!(episode.get_cursor_timestamp(), 5_000_000_000);

        let obs = episode.get_observation(&[], 0.0, None);
        assert_eq!(obs.timestamp_ns, 5_000_000_000);
    }

    #[test]
    fn test_advance_reaches_end() {
        // Bars at 0, 3, 6, 9, 12 seconds. Episode ends at 15 seconds.
        // Free-running cursor: 0 → 5 → 10 → 15 (ok) → 20 (> 15, done).
        let mut bars = Vec::new();
        for i in 0..5 {
            bars.push(Bar {
                timestamp_ns: i * 3_000_000_000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
            });
        }

        let mut episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            15_000_000_000,
        );

        assert!(episode.advance(5_000_000_000));
        assert_eq!(episode.get_cursor_timestamp(), 5_000_000_000);

        assert!(episode.advance(5_000_000_000));
        assert_eq!(episode.get_cursor_timestamp(), 10_000_000_000);

        assert!(episode.advance(5_000_000_000));
        assert_eq!(episode.get_cursor_timestamp(), 15_000_000_000);

        // Next step crosses episode_end_ts → done.
        assert!(!episode.advance(5_000_000_000));
        assert!(episode.is_done());
    }

    #[test]
    fn test_advance_multiple_steps() {
        // Bars at 0..19s (1s intervals). Episode ends at 20s.
        let mut bars = Vec::new();
        for i in 0..20 {
            bars.push(Bar {
                timestamp_ns: i * 1_000_000_000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
            });
        }

        let mut episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            20_000_000_000,
        );

        assert!(episode.advance(5_000_000_000));
        assert_eq!(episode.get_cursor_timestamp(), 5_000_000_000);

        assert!(episode.advance(5_000_000_000));
        assert_eq!(episode.get_cursor_timestamp(), 10_000_000_000);

        assert!(episode.advance(5_000_000_000));
        assert_eq!(episode.get_cursor_timestamp(), 15_000_000_000);

        assert!(episode.advance(5_000_000_000));
        assert_eq!(episode.get_cursor_timestamp(), 20_000_000_000);

        // Next step crosses episode_end_ts → done.
        assert!(!episode.advance(5_000_000_000));
        assert!(episode.is_done());
    }

    #[test]
    fn test_get_observation_timestamp() {
        // Create bars with specific timestamps
        let bars = vec![
            Bar {
                timestamp_ns: 100_000_000_000,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.5,
                volume: 1000.0,
            },
            Bar {
                timestamp_ns: 101_000_000_000,
                open: 100.5,
                high: 101.5,
                low: 100.0,
                close: 101.0,
                volume: 1100.0,
            },
        ];

        let episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            100_000_000_000,
            200_000_000_000,
        );

        let obs = episode.get_observation(&[], 0.0, None);
        assert_eq!(obs.timestamp_ns, 100_000_000_000);
    }

    #[test]
    fn test_episode_state_with_multiple_intervals() {
        // Create bars for multiple time intervals
        let mut m1_bars = Vec::new();
        let mut m5_bars = Vec::new();

        for i in 0..10 {
            m1_bars.push(Bar {
                timestamp_ns: i * 60_000_000_000, // 1 minute intervals
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
            });

            // M5 bars are every 5th M1 bar (at 0 and 5 minutes)
            m5_bars.push(Bar {
                timestamp_ns: i * 5 * 60_000_000_000, // 5 minute intervals
                open: 100.0 + (i * 5) as f64,
                high: 101.0 + (i * 5) as f64,
                low: 99.0 + (i * 5) as f64,
                close: 100.5 + (i * 5) as f64,
                volume: 5000.0,
            });
        }

        let mut bars_map = HashMap::new();
        bars_map.insert("M1".to_string(), m1_bars);
        bars_map.insert("M5".to_string(), m5_bars);

        let mut episode = Episode::new("USDJPY".to_string(), bars_map, 0, 600_000_000_000);

        // Initial cursor sits at episode_start_ts.
        assert_eq!(episode.get_cursor_timestamp(), 0);
        assert!(!episode.is_done());

        // Advance by 5 minutes (300 seconds = 300_000_000_000 ns)
        let still_running = episode.advance(300_000_000_000);

        // Should still be running
        assert!(still_running);

        // Free-running cursor lands exactly at +300s.
        assert_eq!(episode.get_cursor_timestamp(), 300_000_000_000);

        let obs = episode.get_observation(&[], 0.0, None);
        assert_eq!(obs.timestamp_ns, 300_000_000_000);
        // At cursor=300s the M1 bar starting at 300s is still forming;
        // the last completed M1 bar closed at 240s (240+60=300).
        assert_eq!(obs.live_bars["M1"].timestamp_ns, 240_000_000_000);
        // M5 bar starting at 0s completed at 300s; the bar at 300s is forming.
        assert_eq!(obs.live_bars["M5"].timestamp_ns, 0);
    }

    #[test]
    fn test_observation_uses_latest_slower_interval_bar_at_or_before_reference_time() {
        let m1_bars = (0..7)
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

        let mut episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), m1_bars), ("M5".to_string(), m5_bars)]
                .into_iter()
                .collect(),
            0,
            360_000_000_000,
        );

        assert!(episode.advance(360_000_000_000));

        let obs = episode.get_observation(&[], 0.0, None);
        assert_eq!(obs.timestamp_ns, 360_000_000_000);
        // At cursor=360s, M1 bar at 300s completed at 360s; bar at 360s is forming.
        assert_eq!(obs.live_bars["M1"].timestamp_ns, 300_000_000_000);
        // M5 bar at 0s completed at 300s; bar at 300s is forming until 600s.
        assert_eq!(obs.live_bars["M5"].timestamp_ns, 0);
    }

    #[test]
    fn test_observation_includes_recent_and_live_ticks() {
        let bars = (0..3)
            .map(|i| Bar {
                timestamp_ns: i * 60_000_000_000,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.5,
                volume: 1000.0,
            })
            .collect::<Vec<_>>();
        let ticks = vec![
            Tick {
                timestamp_ns: 1_000_000_000,
                bid: 100.10,
                ask: 100.11,
            },
            Tick {
                timestamp_ns: 59_000_000_000,
                bid: 100.20,
                ask: 100.21,
            },
            Tick {
                timestamp_ns: 61_000_000_000,
                bid: 100.30,
                ask: 100.31,
            },
            Tick {
                timestamp_ns: 119_000_000_000,
                bid: 100.40,
                ask: 100.41,
            },
        ];

        let mut episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            180_000_000_000,
        )
        .with_ticks(ticks);

        assert!(episode.advance(60_000_000_000));

        let obs = episode.get_observation(&[], 0.0, Some(0));
        assert_eq!(obs.live_ticks.len(), 1);
        assert_eq!(obs.live_ticks[0].timestamp_ns, 59_000_000_000);

        let recent = episode.ticks_in_range(0, 55_000_000_000);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].timestamp_ns, 1_000_000_000);
    }

    #[test]
    fn test_first_step_observation_includes_ticks_after_bar_open() {
        // Mirrors the real-world S3 case: the first bar timestamp lands on a
        // session-open boundary and the very first tick is a few ms later.
        // After advancing one 5s step the live tick window must capture it.
        let bars = vec![
            Bar {
                timestamp_ns: 0,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.5,
                volume: 1000.0,
            },
            Bar {
                timestamp_ns: 60_000_000_000,
                open: 100.5,
                high: 101.5,
                low: 100.0,
                close: 101.0,
                volume: 1000.0,
            },
        ];
        let ticks = vec![
            Tick {
                timestamp_ns: 29_000_000,
                bid: 100.10,
                ask: 100.12,
            },
            Tick {
                timestamp_ns: 4_500_000_000,
                bid: 100.20,
                ask: 100.22,
            },
        ];

        let mut episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            120_000_000_000,
        )
        .with_ticks(ticks);

        // Simulate environment.reset() pushing the cursor one step in.
        assert!(episode.advance(5_000_000_000));

        let obs = episode.get_observation(&[], 0.0, None);
        assert_eq!(obs.timestamp_ns, 5_000_000_000);
        assert_eq!(obs.live_ticks.len(), 2);
        assert_eq!(obs.live_ticks[0].timestamp_ns, 4_500_000_000);
        assert_eq!(obs.live_ticks[1].timestamp_ns, 29_000_000);
    }

    #[test]
    fn test_forming_bar_synthesizes_partial_ohlc_from_ticks() {
        let bars = vec![
            Bar {
                timestamp_ns: 0,
                open: 100.00,
                high: 999.0, // historical full-period high, must NOT leak
                low: 1.0,    // historical full-period low, must NOT leak
                close: 50.0, // historical close, must NOT leak
                volume: 1000.0,
            },
            Bar {
                timestamp_ns: 60_000_000_000,
                open: 200.0,
                high: 200.0,
                low: 200.0,
                close: 200.0,
                volume: 1000.0,
            },
        ];
        let ticks = vec![
            Tick {
                timestamp_ns: 1_000_000_000,
                bid: 100.10,
                ask: 100.20,
            }, // mid 100.15
            Tick {
                timestamp_ns: 2_000_000_000,
                bid: 100.30,
                ask: 100.40,
            }, // mid 100.35, high
            Tick {
                timestamp_ns: 3_000_000_000,
                bid: 99.90,
                ask: 100.00,
            }, // mid 99.95, low
            Tick {
                timestamp_ns: 4_000_000_000,
                bid: 100.20,
                ask: 100.22,
            }, // mid 100.21, close
            Tick {
                timestamp_ns: 6_000_000_000,
                bid: 105.0,
                ask: 105.0,
            }, // outside window
        ];

        let episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            120_000_000_000,
        )
        .with_ticks(ticks);

        let bar = episode
            .forming_bar("M1", 5_000_000_000)
            .expect("forming bar exists");

        assert_eq!(bar.timestamp_ns, 0);
        assert_eq!(bar.open, 100.00);
        assert!((bar.high - 100.35).abs() < 1e-9);
        assert!((bar.low - 99.95).abs() < 1e-9);
        assert!((bar.close - 100.21).abs() < 1e-9);
        assert_eq!(bar.volume, 4.0);
    }

    #[test]
    fn test_forming_bar_falls_back_to_open_when_no_ticks_in_window() {
        let bars = vec![Bar {
            timestamp_ns: 0,
            open: 100.0,
            high: 999.0,
            low: 1.0,
            close: 50.0,
            volume: 1000.0,
        }];

        let episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            60_000_000_000,
        );

        let bar = episode
            .forming_bar("M1", 1_000_000_000)
            .expect("forming bar exists");
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.high, 100.0);
        assert_eq!(bar.low, 100.0);
        assert_eq!(bar.close, 100.0);
        assert_eq!(bar.volume, 0.0);
    }

    #[test]
    fn test_recent_news_capped_and_sorted_latest_first() {
        let total = (RECENT_WINDOW + 10) as i64;
        let news = (0..total)
            .map(|i| News {
                timestamp_ns: i * 60_000_000_000,
                headline: format!("h{}", i),
                sentiment_score: 0.0,
                source: "test".to_string(),
            })
            .collect::<Vec<_>>();

        let bars = (0..total)
            .map(|i| Bar {
                timestamp_ns: i * 60_000_000_000,
                open: 100.0,
                high: 100.0,
                low: 100.0,
                close: 100.0,
                volume: 0.0,
            })
            .collect::<Vec<_>>();
        let cursor_ts = (total - 1) * 60_000_000_000;

        let mut episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            cursor_ts,
        )
        .with_news(news);
        assert!(episode.advance(cursor_ts));

        let recent = episode.recent_news(cursor_ts);
        assert_eq!(recent.len(), RECENT_WINDOW);
        assert_eq!(recent[0].timestamp_ns, cursor_ts);
        assert!(recent[0].timestamp_ns > recent[1].timestamp_ns);
    }

    #[test]
    fn test_live_bar_and_ta_present_in_observation() {
        let bars = (0..(RECENT_WINDOW as i64 + 10))
            .map(|i| Bar {
                timestamp_ns: i * 60_000_000_000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
            })
            .collect::<Vec<_>>();
        let cursor_ts = (RECENT_WINDOW as i64 + 9) * 60_000_000_000;

        let mut episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            cursor_ts,
        )
        .with_ticks(Vec::new());

        assert!(episode.advance(cursor_ts));

        let obs = episode.get_observation(&[], 0.0, Some(0));

        assert!(obs.live_bars.contains_key("M1"));
        assert_eq!(obs.ta.len(), crate::data_loader::TIME_INTERVALS.len());
    }

    #[test]
    fn test_recent_ticks_returns_all_ticks_in_60s_window_before_live() {
        // Cursor at 120s; live_lower = 120s - 5s = 115s; recent window = [55s, 115s).
        let bars = vec![
            Bar { timestamp_ns: 0, open: 0.0, high: 0.0, low: 0.0, close: 0.0, volume: 0.0 },
            Bar { timestamp_ns: 60_000_000_000, open: 0.0, high: 0.0, low: 0.0, close: 0.0, volume: 0.0 },
            Bar { timestamp_ns: 120_000_000_000, open: 0.0, high: 0.0, low: 0.0, close: 0.0, volume: 0.0 },
        ];
        // 70 ticks at 1s intervals starting at 50s: 50, 51, …, 119.
        let tick_count = 70i64;
        let ticks: Vec<Tick> = (0..tick_count)
            .map(|i| Tick {
                timestamp_ns: 50_000_000_000 + i * 1_000_000_000,
                bid: 100.0,
                ask: 100.01,
            })
            .collect();

        let mut episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            180_000_000_000,
        )
        .with_ticks(ticks);

        // Advance to bar at 120s.
        assert!(episode.advance(120_000_000_000));

        // previous=60s, so live_lower = max(60s, 120s-5s) = 115s.
        // recent window = [115s-60s, 115s) = [55s, 115s) → ticks at 55s–114s = 60 ticks.
        let recent = episode.ticks_in_range(55_000_000_000, 115_000_000_000);
        assert_eq!(recent.len(), 60);
        assert_eq!(recent[0].timestamp_ns, 55_000_000_000);
        assert_eq!(recent[59].timestamp_ns, 114_000_000_000);
        assert!(recent[0].timestamp_ns < recent[1].timestamp_ns);

        // live_ticks = [115s, 120s) → ticks at 115s–119s = 5 ticks.
        let live = episode.ticks_in_range(115_000_000_000, 120_000_000_000);
        assert_eq!(live.len(), 5);
        assert_eq!(live[0].timestamp_ns, 115_000_000_000);
        assert_eq!(live[4].timestamp_ns, 119_000_000_000);
    }

    #[test]
    fn test_episode_done_at_end() {
        // Create bars with specific end timestamp
        let bars = vec![
            Bar {
                timestamp_ns: 0,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.5,
                volume: 1000.0,
            },
            Bar {
                timestamp_ns: 60_000_000_000,
                open: 100.5,
                high: 101.5,
                low: 100.0,
                close: 101.0,
                volume: 1100.0,
            },
        ];

        let mut episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            60_000_000_000, // episode ends at 60 seconds
        );

        // Initial state
        assert!(!episode.is_done());

        // Advance to end
        assert!(episode.advance(60_000_000_000));
        assert!(!episode.is_done());

        // Next advance should be done
        assert!(!episode.advance(60_000_000_000));
        assert!(episode.is_done());
    }

    #[test]
    fn test_advance_with_custom_step_size() {
        let mut bars = Vec::new();
        for i in 0..10 {
            bars.push(Bar {
                timestamp_ns: i * 1_000_000_000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
            });
        }

        let mut episode = Episode::new(
            "USDJPY".to_string(),
            [("M1".to_string(), bars)].into_iter().collect(),
            0,
            10_000_000_000,
        );

        assert!(episode.advance(2_000_000_000));
        assert_eq!(episode.get_cursor_timestamp(), 2_000_000_000);
        assert_eq!(
            episode.get_observation(&[], 0.0, None).timestamp_ns,
            2_000_000_000
        );
    }

    #[test]
    fn test_resolve_episode_bounds_defaults_to_reference_range() {
        let reference_bars = vec![
            Bar {
                timestamp_ns: 10,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.5,
                volume: 1000.0,
            },
            Bar {
                timestamp_ns: 20,
                open: 101.0,
                high: 102.0,
                low: 100.0,
                close: 101.5,
                volume: 1100.0,
            },
        ];

        let (start, end) = resolve_episode_bounds(&reference_bars, "USDJPY", 0, 0).unwrap();

        assert_eq!((start, end), (10, 20));
    }

    #[test]
    fn test_resolve_training_tick_query_uses_resolved_range_when_episode_bounds_unset() {
        // Ticks always need a range so the loader takes the range-list path.
        // With episode_*_ts=0 we fall back to the resolved range.
        let (snapshot_ts, start, end) = resolve_training_tick_query(Some(123), 0, 0, 10, 20);

        assert_eq!(snapshot_ts, Some(123));
        assert_eq!(start, Some(10));
        assert_eq!(end, Some(20));
    }

    #[test]
    fn test_resolve_training_tick_query_keeps_snapshot_ts_with_explicit_bounds() {
        let (snapshot_ts, start, end) = resolve_training_tick_query(Some(123), 1, 2, 10, 20);

        assert_eq!(snapshot_ts, Some(123));
        assert_eq!(start, Some(1));
        assert_eq!(end, Some(2));
    }

    #[test]
    fn test_resolve_training_tick_query_falls_back_to_resolved_range_without_snapshot_ts() {
        let (snapshot_ts, start, end) = resolve_training_tick_query(None, 1, 2, 10, 20);

        assert_eq!(snapshot_ts, None);
        assert_eq!(start, Some(1));
        assert_eq!(end, Some(2));
    }

    #[test]
    fn test_resolve_training_tick_query_falls_back_to_resolved_range_when_all_unset() {
        let (snapshot_ts, start, end) = resolve_training_tick_query(None, 0, 0, 10, 20);

        assert_eq!(snapshot_ts, None);
        assert_eq!(start, Some(10));
        assert_eq!(end, Some(20));
    }

    #[test]
    fn test_step_size_supports_m1_tick_fallback_accepts_whole_minute_multiples() {
        assert!(step_size_supports_m1_tick_fallback(60_000_000_000));
        assert!(step_size_supports_m1_tick_fallback(300_000_000_000));
        assert!(step_size_supports_m1_tick_fallback(3_600_000_000_000));
    }

    #[test]
    fn test_step_size_supports_m1_tick_fallback_rejects_sub_minute_and_zero() {
        assert!(!step_size_supports_m1_tick_fallback(0));
        assert!(!step_size_supports_m1_tick_fallback(-60_000_000_000));
        assert!(!step_size_supports_m1_tick_fallback(30_000_000_000));
        assert!(!step_size_supports_m1_tick_fallback(59_000_000_000));
        assert!(!step_size_supports_m1_tick_fallback(61_000_000_000));
    }

    #[test]
    fn test_synthesize_ticks_from_m1_bars_maps_low_to_bid_and_high_to_ask() {
        let bars = vec![
            Bar { timestamp_ns: 1_000, open: 1.10, high: 1.20, low: 1.05, close: 1.15, volume: 0.0 },
            Bar { timestamp_ns: 2_000, open: 1.15, high: 1.30, low: 1.10, close: 1.25, volume: 0.0 },
        ];

        let ticks = synthesize_ticks_from_m1_bars(&bars, None, None);

        assert_eq!(ticks.len(), 2);
        assert_eq!(ticks[0].timestamp_ns, 1_000);
        assert_eq!(ticks[0].bid, 1.05);
        assert_eq!(ticks[0].ask, 1.20);
        assert_eq!(ticks[1].timestamp_ns, 2_000);
        assert_eq!(ticks[1].bid, 1.10);
        assert_eq!(ticks[1].ask, 1.30);
    }

    #[test]
    fn test_synthesize_ticks_from_m1_bars_filters_to_range() {
        let bars = vec![
            Bar { timestamp_ns: 1_000, open: 1.10, high: 1.20, low: 1.05, close: 1.15, volume: 0.0 },
            Bar { timestamp_ns: 2_000, open: 1.15, high: 1.30, low: 1.10, close: 1.25, volume: 0.0 },
            Bar { timestamp_ns: 3_000, open: 1.25, high: 1.35, low: 1.20, close: 1.30, volume: 0.0 },
        ];

        let ticks = synthesize_ticks_from_m1_bars(&bars, Some(1_500), Some(2_500));

        assert_eq!(ticks.len(), 1);
        assert_eq!(ticks[0].timestamp_ns, 2_000);
    }

    #[test]
    fn test_is_archive_gap_error_detects_all_typed_variants() {
        for typed in [
            DataLoaderError::NoTickSourcesInRange,
            DataLoaderError::NoSourcesMatchedInRange {
                interval: "ticks".to_string(),
                source_uri: "s3://x".to_string(),
                requested_range: String::new(),
                expected_partitions: String::new(),
                candidates: String::new(),
            },
            DataLoaderError::NoSourcesMatchedInRangeUnknownSource {
                interval: "ticks".to_string(),
                requested_range: String::new(),
                candidates: String::new(),
            },
            DataLoaderError::NoSourcesAvailable {
                interval: "ticks".to_string(),
            },
            DataLoaderError::ExactSnapshotKeyMissing {
                key: "s3://x/.../foo.parquet".to_string(),
                snapshot_ts_ns: 0,
                formatted_ts: "1970-01-01T00:00:00+00:00".to_string(),
            },
        ] {
            let err: anyhow::Error = typed.into();
            assert!(
                is_archive_gap_error(&err),
                "expected fallback to trigger for {:#}",
                err
            );
        }
    }

    #[test]
    fn test_is_archive_gap_error_ignores_untyped_errors() {
        // A bare anyhow!() that happens to contain similar wording must NOT
        // satisfy the predicate, that's the whole point of the typed-error
        // refactor.
        let err = anyhow::anyhow!("No parquet sources matched the requested time range manually");
        assert!(!is_archive_gap_error(&err));

        let err = anyhow::anyhow!("AccessDenied calling s3:ListObjectsV2");
        assert!(!is_archive_gap_error(&err));
    }

    #[test]
    fn test_is_archive_gap_error_walks_anyhow_context_chain() {
        let typed = DataLoaderError::NoTickSourcesInRange;
        let root: anyhow::Error = typed.into();
        let wrapped = root.context("Failed to load training ticks for USDJPY");
        assert!(is_archive_gap_error(&wrapped));
    }

}

/// True when an `anyhow::Error` from the data loader indicates the parquet
/// archive has no partitions covering the requested range, applies to any
/// data type (ticks, news, bars). Walks the anyhow error chain so wrapped
/// errors (`.with_context(...)`) are still recognised by their root
/// `DataLoaderError` variant, avoids string matching on the formatted
/// message.
pub(crate) fn is_archive_gap_error(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<DataLoaderError>()
            .is_some_and(DataLoaderError::is_archive_gap)
    })
}

/// Synthesize per-minute ticks from M1 bars when raw tick parquet is absent.
/// Each bar becomes one tick at the bar's timestamp with `bid = bar.low` and
/// `ask = bar.high`. Used only when `step_size_seconds` is a multiple of 60
/// and tick data isn't available.
pub(crate) fn synthesize_ticks_from_m1_bars(
    m1_bars: &[Bar],
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> Vec<Tick> {
    m1_bars
        .iter()
        .filter(|bar| start_timestamp_ns.is_none_or(|s| bar.timestamp_ns >= s))
        .filter(|bar| end_timestamp_ns.is_none_or(|e| bar.timestamp_ns <= e))
        .map(|bar| Tick {
            timestamp_ns: bar.timestamp_ns,
            bid: bar.low,
            ask: bar.high,
        })
        .collect()
}

/// True when `step_size_ns` is a positive whole-minute multiple, the
/// precondition for the M1-bar tick fallback.
pub(crate) fn step_size_supports_m1_tick_fallback(step_size_ns: i64) -> bool {
    step_size_ns > 0 && step_size_ns % ONE_MINUTE_NS == 0
}

/// Initialize an episode by loading bars from S3 parquet files
pub async fn initialize_episode(
    symbol: &str,
    s3_prefix: &str,
    local_cache_dir: &str,
    price_snapshot_ts: Option<i64>,
    episode_start_ts: i64,
    episode_end_ts: i64,
    step_size_ns: i64,
    market_data_cache: &MarketDataCache,
) -> Result<Episode> {
    let mut bars_map = HashMap::new();
    let start_timestamp_ns = (episode_start_ts > 0).then_some(episode_start_ts);
    let end_timestamp_ns = (episode_end_ts > 0).then_some(episode_end_ts);

    // EOH-snapshot parquet files contain cumulative history *up to* their
    // partition timestamp. When the caller pinned a snapshot via
    // `price_snapshot_ts`, honour it. Otherwise fall back to `episode_end_ts`
    // so a 2012-01-31 episode picks the 2012-01-31 snapshot, not "latest",
    // which would load a file whose bars are all after the episode window
    // and get filtered to empty downstream.
    let bar_snapshot_ts = price_snapshot_ts.or(end_timestamp_ns);

    for interval in TIME_INTERVALS {
        let interval_source = build_interval_data_source(s3_prefix, symbol, interval);

        let mut bars = match load_bars_from_parquet_with_range_cached_from_local_cache_dir(
            local_cache_dir,
            market_data_cache,
            &interval_source,
            symbol,
            interval,
            bar_snapshot_ts,
            None, // load full history, episode_start_ts only constrains ticks
            end_timestamp_ns,
        )
        .await
        {
            Ok(bars) => bars,
            Err(err) => {
                return Err(err).with_context(|| {
                    format!(
                        "Failed to load training interval {} for {} from {}",
                        interval, symbol, interval_source
                    )
                })
            }
        };

        if bars.is_empty() {
            return Err(anyhow::anyhow!(
                "No bars loaded for {} {} from {}",
                symbol,
                interval,
                interval_source
            ));
        }

        if episode_end_ts > 0 {
            bars = bars
                .into_iter()
                .filter(|b| b.timestamp_ns <= episode_end_ts)
                .collect();

            if bars.is_empty() {
                return Err(anyhow::anyhow!(
                    "No bars found for {} {} before episode_end_ts {}",
                    symbol,
                    interval,
                    episode_end_ts
                ));
            }
        }

        bars_map.insert(interval.to_string(), bars);
    }

    let reference_bars = bars_map
        .get(TIME_INTERVALS[0])
        .ok_or_else(|| anyhow::anyhow!("Missing reference bars for {}", TIME_INTERVALS[0]))?;
    let (resolved_start_ts, resolved_end_ts) =
        resolve_episode_bounds(reference_bars, symbol, episode_start_ts, episode_end_ts)?;
    let (tick_snapshot_ts, tick_start_ts, tick_end_ts) = resolve_training_tick_query(
        price_snapshot_ts,
        episode_start_ts,
        episode_end_ts,
        resolved_start_ts,
        resolved_end_ts,
    );

    let news_source = build_news_data_source(s3_prefix, symbol);
    let news = match load_news_from_parquet_with_range_cached_from_local_cache_dir(
        local_cache_dir,
        market_data_cache,
        &news_source,
        start_timestamp_ns,
        end_timestamp_ns,
    )
    .await
    {
        Ok(news) => news,
        // News archive starts well after the price archive (no pre-2021 news
        // sources). Treat archive gaps as "no news for this episode", log
        // and proceed with an empty Vec rather than failing the Reset.
        Err(err) if is_archive_gap_error(&err) => {
            warn!(
                "No news parquet sources for {} in episode range, proceeding with empty news. Underlying: {:#}",
                symbol, err
            );
            Vec::new()
        }
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "Failed to load training news for {} from {}",
                    symbol, news_source
                )
            })
        }
    };

    let tick_source = build_tick_data_source(s3_prefix, symbol);
    let tick_load_result = load_ticks_from_parquet_with_range_cached_from_local_cache_dir(
        local_cache_dir,
        market_data_cache,
        &tick_source,
        symbol,
        tick_snapshot_ts,
        tick_start_ts,
        tick_end_ts,
    )
    .await;

    let m1_fallback_allowed = step_size_supports_m1_tick_fallback(step_size_ns);
    let ticks = match tick_load_result {
        Ok(ticks) if !ticks.is_empty() => ticks,
        Ok(_) if m1_fallback_allowed => {
            let synth = synth_m1_ticks_for_range(&bars_map, tick_start_ts, tick_end_ts);
            warn!(
                "No ticks loaded for {} in range (synthesised {} M1-derived ticks (bid=low, ask=high)",
                symbol,
                synth.len()
            );
            synth
        }
        Ok(empty) => empty,
        Err(err) if is_archive_gap_error(&err) && m1_fallback_allowed => {
            let synth = synth_m1_ticks_for_range(&bars_map, tick_start_ts, tick_end_ts);
            warn!(
                "Tick parquet missing for {} in range) synthesised {} M1-derived ticks (bid=low, ask=high)",
                symbol,
                synth.len()
            );
            synth
        }
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "Failed to load training ticks for {} from {}",
                    symbol, tick_source
                )
            })
        }
    };

    Ok(Episode::new(
        symbol.to_string(),
        bars_map,
        resolved_start_ts,
        resolved_end_ts,
    )
    .with_ticks(ticks)
    .with_news(news))
}

fn synth_m1_ticks_for_range(
    bars_map: &HashMap<String, Vec<Bar>>,
    tick_start_ts: Option<i64>,
    tick_end_ts: Option<i64>,
) -> Vec<Tick> {
    bars_map
        .get("M1")
        .map(|bars| synthesize_ticks_from_m1_bars(bars, tick_start_ts, tick_end_ts))
        .unwrap_or_default()
}

/// Preload training market data on server startup.
///
/// ``tick_window`` optionally narrows the tick preload to a specific
/// ``(start_ns, end_ns)`` range, typically the union of the trainer's
/// per-date episode windows, supplied via the modelenv-server CLI
/// (`--training-date-start` etc.). When ``None``, falls back to the full
/// M1 reference bar span, which can span months for accumulator-style
/// EOH snapshots and incur a multi-minute cold start unnecessarily.
///
/// Bars (all `TIME_INTERVALS`) and news are always preloaded with no
/// time range filter; they are single-snapshot files, so the I/O cost
/// doesn't scale with window size.
///
/// Out-of-window tick data is still served correctly: `initialize_episode`
/// calls `load_ticks_from_parquet_with_range_cached_from_local_cache_dir`
/// with the episode's own bounds, which falls through to
/// `ensure_local_cached_s3_source` and downloads any missing parquet
/// hour lazily on Reset().
pub async fn preload_training_market_data(
    symbol: &str,
    s3_prefix: &str,
    local_cache_dir: &str,
    price_snapshot_ts: Option<i64>,
    tick_window: Option<(i64, i64)>,
    market_data_cache: &MarketDataCache,
) -> Result<()> {
    let mut has_reference_interval = false;
    let mut reference_bars = None;

    for interval in TIME_INTERVALS {
        let interval_source = build_interval_data_source(s3_prefix, symbol, interval);
        info!(
            "Preloading training interval {} for {} from {}",
            interval, symbol, interval_source
        );
        match load_bars_from_parquet_with_range_cached_from_local_cache_dir(
            local_cache_dir,
            market_data_cache,
            &interval_source,
            symbol,
            interval,
            price_snapshot_ts,
            None,
            None,
        )
        .await
        {
            Ok(bars) => {
                info!(
                    "Finished preloading training interval {} for {}",
                    interval, symbol
                );
                if *interval == TIME_INTERVALS[0] {
                    has_reference_interval = true;
                    reference_bars = Some(bars);
                }
            }
            Err(err) => {
                return Err(err).with_context(|| {
                    format!(
                        "Failed to preload training interval {} for {} from {}",
                        interval, symbol, interval_source
                    )
                })
            }
        }
    }

    if !has_reference_interval {
        return Err(anyhow::anyhow!(
            "Missing reference bars for {} while preloading training data",
            TIME_INTERVALS[0]
        ));
    }
    let reference_bars = reference_bars.ok_or_else(|| {
        anyhow::anyhow!(
            "Missing reference bars for {} while preloading training data",
            TIME_INTERVALS[0]
        )
    })?;
    // If the caller supplied an explicit tick window (--training-date-start
    // / --training-date-end / --training-hour-start / --training-hour-end),
    // use it. Otherwise default to the full M1 reference span.
    let (resolved_start_ts, resolved_end_ts) = match tick_window {
        Some((start_ns, end_ns)) => (start_ns, end_ns),
        None => resolve_episode_bounds(&reference_bars, symbol, 0, 0)?,
    };
    let (tick_snapshot_ts, tick_start_ts, tick_end_ts) =
        resolve_training_tick_query(price_snapshot_ts, 0, 0, resolved_start_ts, resolved_end_ts);

    let news_source = build_news_data_source(s3_prefix, symbol);
    info!(
        "Preloading training news for {} from {}",
        symbol, news_source
    );
    match load_news_from_parquet_with_range_cached_from_local_cache_dir(
        local_cache_dir,
        market_data_cache,
        &news_source,
        None,
        None,
    )
    .await
    {
        Ok(_) => {
            info!("Finished preloading training news for {}", symbol);
        }
        Err(err) if is_archive_gap_error(&err) => {
            warn!(
                "No news parquet sources for {} in preload range, per-episode loader will proceed with empty news. Underlying: {:#}",
                symbol, err
            );
        }
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "Failed to preload training news for {} from {}",
                    symbol, news_source
                )
            })
        }
    }

    let tick_source = build_tick_data_source(s3_prefix, symbol);
    info!(
        "Preloading training ticks for {} from {}",
        symbol, tick_source
    );
    match load_ticks_from_parquet_with_range_cached_from_local_cache_dir(
        local_cache_dir,
        market_data_cache,
        &tick_source,
        symbol,
        tick_snapshot_ts,
        tick_start_ts,
        tick_end_ts,
    )
    .await
    {
        Ok(_) => {
            info!("Finished preloading training ticks for {}", symbol);
        }
        // Tick archive doesn't cover this range, log and continue. Per-episode
        // ``initialize_episode`` will either fall back to M1-derived ticks (when
        // step_size_seconds is a whole-minute multiple) or surface the failure
        // there with the actual step_size in context.
        Err(err) if is_archive_gap_error(&err) => {
            warn!(
                "No tick parquet sources for {} in preload range, deferring to per-episode M1 fallback. Underlying: {:#}",
                symbol, err
            );
        }
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "Failed to preload training ticks for {} from {}",
                    symbol, tick_source
                )
            })
        }
    }

    Ok(())
}
