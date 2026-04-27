// Episode management module
use anyhow::Result;
use log::warn;
use std::collections::HashMap;

use modelenv_proto::{Bar, BarList, News, Observation};

use crate::data_loader::{
    build_interval_data_source, build_news_data_source,
    load_bars_from_parquet_with_range_cached_from_local_cache_dir,
    load_news_from_parquet_with_range_cached_from_local_cache_dir, TIME_INTERVALS,
};
use crate::market_data_cache::MarketDataCache;
use crate::position::NANOS_PER_DAY;

/// Represents a loaded episode with price bars for all time intervals
#[derive(Clone)]
pub struct Episode {
    pub symbol: String,
    pub bars: HashMap<String, Vec<Bar>>,
    pub news: Vec<News>,
    pub cursor: usize,
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
            news: Vec::new(),
            cursor: 0,
            episode_start_ts,
            episode_end_ts,
            done: false,
        }
    }

    pub fn with_news(mut self, news: Vec<News>) -> Self {
        self.news = news;
        self
    }

    fn interval_cursor_at_or_before(
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

    pub fn current_bar(&self, interval: &str) -> Option<&Bar> {
        let current_timestamp = self.get_cursor_timestamp();
        let interval_cursor = self.interval_cursor_at_or_before(interval, current_timestamp)?;
        self.bars
            .get(interval)
            .and_then(|bars| bars.get(interval_cursor))
    }

    /// Get the current observation for the episode
    pub fn get_observation(
        &self,
        positions: &[modelenv_proto::Position],
        realised_pnl_12m: f64,
    ) -> Observation {
        let mut live_bars = HashMap::new();
        let mut recent_bars = HashMap::new();

        // Get the current timestamp from the cursor
        let current_timestamp = self.get_cursor_timestamp();

        for interval in TIME_INTERVALS {
            if let Some(bars) = self.bars.get(*interval) {
                if let Some(interval_cursor) =
                    self.interval_cursor_at_or_before(interval, current_timestamp)
                {
                    if let Some(latest_bar) = bars.get(interval_cursor) {
                        live_bars.insert(interval.to_string(), latest_bar.clone());
                    }

                    let start_idx = interval_cursor.saturating_sub(63);
                    let end_idx = interval_cursor + 1;
                    let recent: Vec<Bar> = bars
                        .get(start_idx..end_idx)
                        .map(|slice| slice.to_vec())
                        .unwrap_or_default();

                    recent_bars.insert(interval.to_string(), BarList { bars: recent });
                }
            }
        }

        Observation {
            timestamp_ns: current_timestamp,
            symbol: self.symbol.clone(),
            live_bars,
            recent_bars,
            positions: positions.to_vec(),
            realised_pnl_12m,
            recent_fills: Vec::new(),
            indicators: Vec::new(),
            recent_ticks: Vec::new(),
            live_ticks: Vec::new(),
            recent_news: self.recent_news(current_timestamp),
            done: self.done,
        }
    }

    fn recent_news(&self, current_timestamp: i64) -> Vec<News> {
        let mut recent: Vec<News> = self
            .news
            .iter()
            .filter(|item| item.timestamp_ns <= current_timestamp)
            .cloned()
            .collect();
        if recent.len() > 16 {
            recent.drain(0..recent.len() - 16);
        }
        recent
    }

    /// Advance the episode cursor by one step (5 seconds = 5,000,000,000 nanoseconds)
    /// Returns true if the episode is still running, false if done
    pub fn advance(&mut self, step_size_ns: i64) -> bool {
        if self.done {
            return false;
        }

        // Get the current timestamp at the cursor position
        let current_timestamp = self.get_cursor_timestamp();

        // Calculate the target timestamp (current + 5 seconds)
        let target_timestamp = current_timestamp + step_size_ns;

        // Check if target timestamp exceeds episode end
        if target_timestamp > self.episode_end_ts {
            self.done = true;
            return false;
        }

        let Some(reference_bars) = self.bars.get(TIME_INTERVALS[0]) else {
            self.done = true;
            return false;
        };

        let search_start = (self.cursor + 1).min(reference_bars.len());
        if let Some(offset) = reference_bars[search_start..]
            .iter()
            .position(|bar| bar.timestamp_ns >= target_timestamp)
        {
            self.cursor = search_start + offset;
            true
        } else {
            self.done = true;
            false
        }
    }

    /// Check if a day boundary has been crossed between two timestamps
    pub fn has_day_boundary_crossed(&self, from_timestamp_ns: i64, to_timestamp_ns: i64) -> bool {
        let from_day = from_timestamp_ns / NANOS_PER_DAY;
        let to_day = to_timestamp_ns / NANOS_PER_DAY;
        from_day != to_day
    }

    /// Get the timestamp at the current cursor position (uses first interval as reference)
    pub fn get_cursor_timestamp(&self) -> i64 {
        if let Some(bars) = self.bars.get(TIME_INTERVALS[0]) {
            if let Some(bar) = bars.get(self.cursor) {
                return bar.timestamp_ns;
            }
        }
        self.episode_start_ts
    }

    /// Check if the episode is done
    pub fn is_done(&self) -> bool {
        self.done
    }
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

        // Initial cursor should be at 0
        assert_eq!(episode.cursor, 0);

        // Advance by 5 seconds (5_000_000_000 ns)
        let still_running = episode.advance(5_000_000_000);

        // Should still be running
        assert!(still_running);

        // Cursor should have moved to bar at or after 5 seconds (index 5)
        assert_eq!(episode.cursor, 5);

        // Timestamp should be at 5 seconds
        let obs = episode.get_observation(&[], 0.0);
        assert_eq!(obs.timestamp_ns, 5_000_000_000);
    }

    #[test]
    fn test_advance_reaches_end() {
        // Create bars with 3-second intervals - bars at 0, 3, 6, 9, 12 seconds
        // Episode ends at 15 seconds
        // Advancing 5 seconds from 0 should find bar at 6 seconds (index 2)
        // Advancing 5 seconds from 6 should find bar at 12 seconds (index 4) - no bar at 9s >= 11s
        // Advancing 5 seconds from 12 should find bar at 15... but we only have up to 12
        let mut bars = Vec::new();
        for i in 0..5 {
            bars.push(Bar {
                timestamp_ns: i * 3_000_000_000, // 3 second intervals: 0, 3, 6, 9, 12
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
            15_000_000_000, // episode ends at 15 seconds
        );

        // First advance: 0 -> 6 seconds (bar at index 2)
        assert!(episode.advance(5_000_000_000));
        assert_eq!(episode.cursor, 2);

        // Second advance: 6 -> 11 seconds (no bar at 9s >= 11s, so use bar at 12s index 4)
        assert!(episode.advance(5_000_000_000));
        assert_eq!(episode.cursor, 4);

        // Third advance: 12 -> 17 seconds, but no bar at 17 seconds
        // So this should be done
        let still_running = episode.advance(5_000_000_000);
        assert!(!still_running);
        assert!(episode.is_done());
    }

    #[test]
    fn test_advance_multiple_steps() {
        // Create bars with 1-second intervals
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

        // First advance
        assert!(episode.advance(5_000_000_000));
        assert_eq!(episode.cursor, 5);

        // Second advance
        assert!(episode.advance(5_000_000_000));
        assert_eq!(episode.cursor, 10);

        // Third advance
        assert!(episode.advance(5_000_000_000));
        assert_eq!(episode.cursor, 15);

        // Fourth advance - should be done (no bar at 20 seconds)
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
            0,
            200_000_000_000,
        );

        let obs = episode.get_observation(&[], 0.0);
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

        // Initial cursor should be at 0
        assert_eq!(episode.cursor, 0);
        assert!(!episode.is_done());

        // Advance by 5 minutes (300 seconds = 300_000_000_000 ns)
        let still_running = episode.advance(300_000_000_000);

        // Should still be running
        assert!(still_running);

        // Cursor should track the reference M1 interval.
        assert_eq!(episode.cursor, 5);

        let obs = episode.get_observation(&[], 0.0);
        assert_eq!(obs.timestamp_ns, 300_000_000_000);
        assert_eq!(obs.live_bars["M1"].timestamp_ns, 300_000_000_000);
        assert_eq!(obs.live_bars["M5"].timestamp_ns, 300_000_000_000);
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

        let obs = episode.get_observation(&[], 0.0);
        assert_eq!(obs.timestamp_ns, 360_000_000_000);
        assert_eq!(obs.live_bars["M1"].timestamp_ns, 360_000_000_000);
        assert_eq!(obs.live_bars["M5"].timestamp_ns, 300_000_000_000);
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
        assert_eq!(episode.cursor, 2);
        assert_eq!(
            episode.get_observation(&[], 0.0).timestamp_ns,
            2_000_000_000
        );
    }
}

/// Initialize an episode by loading bars from S3 parquet files
pub async fn initialize_episode(
    symbol: &str,
    s3_prefix: &str,
    local_cache_dir: &str,
    price_snapshot_ts: Option<i64>,
    episode_start_ts: i64,
    episode_end_ts: i64,
    market_data_cache: &MarketDataCache,
) -> Result<Episode> {
    let mut bars_map = HashMap::new();
    let start_timestamp_ns = (episode_start_ts > 0).then_some(episode_start_ts);
    let end_timestamp_ns = (episode_end_ts > 0).then_some(episode_end_ts);

    for interval in TIME_INTERVALS {
        let interval_source = build_interval_data_source(s3_prefix, symbol, interval);

        let mut bars = match load_bars_from_parquet_with_range_cached_from_local_cache_dir(
            local_cache_dir,
            market_data_cache,
            &interval_source,
            symbol,
            interval,
            price_snapshot_ts,
            start_timestamp_ns,
            end_timestamp_ns,
        )
        .await
        {
            Ok(bars) => bars,
            Err(err) if is_missing_interval_data_error(&err) => {
                warn!(
                    "Skipping unavailable training interval {} for {} at {}: {}",
                    interval, symbol, interval_source, err
                );
                continue;
            }
            Err(err) => return Err(err),
        };

        if bars.is_empty() {
            return Err(anyhow::anyhow!(
                "No bars loaded for {} {} from {}",
                symbol,
                interval,
                interval_source
            ));
        }

        let start_ts = if episode_start_ts == 0 {
            bars.first().unwrap().timestamp_ns
        } else {
            episode_start_ts
        };

        let end_ts = if episode_end_ts == 0 {
            bars.last().unwrap().timestamp_ns
        } else {
            episode_end_ts
        };

        bars = bars
            .into_iter()
            .filter(|b| b.timestamp_ns >= start_ts && b.timestamp_ns <= end_ts)
            .collect();

        if bars.is_empty() {
            return Err(anyhow::anyhow!(
                "No bars found for {} {} in time range [{}, {}]",
                symbol,
                interval,
                start_ts,
                end_ts
            ));
        }

        bars_map.insert(interval.to_string(), bars);
    }

    let reference_bars = bars_map
        .get(TIME_INTERVALS[0])
        .ok_or_else(|| anyhow::anyhow!("Missing reference bars for {}", TIME_INTERVALS[0]))?;
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
        Err(err) if is_missing_interval_data_error(&err) => {
            warn!(
                "Skipping unavailable training news for {} at {}: {}",
                symbol, news_source, err
            );
            Vec::new()
        }
        Err(err) => return Err(err),
    };

    Ok(Episode::new(
        symbol.to_string(),
        bars_map,
        resolved_start_ts,
        resolved_end_ts,
    )
    .with_news(news))
}

pub async fn preload_training_market_data(
    symbol: &str,
    s3_prefix: &str,
    local_cache_dir: &str,
    price_snapshot_ts: Option<i64>,
    market_data_cache: &MarketDataCache,
) -> Result<()> {
    let mut has_reference_interval = false;

    for interval in TIME_INTERVALS {
        let interval_source = build_interval_data_source(s3_prefix, symbol, interval);
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
            Ok(_) => {
                if *interval == TIME_INTERVALS[0] {
                    has_reference_interval = true;
                }
            }
            Err(err) if is_missing_interval_data_error(&err) => {
                warn!(
                    "Skipping unavailable training interval {} for {} at {} during startup preload: {}",
                    interval, symbol, interval_source, err
                );
            }
            Err(err) => return Err(err),
        }
    }

    if !has_reference_interval {
        return Err(anyhow::anyhow!(
            "Missing reference bars for {} while preloading training data",
            TIME_INTERVALS[0]
        ));
    }

    let news_source = build_news_data_source(s3_prefix, symbol);
    match load_news_from_parquet_with_range_cached_from_local_cache_dir(
        local_cache_dir,
        market_data_cache,
        &news_source,
        None,
        None,
    )
    .await
    {
        Ok(_) => {}
        Err(err) if is_missing_interval_data_error(&err) => {
            warn!(
                "Skipping unavailable training news for {} at {} during startup preload: {}",
                symbol, news_source, err
            );
        }
        Err(err) => return Err(err),
    }

    Ok(())
}

pub(crate) fn is_missing_interval_data_error(err: &anyhow::Error) -> bool {
    let message = err.to_string();
    message.contains("No parquet files found under S3 prefix")
        || message.contains("No parquet files found under s3://")
        || message.contains("No child prefixes found under s3://")
        || message.contains("No parquet files found under local path")
        || message.contains("Local parquet path does not exist")
}
