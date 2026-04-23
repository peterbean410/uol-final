// Episode management module
use anyhow::Result;
use std::collections::HashMap;

use modelenv_proto::{Bar, BarList, Observation};

use crate::data_loader::{load_bars_from_parquet, TIME_INTERVALS};
use crate::position::NANOS_PER_DAY;

/// Represents a loaded episode with price bars for all time intervals
pub struct Episode {
    pub symbol: String,
    pub bars: HashMap<String, Vec<Bar>>,
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
            cursor: 0,
            episode_start_ts,
            episode_end_ts,
            done: false,
        }
    }

    /// Get the current observation for the episode
    pub fn get_observation(&self, positions: &[modelenv_proto::Position], realised_pnl_12m: f64) -> Observation {
        let mut live_bars = HashMap::new();
        let mut recent_bars = HashMap::new();

        // Get the current timestamp from the cursor
        let current_timestamp = self.get_cursor_timestamp();

        for interval in TIME_INTERVALS {
            if let Some(bars) = self.bars.get(*interval) {
                if let Some(latest_bar) = bars.get(self.cursor) {
                    live_bars.insert(interval.to_string(), latest_bar.clone());
                } else if let Some(latest_bar) = bars.last() {
                    // Use the most recent available bar if cursor is beyond available bars
                    live_bars.insert(interval.to_string(), latest_bar.clone());
                }

                // Get recent bars (up to 64) ending at current cursor
                let start_idx = if self.cursor >= 64 { self.cursor - 63 } else { 0 };
                let end_idx = self.cursor + 1;
                let recent: Vec<Bar> = bars
                    .get(start_idx..end_idx)
                    .map(|slice| slice.to_vec())
                    .unwrap_or_default();
                
                recent_bars.insert(
                    interval.to_string(),
                    BarList { bars: recent },
                );
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
            recent_news: Vec::new(),
            done: self.done,
        }
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
        
        // Find the minimum cursor position across all intervals that satisfies the target timestamp
        let mut new_cursor = usize::MAX;
        let mut found_any = false;
        
        for interval in TIME_INTERVALS {
            if let Some(bars) = self.bars.get(*interval) {
                // Find the first bar with timestamp >= target_timestamp
                if let Some(pos) = bars.iter().position(|b| b.timestamp_ns >= target_timestamp) {
                    found_any = true;
                    // Use the minimum cursor position across all intervals
                    if pos < new_cursor {
                        new_cursor = pos;
                    }
                } else {
                    // No bar found at target timestamp - episode is done
                    self.done = true;
                    return false;
                }
            }
        }
        
        if !found_any || new_cursor == usize::MAX {
            self.done = true;
            return false;
        }
        
        self.cursor = new_cursor;
        true
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
}

/// Initialize an episode by loading bars from S3 parquet files
pub async fn initialize_episode(
    symbol: &str,
    s3_prefix: &str,
    episode_start_ts: i64,
    episode_end_ts: i64,
) -> Result<Episode> {
    let mut bars_map = HashMap::new();

    for interval in TIME_INTERVALS {
        // Construct S3 URI for the parquet file
        let s3_uri = format!("{}/{}_{}/data.parquet", s3_prefix, symbol, interval);
        
        // Load bars from S3
        let mut bars = load_bars_from_parquet(&s3_uri, symbol, interval).await?;
        
        // Validate that we have bars
        if bars.is_empty() {
            return Err(anyhow::anyhow!(
                "No bars loaded for {} {} from {}",
                symbol,
                interval,
                s3_uri
            ));
        }

        // If no start timestamp provided (0), use the first bar's timestamp (earliest data)
        // This loads the latest available parquet file with all its data
        let start_ts = if episode_start_ts == 0 {
            bars.first().unwrap().timestamp_ns
        } else {
            episode_start_ts
        };

        // Filter bars to only include those within the episode time range
        bars = bars.into_iter()
            .filter(|b| b.timestamp_ns >= start_ts && b.timestamp_ns <= episode_end_ts)
            .collect();

        // Validate that we have bars after filtering
        if bars.is_empty() {
            return Err(anyhow::anyhow!(
                "No bars found for {} {} in time range [{}, {}]",
                symbol,
                interval,
                start_ts,
                episode_end_ts
            ));
        }

        bars_map.insert(interval.to_string(), bars);
    }

    Ok(Episode::new(
        symbol.to_string(),
        bars_map,
        episode_start_ts,
        episode_end_ts,
    ))
}
