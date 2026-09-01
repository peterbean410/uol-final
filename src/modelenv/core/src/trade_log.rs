//! Optional durable trade log for offline review and debugging.
//!
//! When enabled (via `--trade-log <path>` / `MODELENV_TRADE_LOG`), every order
//! fill and every position close is appended as one JSON object per line
//! (JSONL) to the configured file. The log is purely a side-channel for humans
//! and tooling; it is never read back into the environment and does not affect
//! observations, rewards, or any model-visible state.
//!
//! Records share a `type` discriminator so fills and closes can coexist in a
//! single file:
//!   * `{"type":"fill", ...}`; one execution (open or close leg, incl. partials)
//!   * `{"type":"close", ...}`; a fully closed position with realised PnL
//!
//! Writes are best-effort: an I/O error is logged at warn level and dropped so
//! that a failing log can never interrupt trading or training.

use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use log::warn;
use serde::Serialize;

use crate::environment::Fill;
use crate::position::{ClosedPosition, Side};
use modelenv_proto::FillSide;

/// Appends trade events to a JSONL file. Cheap to share behind `&self`; the
/// internal writer is guarded by a mutex so logging from `&mut self` env
/// methods needs no extra borrow juggling.
pub struct TradeLogger {
    path: String,
    writer: Mutex<BufWriter<fs::File>>,
}

impl TradeLogger {
    /// Open (creating parent directories and the file if needed) the trade log
    /// at `path` in append mode.
    pub fn open(path: &str) -> Result<Self> {
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("creating trade-log directory {}", parent.display())
                })?;
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening trade log {path}"))?;
        Ok(TradeLogger {
            path: path.to_string(),
            writer: Mutex::new(BufWriter::new(file)),
        })
    }

    /// Append a single fill (order execution). `event` distinguishes the open
    /// leg from the close leg for readability.
    pub fn log_fill(&self, symbol: &str, event: FillEvent, fill: &Fill) {
        let record = FillRecord {
            kind: "fill",
            event: event.as_str(),
            symbol,
            order_id: &fill.order_id,
            timestamp_ns: fill.timestamp_ns,
            iso_time: iso_time(fill.timestamp_ns),
            price: fill.price,
            size: fill.size,
            side: fill_side_str(fill.side),
            partial: fill.partial,
        };
        self.write_record(&record);
    }

    /// Append a fully closed position, including realised PnL and hold time.
    pub fn log_close(&self, symbol: &str, closed: &ClosedPosition) {
        let hold_seconds =
            (closed.close_timestamp_ns - closed.open_timestamp_ns) as f64 / 1_000_000_000.0;
        let record = CloseRecord {
            kind: "close",
            symbol,
            position_id: &closed.position_id,
            timestamp_ns: closed.close_timestamp_ns,
            iso_time: iso_time(closed.close_timestamp_ns),
            entry_price: closed.entry_price,
            close_price: closed.close_price,
            volume: closed.volume,
            side: side_str(closed.side),
            realised_pnl: closed.realised_pnl,
            swap: closed.swap,
            total_pnl: closed.realised_pnl + closed.swap,
            open_timestamp_ns: closed.open_timestamp_ns,
            hold_seconds,
        };
        self.write_record(&record);
    }

    fn write_record<T: Serialize>(&self, record: &T) {
        if let Err(err) = self.try_write_record(record) {
            warn!("failed to append to trade log {}: {err:#}", self.path);
        }
    }

    fn try_write_record<T: Serialize>(&self, record: &T) -> Result<()> {
        let mut line = serde_json::to_vec(record).context("serialising trade-log record")?;
        line.push(b'\n');
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| anyhow::anyhow!("trade-log writer mutex poisoned"))?;
        writer.write_all(&line).context("writing trade-log record")?;
                                writer.flush().context("flushing trade log")?;
        Ok(())
    }
}

/// Whether a fill opened exposure or closed it.
#[derive(Clone, Copy)]
pub enum FillEvent {
    Open,
    Close,
}

impl FillEvent {
    fn as_str(self) -> &'static str {
        match self {
            FillEvent::Open => "open",
            FillEvent::Close => "close",
        }
    }
}

#[derive(Serialize)]
struct FillRecord<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    event: &'static str,
    symbol: &'a str,
    order_id: &'a str,
    timestamp_ns: i64,
    iso_time: String,
    price: f64,
    size: f64,
    side: &'static str,
    partial: bool,
}

#[derive(Serialize)]
struct CloseRecord<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    symbol: &'a str,
    position_id: &'a str,
    timestamp_ns: i64,
    iso_time: String,
    entry_price: f64,
    close_price: f64,
    volume: f64,
    side: &'static str,
    realised_pnl: f64,
    swap: f64,
    total_pnl: f64,
    open_timestamp_ns: i64,
    hold_seconds: f64,
}

fn iso_time(timestamp_ns: i64) -> String {
    let secs = timestamp_ns.div_euclid(1_000_000_000);
    let nsecs = timestamp_ns.rem_euclid(1_000_000_000) as u32;
    chrono::DateTime::from_timestamp(secs, nsecs)
        .map(|dt| dt.to_rfc3339())
        .unwrap_or_default()
}

fn fill_side_str(side: FillSide) -> &'static str {
    match side {
        FillSide::Buy => "buy",
        FillSide::Sell => "sell",
    }
}

fn side_str(side: Side) -> &'static str {
    match side {
        Side::Buy => "buy",
        Side::Sell => "sell",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn read_lines(path: &str) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn fill_and_close_round_trip_into_jsonl() {
        let dir = tempfile::tempdir().unwrap();
                let path = dir
            .path()
            .join("logs/trades.jsonl")
            .to_str()
            .unwrap()
            .to_string();

        let logger = TradeLogger::open(&path).unwrap();

        let open_fill = Fill {
            order_id: "USDJPY-1".to_string(),
            timestamp_ns: 1_700_000_000_000_000_000,
            price: 150.123,
            size: 1.0,
            side: FillSide::Buy,
            partial: false,
        };
        logger.log_fill("USDJPY", FillEvent::Open, &open_fill);

        let closed = ClosedPosition {
            position_id: "pos-1".to_string(),
            entry_price: 150.123,
            close_price: 150.523,
            volume: 1.0,
            side: Side::Buy,
            realised_pnl: 0.4,
            swap: -0.01,
            open_timestamp_ns: 1_700_000_000_000_000_000,
            close_timestamp_ns: 1_700_000_060_000_000_000,
        };
        logger.log_close("USDJPY", &closed);

        let close_fill = Fill {
            order_id: "fill_1700000060000000000".to_string(),
            timestamp_ns: 1_700_000_060_000_000_000,
            price: 150.523,
            size: 1.0,
            side: FillSide::Sell,
            partial: false,
        };
        logger.log_fill("USDJPY", FillEvent::Close, &close_fill);

        let lines = read_lines(&path);
        assert_eq!(lines.len(), 3, "expected one line per event");

                assert_eq!(lines[0]["type"], "fill");
        assert_eq!(lines[0]["event"], "open");
        assert_eq!(lines[0]["symbol"], "USDJPY");
        assert_eq!(lines[0]["order_id"], "USDJPY-1");
        assert_eq!(lines[0]["side"], "buy");
        assert_eq!(lines[0]["partial"], false);
        assert_eq!(lines[0]["price"], 150.123);
        assert_eq!(lines[0]["timestamp_ns"], 1_700_000_000_000_000_000_i64);
        assert!(lines[0]["iso_time"].as_str().unwrap().starts_with("2023-11-14T"));

                assert_eq!(lines[1]["type"], "close");
        assert_eq!(lines[1]["position_id"], "pos-1");
        assert_eq!(lines[1]["side"], "buy");
        assert_eq!(lines[1]["entry_price"], 150.123);
        assert_eq!(lines[1]["close_price"], 150.523);
        assert_eq!(lines[1]["realised_pnl"], 0.4);
        assert_eq!(lines[1]["swap"], -0.01);
        assert_eq!(
            lines[1]["total_pnl"].as_f64().unwrap(),
            closed.realised_pnl + closed.swap
        );
        assert_eq!(lines[1]["hold_seconds"], 60.0);

                assert_eq!(lines[2]["type"], "fill");
        assert_eq!(lines[2]["event"], "close");
        assert_eq!(lines[2]["side"], "sell");
    }

    #[test]
    fn open_appends_to_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trades.jsonl").to_str().unwrap().to_string();

        let fill = Fill {
            order_id: "o-1".to_string(),
            timestamp_ns: 1_700_000_000_000_000_000,
            price: 150.0,
            size: 1.0,
            side: FillSide::Buy,
            partial: false,
        };

                TradeLogger::open(&path).unwrap().log_fill("USDJPY", FillEvent::Open, &fill);
        TradeLogger::open(&path).unwrap().log_fill("USDJPY", FillEvent::Open, &fill);

        assert_eq!(read_lines(&path).len(), 2, "second open must append, not truncate");
    }
}
