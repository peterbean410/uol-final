// FX RL Model Environment Server
mod broker;
mod server;

use anyhow::Result;
use env_logger::Env;
use log::{error, info};
use modelenv_core::{
    config::{Config, Mode},
    environment::{default_swap_rate_for, Environment},
};

use crate::server::EnvironmentService;
use modelenv_proto::environment_server::EnvironmentServer;
use tonic::transport::Server;

fn init_logging() {
    let _ = env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .try_init();
}

fn format_timestamp_ns(ns: i64) -> String {
    let secs = ns / 1_000_000_000;
    let nsecs = (ns % 1_000_000_000) as u32;
    chrono::DateTime::from_timestamp(secs, nsecs)
        .map(|d| d.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_else(|| ns.to_string())
}

fn build_environment(config: &Config) -> Environment {
    let mut environment = Environment::new(
        config.mode.clone(),
        config.symbol.clone(),
        config.s3_prefix.clone(),
    )
    .with_local_cache_dir(config.local_cache_dir.clone())
    .with_reward_lambda(config.reward_lambda)
    .with_reward_action_penalty(config.reward_action_penalty)
    .with_reward_holding_penalty(config.reward_holding_penalty)
    .with_disable_hedging(config.disable_hedging)
    .with_leverage(config.leverage)
    .with_trade_log(config.trade_log_path.clone());

    // Swap rates: Training/backtest seeds the built-in per-symbol default table
    // inside Environment::new; Live syncs from the broker. Only override here when
    // a rate is explicitly configured, filling any unspecified side from the
    // mode-appropriate default (the table in Training, 0.0 in Live).
    if config.swap_rate_long.is_some() || config.swap_rate_short.is_some() {
        let (default_long, default_short) = if matches!(config.mode, Mode::Training) {
            default_swap_rate_for(&config.symbol)
        } else {
            (0.0, 0.0)
        };
        environment = environment.with_daily_swap_rate(
            config.symbol.clone(),
            config.swap_rate_long.unwrap_or(default_long),
            config.swap_rate_short.unwrap_or(default_short),
        );
    }

    // Compute the training tick window once so it can both scope the tick
    // preload AND default the bar-snapshot timestamp.
    let training_tick_window = match config.training_tick_window() {
        Ok(window) => window,
        Err(err) => {
            log::warn!(
                "Ignoring training tick window from config (will preload full M1 reference span): {}",
                err
            );
            None
        }
    };

    // Pick a single bar snapshot timestamp for the whole training session.
    // Precedence:
    //   1. Explicit --price-snapshot-ts (CLI/env override wins).
    //   2. End of the training tick window (so every episode in this run
    //      shares one cumulative eod/eoh/eom snapshot file per interval,
    //      avoids re-downloading a fresh parquet for each episode_end_ts).
    //   3. None, loader falls back to the latest available snapshot.
    let effective_price_snapshot_ts = config
        .price_snapshot_ts
        .or_else(|| training_tick_window.map(|(_, end_ns)| end_ns));

    if let Some(price_snapshot_ts) = effective_price_snapshot_ts {
        environment = environment.with_price_snapshot_ts(price_snapshot_ts);
    }

    if let Some((start_ns, end_ns)) = training_tick_window {
        environment = environment.with_training_tick_window(start_ns, end_ns);
    }

    environment
}

fn print_double_bottoms_table(patterns: &[modelenv_proto::DoubleBottomPattern]) {
    if patterns.is_empty() {
        println!("No double-bottom patterns detected.");
        return;
    }

    println!(
        "{:<4} {:<22} {:<12} {:<22} {:<12} {:<12} {:<8} {:<6} {:<10}",
        "#", "Bottom 1", "Low 1", "Bottom 2", "Low 2", "Neckline", "Depth %", "Width", "Confirmed"
    );
    for (i, p) in patterns.iter().enumerate() {
        let ts1 = format_timestamp_ns(p.ts1);
        let ts2 = format_timestamp_ns(p.ts2);
        let conf = if p.confirmed { "Yes" } else { "No" };
        println!(
            "{:<4} {:<22} {:<12.5} {:<22} {:<12.5} {:<12.5} {:<8.2} {:<6} {:<10}",
            i + 1,
            ts1,
            p.low1,
            ts2,
            p.low2,
            p.neckline,
            p.depth_pct,
            p.width_bars,
            conf,
        );
    }
    println!("\nTotal: {} pattern(s)", patterns.len());
}

fn parse_cli_detect_args() -> Option<(String, String, i32, u32, u32, u32, usize, f64, usize)> {
    let args: Vec<String> = std::env::args().collect();
    let has_detect = args.iter().any(|a| a == "--detect-double-bottoms");
    if !has_detect {
        return None;
    }

    fn get_arg(args: &[String], flag: &str) -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    }

    let symbol = get_arg(&args, "--symbol").unwrap_or_else(|| "USDJPY".to_string());
    let interval = get_arg(&args, "--interval").unwrap_or_else(|| "M15".to_string());
    let year: i32 = get_arg(&args, "--year")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2026);
    let month: u32 = get_arg(&args, "--month")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let day: u32 = get_arg(&args, "--day")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let hour: u32 = get_arg(&args, "--hour")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let window: usize = get_arg(&args, "--window")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let tolerance: f64 = get_arg(&args, "--tolerance")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.3);
    let min_width: usize = get_arg(&args, "--min-width")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    Some((symbol, interval, year, month, day, hour, window, tolerance, min_width))
}

async fn run_detect_double_bottoms(
    symbol: &str,
    interval: &str,
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    window: usize,
    tolerance: f64,
    min_width: usize,
    s3_prefix: &str,
) -> Result<()> {
    use modelenv_core::data_loader::build_interval_data_source;
    use modelenv_core::indicators::patterns::detect_double_bottoms;

    // Build the S3 data source prefix using the same function as episode loading
    let source = build_interval_data_source(s3_prefix, symbol, interval);
    println!("Loading bars from {source} ...");

    // Build a timestamp at the requested year/month/day/hour for the upper bound
    let target_dt = chrono::NaiveDate::from_ymd_opt(year, month, day)
        .and_then(|d| d.and_hms_opt(hour, 0, 0))
        .and_then(|dt| dt.and_utc().timestamp_nanos_opt())
        .unwrap_or(0);

    let bars = modelenv_core::data_loader::load_bars_from_parquet_with_end_ts(
        &source, symbol, interval, target_dt,
    )
    .await?;

    if bars.is_empty() {
        println!("No bars loaded.");
        return Ok(());
    }
    println!("Loaded {} bars", bars.len());

    let detection = detect_double_bottoms(&bars, window, tolerance, min_width)?;

    if let Some(min_val) = detection.latest_min {
        println!("Latest Local Minimum: {:.5}", min_val);
    }
    if let Some(max_val) = detection.latest_max {
        println!("Latest Local Maximum: {:.5}", max_val);
    }
    println!();

    let patterns: Vec<modelenv_proto::DoubleBottomPattern> = detection
        .patterns
        .iter()
        .map(|p| modelenv_proto::DoubleBottomPattern {
            idx1: p.idx1 as i64,
            idx2: p.idx2 as i64,
            ts1: p.ts1,
            ts2: p.ts2,
            low1: p.low1,
            low2: p.low2,
            neckline: p.neckline,
            neckline_idx: p.neckline_idx as i64,
            depth_pct: p.depth_pct,
            width_bars: p.width_bars as i64,
            confirmed: p.confirmed,
            min_before_val: p.min_before_val.unwrap_or(0.0),
            min_before_ts: p.min_before_ts.unwrap_or(0),
            max_before_val: p.max_before_val.unwrap_or(0.0),
            max_before_ts: p.max_before_ts.unwrap_or(0),
        })
        .collect();

    print_double_bottoms_table(&patterns);

    if let Some(last) = bars.last() {
        let ts = format_timestamp_ns(last.timestamp_ns);
        println!(
            "\nLatest Bar: {}  Open={:.5}  High={:.5}  Low={:.5}  Close={:.5}",
            ts, last.open, last.high, last.low, last.close
        );
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Check for CLI subcommand before starting the server.
    // Must be checked before Config::load() because Config rejects unknown flags.
    if let Some((symbol, interval, year, month, day, hour, window, tolerance, min_width)) =
        parse_cli_detect_args()
    {
        init_logging();
        // Rewrite argv so Config::load() doesn't see the detect flags.
        let original: Vec<String> = std::env::args().collect();
        let detect_flags_with_val = &[
            "--year", "--month", "--day", "--hour", "--interval",
            "--window", "--tolerance", "--min-width",
        ];
        let detect_flags_bool = &["--detect-double-bottoms"];
        let mut filtered: Vec<String> = Vec::new();
        let mut skip_next = false;
        for a in &original {
            if skip_next {
                skip_next = false;
                continue;
            }
            if detect_flags_bool.contains(&a.as_str()) {
                continue; // boolean flag, no value to skip
            }
            if detect_flags_with_val.contains(&a.as_str()) {
                skip_next = true;
                continue;
            }
            filtered.push(a.clone());
        }
        // Build a Config from the filtered args so --local-cache-dir etc. still work
        let config = Config::load_from_args(&filtered)?;
        let prefix = if config.s3_prefix.starts_with("s3://") {
            config.s3_prefix.clone()
        } else {
            format!("s3://{}", config.s3_prefix)
        };
        return run_detect_double_bottoms(
            &symbol, &interval, year, month, day, hour, window, tolerance, min_width,
            &prefix,
        )
        .await;
    }

    init_logging();

    // Load configuration from command-line arguments and environment variables
    let config = Config::load().map_err(|err| {
        error!("Failed to load configuration: {}", err);
        err
    })?;

    // Log configuration at startup
    config.log();

    // Create the environment
    let mut environment = build_environment(&config);
    if config.mode == Mode::Training {
        info!("Preloading training market data on startup...");
        environment.preload_training_data().await.map_err(|err| {
            error!("Failed to preload training market data: {}", err);
            err
        })?;
        info!("Training market data preload complete");
    }

    // Configure broker gateway if in Production Mode
    if config.mode == Mode::Live {
        info!("Connecting to broker gateway...");

        // Try to create broker gateway connection
        let broker_gateway = broker::try_create_broker_gateway(
            config.broker_gateway.broker_gateway.as_deref(),
            config.broker_gateway.broker_addr.as_deref(),
            config.broker_gateway.ctrader_app_client_id.as_deref(),
            config.broker_gateway.ctrader_app_client_secret.as_deref(),
            config.broker_gateway.ctrader_access_token.as_deref(),
            config.broker_gateway.ctrader_refresh_token.as_deref(),
            config.broker_gateway.ctrader_account.as_deref(),
            config.symbol.as_str(),
        )
        .await
        .map_err(|err| {
            error!("Failed to initialize broker gateway: {}", err);
            err
        })?;

        match broker_gateway {
            Some(bg) => {
                environment = environment.with_broker_gateway(bg);
                info!("Broker gateway connected successfully");
            }
            None => {
                // This shouldn't happen since we checked is_broker_gateway_configured()
                let err = anyhow::anyhow!("Broker gateway not configured but mode is Live");
                error!("Failed to start in live mode: {}", err);
                return Err(err);
            }
        }
    }

    // Create the gRPC service
    let environment_service = EnvironmentService::new(environment);

    // Start the gRPC server
    let addr = config.addr.parse().map_err(|err| {
        error!("Invalid gRPC server address '{}': {}", config.addr, err);
        err
    })?;
    info!("Starting gRPC server on {}", addr);

    Server::builder()
        .add_service(EnvironmentServer::new(environment_service))
        .serve(addr)
        .await
        .map_err(|err| {
            error!("gRPC server exited with error: {}", err);
            err
        })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_environment_applies_reward_config() {
        let mut config = Config::default();
        config.reward_lambda = 2.5;
        config.reward_action_penalty = 0.05;
        config.reward_holding_penalty = 0.0002;

        let environment = build_environment(&config);

        assert_eq!(environment.reward_parameters(), (2.5, 0.05, 0.0002, 0.01));
    }
}
