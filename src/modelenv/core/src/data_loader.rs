// Data loader module for loading price bars from S3 parquet files
use anyhow::{anyhow, Result};
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use chrono::{DateTime, Datelike, Duration, Months, NaiveDate, TimeZone, Timelike, Utc};
use log::info;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs::OpenOptions;
use tokio::process::Command;
use tokio::time::{sleep, Duration as TokioDuration};

use modelenv_proto::{Bar, News, Tick};

use crate::market_data_cache::{CachedLatestSource, MarketDataCache};

/// Supported time intervals for price bars
pub const TIME_INTERVALS: &[&str] = &["M1", "M5", "M15", "H1", "H4", "D1", "W1", "MN"];
pub const DEFAULT_LOCAL_CACHE_DIR: &str = "/tmp/modelenv-cache";
const TRAINING_DATA_BRANCH: &str = "marketdata/interval-price";
const EOH_SNAPSHOT_BRANCH: &str = "marketdata/eoh-snapshot";
const EOD_SNAPSHOT_BRANCH: &str = "marketdata/eod-snapshot";
const EOW_SNAPSHOT_BRANCH: &str = "marketdata/eow-snapshot";
const EOM_SNAPSHOT_BRANCH: &str = "marketdata/eom-snapshot";
const EOD_TICK_SNAPSHOT_BRANCH: &str = "marketdata/eod-tick-snapshot";
const NEWS_DATA_BRANCH: &str = "marketdata/interval-news";
const EOD_NEWS_SNAPSHOT_BRANCH: &str = "marketdata/eod-news-snapshot";
const SOURCE_SELECTION_LOOKBACK_NS: i64 = 31 * 24 * 60 * 60 * 1_000_000_000;
const CACHE_DOWNLOAD_LOCK_STALE_SECS: u64 = 60 * 60;
const CACHE_DOWNLOAD_LOCK_POLL_MS: u64 = 250;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PartitionTier {
    Hour,
    Day,
    Month,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExecutionCadence {
    Hourly,
    Daily,
    Weekly,
    Monthly,
}

#[derive(Clone, Copy, Debug)]
struct IntervalSchedule {
    partition_tier: PartitionTier,
    cadence: ExecutionCadence,
}

fn interval_schedule(interval: &str) -> Result<IntervalSchedule> {
    match interval {
        "M1" | "M5" | "M15" => Ok(IntervalSchedule {
            partition_tier: PartitionTier::Hour,
            cadence: ExecutionCadence::Hourly,
        }),
        "H1" | "H4" | "D1" => Ok(IntervalSchedule {
            partition_tier: PartitionTier::Day,
            cadence: ExecutionCadence::Daily,
        }),
        "W1" => Ok(IntervalSchedule {
            partition_tier: PartitionTier::Month,
            cadence: ExecutionCadence::Weekly,
        }),
        "MN" => Ok(IntervalSchedule {
            partition_tier: PartitionTier::Month,
            cadence: ExecutionCadence::Monthly,
        }),
        other => Err(anyhow!("Unsupported training interval {}", other)),
    }
}

fn interval_storage_name(interval: &str) -> &'static str {
    match interval {
        "M1" => "M1",
        "M5" => "M5",
        "M15" => "M15",
        "H1" => "H1",
        "H4" => "H4",
        "D1" => "D1",
        "W1" => "W1",
        "MN" => "MN1",
        _ => "UNKNOWN",
    }
}

fn news_symbol_storage_name(symbol: &str) -> String {
    let cleaned = symbol.trim().replace('/', "-").to_uppercase();
    if cleaned.contains('-') {
        cleaned
    } else if cleaned.len() == 6 {
        format!("{}-{}", &cleaned[..3], &cleaned[3..])
    } else {
        cleaned
    }
}

fn snapshot_branch(interval: &str) -> &'static str {
    match interval {
        "M1" | "M5" | "M15" => EOH_SNAPSHOT_BRANCH,
        "H1" | "H4" | "D1" => EOD_SNAPSHOT_BRANCH,
        "W1" => EOW_SNAPSHOT_BRANCH,
        "MN" => EOM_SNAPSHOT_BRANCH,
        _ => TRAINING_DATA_BRANCH,
    }
}

fn known_price_data_branches() -> [&'static str; 5] {
    [
        TRAINING_DATA_BRANCH,
        EOH_SNAPSHOT_BRANCH,
        EOD_SNAPSHOT_BRANCH,
        EOW_SNAPSHOT_BRANCH,
        EOM_SNAPSHOT_BRANCH,
    ]
}

/// Normalize the training data prefix to the bucket root or explicit source root.
pub fn normalise_training_data_prefix(prefix: &str) -> String {
    let trimmed = prefix.trim_end_matches('/');
    if trimmed.is_empty() || trimmed.starts_with("file://") || trimmed.ends_with(".parquet") {
        return trimmed.to_string();
    }

    if trimmed.contains("/symbol=") && trimmed.contains("/interval=") {
        trimmed.to_string()
    } else if let Some(branch) = known_price_data_branches()
        .into_iter()
        .find(|branch| trimmed.ends_with(branch))
    {
        trimmed
            .trim_end_matches(branch)
            .trim_end_matches('/')
            .to_string()
    } else {
        trimmed.to_string()
    }
}

/// Build the snapshot parquet prefix for a symbol/interval pair.
pub fn build_interval_data_source(prefix: &str, symbol: &str, time_interval: &str) -> String {
    let base = normalise_training_data_prefix(prefix);
    let storage_interval = interval_storage_name(time_interval);
    if base.ends_with(".parquet") || (base.contains("/symbol=") && base.contains("/interval=")) {
        base
    } else {
        format!(
            "{base}/{}/symbol={symbol}/interval={storage_interval}",
            snapshot_branch(time_interval)
        )
    }
}

pub fn normalise_news_data_prefix(prefix: &str) -> String {
    let trimmed = prefix.trim_end_matches('/');
    if trimmed.is_empty() || trimmed.starts_with("file://") || trimmed.ends_with(".parquet") {
        return trimmed.to_string();
    }

    if trimmed.ends_with(EOD_NEWS_SNAPSHOT_BRANCH)
        || trimmed.contains("/marketdata/eod-news-snapshot/")
    {
        return trimmed.to_string();
    }

    if trimmed.ends_with(NEWS_DATA_BRANCH) || trimmed.contains("/marketdata/interval-news/") {
        return trimmed.replacen(NEWS_DATA_BRANCH, EOD_NEWS_SNAPSHOT_BRANCH, 1);
    }

    if let Some(branch) = known_price_data_branches()
        .into_iter()
        .find(|branch| trimmed.ends_with(branch))
    {
        return format!(
            "{}{}",
            trimmed.trim_end_matches(branch),
            EOD_NEWS_SNAPSHOT_BRANCH
        );
    }

    for branch in known_price_data_branches() {
        let from = format!("/{branch}/");
        if trimmed.contains(&from) {
            return trimmed.replacen(&from, &format!("/{EOD_NEWS_SNAPSHOT_BRANCH}/"), 1);
        }
    }

    format!("{trimmed}/{EOD_NEWS_SNAPSHOT_BRANCH}")
}

pub fn build_news_data_source(prefix: &str, symbol: &str) -> String {
    let base = normalise_news_data_prefix(prefix);
    if base.ends_with(".parquet") {
        base
    } else {
        format!("{base}/symbol={}", news_symbol_storage_name(symbol))
    }
}

pub fn normalise_tick_data_prefix(prefix: &str) -> String {
    let trimmed = prefix.trim_end_matches('/');
    if trimmed.is_empty() || trimmed.starts_with("file://") || trimmed.ends_with(".parquet") {
        return trimmed.to_string();
    }

    if trimmed.ends_with(EOD_TICK_SNAPSHOT_BRANCH)
        || trimmed.contains("/marketdata/eod-tick-snapshot/")
    {
        return trimmed.to_string();
    }

    if trimmed.contains("/symbol=")
        && (trimmed.contains("/interval=ticks")
            || trimmed.contains("/marketdata/eod-tick-snapshot/"))
    {
        return trimmed.to_string();
    }

    if trimmed.ends_with(TRAINING_DATA_BRANCH) || trimmed.contains("/marketdata/interval-price/") {
        return trimmed.replacen(TRAINING_DATA_BRANCH, EOD_TICK_SNAPSHOT_BRANCH, 1);
    }

    if let Some(branch) = known_price_data_branches()
        .into_iter()
        .find(|branch| trimmed.ends_with(branch))
    {
        return format!(
            "{}{}",
            trimmed.trim_end_matches(branch).trim_end_matches('/'),
            format!("/{EOD_TICK_SNAPSHOT_BRANCH}")
        );
    }

    for branch in known_price_data_branches() {
        let from = format!("/{branch}/");
        if trimmed.contains(&from) {
            return trimmed.replacen(&from, &format!("/{EOD_TICK_SNAPSHOT_BRANCH}/"), 1);
        }
    }

    format!("{trimmed}/{EOD_TICK_SNAPSHOT_BRANCH}")
}

pub fn build_tick_data_source(prefix: &str, symbol: &str) -> String {
    let base = normalise_tick_data_prefix(prefix);
    if base.ends_with(".parquet") || base.contains("/symbol=") {
        base
    } else {
        format!("{base}/symbol={symbol}")
    }
}

fn tick_source_schedule(source_uri: &str) -> IntervalSchedule {
    if source_uri.contains(EOD_TICK_SNAPSHOT_BRANCH) {
        IntervalSchedule {
            partition_tier: PartitionTier::Day,
            cadence: ExecutionCadence::Daily,
        }
    } else {
        IntervalSchedule {
            partition_tier: PartitionTier::Hour,
            cadence: ExecutionCadence::Hourly,
        }
    }
}

/// Load price bars from a parquet file (S3 or local file)
pub async fn load_bars_from_parquet(
    s3_uri: &str,
    symbol: &str,
    time_interval: &str,
) -> Result<Vec<Bar>> {
    load_bars_from_parquet_with_range_from_local_cache_dir(
        DEFAULT_LOCAL_CACHE_DIR,
        s3_uri,
        symbol,
        time_interval,
        None,
        None,
        None,
    )
    .await
}

/// Load price bars from a parquet file, stopping at the given end timestamp
pub async fn load_bars_from_parquet_with_end_ts(
    s3_uri: &str,
    symbol: &str,
    time_interval: &str,
    end_timestamp_ns: i64,
) -> Result<Vec<Bar>> {
    let end_timestamp_ns = if end_timestamp_ns > 0 {
        Some(end_timestamp_ns)
    } else {
        None
    };
    load_bars_from_parquet_with_range_from_local_cache_dir(
        DEFAULT_LOCAL_CACHE_DIR,
        s3_uri,
        symbol,
        time_interval,
        None,
        None,
        end_timestamp_ns,
    )
    .await
}

pub async fn load_bars_from_parquet_with_range_cached(
    cache: &MarketDataCache,
    source_uri: &str,
    symbol: &str,
    time_interval: &str,
    snapshot_selection_timestamp_ns: Option<i64>,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> Result<Vec<Bar>> {
    load_bars_from_parquet_with_range_cached_from_local_cache_dir(
        DEFAULT_LOCAL_CACHE_DIR,
        cache,
        source_uri,
        symbol,
        time_interval,
        snapshot_selection_timestamp_ns,
        start_timestamp_ns,
        end_timestamp_ns,
    )
    .await
}

pub(crate) async fn load_bars_from_parquet_with_range_cached_from_local_cache_dir(
    local_cache_dir: &str,
    cache: &MarketDataCache,
    source_uri: &str,
    symbol: &str,
    time_interval: &str,
    snapshot_selection_timestamp_ns: Option<i64>,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> Result<Vec<Bar>> {
    let sources = if source_uri.ends_with(".parquet") {
        vec![source_uri.to_string()]
    } else if source_uri.starts_with("s3://") && !source_uri.ends_with(".parquet") {
        determine_price_snapshot_s3_sources_cached(
            local_cache_dir,
            Some(cache),
            source_uri,
            interval_schedule(time_interval)?,
            snapshot_selection_timestamp_ns,
        )
        .await?
    } else if let Some(cached_sources) =
        cached_price_snapshot_sources(cache, source_uri, snapshot_selection_timestamp_ns).await?
    {
        cached_sources
    } else {
        let selected = select_price_snapshot_sources(
            list_parquet_sources(source_uri).await?,
            snapshot_selection_timestamp_ns,
        )?;
        cache_price_snapshot_selection(
            cache,
            source_uri,
            snapshot_selection_timestamp_ns,
            &selected,
        )
        .await;
        selected
    };

    info!(
        "Resolved parquet source(s) for {} {}: {}",
        symbol,
        time_interval,
        sources.join(", ")
    );

    collect_bars_from_sources(
        local_cache_dir,
        Some(cache),
        sources,
        symbol,
        time_interval,
        start_timestamp_ns,
        end_timestamp_ns,
    )
    .await
}

pub(crate) async fn load_ticks_from_parquet_with_range_cached_from_local_cache_dir(
    local_cache_dir: &str,
    cache: &MarketDataCache,
    source_uri: &str,
    symbol: &str,
    snapshot_selection_timestamp_ns: Option<i64>,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> Result<Vec<Tick>> {
    let schedule = tick_source_schedule(source_uri);
    let sources = if let Some(snapshot_selection_timestamp_ns) = snapshot_selection_timestamp_ns {
        if source_uri.ends_with(".parquet") {
            vec![source_uri.to_string()]
        } else if source_uri.starts_with("s3://") && !source_uri.ends_with(".parquet") {
            determine_price_snapshot_s3_sources_cached(
                local_cache_dir,
                Some(cache),
                source_uri,
                schedule,
                Some(snapshot_selection_timestamp_ns),
            )
            .await?
        } else if let Some(cached_sources) =
            cached_price_snapshot_sources(cache, source_uri, Some(snapshot_selection_timestamp_ns))
                .await?
        {
            cached_sources
        } else {
            let speculative_source = speculative_price_snapshot_source_uri(
                source_uri,
                schedule,
                snapshot_selection_timestamp_ns,
            );
            let speculative_path = Path::new(
                speculative_source
                    .strip_prefix("file://")
                    .unwrap_or(&speculative_source),
            );
            if !speculative_path.exists() {
                return Err(anyhow!(
                    "Exact parquet tick key {} for configured snapshot timestamp {} ({}) was not found",
                    speculative_source,
                    snapshot_selection_timestamp_ns,
                    format_timestamp_ns(snapshot_selection_timestamp_ns)
                ));
            }
            let selected = vec![speculative_source];
            cache_price_snapshot_selection(
                cache,
                source_uri,
                Some(snapshot_selection_timestamp_ns),
                &selected,
            )
            .await;
            selected
        }
    } else {
        let local_cached_sources =
            if source_uri.starts_with("s3://") && !source_uri.ends_with(".parquet") {
                list_existing_local_cached_s3_sources(local_cache_dir, source_uri)?
            } else {
                None
            };
        if let Some(local_sources) = local_cached_sources {
            let selected = if let Some(start_timestamp_ns) = start_timestamp_ns {
                let effective_end_ns = end_timestamp_ns.unwrap_or_else(now_ns);
                if effective_end_ns < start_timestamp_ns {
                    return Err(anyhow!(
                        "Invalid time range for {}: start {} is after end {}",
                        source_uri,
                        start_timestamp_ns,
                        effective_end_ns
                    ));
                }
                select_candidate_sources_with_range_details(
                    local_sources,
                    source_uri,
                    schedule,
                    "ticks",
                    Some(start_timestamp_ns),
                    Some(effective_end_ns),
                )?
            } else if let Some(end_timestamp_ns) = end_timestamp_ns {
                let selected = select_candidate_sources_with_range_details(
                    local_sources,
                    source_uri,
                    schedule,
                    "ticks",
                    None,
                    Some(end_timestamp_ns),
                )?;
                vec![selected.last().cloned().ok_or_else(|| {
                    anyhow!("No parquet sources matched the requested time range for ticks")
                })?]
            } else {
                select_candidate_sources(
                    local_sources,
                    "ticks",
                    None,
                    None,
                    Some(source_uri),
                    Some(schedule),
                )?
            };
            cache_latest_selection(
                cache,
                source_uri,
                start_timestamp_ns,
                end_timestamp_ns,
                &selected,
            )
            .await;
            selected
        } else if let Some(cached_sources) =
            cached_latest_sources(cache, source_uri, start_timestamp_ns, end_timestamp_ns).await?
        {
            cached_sources
        } else if source_uri.starts_with("s3://") && !source_uri.ends_with(".parquet") {
            determine_s3_sources_cached(
                local_cache_dir,
                Some(cache),
                source_uri,
                schedule,
                start_timestamp_ns,
                end_timestamp_ns,
            )
            .await?
        } else {
            let selected = select_candidate_sources_with_range_details(
                list_parquet_sources(source_uri).await?,
                source_uri,
                schedule,
                "ticks",
                start_timestamp_ns,
                end_timestamp_ns,
            )?;
            cache_latest_selection(
                cache,
                source_uri,
                start_timestamp_ns,
                end_timestamp_ns,
                &selected,
            )
            .await;
            selected
        }
    };

    collect_ticks_from_sources(
        local_cache_dir,
        cache,
        sources,
        symbol,
        start_timestamp_ns,
        end_timestamp_ns,
    )
    .await
}

/// Load price bars from a snapshot parquet source, optionally constrained by time range.
pub async fn load_bars_from_parquet_with_range(
    source_uri: &str,
    symbol: &str,
    time_interval: &str,
    snapshot_selection_timestamp_ns: Option<i64>,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> Result<Vec<Bar>> {
    load_bars_from_parquet_with_range_from_local_cache_dir(
        DEFAULT_LOCAL_CACHE_DIR,
        source_uri,
        symbol,
        time_interval,
        snapshot_selection_timestamp_ns,
        start_timestamp_ns,
        end_timestamp_ns,
    )
    .await
}

pub(crate) async fn load_bars_from_parquet_with_range_from_local_cache_dir(
    local_cache_dir: &str,
    source_uri: &str,
    symbol: &str,
    time_interval: &str,
    snapshot_selection_timestamp_ns: Option<i64>,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> Result<Vec<Bar>> {
    let sources = if source_uri.ends_with(".parquet") {
        vec![source_uri.to_string()]
    } else if source_uri.starts_with("s3://") && !source_uri.ends_with(".parquet") {
        determine_price_snapshot_s3_sources_cached(
            local_cache_dir,
            None,
            source_uri,
            interval_schedule(time_interval)?,
            snapshot_selection_timestamp_ns,
        )
        .await?
    } else {
        select_price_snapshot_sources(
            list_parquet_sources(source_uri).await?,
            snapshot_selection_timestamp_ns,
        )?
    };

    collect_bars_from_sources(
        local_cache_dir,
        None,
        sources,
        symbol,
        time_interval,
        start_timestamp_ns,
        end_timestamp_ns,
    )
    .await
}

async fn collect_bars_from_sources(
    local_cache_dir: &str,
    cache: Option<&MarketDataCache>,
    sources: Vec<String>,
    symbol: &str,
    time_interval: &str,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> Result<Vec<Bar>> {
    let mut bars = Vec::new();
    for source in sources {
        if let Some(cache) = cache {
            if let Some(cached_bars) = cache.price_bars(&source).await {
                bars.extend(cached_bars);
                continue;
            }
        }

        let Some(bytes) = try_read_bytes_from_source(local_cache_dir, &source).await? else {
            continue;
        };
        let parsed_bars = parse_bars_from_bytes(bytes, symbol, time_interval, i64::MAX)?;
        if let Some(cache) = cache {
            cache
                .put_price_bars(source.clone(), parsed_bars.clone())
                .await;
        }
        bars.extend(parsed_bars);
    }

    if let Some(start_timestamp_ns) = start_timestamp_ns {
        bars.retain(|bar| bar.timestamp_ns >= start_timestamp_ns);
    }

    if let Some(end_timestamp_ns) = end_timestamp_ns {
        bars.retain(|bar| bar.timestamp_ns <= end_timestamp_ns);
    }

    bars.sort_by_key(|bar| bar.timestamp_ns);

    if bars.is_empty() {
        return Err(anyhow!(
            "Parquet sources contain no bars for {} {}",
            symbol,
            time_interval
        ));
    }

    Ok(bars)
}

pub async fn load_news_from_parquet_with_range_cached(
    cache: &MarketDataCache,
    source_uri: &str,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> Result<Vec<News>> {
    load_news_from_parquet_with_range_cached_from_local_cache_dir(
        DEFAULT_LOCAL_CACHE_DIR,
        cache,
        source_uri,
        start_timestamp_ns,
        end_timestamp_ns,
    )
    .await
}

pub(crate) async fn load_news_from_parquet_with_range_cached_from_local_cache_dir(
    local_cache_dir: &str,
    cache: &MarketDataCache,
    source_uri: &str,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> Result<Vec<News>> {
    let schedule = IntervalSchedule {
        partition_tier: PartitionTier::Day,
        cadence: ExecutionCadence::Daily,
    };
    let local_cached_sources =
        if source_uri.starts_with("s3://") && !source_uri.ends_with(".parquet") {
            list_existing_local_cached_s3_sources(local_cache_dir, source_uri)?
        } else {
            None
        };
    let sources = if let Some(local_sources) = local_cached_sources {
        let selected = if let Some(start_timestamp_ns) = start_timestamp_ns {
            let effective_end_ns = end_timestamp_ns.unwrap_or_else(now_ns);
            if effective_end_ns < start_timestamp_ns {
                return Err(anyhow!(
                    "Invalid time range for {}: start {} is after end {}",
                    source_uri,
                    start_timestamp_ns,
                    effective_end_ns
                ));
            }
            select_candidate_sources(
                local_sources,
                "D1",
                Some(start_timestamp_ns),
                Some(effective_end_ns),
                Some(source_uri),
                Some(schedule),
            )?
        } else if let Some(end_timestamp_ns) = end_timestamp_ns {
            select_candidate_sources(
                local_sources,
                "D1",
                None,
                Some(end_timestamp_ns),
                Some(source_uri),
                Some(schedule),
            )?
        } else {
            select_candidate_sources(
                local_sources,
                "D1",
                None,
                None,
                Some(source_uri),
                Some(schedule),
            )?
        };
        cache_latest_selection(
            cache,
            source_uri,
            start_timestamp_ns,
            end_timestamp_ns,
            &selected,
        )
        .await;
        selected
    } else if let Some(cached_sources) =
        cached_latest_sources(cache, source_uri, start_timestamp_ns, end_timestamp_ns).await?
    {
        cached_sources
    } else if source_uri.starts_with("s3://") && !source_uri.ends_with(".parquet") {
        determine_s3_sources_cached(
            local_cache_dir,
            Some(cache),
            source_uri,
            schedule,
            start_timestamp_ns,
            end_timestamp_ns,
        )
        .await?
    } else {
        let selected = select_candidate_sources(
            list_parquet_sources(source_uri).await?,
            "D1",
            start_timestamp_ns,
            end_timestamp_ns,
            Some(source_uri),
            Some(schedule),
        )?;
        cache_latest_selection(
            cache,
            source_uri,
            start_timestamp_ns,
            end_timestamp_ns,
            &selected,
        )
        .await;
        selected
    };

    collect_news_from_sources(
        local_cache_dir,
        cache,
        sources,
        start_timestamp_ns,
        end_timestamp_ns,
    )
    .await
}

async fn collect_news_from_sources(
    local_cache_dir: &str,
    cache: &MarketDataCache,
    sources: Vec<String>,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> Result<Vec<News>> {
    let mut news = Vec::new();
    for source in sources {
        if let Some(cached_news) = cache.news_items(&source).await {
            news.extend(cached_news);
            continue;
        }

        let Some(bytes) = try_read_bytes_from_source(local_cache_dir, &source).await? else {
            continue;
        };
        let parsed_news = parse_news_from_bytes(bytes, i64::MAX)?;
        cache
            .put_news_items(source.clone(), parsed_news.clone())
            .await;
        news.extend(parsed_news);
    }

    if let Some(start_timestamp_ns) = start_timestamp_ns {
        news.retain(|item| item.timestamp_ns >= start_timestamp_ns);
    }

    if let Some(end_timestamp_ns) = end_timestamp_ns {
        news.retain(|item| item.timestamp_ns <= end_timestamp_ns);
    }

    news.sort_by_key(|item| item.timestamp_ns);
    Ok(news)
}

async fn collect_ticks_from_sources(
    local_cache_dir: &str,
    cache: &MarketDataCache,
    sources: Vec<String>,
    symbol: &str,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> Result<Vec<Tick>> {
    let mut ticks = Vec::new();
    for source in sources {
        if let Some(cached_ticks) = cache.tick_items(&source).await {
            ticks.extend(cached_ticks);
            continue;
        }

        let Some(bytes) = try_read_bytes_from_source(local_cache_dir, &source).await? else {
            continue;
        };
        let parsed_ticks = parse_ticks_from_bytes(bytes, symbol, i64::MAX)?;
        cache
            .put_tick_items(source.clone(), parsed_ticks.clone())
            .await;
        ticks.extend(parsed_ticks);
    }

    if let Some(start_timestamp_ns) = start_timestamp_ns {
        ticks.retain(|item| item.timestamp_ns >= start_timestamp_ns);
    }

    if let Some(end_timestamp_ns) = end_timestamp_ns {
        ticks.retain(|item| item.timestamp_ns <= end_timestamp_ns);
    }

    ticks.sort_by_key(|item| item.timestamp_ns);
    ticks.dedup_by(|left, right| {
        left.timestamp_ns == right.timestamp_ns
            && left.bid.to_bits() == right.bid.to_bits()
            && left.ask.to_bits() == right.ask.to_bits()
    });
    Ok(ticks)
}

async fn cached_latest_sources(
    cache: &MarketDataCache,
    source_uri: &str,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> Result<Option<Vec<String>>> {
    if start_timestamp_ns.is_some() || end_timestamp_ns.is_some() {
        return Ok(None);
    }

    if let Some(cached_latest) = cache.latest_source(source_uri).await {
        return match cached_latest {
            CachedLatestSource::Present(uri) => Ok(Some(vec![uri])),
            CachedLatestSource::Missing(message) => Err(anyhow!(message)),
        };
    }

    Ok(None)
}

async fn cache_latest_selection(
    cache: &MarketDataCache,
    source_uri: &str,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
    selected: &[String],
) {
    if start_timestamp_ns.is_none() && end_timestamp_ns.is_none() && selected.len() == 1 {
        cache
            .put_latest_source(
                source_uri.to_string(),
                CachedLatestSource::Present(selected[0].clone()),
            )
            .await;
    }
}

fn price_snapshot_cache_key(
    source_uri: &str,
    snapshot_selection_timestamp_ns: Option<i64>,
) -> String {
    match snapshot_selection_timestamp_ns {
        Some(value) => format!("{source_uri}#snapshot-ts={value}"),
        None => source_uri.to_string(),
    }
}

async fn cached_price_snapshot_sources(
    cache: &MarketDataCache,
    source_uri: &str,
    snapshot_selection_timestamp_ns: Option<i64>,
) -> Result<Option<Vec<String>>> {
    let cache_key = price_snapshot_cache_key(source_uri, snapshot_selection_timestamp_ns);
    if let Some(cached_latest) = cache.latest_source(&cache_key).await {
        return match cached_latest {
            CachedLatestSource::Present(uri) => Ok(Some(vec![uri])),
            CachedLatestSource::Missing(message) => Err(anyhow!(message)),
        };
    }

    Ok(None)
}

async fn cache_price_snapshot_selection(
    cache: &MarketDataCache,
    source_uri: &str,
    snapshot_selection_timestamp_ns: Option<i64>,
    selected: &[String],
) {
    if selected.len() == 1 {
        cache
            .put_latest_source(
                price_snapshot_cache_key(source_uri, snapshot_selection_timestamp_ns),
                CachedLatestSource::Present(selected[0].clone()),
            )
            .await;
    }
}

fn parse_bars_from_bytes(
    bytes: Bytes,
    symbol: &str,
    time_interval: &str,
    end_timestamp_ns: i64,
) -> Result<Vec<Bar>> {
    let mut parquet_reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .map_err(|e| anyhow!("Failed to create parquet reader: {}", e))?
        .build()
        .map_err(|e| anyhow!("Failed to build parquet reader: {}", e))?;

    let mut bars = Vec::new();

    while let Some(batch) = parquet_reader.next() {
        let batch = batch.map_err(|e| anyhow!("Failed to read batch: {}", e))?;

        for i in 0..batch.num_rows() {
            let bar = parse_bar_from_batch(&batch, i)?;

            // Stop loading if we've passed the end timestamp
            if bar.timestamp_ns > end_timestamp_ns {
                return Ok(bars);
            }

            bars.push(bar);
        }
    }

    if bars.is_empty() {
        return Err(anyhow!(
            "Parquet file contains no bars for {} {}",
            symbol,
            time_interval
        ));
    }

    Ok(bars)
}

async fn list_parquet_sources(source_uri: &str) -> Result<Vec<String>> {
    let local_path = source_uri.strip_prefix("file://").unwrap_or(source_uri);
    list_local_parquet_sources(Path::new(local_path))
}

fn list_existing_local_cached_s3_sources(
    local_cache_dir: &str,
    source_uri: &str,
) -> Result<Option<Vec<String>>> {
    let local_path = local_cache_path_for_s3_source(local_cache_dir, source_uri)?;
    if local_path.is_file() {
        return Ok(Some(vec![local_path.to_string_lossy().to_string()]));
    }

    if !local_path.exists() {
        return Ok(None);
    }

    let mut files = Vec::new();
    collect_local_parquet_files(&local_path, &mut files)?;
    if files.is_empty() {
        return Ok(None);
    }
    files.sort();
    Ok(Some(files))
}

fn list_local_parquet_sources(path: &Path) -> Result<Vec<String>> {
    if path.is_file() {
        return Ok(vec![path.to_string_lossy().to_string()]);
    }

    if !path.exists() {
        return Err(anyhow!(
            "Local parquet path does not exist: {}",
            path.display()
        ));
    }

    let mut files = Vec::new();
    collect_local_parquet_files(path, &mut files)?;
    if files.is_empty() {
        return Err(anyhow!(
            "No parquet files found under local path {}",
            path.display()
        ));
    }
    files.sort();
    Ok(files)
}

fn collect_local_parquet_files(path: &Path, files: &mut Vec<String>) -> Result<()> {
    for entry in std::fs::read_dir(path)
        .map_err(|e| anyhow!("Failed to read local directory {}: {}", path.display(), e))?
    {
        let entry = entry.map_err(|e| anyhow!("Failed to read directory entry: {}", e))?;
        let entry_path = entry.path();
        if entry_path.is_dir() {
            collect_local_parquet_files(&entry_path, files)?;
        } else if entry_path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("parquet"))
        {
            files.push(entry_path.to_string_lossy().to_string());
        }
    }
    Ok(())
}

async fn determine_price_snapshot_s3_sources_cached(
    local_cache_dir: &str,
    cache: Option<&MarketDataCache>,
    source_uri: &str,
    schedule: IntervalSchedule,
    snapshot_selection_timestamp_ns: Option<i64>,
) -> Result<Vec<String>> {
    let cache_key = price_snapshot_cache_key(source_uri, snapshot_selection_timestamp_ns);

    if let Some(snapshot_selection_timestamp_ns) = snapshot_selection_timestamp_ns {
        if let Some(cache) = cache {
            if let Some(cached_latest) = cache.latest_source(&cache_key).await {
                return match cached_latest {
                    CachedLatestSource::Present(uri) => Ok(vec![uri]),
                    CachedLatestSource::Missing(message) => Err(anyhow!(message)),
                };
            }
        }

        let speculative_source = speculative_price_snapshot_source_uri(
            source_uri,
            schedule,
            snapshot_selection_timestamp_ns,
        );
        info!(
            "Resolving timestamped parquet snapshot for {} at {} via {}",
            source_uri,
            format_timestamp_ns(snapshot_selection_timestamp_ns),
            speculative_source
        );
        match resolve_speculative_price_snapshot_source(local_cache_dir, speculative_source.clone())
            .await?
        {
            true => {
                if let Some(cache) = cache {
                    cache_price_snapshot_selection(
                        cache,
                        source_uri,
                        Some(snapshot_selection_timestamp_ns),
                        std::slice::from_ref(&speculative_source),
                    )
                    .await;
                }
                return Ok(vec![speculative_source]);
            }
            false => {
                let err = anyhow!(
                    "Exact parquet snapshot key {} for configured snapshot timestamp {} ({}) was not found in local cache or S3",
                    speculative_source,
                    snapshot_selection_timestamp_ns,
                    format_timestamp_ns(snapshot_selection_timestamp_ns)
                );
                if let Some(cache) = cache {
                    cache
                        .put_latest_source(cache_key, CachedLatestSource::Missing(err.to_string()))
                        .await;
                }
                return Err(err);
            }
        }
    } else if let Some(local_sources) =
        list_existing_local_cached_s3_sources(local_cache_dir, source_uri)?
    {
        let selected =
            select_price_snapshot_sources(local_sources, snapshot_selection_timestamp_ns)?;
        if let Some(cache) = cache {
            cache_price_snapshot_selection(
                cache,
                source_uri,
                snapshot_selection_timestamp_ns,
                &selected,
            )
            .await;
        }
        return Ok(selected);
    }

    if let Some(cache) = cache {
        if let Some(cached_latest) = cache.latest_source(&cache_key).await {
            return match cached_latest {
                CachedLatestSource::Present(uri) => Ok(vec![uri]),
                CachedLatestSource::Missing(message) => Err(anyhow!(message)),
            };
        }
    }

    let latest = find_latest_s3_source(source_uri, schedule).await;
    if let Some(cache) = cache {
        let cached_value = match &latest {
            Ok(uri) => CachedLatestSource::Present(uri.clone()),
            Err(err) => CachedLatestSource::Missing(err.to_string()),
        };
        cache.put_latest_source(cache_key, cached_value).await;
    }

    Ok(vec![latest?])
}

async fn resolve_speculative_price_snapshot_source(
    local_cache_dir: &str,
    speculative_source: String,
) -> Result<bool> {
    let local_path = local_cache_path_for_s3_source(local_cache_dir, &speculative_source)?;
    if local_path.exists() {
        return Ok(true);
    }

    if s3_object_exists(&speculative_source).await? {
        return Ok(true);
    }

    Ok(false)
}

fn speculative_price_snapshot_source_uri(
    source_uri: &str,
    schedule: IntervalSchedule,
    snapshot_selection_timestamp_ns: i64,
) -> String {
    build_s3_object_uri(
        source_uri,
        schedule,
        align_down_execution_dt(
            Utc.timestamp_nanos(snapshot_selection_timestamp_ns),
            schedule.cadence,
        ),
    )
}

async fn determine_s3_sources_cached(
    local_cache_dir: &str,
    cache: Option<&MarketDataCache>,
    source_uri: &str,
    schedule: IntervalSchedule,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> Result<Vec<String>> {
    if source_uri.ends_with(".parquet") {
        return Ok(vec![source_uri.to_string()]);
    }

    if let Some(local_sources) = list_existing_local_cached_s3_sources(local_cache_dir, source_uri)?
    {
        if let Some(start_timestamp_ns) = start_timestamp_ns {
            let effective_end_ns = end_timestamp_ns.unwrap_or_else(now_ns);
            if effective_end_ns < start_timestamp_ns {
                return Err(anyhow!(
                    "Invalid time range for {}: start {} is after end {}",
                    source_uri,
                    start_timestamp_ns,
                    effective_end_ns
                ));
            }

            return select_candidate_sources(
                local_sources,
                "D1",
                Some(start_timestamp_ns),
                Some(effective_end_ns),
                Some(source_uri),
                Some(schedule),
            );
        }

        if let Some(end_timestamp_ns) = end_timestamp_ns {
            return select_candidate_sources(
                local_sources,
                "D1",
                None,
                Some(end_timestamp_ns),
                Some(source_uri),
                Some(schedule),
            );
        }

        return select_candidate_sources(
            local_sources,
            "D1",
            None,
            None,
            Some(source_uri),
            Some(schedule),
        );
    }

    if let Some(start_timestamp_ns) = start_timestamp_ns {
        let effective_end_ns = end_timestamp_ns.unwrap_or_else(now_ns);
        if effective_end_ns < start_timestamp_ns {
            return Err(anyhow!(
                "Invalid time range for {}: start {} is after end {}",
                source_uri,
                start_timestamp_ns,
                effective_end_ns
            ));
        }

        return Ok(generate_s3_sources_for_range(
            source_uri,
            schedule,
            start_timestamp_ns,
            effective_end_ns,
        ));
    }

    if let Some(end_timestamp_ns) = end_timestamp_ns {
        return Ok(vec![
            find_latest_s3_source_at_or_before(source_uri, schedule, end_timestamp_ns).await?,
        ]);
    }

    if let Some(cache) = cache {
        if let Some(cached_latest) = cache.latest_source(source_uri).await {
            return match cached_latest {
                CachedLatestSource::Present(uri) => Ok(vec![uri]),
                CachedLatestSource::Missing(message) => Err(anyhow!(message)),
            };
        }
    }

    let latest = find_latest_s3_source(source_uri, schedule).await;
    if let Some(cache) = cache {
        let cached_value = match &latest {
            Ok(uri) => CachedLatestSource::Present(uri.clone()),
            Err(err) => CachedLatestSource::Missing(err.to_string()),
        };
        cache
            .put_latest_source(source_uri.to_string(), cached_value)
            .await;
    }

    Ok(vec![latest?])
}

fn select_price_snapshot_sources(
    sources: Vec<String>,
    snapshot_selection_timestamp_ns: Option<i64>,
) -> Result<Vec<String>> {
    if sources.is_empty() {
        return Err(anyhow!("No parquet snapshot sources available"));
    }

    let mut stamped_sources: Vec<(Option<i64>, String)> = sources
        .into_iter()
        .map(|source| (extract_source_timestamp_ns(&source), source))
        .collect();
    stamped_sources.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    let candidates_summary = describe_stamped_sources(&stamped_sources);

    let selected = if let Some(snapshot_selection_timestamp_ns) = snapshot_selection_timestamp_ns {
        stamped_sources
            .iter()
            .rev()
            .find(|(timestamp_ns, _)| {
                timestamp_ns.is_some_and(|timestamp| timestamp <= snapshot_selection_timestamp_ns)
            })
            .map(|(_, source)| source.clone())
    } else {
        stamped_sources.last().map(|(_, source)| source.clone())
    };

    selected.map(|source| vec![source]).ok_or_else(|| {
        if let Some(snapshot_selection_timestamp_ns) = snapshot_selection_timestamp_ns {
            anyhow!(
                "No parquet snapshot matched configured snapshot timestamp {} ({}). Candidate snapshots: {}",
                snapshot_selection_timestamp_ns,
                format_timestamp_ns(snapshot_selection_timestamp_ns),
                candidates_summary
            )
        } else {
            anyhow!(
                "No parquet snapshot matched the configured snapshot timestamp. Candidate snapshots: {}",
                candidates_summary
            )
        }
    })
}

fn describe_stamped_sources(stamped_sources: &[(Option<i64>, String)]) -> String {
    const MAX_CANDIDATES: usize = 8;

    let mut described = stamped_sources
        .iter()
        .take(MAX_CANDIDATES)
        .map(|(timestamp_ns, source)| match timestamp_ns {
            Some(timestamp_ns) => format!("{} ({})", source, format_timestamp_ns(*timestamp_ns)),
            None => format!("{source} (timestamp: unknown)"),
        })
        .collect::<Vec<_>>();

    if stamped_sources.len() > MAX_CANDIDATES {
        described.push(format!(
            "... and {} more",
            stamped_sources.len() - MAX_CANDIDATES
        ));
    }

    described.join(", ")
}

fn describe_requested_time_range(
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> String {
    match (start_timestamp_ns, end_timestamp_ns) {
        (Some(start), Some(end)) => format!(
            "{} ({}) to {} ({})",
            start,
            format_timestamp_ns(start),
            end,
            format_timestamp_ns(end)
        ),
        (Some(start), None) => format!("from {} ({}) onward", start, format_timestamp_ns(start)),
        (None, Some(end)) => format!("up to {} ({})", end, format_timestamp_ns(end)),
        (None, None) => "latest available parquet source".to_string(),
    }
}

fn describe_expected_source_range(
    source_uri: &str,
    schedule: IntervalSchedule,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> String {
    let Some(start_or_end) = start_timestamp_ns.or(end_timestamp_ns) else {
        return format!("latest available parquet source under {}", source_uri);
    };
    let end_or_start = end_timestamp_ns
        .or(start_timestamp_ns)
        .unwrap_or(start_or_end);

    let start_source = build_s3_object_uri(
        source_uri,
        schedule,
        align_down_execution_dt(Utc.timestamp_nanos(start_or_end), schedule.cadence),
    );
    let end_source = build_s3_object_uri(
        source_uri,
        schedule,
        align_up_execution_dt(Utc.timestamp_nanos(end_or_start), schedule.cadence),
    );

    if start_source == end_source {
        start_source
    } else {
        format!("{start_source} .. {end_source}")
    }
}

fn select_candidate_sources_with_range_details(
    sources: Vec<String>,
    source_uri: &str,
    schedule: IntervalSchedule,
    time_interval: &str,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
) -> Result<Vec<String>> {
    if sources.is_empty() {
        return Err(anyhow!(
            "No parquet sources available for {}",
            time_interval
        ));
    }

    let mut stamped_sources: Vec<(Option<i64>, String)> = sources
        .into_iter()
        .map(|source| (extract_source_timestamp_ns(&source), source))
        .collect();
    stamped_sources.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

    if start_timestamp_ns.is_none() && end_timestamp_ns.is_none() {
        return Ok(vec![stamped_sources
            .last()
            .map(|(_, source)| source.clone())
            .ok_or_else(|| {
                anyhow!("No parquet sources available for {}", time_interval)
            })?]);
    }

    let selected: Vec<String> = stamped_sources
        .iter()
        .filter(|(timestamp_ns, _)| match timestamp_ns {
            Some(timestamp_ns) => {
                let satisfies_start = start_timestamp_ns
                    .map(|start| *timestamp_ns + SOURCE_SELECTION_LOOKBACK_NS >= start)
                    .unwrap_or(true);
                let satisfies_end = end_timestamp_ns
                    .map(|end| *timestamp_ns <= end)
                    .unwrap_or(true);
                satisfies_start && satisfies_end
            }
            None => true,
        })
        .map(|(_, source)| source.clone())
        .collect();

    if selected.is_empty() {
        return Err(anyhow!(
            "No parquet sources matched the requested time range for {} under {}. Requested range: {}. Expected parquet partitions: {}. Candidate sources: {}",
            time_interval,
            source_uri,
            describe_requested_time_range(start_timestamp_ns, end_timestamp_ns),
            describe_expected_source_range(source_uri, schedule, start_timestamp_ns, end_timestamp_ns),
            describe_stamped_sources(&stamped_sources)
        ));
    }

    Ok(selected)
}

fn format_timestamp_ns(timestamp_ns: i64) -> String {
    Utc.timestamp_nanos(timestamp_ns).to_rfc3339()
}

fn generate_s3_sources_for_range(
    source_uri: &str,
    schedule: IntervalSchedule,
    start_timestamp_ns: i64,
    end_timestamp_ns: i64,
) -> Vec<String> {
    let start_dt =
        align_down_execution_dt(Utc.timestamp_nanos(start_timestamp_ns), schedule.cadence);
    let end_dt = align_up_execution_dt(Utc.timestamp_nanos(end_timestamp_ns), schedule.cadence);

    let mut sources = Vec::new();
    let mut cursor = start_dt;
    while cursor <= end_dt {
        sources.push(build_s3_object_uri(source_uri, schedule, cursor));
        cursor = advance_execution_dt(cursor, schedule.cadence);
    }
    sources
}

async fn find_latest_s3_source(source_uri: &str, schedule: IntervalSchedule) -> Result<String> {
    let (bucket, root_prefix) = parse_s3_uri(source_uri)?;
    let root_prefix = with_trailing_slash(&root_prefix);

    let year_prefix = latest_child_prefix(&bucket, &root_prefix).await?;
    let month_prefix = latest_child_prefix(&bucket, &year_prefix).await?;

    let leaf_prefix = match schedule.partition_tier {
        PartitionTier::Hour => {
            let day_prefix = latest_child_prefix(&bucket, &month_prefix).await?;
            latest_child_prefix(&bucket, &day_prefix).await?
        }
        PartitionTier::Day => latest_child_prefix(&bucket, &month_prefix).await?,
        PartitionTier::Month => month_prefix,
    };

    latest_object_under_prefix(&bucket, &leaf_prefix).await
}

async fn find_latest_s3_source_at_or_before(
    source_uri: &str,
    schedule: IntervalSchedule,
    end_timestamp_ns: i64,
) -> Result<String> {
    let mut cursor =
        align_down_execution_dt(Utc.timestamp_nanos(end_timestamp_ns), schedule.cadence);
    let earliest_supported = Utc
        .with_ymd_and_hms(2010, 1, 1, 0, 0, 0)
        .single()
        .ok_or_else(|| anyhow!("Failed to construct earliest supported timestamp"))?;

    while cursor >= earliest_supported {
        let candidate = build_s3_object_uri(source_uri, schedule, cursor);
        if s3_object_exists(&candidate).await? {
            return Ok(candidate);
        }
        cursor = rewind_execution_dt(cursor, schedule.cadence);
    }

    Err(anyhow!(
        "No parquet files found under S3 prefix {} at or before {}",
        source_uri,
        end_timestamp_ns
    ))
}

fn build_s3_object_uri(
    source_uri: &str,
    schedule: IntervalSchedule,
    execution_dt: DateTime<Utc>,
) -> String {
    let base = source_uri.trim_end_matches('/');
    let timestamp = execution_dt.format("%Y%m%dT%H%M%SZ");
    match schedule.partition_tier {
        PartitionTier::Hour => format!(
            "{base}/year={}/month={:02}/day={:02}/hour={:02}/{}.parquet",
            execution_dt.year(),
            execution_dt.month(),
            execution_dt.day(),
            execution_dt.hour(),
            timestamp
        ),
        PartitionTier::Day => format!(
            "{base}/year={}/month={:02}/day={:02}/{}.parquet",
            execution_dt.year(),
            execution_dt.month(),
            execution_dt.day(),
            timestamp
        ),
        PartitionTier::Month => format!(
            "{base}/year={}/month={:02}/{}.parquet",
            execution_dt.year(),
            execution_dt.month(),
            timestamp
        ),
    }
}

fn align_down_execution_dt(dt: DateTime<Utc>, cadence: ExecutionCadence) -> DateTime<Utc> {
    match cadence {
        ExecutionCadence::Hourly => Utc
            .with_ymd_and_hms(dt.year(), dt.month(), dt.day(), dt.hour(), 0, 0)
            .single()
            .expect("valid hourly truncation"),
        ExecutionCadence::Daily => Utc
            .with_ymd_and_hms(dt.year(), dt.month(), dt.day(), 0, 0, 0)
            .single()
            .expect("valid daily truncation"),
        ExecutionCadence::Weekly => {
            let midnight = Utc
                .with_ymd_and_hms(dt.year(), dt.month(), dt.day(), 0, 0, 0)
                .single()
                .expect("valid weekly truncation");
            midnight - Duration::days(dt.weekday().num_days_from_monday() as i64)
        }
        ExecutionCadence::Monthly => Utc
            .with_ymd_and_hms(dt.year(), dt.month(), 1, 0, 0, 0)
            .single()
            .expect("valid monthly truncation"),
    }
}

fn align_up_execution_dt(dt: DateTime<Utc>, cadence: ExecutionCadence) -> DateTime<Utc> {
    let floored = align_down_execution_dt(dt, cadence);
    if floored == dt {
        floored
    } else {
        advance_execution_dt(floored, cadence)
    }
}

fn advance_execution_dt(dt: DateTime<Utc>, cadence: ExecutionCadence) -> DateTime<Utc> {
    match cadence {
        ExecutionCadence::Hourly => dt + Duration::hours(1),
        ExecutionCadence::Daily => dt + Duration::days(1),
        ExecutionCadence::Weekly => dt + Duration::days(7),
        ExecutionCadence::Monthly => dt
            .checked_add_months(Months::new(1))
            .expect("valid monthly increment"),
    }
}

fn rewind_execution_dt(dt: DateTime<Utc>, cadence: ExecutionCadence) -> DateTime<Utc> {
    match cadence {
        ExecutionCadence::Hourly => dt - Duration::hours(1),
        ExecutionCadence::Daily => dt - Duration::days(1),
        ExecutionCadence::Weekly => dt - Duration::days(7),
        ExecutionCadence::Monthly => dt
            .checked_sub_months(Months::new(1))
            .expect("valid monthly decrement"),
    }
}

fn with_trailing_slash(prefix: &str) -> String {
    if prefix.ends_with('/') {
        prefix.to_string()
    } else {
        format!("{prefix}/")
    }
}

async fn latest_child_prefix(bucket: &str, prefix: &str) -> Result<String> {
    let output = Command::new("aws")
        .arg("s3api")
        .arg("list-objects-v2")
        .arg("--bucket")
        .arg(bucket)
        .arg("--prefix")
        .arg(prefix)
        .arg("--delimiter")
        .arg("/")
        .arg("--query")
        .arg("CommonPrefixes[].Prefix")
        .arg("--output")
        .arg("text")
        .output()
        .await
        .map_err(|e| anyhow!("Failed to invoke aws CLI for s3://{bucket}/{prefix}: {}", e))?;

    if !output.status.success() {
        return Err(anyhow!(
            "Failed to list child prefixes under s3://{bucket}/{prefix}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let mut prefixes: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .filter(|value| !value.is_empty() && *value != "None")
        .map(|value| value.to_string())
        .collect();
    prefixes.sort();
    prefixes
        .pop()
        .ok_or_else(|| anyhow!("No child prefixes found under s3://{bucket}/{prefix}"))
}

async fn latest_object_under_prefix(bucket: &str, prefix: &str) -> Result<String> {
    let output = Command::new("aws")
        .arg("s3api")
        .arg("list-objects-v2")
        .arg("--bucket")
        .arg(bucket)
        .arg("--prefix")
        .arg(prefix)
        .arg("--query")
        .arg("Contents[].Key")
        .arg("--output")
        .arg("text")
        .output()
        .await
        .map_err(|e| anyhow!("Failed to invoke aws CLI for s3://{bucket}/{prefix}: {}", e))?;

    if !output.status.success() {
        return Err(anyhow!(
            "Failed to list objects under s3://{bucket}/{prefix}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let mut keys: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .filter(|value| value.ends_with(".parquet"))
        .map(|value| value.to_string())
        .collect();
    keys.sort();
    let key = keys
        .pop()
        .ok_or_else(|| anyhow!("No parquet files found under s3://{bucket}/{prefix}"))?;
    Ok(format!("s3://{bucket}/{key}"))
}

async fn s3_object_exists(source_uri: &str) -> Result<bool> {
    let (bucket, key) = parse_s3_uri(source_uri)?;
    let output = Command::new("aws")
        .arg("s3api")
        .arg("head-object")
        .arg("--bucket")
        .arg(&bucket)
        .arg("--key")
        .arg(&key)
        .output()
        .await
        .map_err(|e| anyhow!("Failed to invoke aws CLI for {}: {}", source_uri, e))?;

    if output.status.success() {
        return Ok(true);
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    if stderr.contains("404") || stderr.contains("Not Found") || stderr.contains("HeadObject") {
        return Ok(false);
    }

    Err(anyhow!(
        "Failed to check object existence for {}: {}",
        source_uri,
        stderr.trim()
    ))
}

fn select_candidate_sources(
    sources: Vec<String>,
    time_interval: &str,
    start_timestamp_ns: Option<i64>,
    end_timestamp_ns: Option<i64>,
    source_uri: Option<&str>,
    schedule: Option<IntervalSchedule>,
) -> Result<Vec<String>> {
    if sources.is_empty() {
        return Err(anyhow!(
            "No parquet sources available for {}",
            time_interval
        ));
    }

    let mut stamped_sources: Vec<(Option<i64>, String)> = sources
        .into_iter()
        .map(|source| (extract_source_timestamp_ns(&source), source))
        .collect();
    stamped_sources.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

    if start_timestamp_ns.is_none() && end_timestamp_ns.is_none() {
        return Ok(vec![stamped_sources
            .last()
            .map(|(_, source)| source.clone())
            .ok_or_else(|| {
                anyhow!("No parquet sources available for {}", time_interval)
            })?]);
    }

    let selected: Vec<String> = stamped_sources
        .iter()
        .filter(|(timestamp_ns, _)| match timestamp_ns {
            Some(timestamp_ns) => {
                let satisfies_start = start_timestamp_ns
                    .map(|start| *timestamp_ns + SOURCE_SELECTION_LOOKBACK_NS >= start)
                    .unwrap_or(true);
                let satisfies_end = end_timestamp_ns
                    .map(|end| *timestamp_ns <= end)
                    .unwrap_or(true);
                satisfies_start && satisfies_end
            }
            None => true,
        })
        .map(|(_, source)| source.clone())
        .collect();

    if selected.is_empty() {
        return Err(if let (Some(uri), Some(sched)) = (source_uri, schedule) {
            anyhow!(
                "No parquet sources matched the requested time range for {} under {}. Requested range: {}. Expected parquet partitions: {}. Candidate sources: {}",
                time_interval,
                uri,
                describe_requested_time_range(start_timestamp_ns, end_timestamp_ns),
                describe_expected_source_range(uri, sched, start_timestamp_ns, end_timestamp_ns),
                describe_stamped_sources(&stamped_sources)
            )
        } else {
            anyhow!(
                "No parquet sources matched the requested time range for {}. Requested range: {}. Candidate sources: {}",
                time_interval,
                describe_requested_time_range(start_timestamp_ns, end_timestamp_ns),
                describe_stamped_sources(&stamped_sources)
            )
        });
    }

    Ok(selected)
}

fn extract_source_timestamp_ns(source: &str) -> Option<i64> {
    let filename = source.rsplit('/').next()?.strip_suffix(".parquet")?;
    if filename.len() != 16 || !filename.ends_with('Z') {
        return None;
    }

    let year = filename[0..4].parse::<i32>().ok()?;
    let month = filename[4..6].parse::<u32>().ok()?;
    let day = filename[6..8].parse::<u32>().ok()?;
    let hour = filename[9..11].parse::<u32>().ok()?;
    let minute = filename[11..13].parse::<u32>().ok()?;
    let second = filename[13..15].parse::<u32>().ok()?;

    let datetime = NaiveDate::from_ymd_opt(year, month, day)?
        .and_hms_opt(hour, minute, second)?
        .and_utc();
    Some(datetime.timestamp_nanos_opt()? as i64)
}

fn source_partition_descriptor(source_uri: &str) -> String {
    let segments = source_uri
        .split('/')
        .filter(|segment| {
            segment.starts_with("year=")
                || segment.starts_with("month=")
                || segment.starts_with("day=")
                || segment.starts_with("hour=")
        })
        .collect::<Vec<_>>();

    if segments.is_empty() {
        "unpartitioned".to_string()
    } else {
        segments.join("/")
    }
}

async fn try_read_bytes_from_source(local_cache_dir: &str, source: &str) -> Result<Option<Bytes>> {
    if source.starts_with("s3://") {
        let Some(local_path) = ensure_local_cached_s3_source(local_cache_dir, source).await? else {
            return Ok(None);
        };
        let bytes = tokio::fs::read(&local_path)
            .await
            .map_err(|e| anyhow!("Failed to read cached file {}: {}", local_path.display(), e))?;
        return Ok(Some(Bytes::from(bytes)));
    }

    let local_path = source.strip_prefix("file://").unwrap_or(source);
    let bytes = tokio::fs::read(local_path)
        .await
        .map_err(|e| anyhow!("Failed to read file {}: {}", local_path, e))?;
    Ok(Some(Bytes::from(bytes)))
}

fn local_cache_path_for_s3_source(local_cache_dir: &str, source_uri: &str) -> Result<PathBuf> {
    let (bucket, key) = parse_s3_uri(source_uri)?;
    Ok(Path::new(local_cache_dir).join(bucket).join(key))
}

fn cache_download_lock_path(local_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.lock", local_path.display()))
}

async fn ensure_local_cached_s3_source(
    local_cache_dir: &str,
    source_uri: &str,
) -> Result<Option<PathBuf>> {
    let local_path = local_cache_path_for_s3_source(local_cache_dir, source_uri)?;
    if tokio::fs::try_exists(&local_path)
        .await
        .map_err(|e| anyhow!("Failed to stat cached file {}: {}", local_path.display(), e))?
    {
        return Ok(Some(local_path));
    }

    let parent = local_path.parent().ok_or_else(|| {
        anyhow!(
            "Cached parquet path {} has no parent directory",
            local_path.display()
        )
    })?;
    tokio::fs::create_dir_all(parent).await.map_err(|e| {
        anyhow!(
            "Failed to create cache directory {}: {}",
            parent.display(),
            e
        )
    })?;

    let lock_path = cache_download_lock_path(&local_path);
    let partition = source_partition_descriptor(source_uri);
    let mut waiting_for_existing_download = false;
    loop {
        if tokio::fs::try_exists(&local_path)
            .await
            .map_err(|e| anyhow!("Failed to stat cached file {}: {}", local_path.display(), e))?
        {
            return Ok(Some(local_path));
        }

        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
            .await
        {
            Ok(_) => {
                let download_result =
                    download_s3_source_to_local_cache(source_uri, &local_path).await;
                let cleanup_result = remove_file_if_exists(&lock_path).await;
                match (download_result, cleanup_result) {
                    (Ok(downloaded), Ok(())) => return Ok(downloaded.then_some(local_path)),
                    (Err(err), Ok(())) => return Err(err),
                    (Ok(_), Err(err)) => return Err(err),
                    (Err(download_err), Err(cleanup_err)) => {
                        return Err(anyhow!(
                            "Failed to download {} and failed to remove lock {}: {}; {}",
                            source_uri,
                            lock_path.display(),
                            download_err,
                            cleanup_err
                        ))
                    }
                }
            }
            Err(err) if err.kind() == ErrorKind::AlreadyExists => {
                if !waiting_for_existing_download {
                    info!(
                        "Waiting for parquet cache download already in progress for {} ({})",
                        source_uri, partition
                    );
                    waiting_for_existing_download = true;
                }
                if cache_download_lock_is_stale(&lock_path).await? {
                    info!(
                        "Found stale parquet cache download lock for {} ({}) at {}; retrying",
                        source_uri,
                        partition,
                        lock_path.display()
                    );
                    remove_file_if_exists(&lock_path).await?;
                    waiting_for_existing_download = false;
                    continue;
                }
                sleep(TokioDuration::from_millis(CACHE_DOWNLOAD_LOCK_POLL_MS)).await;
            }
            Err(err) => {
                return Err(anyhow!(
                    "Failed to create cache download lock {}: {}",
                    lock_path.display(),
                    err
                ))
            }
        }
    }
}

async fn download_s3_source_to_local_cache(source_uri: &str, local_path: &Path) -> Result<bool> {
    if tokio::fs::try_exists(local_path)
        .await
        .map_err(|e| anyhow!("Failed to stat cached file {}: {}", local_path.display(), e))?
    {
        return Ok(true);
    }

    let partition = source_partition_descriptor(source_uri);
    let temp_path = PathBuf::from(format!(
        "{}.download-{}.tmp",
        local_path.display(),
        now_ns()
    ));
    info!(
        "Starting parquet cache download for {} ({}) -> {}",
        source_uri,
        partition,
        local_path.display()
    );
    let output = Command::new("aws")
        .arg("s3")
        .arg("cp")
        .arg(source_uri)
        .arg(&temp_path)
        .output()
        .await
        .map_err(|e| anyhow!("Failed to invoke aws CLI for {}: {}", source_uri, e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        remove_file_if_exists(&temp_path).await?;
        if stderr.contains("404") || stderr.contains("Not Found") || stderr.contains("HeadObject") {
            info!(
                "Parquet cache download source not found for {} ({})",
                source_uri, partition
            );
            return Ok(false);
        }
        return Err(anyhow!(
            "Failed to download file {} to {}: {}",
            source_uri,
            temp_path.display(),
            stderr.trim()
        ));
    }

    if tokio::fs::try_exists(local_path)
        .await
        .map_err(|e| anyhow!("Failed to stat cached file {}: {}", local_path.display(), e))?
    {
        remove_file_if_exists(&temp_path).await?;
        return Ok(true);
    }

    tokio::fs::rename(&temp_path, local_path)
        .await
        .map_err(|e| {
            anyhow!(
                "Failed to move downloaded file {} into cache path {}: {}",
                temp_path.display(),
                local_path.display(),
                e
            )
        })?;
    info!(
        "Finished parquet cache download for {} ({}) -> {}",
        source_uri,
        partition,
        local_path.display()
    );
    Ok(true)
}

async fn cache_download_lock_is_stale(lock_path: &Path) -> Result<bool> {
    let metadata = match tokio::fs::metadata(lock_path).await {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(false),
        Err(err) => {
            return Err(anyhow!(
                "Failed to stat cache download lock {}: {}",
                lock_path.display(),
                err
            ))
        }
    };

    let modified = metadata.modified().map_err(|e| {
        anyhow!(
            "Failed to inspect cache download lock timestamp {}: {}",
            lock_path.display(),
            e
        )
    })?;
    let age = SystemTime::now().duration_since(modified).map_err(|e| {
        anyhow!(
            "Failed to compute cache download lock age {}: {}",
            lock_path.display(),
            e
        )
    })?;
    Ok(age > TokioDuration::from_secs(CACHE_DOWNLOAD_LOCK_STALE_SECS))
}

async fn remove_file_if_exists(path: &Path) -> Result<()> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => Err(anyhow!("Failed to remove file {}: {}", path.display(), err)),
    }
}

fn parse_s3_uri(uri: &str) -> Result<(String, String)> {
    let without_scheme = uri
        .strip_prefix("s3://")
        .ok_or_else(|| anyhow!("Invalid s3 URI: {}", uri))?;
    let mut parts = without_scheme.splitn(2, '/');
    let bucket = parts
        .next()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("Invalid s3 URI bucket: {}", uri))?;
    let key = parts.next().unwrap_or_default().trim_end_matches('/');
    Ok((bucket.to_string(), key.to_string()))
}

/// Parse timestamp from parquet data (handles both nanosecond and millisecond formats)
pub fn parse_timestamp(value: &arrow::array::Int64Array, index: usize) -> i64 {
    let ts = value.value(index);
    // If timestamp is in milliseconds (less than 1600000000000), convert to nanoseconds
    if ts < 1600000000000 {
        ts * 1_000_000
    } else {
        ts
    }
}

fn column_by_name_case_insensitive<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Option<&'a dyn Array> {
    if let Some(column) = batch.column_by_name(name) {
        return Some(column.as_ref());
    }

    batch
        .schema()
        .fields()
        .iter()
        .position(|field| field.name().eq_ignore_ascii_case(name))
        .map(|index| batch.column(index).as_ref())
}

/// Parse a Bar from a RecordBatch at the given index
pub fn parse_bar_from_batch(batch: &RecordBatch, index: usize) -> Result<Bar> {
    let timestamp = column_by_name_case_insensitive(batch, "timestamp")
        .ok_or_else(|| anyhow!("Missing timestamp column"))?;

    let open = column_by_name_case_insensitive(batch, "open")
        .ok_or_else(|| anyhow!("Missing open column"))?;

    let high = column_by_name_case_insensitive(batch, "high")
        .ok_or_else(|| anyhow!("Missing high column"))?;

    let low = column_by_name_case_insensitive(batch, "low")
        .ok_or_else(|| anyhow!("Missing low column"))?;

    let close = column_by_name_case_insensitive(batch, "close")
        .ok_or_else(|| anyhow!("Missing close column"))?;

    // Volume is optional - default to 0.0 if missing
    let volume = column_by_name_case_insensitive(batch, "volume")
        .map(|value| {
            if let Some(value) = value.as_any().downcast_ref::<arrow::array::Float64Array>() {
                value.value(index)
            } else if let Some(value) = value.as_any().downcast_ref::<arrow::array::Int64Array>() {
                value.value(index) as f64
            } else if let Some(value) = value.as_any().downcast_ref::<arrow::array::UInt64Array>() {
                value.value(index) as f64
            } else {
                0.0
            }
        })
        .unwrap_or(0.0);

    let timestamp_ns = if let Some(arr) = timestamp
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
    {
        parse_timestamp(arr, index)
    } else if let Some(arr) = timestamp
        .as_any()
        .downcast_ref::<arrow::array::TimestampNanosecondArray>()
    {
        arr.value(index)
    } else if let Some(arr) = timestamp
        .as_any()
        .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
    {
        arr.value(index) * 1_000
    } else if let Some(arr) = timestamp
        .as_any()
        .downcast_ref::<arrow::array::TimestampMillisecondArray>()
    {
        arr.value(index) * 1_000_000
    } else if let Some(arr) = timestamp
        .as_any()
        .downcast_ref::<arrow::array::TimestampSecondArray>()
    {
        arr.value(index) * 1_000_000_000
    } else {
        return Err(anyhow!("Invalid timestamp format"));
    };

    let open = open
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .ok_or_else(|| anyhow!("Invalid open format"))?
        .value(index);

    let high = high
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .ok_or_else(|| anyhow!("Invalid high format"))?
        .value(index);

    let low = low
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .ok_or_else(|| anyhow!("Invalid low format"))?
        .value(index);

    let close = close
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .ok_or_else(|| anyhow!("Invalid close format"))?
        .value(index);

    Ok(Bar {
        timestamp_ns,
        open,
        high,
        low,
        close,
        volume,
    })
}

fn parse_news_from_bytes(bytes: Bytes, end_timestamp_ns: i64) -> Result<Vec<News>> {
    let mut parquet_reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .map_err(|e| anyhow!("Failed to create parquet reader: {}", e))?
        .build()
        .map_err(|e| anyhow!("Failed to build parquet reader: {}", e))?;

    let mut news = Vec::new();

    while let Some(batch) = parquet_reader.next() {
        let batch = batch.map_err(|e| anyhow!("Failed to read batch: {}", e))?;
        for i in 0..batch.num_rows() {
            let item = parse_news_from_batch(&batch, i)?;
            if item.timestamp_ns > end_timestamp_ns {
                return Ok(news);
            }
            news.push(item);
        }
    }

    Ok(news)
}

fn parse_ticks_from_bytes(bytes: Bytes, symbol: &str, end_timestamp_ns: i64) -> Result<Vec<Tick>> {
    let mut parquet_reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .map_err(|e| anyhow!("Failed to create parquet reader: {}", e))?
        .build()
        .map_err(|e| anyhow!("Failed to build parquet reader: {}", e))?;

    let mut ticks = Vec::new();
    let mut last_bid: Option<f64> = None;
    let mut last_ask: Option<f64> = None;
    let mut current_ts: Option<i64> = None;

    while let Some(batch) = parquet_reader.next() {
        let batch = batch.map_err(|e| anyhow!("Failed to read batch: {}", e))?;
        for i in 0..batch.num_rows() {
            if !tick_symbol_matches(&batch, i, symbol)? {
                continue;
            }

            let update = parse_tick_quote_update(&batch, i)?;
            if update.timestamp_ns > end_timestamp_ns {
                emit_pending_tick(&mut ticks, current_ts, last_bid, last_ask);
                return Ok(ticks);
            }

            match current_ts {
                Some(prev_ts) if prev_ts != update.timestamp_ns => {
                    emit_pending_tick(&mut ticks, Some(prev_ts), last_bid, last_ask);
                    current_ts = Some(update.timestamp_ns);
                }
                None => current_ts = Some(update.timestamp_ns),
                _ => {}
            }

            if let Some(bid) = update.bid {
                last_bid = Some(bid);
            }
            if let Some(ask) = update.ask {
                last_ask = Some(ask);
            }
        }
    }

    emit_pending_tick(&mut ticks, current_ts, last_bid, last_ask);
    Ok(ticks)
}

fn emit_pending_tick(
    ticks: &mut Vec<Tick>,
    timestamp_ns: Option<i64>,
    bid: Option<f64>,
    ask: Option<f64>,
) {
    if let (Some(ts), Some(bid), Some(ask)) = (timestamp_ns, bid, ask) {
        if ticks.last().map(|t| t.timestamp_ns) != Some(ts) {
            ticks.push(Tick {
                timestamp_ns: ts,
                bid,
                ask,
            });
        }
    }
}

struct QuoteUpdate {
    timestamp_ns: i64,
    bid: Option<f64>,
    ask: Option<f64>,
}

fn parse_tick_quote_update(batch: &RecordBatch, index: usize) -> Result<QuoteUpdate> {
    let timestamp = column_by_name_case_insensitive(batch, "timestamp_ns")
        .or_else(|| column_by_name_case_insensitive(batch, "timestamp"))
        .or_else(|| column_by_name_case_insensitive(batch, "tick_date_raw"))
        .or_else(|| column_by_name_case_insensitive(batch, "tick_date"))
        .ok_or_else(|| anyhow!("Missing tick timestamp column"))?;
    let timestamp_ns = parse_timestamp_value(timestamp, index)?;

    let bid_col = column_by_name_case_insensitive(batch, "bid");
    let ask_col = column_by_name_case_insensitive(batch, "ask")
        .or_else(|| column_by_name_case_insensitive(batch, "offer"));

    if let (Some(bid), Some(ask)) = (bid_col, ask_col) {
        return Ok(QuoteUpdate {
            timestamp_ns,
            bid: Some(parse_numeric_value(bid, index)?),
            ask: Some(parse_numeric_value(ask, index)?),
        });
    }

    let price_col = bid_col
        .or(ask_col)
        .or_else(|| column_by_name_case_insensitive(batch, "price"))
        .ok_or_else(|| {
            anyhow!("Missing tick price column (need bid/ask, bid, ask, offer, or price)")
        })?;
    let price = parse_numeric_value(price_col, index)?;

    if let Some(type_col) = column_by_name_case_insensitive(batch, "type") {
        let raw = string_value(type_col, index)
            .ok_or_else(|| anyhow!("Invalid tick Type format"))?;
        let side = raw.trim().to_ascii_lowercase();
        return match side.as_str() {
            "bid" => Ok(QuoteUpdate {
                timestamp_ns,
                bid: Some(price),
                ask: None,
            }),
            "ask" | "offer" => Ok(QuoteUpdate {
                timestamp_ns,
                bid: None,
                ask: Some(price),
            }),
            other => Err(anyhow!("Unknown tick Type value: {other}")),
        };
    }

    Ok(QuoteUpdate {
        timestamp_ns,
        bid: Some(price),
        ask: Some(price),
    })
}

fn parse_news_from_batch(batch: &RecordBatch, index: usize) -> Result<News> {
    let timestamp = column_by_name_case_insensitive(batch, "timestamp_ns")
        .or_else(|| column_by_name_case_insensitive(batch, "timestamp"))
        .or_else(|| column_by_name_case_insensitive(batch, "date"))
        .or_else(|| column_by_name_case_insensitive(batch, "published_at"))
        .ok_or_else(|| anyhow!("Missing news timestamp column"))?;

    let headline = column_by_name_case_insensitive(batch, "headline")
        .or_else(|| column_by_name_case_insensitive(batch, "title"))
        .ok_or_else(|| anyhow!("Missing news headline column"))?;

    let source = column_by_name_case_insensitive(batch, "source")
        .or_else(|| column_by_name_case_insensitive(batch, "source_name"))
        .ok_or_else(|| anyhow!("Missing news source column"))?;

    let sentiment_score = column_by_name_case_insensitive(batch, "sentiment_score")
        .or_else(|| column_by_name_case_insensitive(batch, "sentiment"))
        .map(|value| parse_sentiment_score(value, index))
        .transpose()?
        .unwrap_or(0.0);

    Ok(News {
        timestamp_ns: parse_timestamp_value(timestamp, index)?,
        headline: string_value(headline, index)
            .ok_or_else(|| anyhow!("Invalid news headline format"))?,
        sentiment_score,
        source: string_value(source, index).ok_or_else(|| anyhow!("Invalid news source format"))?,
    })
}

fn tick_symbol_matches(batch: &RecordBatch, index: usize, expected_symbol: &str) -> Result<bool> {
    let Some(symbol_value) = column_by_name_case_insensitive(batch, "symbol") else {
        return Ok(true);
    };

    let actual =
        string_value(symbol_value, index).ok_or_else(|| anyhow!("Invalid tick symbol format"))?;
    Ok(normalise_market_symbol(&actual) == normalise_market_symbol(expected_symbol))
}

fn parse_timestamp_value(value: &dyn Array, index: usize) -> Result<i64> {
    if let Some(arr) = value.as_any().downcast_ref::<arrow::array::Int64Array>() {
        Ok(parse_timestamp(arr, index))
    } else if let Some(arr) = value
        .as_any()
        .downcast_ref::<arrow::array::TimestampNanosecondArray>()
    {
        Ok(arr.value(index))
    } else if let Some(arr) = value
        .as_any()
        .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
    {
        Ok(arr.value(index) * 1_000)
    } else if let Some(arr) = value
        .as_any()
        .downcast_ref::<arrow::array::TimestampMillisecondArray>()
    {
        Ok(arr.value(index) * 1_000_000)
    } else if let Some(arr) = value
        .as_any()
        .downcast_ref::<arrow::array::TimestampSecondArray>()
    {
        Ok(arr.value(index) * 1_000_000_000)
    } else if let Some(raw) = string_value(value, index) {
        parse_timestamp_string(&raw)
    } else {
        Err(anyhow!("Invalid timestamp format"))
    }
}

fn parse_timestamp_string(raw: &str) -> Result<i64> {
    if let Ok(ts) = DateTime::parse_from_rfc3339(raw) {
        return Ok(ts
            .timestamp_nanos_opt()
            .ok_or_else(|| anyhow!("Invalid timestamp"))? as i64);
    }

    if let Ok(ts) = DateTime::parse_from_rfc2822(raw) {
        return Ok(ts
            .timestamp_nanos_opt()
            .ok_or_else(|| anyhow!("Invalid timestamp"))? as i64);
    }

    Err(anyhow!("Unsupported timestamp string {}", raw))
}

fn parse_numeric_value(value: &dyn Array, index: usize) -> Result<f64> {
    if let Some(arr) = value.as_any().downcast_ref::<arrow::array::Float64Array>() {
        Ok(arr.value(index))
    } else if let Some(arr) = value.as_any().downcast_ref::<arrow::array::Float32Array>() {
        Ok(arr.value(index) as f64)
    } else if let Some(arr) = value.as_any().downcast_ref::<arrow::array::Int64Array>() {
        Ok(arr.value(index) as f64)
    } else if let Some(arr) = value.as_any().downcast_ref::<arrow::array::Int32Array>() {
        Ok(arr.value(index) as f64)
    } else if let Some(arr) = value.as_any().downcast_ref::<arrow::array::UInt64Array>() {
        Ok(arr.value(index) as f64)
    } else if let Some(arr) = value.as_any().downcast_ref::<arrow::array::UInt32Array>() {
        Ok(arr.value(index) as f64)
    } else if let Some(raw) = string_value(value, index) {
        raw.parse::<f64>()
            .map_err(|_| anyhow!("Unsupported numeric value {}", raw))
    } else {
        Err(anyhow!("Invalid numeric format"))
    }
}

fn normalise_market_symbol(symbol: &str) -> String {
    symbol
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase()
}

fn string_value(value: &dyn Array, index: usize) -> Option<String> {
    if let Some(arr) = value.as_any().downcast_ref::<arrow::array::StringArray>() {
        Some(arr.value(index).to_string())
    } else if let Some(arr) = value
        .as_any()
        .downcast_ref::<arrow::array::LargeStringArray>()
    {
        Some(arr.value(index).to_string())
    } else {
        None
    }
}

fn parse_sentiment_score(value: &dyn Array, index: usize) -> Result<f64> {
    if let Some(arr) = value.as_any().downcast_ref::<arrow::array::Float64Array>() {
        Ok(arr.value(index))
    } else if let Some(arr) = value.as_any().downcast_ref::<arrow::array::Int64Array>() {
        Ok(arr.value(index) as f64)
    } else if let Some(arr) = value.as_any().downcast_ref::<arrow::array::UInt64Array>() {
        Ok(arr.value(index) as f64)
    } else if let Some(raw) = string_value(value, index) {
        Ok(match raw.trim().to_ascii_lowercase().as_str() {
            "positive" => 1.0,
            "negative" => -1.0,
            "neutral" => 0.0,
            other => other
                .parse::<f64>()
                .map_err(|_| anyhow!("Unsupported sentiment value {}", raw))?,
        })
    } else {
        Err(anyhow!("Invalid sentiment format"))
    }
}

/// Load bars from a parquet file and parse them into Bar messages
pub async fn load_and_parse_bars(s3_uri: &str) -> Result<Vec<Bar>> {
    load_bars_from_parquet(s3_uri, "", "").await
}

/// Get the current timestamp in nanoseconds
pub fn now_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex, OnceLock};
    use tempfile::tempdir;

    use crate::market_data_cache::MarketDataCache;

    fn write_test_parquet(path: &PathBuf, timestamps: &[i64], opens: &[f64]) -> Result<()> {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("timestamp", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("open", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("high", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("low", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("close", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("volume", arrow::datatypes::DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(timestamps.to_vec())),
                Arc::new(Float64Array::from(opens.to_vec())),
                Arc::new(Float64Array::from(
                    opens.iter().map(|value| value + 0.5).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    opens.iter().map(|value| value - 0.5).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    opens.iter().map(|value| value + 0.25).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(vec![1.0; timestamps.len()])),
            ],
        )?;

        let file = std::fs::File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    fn write_tick_parquet(
        path: &PathBuf,
        timestamps_micros: &[i64],
        symbols: &[&str],
        prices: &[f64],
        sizes: &[i64],
    ) -> Result<()> {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new(
                "Timestamp",
                arrow::datatypes::DataType::Timestamp(
                    arrow::datatypes::TimeUnit::Microsecond,
                    None,
                ),
                false,
            ),
            arrow::datatypes::Field::new("Symbol", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("Price", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("Size", arrow::datatypes::DataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow::array::TimestampMicrosecondArray::from(
                    timestamps_micros.to_vec(),
                )),
                Arc::new(StringArray::from(
                    symbols
                        .iter()
                        .map(|value| (*value).to_string())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(prices.to_vec())),
                Arc::new(Int64Array::from(sizes.to_vec())),
            ],
        )?;

        let file = std::fs::File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    fn write_typed_tick_parquet(
        path: &PathBuf,
        timestamps_micros: &[i64],
        symbols: &[&str],
        types: &[&str],
        prices: &[f64],
    ) -> Result<()> {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new(
                "Timestamp",
                arrow::datatypes::DataType::Timestamp(
                    arrow::datatypes::TimeUnit::Microsecond,
                    None,
                ),
                false,
            ),
            arrow::datatypes::Field::new("Symbol", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("Type", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("Price", arrow::datatypes::DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow::array::TimestampMicrosecondArray::from(
                    timestamps_micros.to_vec(),
                )),
                Arc::new(StringArray::from(
                    symbols
                        .iter()
                        .map(|value| (*value).to_string())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    types
                        .iter()
                        .map(|value| (*value).to_string())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(prices.to_vec())),
            ],
        )?;

        let file = std::fs::File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    fn path_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn write_news_parquet(path: &PathBuf, timestamps: &[&str], headlines: &[&str]) -> Result<()> {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("date", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("title", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("source_name", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("sentiment", arrow::datatypes::DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(timestamps.to_vec())),
                Arc::new(StringArray::from(headlines.to_vec())),
                Arc::new(StringArray::from(vec!["Source"; timestamps.len()])),
                Arc::new(StringArray::from(vec!["Positive"; timestamps.len()])),
            ],
        )?;

        let file = std::fs::File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    #[test]
    fn normalises_bucket_root_to_snapshot_parent() {
        assert_eq!(
            normalise_training_data_prefix("s3://prod-fintech-forex-sg-731833471586"),
            "s3://prod-fintech-forex-sg-731833471586"
        );
        assert_eq!(
            normalise_training_data_prefix(
                "s3://prod-fintech-forex-sg-731833471586/marketdata/eoh-snapshot"
            ),
            "s3://prod-fintech-forex-sg-731833471586"
        );
    }

    #[test]
    fn normalises_news_prefix_from_bucket_or_price_branch() {
        assert_eq!(
            normalise_news_data_prefix("s3://prod-fintech-forex-sg-731833471586"),
            "s3://prod-fintech-forex-sg-731833471586/marketdata/eod-news-snapshot"
        );
        assert_eq!(
            normalise_news_data_prefix(
                "s3://prod-fintech-forex-sg-731833471586/marketdata/eod-snapshot"
            ),
            "s3://prod-fintech-forex-sg-731833471586/marketdata/eod-news-snapshot"
        );
    }

    #[test]
    fn builds_snapshot_interval_source() {
        assert_eq!(
            build_interval_data_source("s3://bucket", "USDJPY", "M1"),
            "s3://bucket/marketdata/eoh-snapshot/symbol=USDJPY/interval=M1"
        );
        assert_eq!(
            build_interval_data_source("s3://bucket", "USDJPY", "H1"),
            "s3://bucket/marketdata/eod-snapshot/symbol=USDJPY/interval=H1"
        );
        assert_eq!(
            build_interval_data_source("s3://bucket", "USDJPY", "W1"),
            "s3://bucket/marketdata/eow-snapshot/symbol=USDJPY/interval=W1"
        );
        assert_eq!(
            build_interval_data_source("s3://bucket", "USDJPY", "MN"),
            "s3://bucket/marketdata/eom-snapshot/symbol=USDJPY/interval=MN1"
        );
        assert_eq!(
            build_news_data_source("s3://bucket", "USDJPY"),
            "s3://bucket/marketdata/eod-news-snapshot/symbol=USD-JPY"
        );
        assert_eq!(
            build_tick_data_source("s3://bucket", "USDJPY"),
            "s3://bucket/marketdata/eod-tick-snapshot/symbol=USDJPY"
        );
    }

    #[test]
    fn builds_exact_s3_keys_from_dag_convention() {
        let weekly = build_s3_object_uri(
            "s3://bucket/marketdata/eow-snapshot/symbol=USDJPY/interval=W1",
            interval_schedule("W1").unwrap(),
            Utc.with_ymd_and_hms(2024, 12, 16, 0, 0, 0)
                .single()
                .unwrap(),
        );
        let monthly = build_s3_object_uri(
            "s3://bucket/marketdata/eom-snapshot/symbol=USDJPY/interval=MN1",
            interval_schedule("MN").unwrap(),
            Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).single().unwrap(),
        );

        assert_eq!(
            weekly,
            "s3://bucket/marketdata/eow-snapshot/symbol=USDJPY/interval=W1/year=2024/month=12/20241216T000000Z.parquet"
        );
        assert_eq!(
            monthly,
            "s3://bucket/marketdata/eom-snapshot/symbol=USDJPY/interval=MN1/year=2025/month=01/20250101T000000Z.parquet"
        );
    }

    #[test]
    fn builds_speculative_snapshot_key_from_requested_timestamp() {
        let key = speculative_price_snapshot_source_uri(
            "s3://bucket/marketdata/eod-snapshot/symbol=USDJPY/interval=H1",
            interval_schedule("H1").unwrap(),
            1_640_390_400_000_000_000,
        );

        assert_eq!(
            key,
            "s3://bucket/marketdata/eod-snapshot/symbol=USDJPY/interval=H1/year=2021/month=12/day=25/20211225T000000Z.parquet"
        );
    }

    #[test]
    fn snapshot_selection_error_lists_candidates_and_requested_timestamp() {
        let err = select_price_snapshot_sources(
            vec![
                "s3://bucket/marketdata/eoh-snapshot/symbol=USDJPY/interval=M1/year=2022/month=01/day=01/hour=00/20220101T000000Z.parquet".to_string(),
                "s3://bucket/marketdata/eoh-snapshot/symbol=USDJPY/interval=M1/year=2022/month=01/day=02/hour=00/20220102T000000Z.parquet".to_string(),
            ],
            Some(1_640_390_400_000_000_000),
        )
        .unwrap_err();

        let message = err.to_string();
        assert!(message.contains("1640390400000000000"));
        assert!(message.contains("2021-12-25T00:00:00+00:00"));
        assert!(message.contains("20220101T000000Z.parquet"));
        assert!(message.contains("20220102T000000Z.parquet"));
    }

    #[tokio::test]
    async fn loads_latest_local_snapshot_by_default() {
        let dir = tempdir().unwrap();
        let base = dir
            .path()
            .join("marketdata/eoh-snapshot/symbol=USDJPY/interval=M1");
        std::fs::create_dir_all(base.join("year=2012/month=01/day=02/hour=05")).unwrap();
        std::fs::create_dir_all(base.join("year=2012/month=01/day=02/hour=06")).unwrap();

        write_test_parquet(
            &base.join("year=2012/month=01/day=02/hour=05/20120102T050000Z.parquet"),
            &[1_325_480_400_000_000_000, 1_325_480_460_000_000_000],
            &[100.0, 101.0],
        )
        .unwrap();
        write_test_parquet(
            &base.join("year=2012/month=01/day=02/hour=06/20120102T060000Z.parquet"),
            &[
                1_325_480_400_000_000_000,
                1_325_480_460_000_000_000,
                1_325_484_000_000_000_000,
                1_325_484_060_000_000_000,
            ],
            &[100.0, 101.0, 102.0, 103.0],
        )
        .unwrap();

        let bars = load_bars_from_parquet(
            dir.path()
                .join("marketdata/eoh-snapshot")
                .to_string_lossy()
                .as_ref(),
            "USDJPY",
            "M1",
        )
        .await
        .unwrap();

        assert_eq!(bars.len(), 4);
        assert_eq!(bars[0].open, 100.0);
        assert_eq!(bars[3].open, 103.0);
    }

    #[tokio::test]
    async fn filters_by_episode_end_after_selecting_latest_snapshot() {
        let dir = tempdir().unwrap();
        let base = dir
            .path()
            .join("marketdata/eoh-snapshot/symbol=USDJPY/interval=M1");
        std::fs::create_dir_all(base.join("year=2012/month=01/day=02/hour=05")).unwrap();
        std::fs::create_dir_all(base.join("year=2012/month=01/day=02/hour=06")).unwrap();

        write_test_parquet(
            &base.join("year=2012/month=01/day=02/hour=05/20120102T050000Z.parquet"),
            &[1_325_480_400_000_000_000, 1_325_480_460_000_000_000],
            &[100.0, 101.0],
        )
        .unwrap();
        write_test_parquet(
            &base.join("year=2012/month=01/day=02/hour=06/20120102T060000Z.parquet"),
            &[
                1_325_480_400_000_000_000,
                1_325_480_460_000_000_000,
                1_325_484_000_000_000_000,
                1_325_484_060_000_000_000,
            ],
            &[100.0, 101.0, 102.0, 103.0],
        )
        .unwrap();

        let bars = load_bars_from_parquet_with_range(
            dir.path()
                .join("marketdata/eoh-snapshot")
                .to_string_lossy()
                .as_ref(),
            "USDJPY",
            "M1",
            None,
            None,
            Some(1_325_480_999_000_000_000),
        )
        .await
        .unwrap();

        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].open, 100.0);
        assert_eq!(bars[1].open, 101.0);
    }

    #[tokio::test]
    async fn loads_snapshot_at_or_before_server_selection_timestamp() {
        let dir = tempdir().unwrap();
        let base = dir
            .path()
            .join("marketdata/eoh-snapshot/symbol=USDJPY/interval=M1");
        std::fs::create_dir_all(base.join("year=2012/month=01/day=02/hour=05")).unwrap();
        std::fs::create_dir_all(base.join("year=2012/month=01/day=02/hour=06")).unwrap();

        write_test_parquet(
            &base.join("year=2012/month=01/day=02/hour=05/20120102T050000Z.parquet"),
            &[1_325_480_400_000_000_000, 1_325_480_460_000_000_000],
            &[100.0, 101.0],
        )
        .unwrap();
        write_test_parquet(
            &base.join("year=2012/month=01/day=02/hour=06/20120102T060000Z.parquet"),
            &[
                1_325_480_400_000_000_000,
                1_325_480_460_000_000_000,
                1_325_484_000_000_000_000,
                1_325_484_060_000_000_000,
            ],
            &[100.0, 101.0, 102.0, 103.0],
        )
        .unwrap();

        let bars = load_bars_from_parquet_with_range(
            dir.path()
                .join("marketdata/eoh-snapshot")
                .to_string_lossy()
                .as_ref(),
            "USDJPY",
            "M1",
            Some(1_325_480_999_000_000_000),
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].open, 100.0);
        assert_eq!(bars[1].open, 101.0);
    }

    #[tokio::test]
    async fn reuses_cached_price_bars_after_source_is_removed() {
        let dir = tempdir().unwrap();
        let base = dir.path().join(
            "marketdata/eoh-snapshot/symbol=USDJPY/interval=M1/year=2012/month=01/day=02/hour=06",
        );
        std::fs::create_dir_all(&base).unwrap();
        let parquet_path = base.join("20120102T060000Z.parquet");
        write_test_parquet(
            &parquet_path,
            &[1_325_484_000_000_000_000, 1_325_484_060_000_000_000],
            &[102.0, 103.0],
        )
        .unwrap();

        let cache = MarketDataCache::new();
        let source_root = dir
            .path()
            .join("marketdata/eoh-snapshot")
            .to_string_lossy()
            .to_string();

        let first = load_bars_from_parquet_with_range_cached(
            &cache,
            &source_root,
            "USDJPY",
            "M1",
            None,
            None,
            None,
        )
        .await
        .unwrap();
        assert_eq!(first.len(), 2);

        std::fs::remove_file(&parquet_path).unwrap();

        let second = load_bars_from_parquet_with_range_cached(
            &cache,
            &source_root,
            "USDJPY",
            "M1",
            None,
            None,
            None,
        )
        .await
        .unwrap();
        assert_eq!(second.len(), 2);
        assert_eq!(second[0].open, 102.0);
    }

    #[tokio::test]
    async fn reuses_cached_news_after_source_is_removed() {
        let dir = tempdir().unwrap();
        let base = dir
            .path()
            .join("marketdata/eod-news-snapshot/symbol=USD-JPY/year=2026/month=01/day=01");
        std::fs::create_dir_all(&base).unwrap();
        let parquet_path = base.join("20260101T000000Z.parquet");
        write_news_parquet(
            &parquet_path,
            &[
                "Thu, 01 Jan 2026 09:00:00 -0400",
                "Thu, 01 Jan 2026 12:00:00 -0400",
            ],
            &["Headline A", "Headline B"],
        )
        .unwrap();

        let cache = MarketDataCache::new();
        let source_root = dir
            .path()
            .join("marketdata/eod-news-snapshot")
            .to_string_lossy()
            .to_string();

        let first = load_news_from_parquet_with_range_cached(&cache, &source_root, None, None)
            .await
            .unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].headline, "Headline A");
        assert_eq!(first[0].sentiment_score, 1.0);

        std::fs::remove_file(&parquet_path).unwrap();

        let second = load_news_from_parquet_with_range_cached(&cache, &source_root, None, None)
            .await
            .unwrap();
        assert_eq!(second.len(), 2);
        assert_eq!(second[1].headline, "Headline B");
    }

    #[test]
    fn maps_s3_source_to_local_cache_path() {
        let local_path =
            local_cache_path_for_s3_source("/cache/modelenv", "s3://bucket/path/file.parquet")
                .unwrap();
        assert_eq!(
            local_path,
            Path::new("/cache/modelenv")
                .join("bucket")
                .join("path/file.parquet")
        );
    }

    #[tokio::test]
    async fn reads_s3_source_from_existing_local_cache_file() {
        let dir = tempdir().unwrap();
        let local_path = local_cache_path_for_s3_source(
            dir.path().to_string_lossy().as_ref(),
            "s3://bucket/path/file.parquet",
        )
        .unwrap();
        std::fs::create_dir_all(local_path.parent().unwrap()).unwrap();
        std::fs::write(&local_path, b"cached parquet bytes").unwrap();

        let bytes = try_read_bytes_from_source(
            dir.path().to_string_lossy().as_ref(),
            "s3://bucket/path/file.parquet",
        )
        .await
        .unwrap()
        .unwrap();

        assert_eq!(bytes, Bytes::from_static(b"cached parquet bytes"));
    }

    #[tokio::test]
    async fn prefers_local_cached_price_snapshot_sources_over_stale_latest_cache() {
        let dir = tempdir().unwrap();
        let cache_dir = dir.path().to_string_lossy().to_string();
        let source_uri = "s3://bucket/marketdata/eoh-snapshot/symbol=USDJPY/interval=M1";
        let cached_file = local_cache_path_for_s3_source(
            &cache_dir,
            "s3://bucket/marketdata/eoh-snapshot/symbol=USDJPY/interval=M1/year=2012/month=01/day=02/hour=06/20120102T060000Z.parquet",
        )
        .unwrap();
        std::fs::create_dir_all(cached_file.parent().unwrap()).unwrap();
        write_test_parquet(
            &cached_file,
            &[1_325_484_000_000_000_000, 1_325_484_060_000_000_000],
            &[102.0, 103.0],
        )
        .unwrap();

        let cache = MarketDataCache::new();
        cache
            .put_latest_source(
                price_snapshot_cache_key(source_uri, None),
                CachedLatestSource::Missing("stale missing cache entry".to_string()),
            )
            .await;

        let bars = load_bars_from_parquet_with_range_cached_from_local_cache_dir(
            &cache_dir, &cache, source_uri, "USDJPY", "M1", None, None, None,
        )
        .await
        .unwrap();

        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].open, 102.0);
        assert_eq!(bars[1].open, 103.0);
    }

    #[tokio::test]
    async fn cached_snapshot_loader_uses_speculative_s3_key_when_local_cache_is_future_only() {
        let _guard = path_env_lock().lock().unwrap();

        let dir = tempdir().unwrap();
        let cache_dir = dir.path().to_string_lossy().to_string();
        let source_uri = "s3://bucket/marketdata/eod-snapshot/symbol=USDJPY/interval=H1";
        let future_cached_file = local_cache_path_for_s3_source(
            &cache_dir,
            "s3://bucket/marketdata/eod-snapshot/symbol=USDJPY/interval=H1/year=2025/month=01/day=13/20250113T000000Z.parquet",
        )
        .unwrap();
        std::fs::create_dir_all(future_cached_file.parent().unwrap()).unwrap();
        write_test_parquet(&future_cached_file, &[1_736_726_400_000_000_000], &[155.0]).unwrap();

        let requested_cached_file = local_cache_path_for_s3_source(
            &cache_dir,
            "s3://bucket/marketdata/eod-snapshot/symbol=USDJPY/interval=H1/year=2021/month=12/day=25/20211225T000000Z.parquet",
        )
        .unwrap();
        let fixture_file = dir.path().join("requested.parquet");
        write_test_parquet(
            &fixture_file,
            &[1_640_390_400_000_000_000, 1_640_390_460_000_000_000],
            &[120.0, 121.0],
        )
        .unwrap();

        let fake_bin_dir = dir.path().join("bin");
        std::fs::create_dir_all(&fake_bin_dir).unwrap();
        let fake_aws = fake_bin_dir.join("aws");
        std::fs::write(
            &fake_aws,
            format!(
                "#!/bin/sh\nif [ \"$1\" = \"s3api\" ] && [ \"$2\" = \"head-object\" ]; then\n  case \"$*\" in\n    *'marketdata/eod-snapshot/symbol=USDJPY/interval=H1/year=2021/month=12/day=25/20211225T000000Z.parquet'*) exit 0 ;;\n  esac\n  echo '404' >&2\n  exit 255\nfi\nif [ \"$1\" = \"s3\" ] && [ \"$2\" = \"cp\" ]; then\n  src=\"$3\"\n  dest=\"$4\"\n  case \"$src\" in\n    *'marketdata/eod-snapshot/symbol=USDJPY/interval=H1/year=2021/month=12/day=25/20211225T000000Z.parquet'*) cp '{}' \"$dest\"; exit 0 ;;\n  esac\n  echo '404' >&2\n  exit 255\nfi\necho 'unsupported aws invocation' >&2\nexit 1\n",
                fixture_file.display()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&fake_aws).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&fake_aws, perms).unwrap();
        }

        let original_path = std::env::var_os("PATH");
        let new_path = match &original_path {
            Some(path) => format!("{}:{}", fake_bin_dir.display(), path.to_string_lossy()),
            None => fake_bin_dir.display().to_string(),
        };
        std::env::set_var("PATH", &new_path);

        let cache = MarketDataCache::new();
        let bars = load_bars_from_parquet_with_range_cached_from_local_cache_dir(
            &cache_dir,
            &cache,
            source_uri,
            "USDJPY",
            "H1",
            Some(1_640_390_400_000_000_000),
            None,
            None,
        )
        .await
        .unwrap();

        match original_path {
            Some(path) => std::env::set_var("PATH", path),
            None => std::env::remove_var("PATH"),
        }

        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].open, 120.0);
        assert_eq!(bars[1].open, 121.0);
        assert!(requested_cached_file.exists());
    }

    #[tokio::test]
    async fn timestamped_snapshot_lookup_checks_speculative_s3_key_before_unrelated_local_cache() {
        let _guard = path_env_lock().lock().unwrap();

        let dir = tempdir().unwrap();
        let cache_dir = dir.path().to_string_lossy().to_string();
        let source_uri = "s3://bucket/marketdata/eod-snapshot/symbol=USDJPY/interval=H1";
        let future_cached_file = local_cache_path_for_s3_source(
            &cache_dir,
            "s3://bucket/marketdata/eod-snapshot/symbol=USDJPY/interval=H1/year=2025/month=01/day=13/20250113T000000Z.parquet",
        )
        .unwrap();
        std::fs::create_dir_all(future_cached_file.parent().unwrap()).unwrap();
        write_test_parquet(&future_cached_file, &[1_736_726_400_000_000_000], &[155.0]).unwrap();

        let fake_bin_dir = dir.path().join("bin");
        std::fs::create_dir_all(&fake_bin_dir).unwrap();
        let fake_aws = fake_bin_dir.join("aws");
        std::fs::write(
            &fake_aws,
            r#"#!/bin/sh
if printf '%s\n' "$*" | grep -q 'head-object' && printf '%s\n' "$*" | grep -q 'marketdata/eod-snapshot/symbol=USDJPY/interval=H1/year=2021/month=12/day=25/20211225T000000Z.parquet'; then
  exit 0
fi
echo "404" >&2
exit 255
"#,
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&fake_aws).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&fake_aws, perms).unwrap();
        }

        let original_path = std::env::var_os("PATH");
        let new_path = match &original_path {
            Some(path) => format!("{}:{}", fake_bin_dir.display(), path.to_string_lossy()),
            None => fake_bin_dir.display().to_string(),
        };
        std::env::set_var("PATH", &new_path);

        let resolved = determine_price_snapshot_s3_sources_cached(
            &cache_dir,
            None,
            source_uri,
            interval_schedule("H1").unwrap(),
            Some(1_640_390_400_000_000_000),
        )
        .await
        .unwrap();

        match original_path {
            Some(path) => std::env::set_var("PATH", path),
            None => std::env::remove_var("PATH"),
        }

        assert_eq!(
            resolved,
            vec![
                "s3://bucket/marketdata/eod-snapshot/symbol=USDJPY/interval=H1/year=2021/month=12/day=25/20211225T000000Z.parquet"
                    .to_string()
            ]
        );
    }

    #[tokio::test]
    async fn timestamped_snapshot_lookup_errors_when_exact_key_is_missing() {
        let _guard = path_env_lock().lock().unwrap();

        let dir = tempdir().unwrap();
        let cache_dir = dir.path().to_string_lossy().to_string();
        let source_uri = "s3://bucket/marketdata/eod-snapshot/symbol=USDJPY/interval=H1";
        let future_cached_file = local_cache_path_for_s3_source(
            &cache_dir,
            "s3://bucket/marketdata/eod-snapshot/symbol=USDJPY/interval=H1/year=2025/month=01/day=13/20250113T000000Z.parquet",
        )
        .unwrap();
        std::fs::create_dir_all(future_cached_file.parent().unwrap()).unwrap();
        write_test_parquet(&future_cached_file, &[1_736_726_400_000_000_000], &[155.0]).unwrap();

        let fake_bin_dir = dir.path().join("bin");
        std::fs::create_dir_all(&fake_bin_dir).unwrap();
        let fake_aws = fake_bin_dir.join("aws");
        std::fs::write(
            &fake_aws,
            r#"#!/bin/sh
echo "404" >&2
exit 255
"#,
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&fake_aws).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&fake_aws, perms).unwrap();
        }

        let original_path = std::env::var_os("PATH");
        let new_path = match &original_path {
            Some(path) => format!("{}:{}", fake_bin_dir.display(), path.to_string_lossy()),
            None => fake_bin_dir.display().to_string(),
        };
        std::env::set_var("PATH", &new_path);

        let err = determine_price_snapshot_s3_sources_cached(
            &cache_dir,
            None,
            source_uri,
            interval_schedule("H1").unwrap(),
            Some(1_640_390_400_000_000_000),
        )
        .await
        .unwrap_err();

        match original_path {
            Some(path) => std::env::set_var("PATH", path),
            None => std::env::remove_var("PATH"),
        }

        let message = err.to_string();
        assert!(message.contains("Exact parquet snapshot key"));
        assert!(message.contains("20211225T000000Z.parquet"));
        assert!(message.contains("2021-12-25T00:00:00+00:00"));
    }

    #[tokio::test]
    async fn loads_news_from_local_cached_s3_prefix_without_hitting_s3() {
        let dir = tempdir().unwrap();
        let cache_dir = dir.path().to_string_lossy().to_string();
        let source_uri = "s3://bucket/marketdata/eod-news-snapshot/symbol=USD-JPY";
        let day_one = local_cache_path_for_s3_source(
            &cache_dir,
            "s3://bucket/marketdata/eod-news-snapshot/symbol=USD-JPY/year=2026/month=01/day=01/20260101T000000Z.parquet",
        )
        .unwrap();
        let day_two = local_cache_path_for_s3_source(
            &cache_dir,
            "s3://bucket/marketdata/eod-news-snapshot/symbol=USD-JPY/year=2026/month=01/day=02/20260102T000000Z.parquet",
        )
        .unwrap();
        std::fs::create_dir_all(day_one.parent().unwrap()).unwrap();
        std::fs::create_dir_all(day_two.parent().unwrap()).unwrap();
        write_news_parquet(
            &day_one,
            &["Thu, 01 Jan 2026 09:00:00 -0400"],
            &["Headline A"],
        )
        .unwrap();
        write_news_parquet(
            &day_two,
            &["Fri, 02 Jan 2026 09:00:00 -0400"],
            &["Headline B"],
        )
        .unwrap();

        let cache = MarketDataCache::new();
        let news = load_news_from_parquet_with_range_cached_from_local_cache_dir(
            &cache_dir,
            &cache,
            source_uri,
            Some(1_767_254_400_000_000_000),
            Some(1_767_427_199_000_000_000),
        )
        .await
        .unwrap();

        assert_eq!(news.len(), 2);
        assert_eq!(news[0].headline, "Headline A");
        assert_eq!(news[1].headline, "Headline B");
    }

    #[tokio::test]
    async fn waits_for_existing_download_lock_to_resolve_before_reusing_file() {
        let dir = tempdir().unwrap();
        let cache_dir = dir.path().to_string_lossy().to_string();
        let local_path =
            local_cache_path_for_s3_source(&cache_dir, "s3://bucket/path/file.parquet").unwrap();
        std::fs::create_dir_all(local_path.parent().unwrap()).unwrap();
        let lock_path = cache_download_lock_path(&local_path);
        std::fs::write(&lock_path, b"locked").unwrap();

        let delayed_local_path = local_path.clone();
        let delayed_lock_path = lock_path.clone();
        tokio::spawn(async move {
            sleep(TokioDuration::from_millis(CACHE_DOWNLOAD_LOCK_POLL_MS * 2)).await;
            tokio::fs::write(&delayed_local_path, b"ready")
                .await
                .unwrap();
            tokio::fs::remove_file(&delayed_lock_path).await.unwrap();
        });

        let resolved = ensure_local_cached_s3_source(&cache_dir, "s3://bucket/path/file.parquet")
            .await
            .unwrap()
            .unwrap();

        assert_eq!(resolved, local_path);
    }

    #[test]
    fn parses_case_insensitive_schema_and_timestamp_microseconds() {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new(
                "Timestamp",
                arrow::datatypes::DataType::Timestamp(
                    arrow::datatypes::TimeUnit::Microsecond,
                    None,
                ),
                false,
            ),
            arrow::datatypes::Field::new("Open", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("High", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("Low", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("Close", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("Volume", arrow::datatypes::DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::TimestampMicrosecondArray::from(vec![
                    1_325_480_400_000_000,
                ])),
                Arc::new(Float64Array::from(vec![76.947])),
                Arc::new(Float64Array::from(vec![76.971])),
                Arc::new(Float64Array::from(vec![76.946])),
                Arc::new(Float64Array::from(vec![76.966])),
                Arc::new(Int64Array::from(vec![4])),
            ],
        )
        .unwrap();

        let bar = parse_bar_from_batch(&batch, 0).unwrap();
        assert_eq!(bar.timestamp_ns, 1_325_480_400_000_000_000);
        assert_eq!(bar.volume, 4.0);
        assert_eq!(bar.open, 76.947);
    }

    #[test]
    fn parses_news_batch_with_rfc2822_date_and_string_sentiment() {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("date", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("title", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("source_name", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("sentiment", arrow::datatypes::DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["Thu, 01 Jan 2026 09:00:00 -0400"])),
                Arc::new(StringArray::from(vec!["Headline"])),
                Arc::new(StringArray::from(vec!["Source"])),
                Arc::new(StringArray::from(vec!["Negative"])),
            ],
        )
        .unwrap();

        let item = parse_news_from_batch(&batch, 0).unwrap();
        assert_eq!(item.headline, "Headline");
        assert_eq!(item.source, "Source");
        assert_eq!(item.sentiment_score, -1.0);
        assert!(item.timestamp_ns > 0);
    }

    #[test]
    fn parses_tick_batch_with_case_insensitive_schema() {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new(
                "Timestamp",
                arrow::datatypes::DataType::Timestamp(
                    arrow::datatypes::TimeUnit::Microsecond,
                    None,
                ),
                false,
            ),
            arrow::datatypes::Field::new("Price", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("Size", arrow::datatypes::DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::TimestampMicrosecondArray::from(vec![
                    1_325_480_400_000_000,
                ])),
                Arc::new(Float64Array::from(vec![76.947])),
                Arc::new(Int64Array::from(vec![3])),
            ],
        )
        .unwrap();

        let update = parse_tick_quote_update(&batch, 0).unwrap();
        assert_eq!(update.timestamp_ns, 1_325_480_400_000_000_000);
        assert_eq!(update.bid, Some(76.947));
        assert_eq!(update.ask, Some(76.947));
    }

    #[test]
    fn parses_tick_quote_update_from_type_column() {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new(
                "Timestamp",
                arrow::datatypes::DataType::Timestamp(
                    arrow::datatypes::TimeUnit::Microsecond,
                    None,
                ),
                false,
            ),
            arrow::datatypes::Field::new("Symbol", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("Type", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("Price", arrow::datatypes::DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::TimestampMicrosecondArray::from(vec![
                    1_325_480_400_000_000,
                    1_325_480_400_000_000,
                ])),
                Arc::new(StringArray::from(vec!["USDJPY", "USDJPY"])),
                Arc::new(StringArray::from(vec!["Ask", "Bid"])),
                Arc::new(Float64Array::from(vec![158.779, 158.766])),
            ],
        )
        .unwrap();

        let ask_update = parse_tick_quote_update(&batch, 0).unwrap();
        assert_eq!(ask_update.bid, None);
        assert_eq!(ask_update.ask, Some(158.779));

        let bid_update = parse_tick_quote_update(&batch, 1).unwrap();
        assert_eq!(bid_update.bid, Some(158.766));
        assert_eq!(bid_update.ask, None);
    }

    #[tokio::test]
    async fn merges_bid_and_ask_rows_into_paired_ticks() {
        let dir = tempdir().unwrap();
        let base = dir
            .path()
            .join("marketdata/eod-tick-snapshot/symbol=USDJPY/year=2012/month=01/day=02");
        std::fs::create_dir_all(&base).unwrap();

        write_typed_tick_parquet(
            &base.join("20120102T000000Z.parquet"),
            &[
                1_325_484_000_000_000,
                1_325_484_000_000_000,
                1_325_484_001_000_000,
                1_325_484_002_000_000,
            ],
            &["USDJPY"; 4],
            &["Ask", "Bid", "Bid", "Ask"],
            &[76.973, 76.971, 76.972, 76.974],
        )
        .unwrap();

        let cache = MarketDataCache::new();
        let ticks = load_ticks_from_parquet_with_range_cached_from_local_cache_dir(
            &dir.path().to_string_lossy(),
            &cache,
            &build_tick_data_source(&dir.path().to_string_lossy(), "USDJPY"),
            "USDJPY",
            None,
            Some(1_325_480_400_000_000_000),
            Some(1_325_484_002_000_000_000),
        )
        .await
        .unwrap();

        assert_eq!(ticks.len(), 3);
        assert_eq!(ticks[0].timestamp_ns, 1_325_484_000_000_000_000);
        assert_eq!(ticks[0].bid, 76.971);
        assert_eq!(ticks[0].ask, 76.973);
        assert_eq!(ticks[1].timestamp_ns, 1_325_484_001_000_000_000);
        assert_eq!(ticks[1].bid, 76.972);
        assert_eq!(ticks[1].ask, 76.973);
        assert_eq!(ticks[2].timestamp_ns, 1_325_484_002_000_000_000);
        assert_eq!(ticks[2].bid, 76.972);
        assert_eq!(ticks[2].ask, 76.974);
    }

    #[tokio::test]
    async fn loads_ticks_from_local_eod_snapshots() {
        let dir = tempdir().unwrap();
        let base = dir
            .path()
            .join("marketdata/eod-tick-snapshot/symbol=USDJPY/year=2012/month=01/day=02");
        std::fs::create_dir_all(&base).unwrap();

        write_tick_parquet(
            &base.join("20120102T000000Z.parquet"),
            &[1_325_484_000_000_000, 1_325_484_001_000_000],
            &["USDJPY", "USDJPY"],
            &[76.971, 76.972],
            &[4, 5],
        )
        .unwrap();

        let cache = MarketDataCache::new();
        let ticks = load_ticks_from_parquet_with_range_cached_from_local_cache_dir(
            &dir.path().to_string_lossy(),
            &cache,
            &build_tick_data_source(&dir.path().to_string_lossy(), "USDJPY"),
            "USDJPY",
            None,
            Some(1_325_480_400_000_000_000),
            Some(1_325_484_001_000_000_000),
        )
        .await
        .unwrap();

        assert_eq!(ticks.len(), 2);
        assert_eq!(ticks[0].bid, 76.971);
        assert_eq!(ticks[0].ask, 76.971);
        assert_eq!(ticks[1].bid, 76.972);
        assert_eq!(ticks[1].ask, 76.972);
    }

    #[tokio::test]
    async fn tick_range_error_lists_expected_partitions_and_candidates() {
        let dir = tempdir().unwrap();
        let base = dir.path().join(
            "prod-fintech-forex-sg-731833471586/marketdata/eod-tick-snapshot/symbol=USDJPY/year=2025/month=01/day=13",
        );
        std::fs::create_dir_all(&base).unwrap();

        write_tick_parquet(
            &base.join("20250113T000000Z.parquet"),
            &[1_736_726_400_000_000],
            &["USDJPY"],
            &[155.0],
            &[1],
        )
        .unwrap();

        let cache = MarketDataCache::new();
        let source_uri =
            "s3://prod-fintech-forex-sg-731833471586/marketdata/eod-tick-snapshot/symbol=USDJPY";
        let err = load_ticks_from_parquet_with_range_cached_from_local_cache_dir(
            &dir.path().to_string_lossy(),
            &cache,
            source_uri,
            "USDJPY",
            None,
            Some(1_596_844_800_000_000_000),
            Some(1_596_848_400_000_000_000),
        )
        .await
        .unwrap_err();

        let message = err.to_string();
        assert!(message.contains("Requested range: 1596844800000000000"));
        assert!(message.contains("2020-08-08T00:00:00+00:00"));
        assert!(message.contains(
            "Expected parquet partitions: s3://prod-fintech-forex-sg-731833471586/marketdata/eod-tick-snapshot/symbol=USDJPY/year=2020/month=08/day=08/20200808T000000Z.parquet .. s3://prod-fintech-forex-sg-731833471586/marketdata/eod-tick-snapshot/symbol=USDJPY/year=2020/month=08/day=09/20200809T000000Z.parquet"
        ));
        assert!(message.contains("Candidate sources:"));
        assert!(message.contains("20250113T000000Z.parquet"));
    }

    #[tokio::test]
    async fn timestamped_tick_lookup_uses_exact_daily_snapshot() {
        let dir = tempdir().unwrap();
        let base = dir
            .path()
            .join("marketdata/eod-tick-snapshot/symbol=USDJPY/year=2012/month=01/day=02");
        std::fs::create_dir_all(&base).unwrap();

        write_tick_parquet(
            &base.join("20120102T000000Z.parquet"),
            &[1_325_480_400_000_000, 1_325_480_401_000_000],
            &["USDJPY", "USDJPY"],
            &[76.947, 76.948],
            &[2, 3],
        )
        .unwrap();

        let cache = MarketDataCache::new();
        let ticks = load_ticks_from_parquet_with_range_cached_from_local_cache_dir(
            &dir.path().to_string_lossy(),
            &cache,
            &build_tick_data_source(&dir.path().to_string_lossy(), "USDJPY"),
            "USDJPY",
            Some(1_325_484_123_000_000_000),
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(ticks.len(), 2);
        assert_eq!(ticks[0].bid, 76.947);
        assert_eq!(ticks[0].ask, 76.947);
        assert_eq!(ticks[1].bid, 76.948);
        assert_eq!(ticks[1].ask, 76.948);
    }
}
