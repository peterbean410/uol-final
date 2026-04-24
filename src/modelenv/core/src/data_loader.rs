// Data loader module for loading price bars from S3 parquet files
use anyhow::{anyhow, Result};
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::time::{SystemTime, UNIX_EPOCH};

use modelenv_proto::Bar;

/// Supported time intervals for price bars
pub const TIME_INTERVALS: &[&str] = &["M1", "M5", "M15", "H1", "H4", "D1", "W1", "MN"];

/// Load price bars from a parquet file (S3 or local file)
pub async fn load_bars_from_parquet(
    s3_uri: &str,
    symbol: &str,
    time_interval: &str,
) -> Result<Vec<Bar>> {
    load_bars_from_parquet_with_end_ts(s3_uri, symbol, time_interval, i64::MAX).await
}

/// Load price bars from a parquet file, stopping at the given end timestamp
pub async fn load_bars_from_parquet_with_end_ts(
    s3_uri: &str,
    symbol: &str,
    time_interval: &str,
    end_timestamp_ns: i64,
) -> Result<Vec<Bar>> {
    // For now, only support local file:// URIs
    if s3_uri.starts_with("file://") {
        let local_path = s3_uri.strip_prefix("file://")
            .ok_or_else(|| anyhow!("Invalid file:// URI"))?;
        return load_bars_from_local_file_with_end_ts(local_path, symbol, time_interval, end_timestamp_ns).await;
    }
    
    // For local testing, try as a local file path
    load_bars_from_local_file_with_end_ts(s3_uri, symbol, time_interval, end_timestamp_ns).await
}

/// Load price bars from a local file
async fn load_bars_from_local_file(
    file_path: &str,
    symbol: &str,
    time_interval: &str,
) -> Result<Vec<Bar>> {
    load_bars_from_local_file_with_end_ts(file_path, symbol, time_interval, i64::MAX).await
}

/// Load price bars from a local file, stopping at the given end timestamp
async fn load_bars_from_local_file_with_end_ts(
    file_path: &str,
    symbol: &str,
    time_interval: &str,
    end_timestamp_ns: i64,
) -> Result<Vec<Bar>> {
    // Read the entire file into memory first
    let bytes = tokio::fs::read(file_path).await
        .map_err(|e| anyhow!("Failed to read file {}: {}", file_path, e))?;
    
    // Convert to bytes::Bytes for the parquet reader
    let bytes = bytes::Bytes::from(bytes);
    
    // Create a reader from the bytes
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

/// Parse a Bar from a RecordBatch at the given index
pub fn parse_bar_from_batch(batch: &RecordBatch, index: usize) -> Result<Bar> {
    let timestamp = batch
        .column_by_name("timestamp")
        .ok_or_else(|| anyhow!("Missing timestamp column"))?;
    
    let open = batch
        .column_by_name("open")
        .ok_or_else(|| anyhow!("Missing open column"))?;
    
    let high = batch
        .column_by_name("high")
        .ok_or_else(|| anyhow!("Missing high column"))?;
    
    let low = batch
        .column_by_name("low")
        .ok_or_else(|| anyhow!("Missing low column"))?;
    
    let close = batch
        .column_by_name("close")
        .ok_or_else(|| anyhow!("Missing close column"))?;
    
    // Volume is optional - default to 0.0 if missing
    let volume = batch
        .column_by_name("volume")
        .map(|v| {
            if let Some(v) = v.as_any().downcast_ref::<arrow::array::Float64Array>() {
                v.value(index)
            } else {
                0.0
            }
        })
        .unwrap_or(0.0);
    
    let timestamp_ns = match timestamp.as_any().downcast_ref::<arrow::array::Int64Array>() {
        Some(arr) => parse_timestamp(arr, index),
        None => return Err(anyhow!("Invalid timestamp format")),
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

/// Load bars from a parquet file and parse them into Bar messages
pub async fn load_and_parse_bars(_s3_uri: &str) -> Result<Vec<Bar>> {
    // This is a placeholder - actual implementation would:
    // 1. Download parquet file from S3
    // 2. Create ParquetRecordBatchReaderBuilder
    // 3. Read and parse each batch
    
    // For now, return empty vector
    Ok(Vec::new())
}

/// Get the current timestamp in nanoseconds
pub fn now_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as i64
}
