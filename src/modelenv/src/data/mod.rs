use crate::proto;
use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, Float64Array, Int64Array, TimestampNanosecondArray};
use futures::StreamExt;
use object_store::path::Path as ObjectPath;
use object_store::{parse_url, ObjectStore};
use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use std::sync::Arc;
use url::Url;

/// Loads bars from parquet files laid out under the marketdata S3 prefix.
/// We do not synthesise prices here; every bar comes from a row on disk.
pub struct ParquetBarLoader {
    store: Arc<dyn ObjectStore>,
    base: ObjectPath,
}

impl ParquetBarLoader {
    pub fn from_s3_prefix(prefix: &str) -> Result<Self> {
        let url = Url::parse(prefix).with_context(|| format!("invalid prefix url: {prefix}"))?;
        let (store, base) = parse_url(&url).with_context(|| format!("unsupported object store: {prefix}"))?;
        Ok(Self { store: Arc::from(store), base })
    }

    pub async fn load_bars(&self, relative_path: &str) -> Result<Vec<proto::Bar>> {
        let path = if self.base.as_ref().is_empty() {
            ObjectPath::from(relative_path)
        } else {
            ObjectPath::from(format!("{}/{}", self.base.as_ref().trim_end_matches('/'), relative_path.trim_start_matches('/')))
        };

        let meta = self.store.head(&path).await.with_context(|| format!("head {path}"))?;
        let reader = ParquetObjectReader::new(self.store.clone(), meta);
        let builder = ParquetRecordBatchStreamBuilder::new(reader).await?;
        let mut stream = builder.build()?;

        let mut bars = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let ts = batch
                .column_by_name("timestamp")
                .or_else(|| batch.column_by_name("ts"))
                .ok_or_else(|| anyhow!("missing timestamp column"))?;
            let open = col_f64(&batch, "open")?;
            let high = col_f64(&batch, "high")?;
            let low = col_f64(&batch, "low")?;
            let close = col_f64(&batch, "close")?;
            let volume = col_f64(&batch, "volume").unwrap_or_else(|_| Float64Array::from(vec![0.0; batch.num_rows()]));

            let ts_ns: Vec<i64> = if let Some(arr) = ts.as_any().downcast_ref::<TimestampNanosecondArray>() {
                (0..arr.len()).map(|i| arr.value(i)).collect()
            } else if let Some(arr) = ts.as_any().downcast_ref::<Int64Array>() {
                (0..arr.len()).map(|i| arr.value(i)).collect()
            } else {
                return Err(anyhow!("unsupported timestamp column type"));
            };

            for i in 0..batch.num_rows() {
                bars.push(proto::Bar {
                    timestamp_ns: ts_ns[i],
                    open: open.value(i),
                    high: high.value(i),
                    low: low.value(i),
                    close: close.value(i),
                    volume: volume.value(i),
                });
            }
        }
        Ok(bars)
    }
}

fn col_f64(batch: &arrow::record_batch::RecordBatch, name: &str) -> Result<Float64Array> {
    let arr = batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("missing column: {name}"))?;
    arr.as_any()
        .downcast_ref::<Float64Array>()
        .cloned()
        .ok_or_else(|| anyhow!("column {name} is not f64"))
}
