use std::collections::HashMap;
use std::sync::Arc;

use modelenv_proto::{Bar, News, Tick};
use tokio::sync::Mutex;

#[derive(Clone, Debug)]
pub enum CachedLatestSource {
    Present(String),
    Missing(String),
}

#[derive(Default)]
struct MarketDataCacheState {
    latest_sources: HashMap<String, CachedLatestSource>,
    price_bars: HashMap<String, Vec<Bar>>,
    news_items: HashMap<String, Vec<News>>,
    tick_items: HashMap<String, Vec<Tick>>,
    s3_listings: HashMap<String, Vec<String>>,
}

#[derive(Clone, Default)]
pub struct MarketDataCache {
    state: Arc<Mutex<MarketDataCacheState>>,
}

impl MarketDataCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn latest_source(&self, cache_key: &str) -> Option<CachedLatestSource> {
        self.state
            .lock()
            .await
            .latest_sources
            .get(cache_key)
            .cloned()
    }

    pub async fn put_latest_source(&self, cache_key: String, value: CachedLatestSource) {
        self.state
            .lock()
            .await
            .latest_sources
            .insert(cache_key, value);
    }

    pub async fn price_bars(&self, source: &str) -> Option<Vec<Bar>> {
        self.state.lock().await.price_bars.get(source).cloned()
    }

    pub async fn put_price_bars(&self, source: String, bars: Vec<Bar>) {
        self.state.lock().await.price_bars.insert(source, bars);
    }

    pub async fn news_items(&self, source: &str) -> Option<Vec<News>> {
        self.state.lock().await.news_items.get(source).cloned()
    }

    pub async fn put_news_items(&self, source: String, news: Vec<News>) {
        self.state.lock().await.news_items.insert(source, news);
    }

    pub async fn tick_items(&self, source: &str) -> Option<Vec<Tick>> {
        self.state.lock().await.tick_items.get(source).cloned()
    }

    pub async fn put_tick_items(&self, source: String, ticks: Vec<Tick>) {
        self.state.lock().await.tick_items.insert(source, ticks);
    }

    /// Cached output of `aws s3api list-objects-v2 --prefix <source_uri>`.
    /// Keyed by the S3 prefix URI. The listing is reused across the session
    /// so per-episode tick / bar resolution doesn't re-pay the ~10-15s
    /// `list-objects-v2` cost when the bucket has tens of thousands of keys.
    pub async fn s3_listing(&self, source_uri: &str) -> Option<Vec<String>> {
        self.state.lock().await.s3_listings.get(source_uri).cloned()
    }

    pub async fn put_s3_listing(&self, source_uri: String, keys: Vec<String>) {
        self.state.lock().await.s3_listings.insert(source_uri, keys);
    }
}
