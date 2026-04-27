use std::collections::HashMap;
use std::sync::Arc;

use modelenv_proto::{Bar, News};
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
}
