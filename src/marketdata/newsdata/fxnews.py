"""
Retrieve news for an FX currency pair from forexnewsapi.com.

Usage:
    from marketdata.newsdata.fxnews import get_news_data

    news = get_news_data("USD-JPY", start_date="04162026", end_date="04172026")
"""

import requests

from commons.python.appconfig import AppConfig

NEWS_API_URL = "https://forexnewsapi.com/api/v1"
DEFAULT_ITEMS = 100
DEFAULT_PAGE = 1
REQUEST_TIMEOUT_SECONDS = 30


def get_news_data(
    fx_currency_pair: str,
    start_date: str,
    end_date: str,
    api_key: str | None = None,
    page: int = DEFAULT_PAGE,
    items: int = DEFAULT_ITEMS,
) -> list[dict]:
    """Retrieve news items for an FX currency pair within a date range.

    Args:
        fx_currency_pair: FX pair in dash format (e.g. "USD-JPY", "EUR-USD").
        start_date: Start date in MMDDYYYY format (e.g. "04162026").
        end_date: End date in MMDDYYYY format (e.g. "04172026").
        api_key: forexnewsapi.com API token. If None or empty,
            falls back to AppConfig().fxnews_api_key.
        page: Page number to fetch (1-indexed).
        items: Number of items per page.

    Returns:
        A list of news dicts, each with keys: news_url, image_url, title,
        text, source_name, date, topics, sentiment, type, currency.

    Raises:
        RuntimeError: If no API key can be resolved.
        requests.HTTPError: If the upstream API returns a non-2xx response.
    """
    if not api_key:
        api_key = AppConfig().fxnews_api_key
    if not api_key:
        raise RuntimeError(
            "Missing FXNEWS_API_KEY for forexnewsapi.com. "
            "Set the FXNEWS_API_KEY environment variable or pass api_key explicitly."
        )

    params = {
        "currencypair": fx_currency_pair,
        "items": items,
        "page": page,
        "token": api_key,
        "date": f"{start_date}-{end_date}",
    }

    resp = requests.get(NEWS_API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("data", [])
