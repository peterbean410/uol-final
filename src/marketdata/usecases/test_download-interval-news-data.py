"""
Unit tests for download-interval-news-data.

Mocks the upstream news API, S3 upload, time.sleep, and the current time.

Run:
    python -m pytest "marketdata/usecases/test_download-interval-news-data.py" -v
"""

import importlib
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest
import requests

# The module uses a hyphenated filename, so import via importlib
_mod = importlib.import_module("marketdata.usecases.download-interval-news-data")
download_news_data = _mod.download_news_data
_build_s3_key = _mod._build_s3_key
_fetch_page_with_retry = _mod._fetch_page_with_retry
_fetch_all_pages = _mod._fetch_all_pages

BUCKET = "prod-fintech-forex-sg-731833471586"
FX_CURRENCY_PAIR = "USD-JPY"
INTERVAL = "D1"
API_KEY = "TEST_TOKEN"
FIXED_NOW = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

SAMPLE_NEWS = [
    {
        "news_url": "https://example.com/a",
        "image_url": "https://example.com/a.jpg",
        "title": "Headline A",
        "text": "Body A",
        "source_name": "Source A",
        "date": "Wed, 31 Dec 2025 23:30:00 -0400",
        "topics": ["USA"],
        "sentiment": "Neutral",
        "type": "Article",
        "currency": ["USD-JPY"],
    },
    {
        "news_url": "https://example.com/b",
        "image_url": "https://example.com/b.jpg",
        "title": "Headline B",
        "text": "Body B",
        "source_name": "Source B",
        "date": "Thu, 01 Jan 2026 09:00:00 -0400",
        "topics": ["Japan"],
        "sentiment": "Positive",
        "type": "Article",
        "currency": ["USD-JPY"],
    },
]


def _make_news_item(idx: int) -> dict:
    return {
        "news_url": f"https://example.com/{idx}",
        "image_url": f"https://example.com/{idx}.jpg",
        "title": f"Headline {idx}",
        "text": f"Body {idx}",
        "source_name": "Source",
        "date": "Thu, 01 Jan 2026 09:00:00 -0400",
        "topics": ["X"],
        "sentiment": "Neutral",
        "type": "Article",
        "currency": ["USD-JPY"],
    }


# ── Test 1: date range computed from execution timestamp ─────────────

@patch.object(_mod, "_upload_to_s3")
@patch.object(_mod, "get_news_data")
def test_correct_date_range_passed_to_api(mock_get_news, mock_upload):
    """start_date = end_dt - 1 day, both formatted MMDDYYYY."""
    mock_get_news.return_value = SAMPLE_NEWS

    download_news_data(FX_CURRENCY_PAIR, INTERVAL, FIXED_NOW, API_KEY, BUCKET)

    mock_get_news.assert_called_once()
    kwargs = mock_get_news.call_args.kwargs
    assert kwargs["start_date"] == "12312025"
    assert kwargs["end_date"] == "01012026"
    assert kwargs["api_key"] == API_KEY
    assert kwargs["page"] == 1
    assert kwargs["items"] == _mod.DEFAULT_PAGE_SIZE


# ── Test 2: news data parsed into DataFrame and uploaded ─────────────

@patch.object(_mod, "_upload_to_s3")
@patch.object(_mod, "get_news_data")
def test_news_parsed_into_dataframe(mock_get_news, mock_upload):
    mock_get_news.return_value = SAMPLE_NEWS

    df = download_news_data(FX_CURRENCY_PAIR, INTERVAL, FIXED_NOW, API_KEY, BUCKET)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    expected_cols = {
        "news_url", "image_url", "title", "text", "source_name",
        "date", "topics", "sentiment", "type", "currency",
    }
    assert expected_cols.issubset(set(df.columns))
    mock_upload.assert_called_once()


# ── Test 3: S3 upload path and filename format ───────────────────────

@patch.object(_mod, "_upload_to_s3")
@patch.object(_mod, "get_news_data")
def test_s3_upload_path_and_filename(mock_get_news, mock_upload):
    mock_get_news.return_value = SAMPLE_NEWS

    download_news_data(FX_CURRENCY_PAIR, INTERVAL, FIXED_NOW, API_KEY, BUCKET)

    args, _ = mock_upload.call_args
    uploaded_df, bucket, key = args
    assert bucket == BUCKET
    assert isinstance(uploaded_df, pd.DataFrame)
    assert key == (
        "marketdata/interval-news/symbol=USD-JPY/interval=D1/"
        "year=2026/month=01/day=01/20260101T100000Z.parquet"
    )


def test_build_s3_key_format():
    key = _build_s3_key(FX_CURRENCY_PAIR, INTERVAL, FIXED_NOW)
    assert key == (
        "marketdata/interval-news/symbol=USD-JPY/interval=D1/"
        "year=2026/month=01/day=01/20260101T100000Z.parquet"
    )


# ── Test 4: empty data results in no upload ──────────────────────────

@patch.object(_mod, "_upload_to_s3")
@patch.object(_mod, "get_news_data")
def test_empty_data_skips_upload(mock_get_news, mock_upload):
    mock_get_news.return_value = []

    df = download_news_data(FX_CURRENCY_PAIR, INTERVAL, FIXED_NOW, API_KEY, BUCKET)

    assert df.empty
    mock_upload.assert_not_called()


# ── Test 5: retry succeeds on 2nd attempt with backoff ───────────────

@patch.object(_mod.time, "sleep")
@patch.object(_mod, "get_news_data")
def test_retry_recovers_after_transient_error(mock_get_news, mock_sleep):
    mock_get_news.side_effect = [
        requests.ConnectionError("boom"),
        SAMPLE_NEWS,
    ]

    result = _fetch_page_with_retry(
        FX_CURRENCY_PAIR, "12312025", "01012026", API_KEY, page=1, items=_mod.DEFAULT_PAGE_SIZE
    )

    assert result == SAMPLE_NEWS
    assert mock_get_news.call_count == 2
    mock_sleep.assert_called_once_with(_mod.INITIAL_BACKOFF_SECONDS)


# ── Test 6: retry exhausts and raises after MAX_RETRIES failures ─────

@patch.object(_mod.time, "sleep")
@patch.object(_mod, "get_news_data")
def test_retry_exhaustion_raises_and_logs(mock_get_news, mock_sleep, caplog):
    err = requests.ConnectionError("permanent")
    mock_get_news.side_effect = [err, err, err]

    with caplog.at_level("ERROR", logger=_mod.logger.name):
        with pytest.raises(requests.ConnectionError, match="permanent"):
            _fetch_page_with_retry(
                FX_CURRENCY_PAIR, "12312025", "01012026", API_KEY,
                page=1, items=_mod.DEFAULT_PAGE_SIZE,
            )

    assert mock_get_news.call_count == _mod.MAX_RETRIES
    # exponential backoff: 1s after attempt 1, 2s after attempt 2, no sleep after final attempt
    sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
    assert sleep_calls == [1, 2]
    assert any("failed after" in r.message for r in caplog.records)


# ── Test 7: API-error (RuntimeError) also retries ────────────────────

@patch.object(_mod.time, "sleep")
@patch.object(_mod, "get_news_data")
def test_runtime_error_triggers_retry(mock_get_news, mock_sleep):
    mock_get_news.side_effect = [
        RuntimeError("rate limit"),
        RuntimeError("token expired"),
        SAMPLE_NEWS,
    ]

    result = _fetch_page_with_retry(
        FX_CURRENCY_PAIR, "12312025", "01012026", API_KEY, page=1, items=_mod.DEFAULT_PAGE_SIZE
    )

    assert result == SAMPLE_NEWS
    assert mock_get_news.call_count == 3
    sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
    assert sleep_calls == [1, 2]


# ── Test 8: pagination, full page triggers next fetch ───────────────

@patch.object(_mod, "get_news_data")
def test_pagination_loops_until_partial_page(mock_get_news):
    """When a page returns exactly `items` records, fetch the next page;
    stop when a page returns fewer than `items`."""
    items = 3
    full_page = [_make_news_item(i) for i in range(items)]
    partial_page = [_make_news_item(i) for i in range(items, items + 2)]

    mock_get_news.side_effect = [full_page, full_page, partial_page]

    result = _fetch_all_pages(
        FX_CURRENCY_PAIR, "12312025", "01012026", API_KEY, items=items
    )

    assert len(result) == items * 2 + 2
    assert mock_get_news.call_count == 3
    pages_requested = [c.kwargs["page"] for c in mock_get_news.call_args_list]
    assert pages_requested == [1, 2, 3]
    items_requested = [c.kwargs["items"] for c in mock_get_news.call_args_list]
    assert items_requested == [items, items, items]


# ── Test 9: pagination, first short page exits immediately ──────────

@patch.object(_mod, "get_news_data")
def test_pagination_single_short_page(mock_get_news):
    items = 100
    mock_get_news.return_value = [_make_news_item(i) for i in range(5)]

    result = _fetch_all_pages(
        FX_CURRENCY_PAIR, "12312025", "01012026", API_KEY, items=items
    )

    assert len(result) == 5
    assert mock_get_news.call_count == 1
