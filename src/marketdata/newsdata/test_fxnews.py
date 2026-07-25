"""
Integration tests for fxnews.get_news_data.

These tests exercise the public interface of get_news_data. The upstream
forexnewsapi.com call is mocked so the suite can run without a live
API key; the contract of parameter shaping and response parsing is
still fully verified.

Run:
    python -m pytest marketdata/newsdata/test_fxnews.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from marketdata.newsdata import fxnews
from marketdata.newsdata.fxnews import get_news_data


SAMPLE_RESPONSE = {
    "data": [
        {
            "news_url": "https://www.actionforex.com/example",
            "image_url": "https://forexnewsapi.snapi.dev/images/v1/placeholders/forex/f38.jpg",
            "title": "USD/JPY Forms a Major Head & Shoulders Pattern",
            "text": "USD/JPY was once again the main target for US Dollar bulls.",
            "source_name": "Action Forex",
            "date": "Fri, 17 Apr 2026 22:02:03 -0400",
            "topics": ["Japan", "Oil", "USA"],
            "sentiment": "Negative",
            "type": "Article",
            "currency": ["USD-JPY"],
        }
    ],
    "total_pages": 200,
    "total_items": 10000,
}

EXPECTED_FIELDS = {
    "news_url", "image_url", "title", "text", "source_name",
    "date", "topics", "sentiment", "type", "currency",
}


def _mock_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


@patch.object(fxnews.requests, "get")
def test_returns_list_of_news_items(mock_get):
    mock_get.return_value = _mock_response(SAMPLE_RESPONSE)

    news = get_news_data("USD-JPY", "04162026", "04172026", api_key="k")

    assert isinstance(news, list)
    assert len(news) == 1


@patch.object(fxnews.requests, "get")
def test_news_items_contain_expected_fields(mock_get):
    mock_get.return_value = _mock_response(SAMPLE_RESPONSE)

    news = get_news_data("USD-JPY", "04162026", "04172026", api_key="k")

    assert EXPECTED_FIELDS.issubset(news[0].keys())


@patch.object(fxnews.requests, "get")
def test_request_targets_expected_endpoint_and_params(mock_get):
    mock_get.return_value = _mock_response(SAMPLE_RESPONSE)

    get_news_data("EUR-USD", "04162026", "04172026", api_key="SECRET")

    args, kwargs = mock_get.call_args
    assert args[0] == fxnews.NEWS_API_URL
    params = kwargs["params"]
    assert params["currencypair"] == "EUR-USD"
    assert params["token"] == "SECRET"
    assert params["date"] == "04162026-04172026"
    assert params["items"] == fxnews.DEFAULT_ITEMS
    assert params["page"] == fxnews.DEFAULT_PAGE


@patch.object(fxnews.requests, "get")
def test_empty_data_returns_empty_list(mock_get):
    mock_get.return_value = _mock_response({"data": [], "total_pages": 0, "total_items": 0})

    news = get_news_data("USD-JPY", "04162026", "04172026", api_key="k")

    assert news == []


@patch.object(fxnews.requests, "get")
def test_missing_data_key_returns_empty_list(mock_get):
    mock_get.return_value = _mock_response({})

    news = get_news_data("USD-JPY", "04162026", "04172026", api_key="k")

    assert news == []


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setenv("FXNEWS_API_KEY", "")
    with pytest.raises(RuntimeError, match="Missing FXNEWS_API_KEY"):
        get_news_data("USD-JPY", "04162026", "04172026", api_key="")
