"""
Tests for news_labeller.

The LLM endpoint is mocked so the suite runs without a live model, while still
verifying request shaping, response parsing, dedup, ordering and the
fail-soft contract that a labelling outage returns None rather than raising.

Run:
    python -m pytest marketdata/newsdata/test_news_labeller.py -v
"""

import json
from unittest.mock import MagicMock, patch

import requests

from marketdata.newsdata import news_labeller
from marketdata.newsdata.news_labeller import (
    LABEL_COLUMNS,
    label_headline,
    label_headlines,
)

ENDPOINT = "http://litellm.default.svc.cluster.local:4000"


def _mock_response(content: str):
    resp = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    resp.raise_for_status.return_value = None
    return resp


@patch.object(news_labeller.requests, "post")
def test_parses_both_flags(mock_post):
    mock_post.return_value = _mock_response(
        '{"boj_policy": true, "jpy_intervention": false}'
    )

    result = label_headline("USD/JPY slips as BoJ holds rates", ENDPOINT)

    assert result == {"boj_policy": True, "jpy_intervention": False}


@patch.object(news_labeller.requests, "post")
def test_tolerates_markdown_fenced_json(mock_post):
    mock_post.return_value = _mock_response(
        '```json\n{"boj_policy": false, "jpy_intervention": true}\n```'
    )

    result = label_headline("Japan warns against sharp currency moves", ENDPOINT)

    assert result == {"boj_policy": False, "jpy_intervention": True}


@patch.object(news_labeller.requests, "post")
def test_request_shape_and_auth_header(mock_post):
    mock_post.return_value = _mock_response(
        '{"boj_policy": true, "jpy_intervention": true}'
    )

    label_headline("BoJ decision", ENDPOINT, model="gemma-4-31b-it", api_key="SECRET")

    args, kwargs = mock_post.call_args
    assert args[0] == f"{ENDPOINT}/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer SECRET"
    body = kwargs["json"]
    assert body["model"] == "gemma-4-31b-it"
    assert body["temperature"] == 0
    assert body["messages"][1]["content"] == "BoJ decision"


@patch.object(news_labeller.requests, "post")
def test_no_auth_header_when_key_absent(mock_post):
    mock_post.return_value = _mock_response(
        '{"boj_policy": true, "jpy_intervention": false}'
    )

    label_headline("BoJ decision", ENDPOINT)

    assert "Authorization" not in mock_post.call_args.kwargs["headers"]


@patch.object(news_labeller.time, "sleep", lambda _: None)
@patch.object(news_labeller.requests, "post")
def test_returns_none_after_persistent_failure(mock_post):
    mock_post.side_effect = requests.ConnectionError("endpoint down")

    assert label_headline("BoJ decision", ENDPOINT) is None
    assert mock_post.call_count == news_labeller.MAX_RETRIES


@patch.object(news_labeller.time, "sleep", lambda _: None)
@patch.object(news_labeller.requests, "post")
def test_retries_then_succeeds(mock_post):
    mock_post.side_effect = [
        requests.Timeout("slow"),
        _mock_response('{"boj_policy": true, "jpy_intervention": false}'),
    ]

    assert label_headline("BoJ decision", ENDPOINT) == {
        "boj_policy": True,
        "jpy_intervention": False,
    }


@patch.object(news_labeller.time, "sleep", lambda _: None)
@patch.object(news_labeller.requests, "post")
def test_malformed_json_returns_none(mock_post):
    mock_post.return_value = _mock_response("I think this one is about the BOJ.")

    assert label_headline("BoJ decision", ENDPOINT) is None


def test_blank_title_returns_none_without_calling_endpoint():
    with patch.object(news_labeller.requests, "post") as mock_post:
        assert label_headline("   ", ENDPOINT) is None
        mock_post.assert_not_called()


@patch.object(news_labeller.requests, "post")
def test_duplicate_titles_labelled_once_and_order_preserved(mock_post):
    def respond(*_args, **kwargs):
        title = kwargs["json"]["messages"][1]["content"]
        return _mock_response(
            json.dumps({"boj_policy": title.startswith("BoJ"), "jpy_intervention": False})
        )

    mock_post.side_effect = respond

    titles = ["BoJ holds", "Chart update", "BoJ holds", "Chart update", "BoJ holds"]
    results = label_headlines(titles, ENDPOINT, max_workers=2)

    assert [r["boj_policy"] for r in results] == [True, False, True, False, True]
    assert mock_post.call_count == 2


def test_empty_input_returns_empty_list():
    assert label_headlines([], ENDPOINT) == []


def test_label_columns_are_the_documented_pair():
    assert LABEL_COLUMNS == ("boj_policy", "jpy_intervention")
