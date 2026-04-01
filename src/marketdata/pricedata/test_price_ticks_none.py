"""
Integration test for price_ticks.get_price_ticks with quote_type=None.

This test is in a separate file because Twisted's reactor cannot be
restarted within the same process, and quote_type=None requires two
sequential reactor runs (one for BID, one for ASK).

Run:
    python -m pytest marketdata/pricedata/test_price_ticks_none.py -v
"""

import pytest
import pandas as pd

from marketdata.pricedata.price_ticks import get_price_ticks


@pytest.fixture(scope="module")
def ticks_both():
    return get_price_ticks("USDJPY", num_ticks=50, quote_type=None)


def test_get_price_ticks_none_returns_dataframe(ticks_both):
    assert isinstance(ticks_both, pd.DataFrame)
    print(ticks_both)
    assert not ticks_both.empty


def test_get_price_ticks_none_contains_both_types(ticks_both):
    types = set(ticks_both["Type"].unique())
    assert types == {"Bid", "Ask"}


def test_get_price_ticks_none_has_expected_columns(ticks_both):
    expected = ["Timestamp", "Symbol", "Type", "Price"]
    assert list(ticks_both.columns) == expected


def test_get_price_ticks_none_timestamps_sorted(ticks_both):
    assert ticks_both["Timestamp"].is_monotonic_increasing


def test_get_price_ticks_none_prices_positive(ticks_both):
    assert (ticks_both["Price"] > 0).all()
