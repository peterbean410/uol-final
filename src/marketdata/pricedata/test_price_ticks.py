"""
Integration tests for price_ticks.get_price_ticks.

These tests connect to the live cTrader Open API and require valid
credentials in .env (APP_CLIENT_ID, APP_CLIENT_SECRET, ACCESS_TOKEN,
DEFAULT_ACCOUNT_ID).

Run:
    python -m pytest marketdata/pricedata/test_price_ticks.py -v
"""

import pytest
import pandas as pd

from marketdata.pricedata.price_ticks import get_price_ticks


EXPECTED_COLUMNS = ["Timestamp", "Symbol", "Type", "Price"]


@pytest.fixture(scope="module")
def ticks_bid():
    return get_price_ticks("USDJPY", quote_type="bid")


def test_get_price_ticks_bid_returns_dataframe(ticks_bid):
    assert isinstance(ticks_bid, pd.DataFrame)
    assert not ticks_bid.empty


def test_get_price_ticks_bid_has_expected_columns(ticks_bid):
    assert list(ticks_bid.columns) == EXPECTED_COLUMNS


def test_get_price_ticks_bid_type_is_bid(ticks_bid):
    assert (ticks_bid["Type"] == "Bid").all()


def test_get_price_ticks_symbol_matches(ticks_bid):
    assert (ticks_bid["Symbol"] == "USDJPY").all()


def test_get_price_ticks_prices_are_positive(ticks_bid):
    assert (ticks_bid["Price"] > 0).all()


def test_get_price_ticks_timestamps_are_sorted(ticks_bid):
    assert ticks_bid["Timestamp"].is_monotonic_increasing


def test_get_price_ticks_num_ticks_limits_results(ticks_bid):
    assert len(ticks_bid) <= 100


def test_get_price_ticks_invalid_quote_type_raises():
    with pytest.raises(ValueError, match="Unsupported quote_type"):
        get_price_ticks("USDJPY", quote_type="invalid")


def test_get_price_ticks_none_quote_type_does_not_raise_validation():
    """Verify quote_type=None passes validation (does not raise ValueError)."""
    # We only test that None is accepted as a valid quote_type value.
    # The full None integration test (fetching both BID+ASK) is in
    # test_price_ticks_none.py to avoid Twisted reactor reuse issues.
    try:
        get_price_ticks("USDJPY", num_ticks=1, quote_type=None)
    except ValueError:
        pytest.fail("quote_type=None should be accepted")
    except RuntimeError:
        # RuntimeError from reactor reuse is expected in the same process;
        # the important thing is no ValueError was raised.
        pass
