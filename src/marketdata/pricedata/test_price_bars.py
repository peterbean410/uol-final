"""
Integration tests for price_bars.get_price_bars.

These tests connect to the live cTrader Open API and require valid
credentials in .env (APP_CLIENT_ID, APP_CLIENT_SECRET, ACCESS_TOKEN,
DEFAULT_ACCOUNT_ID).

Run:
    python -m pytest marketdata/pricedata/test_price_bars.py -v
"""

import pytest
import pandas as pd

from marketdata.pricedata.price_bars import get_price_bars


@pytest.fixture(scope="module")
def bars_60m():
    return get_price_bars("USDJPY", time_interval=60)


def test_get_price_bars_returns_dataframe(bars_60m):
    assert isinstance(bars_60m, pd.DataFrame)
    assert not bars_60m.empty


def test_get_price_bars_has_expected_columns(bars_60m):
    expected = ["Timestamp", "Symbol", "Open", "High", "Low", "Close", "Volume"]
    assert list(bars_60m.columns) == expected


def test_get_price_bars_symbol_matches(bars_60m):
    assert (bars_60m["Symbol"] == "USDJPY").all()


def test_get_price_bars_prices_are_positive(bars_60m):
    for col in ["Open", "High", "Low", "Close"]:
        assert (bars_60m[col] > 0).all(), f"{col} contains non-positive values"


def test_get_price_bars_high_gte_low(bars_60m):
    assert (bars_60m["High"] >= bars_60m["Low"]).all()


def test_get_price_bars_invalid_interval_raises():
    with pytest.raises(ValueError, match="Unsupported time_interval"):
        get_price_bars("USDJPY", time_interval=7)
