from types import SimpleNamespace

from marketdata.pricedata.price_bars import (
    DEFAULT_PRICE_SCALE,
    _decode_bar_ohlc,
    _symbol_price_scale,
)


def test_symbol_price_scale_uses_symbol_digits():
    symbol = SimpleNamespace(digits=3)
    assert _symbol_price_scale(symbol) == 3


def test_symbol_price_scale_falls_back_when_digits_missing():
    symbol = SimpleNamespace()
    assert _symbol_price_scale(symbol) == DEFAULT_PRICE_SCALE


def test_symbol_price_scale_falls_back_when_digits_invalid():
    symbol = SimpleNamespace(digits=-1)
    assert _symbol_price_scale(symbol) == DEFAULT_PRICE_SCALE


def test_decode_bar_ohlc_respects_scale():
    open_price, high, low, close = _decode_bar_ohlc(
        low_units=135560,
        delta_open_units=7,
        delta_high_units=20,
        delta_close_units=9,
        scale=3,
    )

    assert open_price == 135.567
    assert high == 135.58
    assert low == 135.56
    assert close == 135.569
