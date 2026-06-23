"""
Retrieve price bar (OHLCV) data for an FX symbol from cTrader Open API.

Usage:
    from marketdata.pricedata.price_bars import get_price_bars

    df = get_price_bars("USDJPY", time_interval=15)
"""

import threading
from datetime import datetime, timedelta, timezone

import pandas as pd
from twisted.internet import reactor
from ctrader_open_api import Client, TcpProtocol, EndPoints, Protobuf
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAGetAccountListByAccessTokenReq,
    ProtoOAAccountAuthReq,
    ProtoOASymbolsListReq,
    ProtoOAGetTrendbarsReq,
)

from commons.python.appconfig import AppConfig

INTERVAL_TO_PERIOD = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 6,
    15: 7, 30: 8, 60: 9, 240: 10, 720: 11, 1440: 12,
    10080: 13,
}

DEFAULT_PRICE_SCALE = 5

# cTrader Open API per-request response timeout. The library default is 5s, which
# is too aggressive for a busy/slow broker endpoint and was the dominant cause of
# transient `twisted.internet.defer.TimeoutError: (5, 'Deferred')` download
# failures (one slow auth/trendbars reply failed the whole task).
RESPONSE_TIMEOUT_SECONDS = 30
# Overall cap on the full auth -> trendbars fetch (several sequential requests).
FETCH_TIMEOUT_SECONDS = 120


def _symbol_price_scale(symbol) -> int:
    """Return number of decimal places for a cTrader symbol."""
    digits = getattr(symbol, "digits", None)
    if isinstance(digits, int) and digits >= 0:
        return digits
    return DEFAULT_PRICE_SCALE


def _decode_bar_ohlc(
    low_units: int,
    delta_open_units: int,
    delta_high_units: int,
    delta_close_units: int,
    scale: int,
) -> tuple[float, float, float, float]:
    """Decode trendbar integer price units into scaled OHLC floats."""
    divisor = 10 ** scale
    low = round(low_units / divisor, scale)
    open_price = round((low_units + delta_open_units) / divisor, scale)
    high = round((low_units + delta_high_units) / divisor, scale)
    close = round((low_units + delta_close_units) / divisor, scale)
    return open_price, high, low, close


class _PriceBarFetcher:
    """Internal class that connects, authenticates, and fetches trendbars."""

    def __init__(self, host, port, app_id, app_secret, access_token,
                 account_id, symbol_name, period, from_ts, to_ts):
        self.client = Client(host, port, TcpProtocol, numberOfMessagesToSendPerSecond=40)
        self.app_id = app_id
        self.app_secret = app_secret
        self.access_token = access_token
        self.account_id = int(account_id)
        self.symbol_name = symbol_name
        self.period = period
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.symbol_id = None
        self.price_scale = DEFAULT_PRICE_SCALE
        self.result_df = None
        self.error = None

    def run(self):
        self.client.setConnectedCallback(self._on_connected)
        self.client.setMessageReceivedCallback(self._on_message)
        self.client.setDisconnectedCallback(
            lambda c, r: None
        )
        self.client.startService()
        reactor.run(installSignalHandlers=False)

    def _send(self, msg):
        d = self.client.send(msg, responseTimeoutInSeconds=RESPONSE_TIMEOUT_SECONDS)
        d.addErrback(lambda f: self._fail(f"Send error: {f}"))
        return d

    def _fail(self, message):
        self.error = message
        reactor.stop()

    def _on_connected(self, _):
        req = ProtoOAApplicationAuthReq()
        req.clientId = self.app_id
        req.clientSecret = self.app_secret
        self._send(req)

    def _on_message(self, _client, message):
        payload = Protobuf.extract(message)
        name = payload.__class__.__name__

        if name == "ProtoOAApplicationAuthRes":
            req = ProtoOAGetAccountListByAccessTokenReq()
            req.accessToken = self.access_token
            self._send(req)
        elif name == "ProtoOAGetAccountListByAccessTokenRes":
            ids = [a.ctidTraderAccountId for a in payload.ctidTraderAccount]
            if self.account_id not in ids:
                self._fail(f"Account {self.account_id} not found. Available: {ids}")
                return
            req = ProtoOAAccountAuthReq()
            req.ctidTraderAccountId = self.account_id
            req.accessToken = self.access_token
            self._send(req)
        elif name == "ProtoOAAccountAuthRes":
            req = ProtoOASymbolsListReq()
            req.ctidTraderAccountId = self.account_id
            self._send(req)
        elif name == "ProtoOASymbolsListRes":
            sym = next((s for s in payload.symbol if s.symbolName == self.symbol_name), None)
            if not sym:
                self._fail(f"Symbol not found: {self.symbol_name}")
                return
            self.symbol_id = sym.symbolId
            self.price_scale = _symbol_price_scale(sym)
            req = ProtoOAGetTrendbarsReq()
            req.ctidTraderAccountId = self.account_id
            req.symbolId = self.symbol_id
            req.period = self.period
            req.fromTimestamp = self.from_ts
            req.toTimestamp = self.to_ts
            self._send(req)
        elif name == "ProtoOAGetTrendbarsRes":
            self._handle_trendbars(payload)
        elif name == "ProtoOAErrorRes":
            self._fail(f"API error: {payload}")
        elif name == "ProtoOAAccountsTokenInvalidatedEvent":
            self._fail("Access token invalidated.")

    def _handle_trendbars(self, payload):
        bars = list(payload.trendbar)
        if not bars:
            self.result_df = pd.DataFrame(
                columns=["Timestamp", "Symbol", "Open", "High", "Low", "Close", "Volume"]
            )
            reactor.stop()
            return

        rows = []
        for bar in bars:
            open_price, high, low, close = _decode_bar_ohlc(
                low_units=bar.low,
                delta_open_units=bar.deltaOpen,
                delta_high_units=bar.deltaHigh,
                delta_close_units=bar.deltaClose,
                scale=self.price_scale,
            )
            rows.append({
                "Timestamp": datetime.fromtimestamp(
                    bar.utcTimestampInMinutes * 60, tz=timezone.utc
                ),
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": bar.volume,
            })

        df = pd.DataFrame(rows)
        df["Symbol"] = self.symbol_name
        self.result_df = df[["Timestamp", "Symbol", "Open", "High", "Low", "Close", "Volume"]]
        reactor.stop()


def get_price_bars(
    fx_symbol: str,
    time_interval: int = 15,
    end_dt: datetime | None = None,
    time_window_minutes: int | None = None,
) -> pd.DataFrame:
    """Retrieve OHLCV price bars for an FX symbol.

    Args:
        fx_symbol: The FX symbol (e.g. "USDJPY", "EURUSD").
        time_interval: Bar interval in minutes. Valid values:
            1, 2, 3, 4, 5, 10, 15, 30, 60, 240, 720, 1440.
        end_dt: End of the data window (defaults to now UTC).
        time_window_minutes: How far back from end_dt to fetch
            (defaults to 60 * time_interval).

    Returns:
        A DataFrame with columns: Timestamp, Symbol, Open, High, Low, Close, Volume.

    Raises:
        ValueError: If time_interval is not a supported value.
        RuntimeError: If the API call fails.
    """
    if time_interval not in INTERVAL_TO_PERIOD:
        raise ValueError(
            f"Unsupported time_interval={time_interval}. "
            f"Choose from: {sorted(INTERVAL_TO_PERIOD.keys())}"
        )

    config = AppConfig()

    if not all([config.app_client_id, config.app_client_secret, config.default_account_id]):
        raise RuntimeError(
            "Missing credentials in AppConfig. "
            "Ensure APP_CLIENT_ID, APP_CLIENT_SECRET, and DEFAULT_ACCOUNT_ID are set."
        )

    # Always refresh the access token before calling the API
    config.refresh_access_token()

    host = EndPoints.PROTOBUF_DEMO_HOST if config.default_mode == "demo" else EndPoints.PROTOBUF_LIVE_HOST
    port = EndPoints.PROTOBUF_PORT
    period = INTERVAL_TO_PERIOD[time_interval]

    if end_dt is None:
        end_dt = datetime.now(tz=timezone.utc)
    if time_window_minutes is None:
        time_window_minutes = 60 * time_interval

    to_ts = int(end_dt.timestamp() * 1000)
    from_ts = int((end_dt - timedelta(minutes=time_window_minutes)).timestamp() * 1000)

    fetcher = _PriceBarFetcher(
        host, port, config.app_client_id, config.app_client_secret, config.access_token,
        config.default_account_id, fx_symbol, period, from_ts, to_ts,
    )

    thread = threading.Thread(target=fetcher.run, daemon=True)
    thread.start()
    thread.join(timeout=FETCH_TIMEOUT_SECONDS)

    if fetcher.error:
        raise RuntimeError(fetcher.error)
    if fetcher.result_df is None:
        raise RuntimeError("Timed out waiting for price bar data.")

    return fetcher.result_df
