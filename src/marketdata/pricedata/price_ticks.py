"""
Retrieve the last N price tick updates for an FX symbol from cTrader Open API.

Usage:
    from marketdata.pricedata.price_ticks import get_price_ticks

    df = get_price_ticks("USDJPY", num_ticks=100)                    # both BID and ASK (default)
    df = get_price_ticks("USDJPY", num_ticks=100, quote_type="bid") # BID only
    df = get_price_ticks("USDJPY", num_ticks=100, quote_type="ask") # ASK only
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
    ProtoOAGetTickDataReq,
)

from commons.python.appconfig import AppConfig

PRICE_DIVISOR = 100_000

# Quote type mapping
QUOTE_TYPE_BID = 1
QUOTE_TYPE_ASK = 2


class _PriceTickFetcher:
    """Internal class that connects, authenticates, and fetches tick data.

    Supports fetching a single quote type, or both BID and ASK sequentially
    within a single reactor session (when quote_types has two entries).
    """

    def __init__(self, host, port, app_id, app_secret, access_token,
                 account_id, symbol_name, quote_types, from_ts, to_ts, num_ticks):
        self.client = Client(host, port, TcpProtocol, numberOfMessagesToSendPerSecond=40)
        self.app_id = app_id
        self.app_secret = app_secret
        self.access_token = access_token
        self.account_id = int(account_id)
        self.symbol_name = symbol_name
        self.quote_types = list(quote_types)  # e.g. [1] for BID, [1, 2] for both
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.num_ticks = num_ticks
        self.symbol_id = None

        # State for current fetch phase
        self._phase_idx = 0
        self._current_qt = self.quote_types[0]
        self._phase_ticks = []

        # Accumulated results per quote type
        self._results: dict[int, list[dict]] = {}

        self.result_df = None
        self.error = None

    def run(self):
        self.client.setConnectedCallback(self._on_connected)
        self.client.setMessageReceivedCallback(self._on_message)
        self.client.setDisconnectedCallback(lambda c, r: None)
        self.client.startService()
        reactor.run(installSignalHandlers=False)

    def _send(self, msg):
        d = self.client.send(msg)
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
            self._start_phase()
        elif name == "ProtoOAGetTickDataRes":
            self._handle_tick_data(payload)
        elif name == "ProtoOAErrorRes":
            self._fail(f"API error: {payload}")
        elif name == "ProtoOAAccountsTokenInvalidatedEvent":
            self._fail("Access token invalidated.")

    def _start_phase(self):
        """Begin fetching ticks for the current quote type phase."""
        self._current_qt = self.quote_types[self._phase_idx]
        self._phase_ticks = []
        self._request_ticks(self.from_ts)

    def _request_ticks(self, from_ts):
        req = ProtoOAGetTickDataReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = self.symbol_id
        req.type = self._current_qt
        req.fromTimestamp = from_ts
        req.toTimestamp = self.to_ts
        self._send(req)

    def _handle_tick_data(self, payload):
        ticks = list(payload.tickData)

        if not ticks and not self._phase_ticks:
            self._results[self._current_qt] = []
            self._advance_or_finalize()
            return

        decoded = self._decode_ticks(ticks)
        self._phase_ticks.extend(decoded)

        if payload.hasMore and decoded:
            last_ts = decoded[-1]["timestamp_ms"]
            self._request_ticks(last_ts + 1)
            return

        kept = self._phase_ticks if self.num_ticks is None else self._phase_ticks[-self.num_ticks:]
        self._results[self._current_qt] = kept
        self._advance_or_finalize()

    def _advance_or_finalize(self):
        """Move to the next quote type phase, or finalize if all done."""
        self._phase_idx += 1
        if self._phase_idx < len(self.quote_types):
            self._start_phase()
        else:
            self._finalize()

    def _decode_ticks(self, ticks):
        decoded = []
        prev_ts = 0
        prev_tick = 0

        for t in ticks:
            abs_ts = prev_ts + t.timestamp
            abs_tick = prev_tick + t.tick
            prev_ts = abs_ts
            prev_tick = abs_tick
            decoded.append({
                "timestamp_ms": abs_ts,
                "price_raw": abs_tick,
            })

        return decoded

    def _finalize(self):
        rows = []
        for qt, ticks in self._results.items():
            quote_label = "Bid" if qt == QUOTE_TYPE_BID else "Ask"
            for t in ticks:
                rows.append({
                    "Timestamp": datetime.fromtimestamp(
                        t["timestamp_ms"] / 1000, tz=timezone.utc
                    ),
                    "Symbol": self.symbol_name,
                    "Type": quote_label,
                    "Price": t["price_raw"] / PRICE_DIVISOR,
                })

        columns = ["Timestamp", "Symbol", "Type", "Price"]
        df = pd.DataFrame(rows, columns=columns)
        if not df.empty:
            df.sort_values("Timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
        self.result_df = df[columns]
        reactor.stop()


def get_price_ticks(
    fx_symbol: str,
    num_ticks: int | None = 100,
    quote_type: str | None = None,
    end_dt: datetime | None = None,
    time_window_minutes: int = 60,
) -> pd.DataFrame:
    """Retrieve price tick updates for an FX symbol within a configurable time window.

    Args:
        fx_symbol: The FX symbol (e.g. "USDJPY", "EURUSD").
        num_ticks: Number of the last price tick updates to retrieve per quote type.
            When None, all ticks in the window are returned.
        quote_type: "bid", "ask", or None. When None, fetches and returns
            both BID and ASK price ticks combined into a single DataFrame.
        end_dt: End of the data window. Defaults to datetime.now(tz=timezone.utc).
        time_window_minutes: How far back from end_dt to fetch. Defaults to 60.

    Returns:
        A DataFrame with columns: Timestamp, Symbol, Type, Price.

    Raises:
        ValueError: If quote_type is not "bid", "ask", or None.
        RuntimeError: If the API call fails.
    """
    if quote_type is not None and quote_type not in ("bid", "ask"):
        raise ValueError(f"Unsupported quote_type={quote_type!r}. Choose 'bid', 'ask', or None.")

    config = AppConfig()

    if not all([config.app_client_id, config.app_client_secret, config.default_account_id]):
        raise RuntimeError(
            "Missing credentials in AppConfig. "
            "Ensure APP_CLIENT_ID, APP_CLIENT_SECRET, and DEFAULT_ACCOUNT_ID are set."
        )

    config.refresh_access_token()

    host = EndPoints.PROTOBUF_DEMO_HOST if config.default_mode == "demo" else EndPoints.PROTOBUF_LIVE_HOST
    port = EndPoints.PROTOBUF_PORT

    if quote_type is None:
        quote_types = [QUOTE_TYPE_BID, QUOTE_TYPE_ASK]
    elif quote_type == "bid":
        quote_types = [QUOTE_TYPE_BID]
    else:
        quote_types = [QUOTE_TYPE_ASK]

    if end_dt is None:
        end_dt = datetime.now(tz=timezone.utc)
    to_ts = int(end_dt.timestamp() * 1000)
    from_ts = int((end_dt - timedelta(minutes=time_window_minutes)).timestamp() * 1000)

    fetcher = _PriceTickFetcher(
        host, port, config.app_client_id, config.app_client_secret, config.access_token,
        config.default_account_id, fx_symbol, quote_types, from_ts, to_ts, num_ticks,
    )

    thread = threading.Thread(target=fetcher.run, daemon=True)
    thread.start()
    thread.join(timeout=60)

    if fetcher.error:
        raise RuntimeError(fetcher.error)
    if fetcher.result_df is None:
        raise RuntimeError("Timed out waiting for price tick data.")

    return fetcher.result_df
