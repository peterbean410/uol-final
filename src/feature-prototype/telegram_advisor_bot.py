#!/usr/bin/env python3
"""Telegram advisor bot for the USD/JPY DQN+PF system, long-polling (getUpdates).

A non-technical user messages the bot and gets the advisor's recommendation,
computed through the **real** production screen + gate (`IntegrationLayer`):

    /advice   -> screened action + forecast (mu, sigma) + screen reason + gate state
    /start, /help -> usage

Receive/send is the Telegram Bot HTTP API via long-polling (no webhook):
``getUpdates`` with a 30s long poll to *receive*, ``sendMessage`` to *send*.

Config (never hard-coded), read from environment or a sibling ``.env`` file:
    TELEGRAM_BOT_TOKEN          (required)  BotFather token
    TELEGRAM_ALLOWED_CHAT_IDS   (optional)  comma-separated chat ids allowed to use it

Run from `src/` (so `dqnpf` resolves), not from this directory:
    PYTHONPATH=. python feature-prototype/telegram_advisor_bot.py
"""

from __future__ import annotations

import logging
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

_THIS = Path(__file__).resolve()
_PROTO_DIR = _THIS.parent
_PKG_ROOT = next(
    (p for p in _THIS.parents if (p / "dqnpf").is_dir()), _PROTO_DIR.parent
)
for _p in (str(_PKG_ROOT), str(_PROTO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data as data_mod  # noqa: E402
import llm  # noqa: E402
import signals as sig_mod  # noqa: E402
from policy import InferencePolicy, ReplayTradingEnv  # noqa: E402
from dqnpf.action_mapper import ACTION_NAMES  # noqa: E402
from dqnpf.config import IntegrationConfig  # noqa: E402
from dqnpf.integration import IntegrationLayer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tg-advisor")
logging.getLogger("dqnpf.integration").setLevel(logging.WARNING)

VARIANCE_THRESHOLD = 3.0
DIRECTIONAL_TOLERANCE = 1.0
HELP = (
    "USD/JPY advisor bot.\n"
    "  /advice (get the current recommendation\n"
    "  /help) this message\n"
    "Or just ask in plain language (e.g. \"what's your view on USD/JPY?\") ("
    "if a local language model is configured I'll answer in prose, grounded in "
    "the live recommendation.\n\n"
    "Educational/research only) not financial advice."
)


def _load_dotenv() -> None:
    env_path = _PROTO_DIR / ".env"
    if not env_path.exists():
        return
    import os

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


@dataclass
class Advice:
    final_action: str
    dqn_action: str
    screened: bool
    reason: str
    mu_bps: float
    sigma_bps: float
    high_sigma: bool
    gate_active: bool
    price: float
    bar_time_utc: str
    data_source: str


LIVE_WINDOW_DAYS = 4
LIVE_CACHE_TTL_SECONDS = 600


def _live_slice() -> dict:
    """Loader arguments for a trailing window that ends at the present moment.

    The prototype pins a fixed January 2024 window so its reported numbers stay
    reproducible. A bot answering "what should I do now" must not inherit that
    pin, so it asks for the last fortnight and lets the cache expire between
    polls. The end date is tomorrow because the tick range is half-open.
    """
    today = datetime.now(timezone.utc).date()
    return {
        "cache_path": _PROTO_DIR / "cache" / "usdjpy_m5_live.parquet",
        "start_date": str(today - timedelta(days=LIVE_WINDOW_DAYS)),
        "end_date": str(today + timedelta(days=1)),
        "max_age_seconds": LIVE_CACHE_TTL_SECONDS,
        "synthetic_end_now": True,
    }


_slice_lock = threading.Lock()
_slice: "data_mod.PriceData | None" = None


def _refresh_slice() -> None:
    """Fetch the trailing window and publish it for the request path to read."""
    global _slice
    try:
        fresh = data_mod.load_usdjpy_m5(**_live_slice())
    except Exception:
        logger.exception("price refresh failed; keeping the previous slice")
        return
    with _slice_lock:
        _slice = fresh
    logger.info("price slice refreshed: %s (%d bars)", fresh.source, fresh.n_bars)


def current_slice() -> "data_mod.PriceData":
    """Return the most recent slice, fetching once if the refresher has not run.

    Downloading a window of tick files takes minutes, so the Telegram handler
    must never do it inline: a blocked handler stops the bot answering at all.
    """
    with _slice_lock:
        if _slice is not None:
            return _slice
    _refresh_slice()
    with _slice_lock:
        if _slice is None:
            raise RuntimeError("no USD/JPY price data available yet")
        return _slice


def start_price_refresher(interval: float = LIVE_CACHE_TTL_SECONDS) -> None:
    """Keep the slice current in the background, off the request path."""

    def loop() -> None:
        while True:
            _refresh_slice()
            time.sleep(interval)

    threading.Thread(target=loop, name="price-refresh", daemon=True).start()


@dataclass
class _Act:
    action: int
    action_name: str = ""


def compute_advice() -> Advice:
    """Compute the advisor's recommendation for the latest available M5 bar."""
    price = current_slice()
    close = price.frame["close"].to_numpy()
    ts = price.frame["timestamp_ns"].to_numpy()
    t = len(close) - 1

    sig = sig_mod.informative_signals(close)
    mu = float(sig.mu_bps[t])
    sigma = float(sig.sigma_bps[t])

    obs = ReplayTradingEnv(close).observe_at(t, pos=0)
    dqn_action = int(InferencePolicy().act(obs))

    cfg = IntegrationConfig(
        symbol="USDJPY",
        variance_threshold=VARIANCE_THRESHOLD,
        screen_profit_window_sessions=3,
        pip_size=data_mod.PIP_SIZE,
    )
    layer = IntegrationLayer(dqn=None, forecaster_bridge=None, signal_cache=None, config=cfg)  # type: ignore[arg-type]
    layer.begin_session()
    screened = layer.screen(
        _Act(dqn_action, ACTION_NAMES[dqn_action]),
        mu,
        sigma,
        timestamp_ns=int(ts[t]),
        price=float(close[t]),
    )
    import pandas as pd

    bar_time = pd.Timestamp(int(ts[t]), unit="ns", tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    return Advice(
        final_action=screened.action_name,
        dqn_action=ACTION_NAMES[dqn_action],
        screened=screened.screened,
        reason=screened.reason,
        mu_bps=mu,
        sigma_bps=sigma,
        high_sigma=sigma > VARIANCE_THRESHOLD,
        gate_active=screened.gate_active,
        price=float(close[t]),
        bar_time_utc=bar_time,
        data_source=price.source,
    )


def _provenance(a: Advice) -> str:
    """The bar an answer is derived from, and where that bar came from.

    Every reply carries this. The advisor reasons over a fixed historical slice,
    so a prose answer without the bar timestamp reads as if it described the
    present market.
    """
    return f"(bar {a.bar_time_utc} · source: {a.data_source} · not financial advice)"


def format_advice(a: Advice) -> str:
    regime = "high-uncertainty" if a.high_sigma else "normal"
    if a.screened:
        rec_line = f"Recommendation: {a.final_action}  (DQN proposed {a.dqn_action}, screened: {a.reason})"
    else:
        rec_line = f"Recommendation: {a.final_action}"
    return (
        f"USD/JPY advice (bar {a.bar_time_utc}\n"
        f"{rec_line}\n"
        f"Forecast: mu={a.mu_bps:+.2f} bp, sigma={a.sigma_bps:.2f} bp ({regime} regime)\n"
        f"Screen: {a.reason} | profitability gate: {'active' if a.gate_active else 'bypassed'}\n"
        f"Last price: {a.price:.3f}\n"
        f"(source: {a.data_source}; educational/research only) not financial advice)"
    )


def render_price_chart(lookback_bars: int = 72) -> tuple[bytes, float, int, str]:
    """Render recent USD/JPY M5 price movement to a PNG. Returns
    (png_bytes, last_price, n_bars, data_source)."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    price = current_slice()
    close = price.frame["close"].to_numpy()
    ts = price.frame["timestamp_ns"].to_numpy()
    try:
        want = int(lookback_bars)
    except (TypeError, ValueError):
        want = 72
    n = max(12, min(want or 72, len(close), 288))
    close = close[-n:]
    stamps = pd.to_datetime(ts[-n:], utc=True)
    x = range(n)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, close, color="#3a86ff", linewidth=1.5)
    ax.scatter([n - 1], [close[-1]], color="#ef476f", s=28, zorder=5)
    ax.annotate(
        f"{close[-1]:.3f}", (n - 1, close[-1]),
        textcoords="offset points", xytext=(6, 0),
        color="#ef476f", fontsize=10, fontweight="bold",
    )
    n_ticks = min(6, n)
    tick_pos = [int(round(i * (n - 1) / (n_ticks - 1))) for i in range(n_ticks)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([stamps[p].strftime("%m-%d %H:%M") for p in tick_pos],
                       rotation=30, ha="right")
    ax.set_title(f"USD/JPY, last {n} × 5-min bars (UTC)")
    ax.set_ylabel("price")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue(), float(close[-1]), n, price.source


CHART_TOOL = {
    "type": "function",
    "function": {
        "name": "send_price_chart",
        "description": (
            "Render and send the user a chart (PNG) of recent USD/JPY price movement "
            "from 5-minute bars. Call this whenever the user wants to SEE the price, a "
            "chart, a graph, the trend, or how USD/JPY is moving."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lookback_bars": {
                    "type": "integer",
                    "description": "How many recent 5-minute bars to plot (default 72 ≈ 6 hours).",
                    "minimum": 12,
                    "maximum": 288,
                }
            },
        },
    },
}


class TelegramBot:
    def __init__(self, token: str, allowed_chat_ids: set[int] | None = None):
        self._base = f"https://api.telegram.org/bot{token}"
        self._token = token
        self._username: str | None = None
        self._allowed = allowed_chat_ids or set()

    def _call(self, method: str, *, params=None, timeout=10):
        r = requests.get(f"{self._base}/{method}", params=params or {}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"telegram {method} failed: {data}")
        return data["result"]

    def _redact(self, value: object) -> str:
        """Strip the bot token from anything bound for a log line.

        ``requests`` embeds the full request URL (token included) in its
        exception text, so logging the exception verbatim would publish the
        credential to stdout and, in the cluster, to log aggregation.
        """
        return str(value).replace(self._token, "<redacted>")

    def get_me(self) -> dict:
        return self._call("getMe")

    def send_message(self, chat_id: int, text: str) -> None:
        requests.post(
            f"{self._base}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=15,
        ).raise_for_status()

    def send_photo(self, chat_id: int, image_bytes: bytes, caption: str | None = None) -> None:
        data = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        requests.post(
            f"{self._base}/sendPhoto",
            data=data,
            files={"photo": ("usdjpy.png", image_bytes, "image/png")},
            timeout=30,
        ).raise_for_status()

    def get_updates(self, offset: int | None, long_poll: int = 30):
        params = {"timeout": long_poll}
        if offset is not None:
            params["offset"] = offset
        return self._call("getUpdates", params=params, timeout=long_poll + 10)

    def _authorised(self, chat_id: int) -> bool:
        return not self._allowed or chat_id in self._allowed

    def handle(self, message: dict) -> None:
        chat_id = message.get("chat", {}).get("id")
        text = (message.get("text") or "").strip()
        if chat_id is None:
            return
        if not self._authorised(chat_id):
            logger.warning("ignoring message from unauthorised chat %s", chat_id)
            return
        logger.info("message from chat %s: %r", chat_id, text)
        if self._username:
            text = re.sub(rf"@{re.escape(self._username)}\b", "", text,
                          flags=re.IGNORECASE).strip()
        is_command = text.startswith("/")
        head = text.lower().lstrip("/").split("@")[0].split()
        cmd = head[0] if head else ""
        try:
            if is_command and cmd in ("start", "help"):
                self.send_message(chat_id, HELP)
            elif is_command and cmd in ("advice", "advise", "rec", "recommendation"):
                advice = compute_advice()
                structured = format_advice(advice)
                prose = llm.narrate(structured)
                self.send_message(
                    chat_id,
                    prose + "\n\n" + _provenance(advice) if prose else structured,
                )
            elif is_command:
                self.send_message(chat_id, HELP)
            elif not text:
                advice = compute_advice()
                structured = format_advice(advice)
                prose = llm.narrate(structured)
                self.send_message(
                    chat_id,
                    prose + "\n\n" + _provenance(advice) if prose else structured,
                )
            else:
                advice = compute_advice()
                structured = format_advice(advice)

                def _execute_tool(name: str, args: dict) -> str:
                    if name == "send_price_chart":
                        png, last, n, src = render_price_chart(args.get("lookback_bars", 72))
                        self.send_photo(
                            chat_id, png,
                            caption=f"USD/JPY, last {n} × 5-min bars (last {last:.3f}, {src})",
                        )
                        return f"Chart sent to the user: last {n} 5-minute bars, latest price {last:.3f}."
                    return f"unknown tool {name}"

                prose = llm.narrate(
                    structured, user_message=text,
                    tools=[CHART_TOOL], execute_tool=_execute_tool,
                )
                if prose:
                    self.send_message(chat_id, prose + "\n\n" + _provenance(advice))
                elif llm.available():
                    self.send_message(
                        chat_id,
                        structured
                        + "\n\n(The language model did not answer, so this is the"
                          " recommendation itself rather than an explanation of it.)",
                    )
                else:
                    self.send_message(
                        chat_id,
                        structured
                        + "\n\n(Conversational mode needs a local LLM (set LLM_BASE_URL; see README.)",
                    )
        except Exception as exc:  # noqa: BLE001 - never let one message kill the loop
            logger.exception("failed to handle message")
            self.send_message(chat_id, f"Sorry) could not compute advice ({exc}).")

    def run(self) -> None:
        me = self.get_me()
        self._username = me.get("username")
        logger.info("bot @%s started (long-polling); allowed=%s",
                    self._username, self._allowed or "ALL")
        start_price_refresher()
        offset: int | None = None
        while True:
            try:
                updates = self.get_updates(offset)
            except requests.RequestException as exc:
                logger.warning("getUpdates error: %s; retrying in 3s", self._redact(exc))
                time.sleep(3)
                continue
            for upd in updates:
                offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("edited_message")
                if msg:
                    try:
                        self.handle(msg)
                    except Exception:  # noqa: BLE001 - one bad message must not
                        logger.exception("unhandled error while handling a message")


def _allowed_from_env() -> set[int]:
    import os

    raw = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "").strip()
    return {int(x) for x in raw.replace(" ", "").split(",") if x} if raw else set()


def main() -> None:
    _load_dotenv()
    import os

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print(
            "TELEGRAM_BOT_TOKEN not set.\n"
            "  export TELEGRAM_BOT_TOKEN=<your BotFather token>\n"
            "  (or put it in feature-prototype/.env)\n"
            "Then, from src/: PYTHONPATH=. python feature-prototype/telegram_advisor_bot.py",
            file=sys.stderr,
        )
        sys.exit(1)

    bot = TelegramBot(token, allowed_chat_ids=_allowed_from_env())
    bot.run()


if __name__ == "__main__":
    main()
