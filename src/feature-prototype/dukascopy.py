"""Download real USD/JPY tick data from Dukascopy (free, no auth).

Dukascopy publishes historical FX ticks as one LZMA-compressed ``.bi5`` file per
hour at:

    https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM0}/{DD}/{HH}h_ticks.bi5

where ``MM0`` is the **0-indexed** month (January = 00). Each decompressed record
is 20 bytes, big-endian ``>IIIff``:

    ms_from_hour (uint32), ask_int (uint32), bid_int (uint32),
    ask_volume (float32), bid_volume (float32)

Integer prices are scaled by ``point_divisor`` (1000 for 3-decimal JPY pairs).

This module is the prototype's real-data *download* stage; `aggregate_m5` (in
`data.py`) is the *aggregation* stage that turns these ticks into M5 bars,
mirroring the project's own ticks -> interval-price pipeline (`marketdata/`).
"""

from __future__ import annotations

import logging
import lzma
import struct
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BASE = "https://datafeed.dukascopy.com/datafeed"
_REC = struct.Struct(">IIIff")
_UA = "Mozilla/5.0 (prototype-research; USDJPY tick fetch)"


def _hour_url(symbol: str, dt: datetime) -> str:
    # NB: month is 0-indexed in Dukascopy paths.
    return f"{_BASE}/{symbol}/{dt.year:04d}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"


def _fetch_hour_raw(symbol: str, dt: datetime, cache_dir: Path, retries: int = 3) -> bytes:
    """Return the raw ``.bi5`` bytes for one hour, using a per-hour disk cache.

    An empty (0-byte) result means a no-trading hour (weekend/holiday) and is
    cached as such so re-runs do not re-request it.
    """
    cpath = cache_dir / symbol / f"{dt.year:04d}" / f"{dt.month:02d}" / f"{dt.day:02d}" / f"{dt.hour:02d}h.bi5"
    if cpath.exists():
        return cpath.read_bytes()
    cpath.parent.mkdir(parents=True, exist_ok=True)
    url = _hour_url(symbol, dt)
    body = b""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):  # no data for this hour
                body = b""
                break
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
    cpath.write_bytes(body)
    return body


def _decode_hour(body: bytes, hour_start: datetime, point_divisor: float) -> list[tuple]:
    if not body:
        return []
    try:
        raw = lzma.decompress(body)
    except lzma.LZMAError:
        return []
    base_ns = int(hour_start.timestamp()) * 1_000_000_000
    rows = []
    for ms, ask_i, bid_i, av, bv in _REC.iter_unpack(raw):
        rows.append(
            (
                base_ns + int(ms) * 1_000_000,  # timestamp_ns
                bid_i / point_divisor,
                ask_i / point_divisor,
                float(av) + float(bv),  # combined volume (millions)
            )
        )
    return rows


def fetch_ticks(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    point_divisor: float = 1000.0,
    cache_dir: str | Path = "feature-prototype/cache/dukascopy",
    max_workers: int = 12,
) -> pd.DataFrame:
    """Download ticks for ``[start, end)`` and return a tidy tick DataFrame.

    Columns: ``timestamp_ns, bid, ask, mid, volume`` (UTC, ascending).
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    cache_dir = Path(cache_dir)

    hours = []
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur < end:
        hours.append(cur)
        cur += timedelta(hours=1)
    logger.info("dukascopy: fetching %d hourly tick files for %s", len(hours), symbol)

    def _one(dt):
        return dt, _fetch_hour_raw(symbol, dt, cache_dir)

    raw_by_hour: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for dt, body in ex.map(_one, hours):
            raw_by_hour[dt] = body

    rows: list[tuple] = []
    empty = 0
    for dt in hours:
        decoded = _decode_hour(raw_by_hour[dt], dt, point_divisor)
        if not decoded:
            empty += 1
        rows.extend(decoded)

    if not rows:
        raise RuntimeError("dukascopy returned no ticks for the requested window")
    df = pd.DataFrame(rows, columns=["timestamp_ns", "bid", "ask", "volume"])
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df.sort_values("timestamp_ns").reset_index(drop=True)
    logger.info(
        "dukascopy: %d ticks across %d hours (%d empty hours skipped)",
        len(df), len(hours), empty,
    )
    return df[["timestamp_ns", "bid", "ask", "mid", "volume"]]
