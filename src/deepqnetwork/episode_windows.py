"""Per-date episode-window generation shared by DQN training and evaluation.

Both the DQN training loop (``deepqnetwork/train.py``) and the DQNPF intraday
backtest (``tradingmodel/intraday/dqnpf/backtest.py``) slice a calendar date
range into one episode per date, each bounded by an hour-of-day window. Keeping
this in one place guarantees the backtest evaluates the policy on the *same*
session windows it trained on (see the train/eval parity requirement).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone


def iter_date_episodes(
    date_start: str,
    date_end: str,
    hour_start: int,
    hour_end: int,
) -> list[tuple[int, int]]:
    """Generate per-date ``(episode_start_ts, episode_end_ts)`` pairs.

    Each calendar date from ``date_start`` through ``date_end`` (inclusive)
    produces one pair.  The Unix timestamps are computed from the configured
    hour-of-day window.

    If ``hour_end >= 24`` the end timestamp rolls into the next calendar
    day.  For example, ``hour_start=15, hour_end=39`` means 15:00 UTC today
    through 15:00 UTC tomorrow (a 24-hour window).

    Args:
        date_start: ISO date string for the first date (e.g. ``"2012-01-01"``).
        date_end: ISO date string for the last date (e.g. ``"2022-12-31"``).
        hour_start: Hour (0-23) at which each episode begins.
        hour_end: Hour at which each episode ends. 0-23 = same day;
            24+ = rolls into the next calendar day (e.g. 39 = 15:00 next day).

    Returns:
        List of ``(episode_start_ts, episode_end_ts)`` tuples, one per date.
    """
    ds = date.fromisoformat(date_start)
    de = date.fromisoformat(date_end)
    if ds > de:
        raise ValueError(
            f"date_start ({date_start}) must be <= date_end ({date_end})"
        )

    episodes: list[tuple[int, int]] = []
    current = ds
    while current <= de:
        dt_start = datetime(
            current.year, current.month, current.day,
            hour_start, 0, 0, tzinfo=timezone.utc,
        )
        end_date = current
        end_hour = hour_end
        if hour_end >= 24:
            extra_days = hour_end // 24
            end_hour = hour_end % 24
            end_date = current + timedelta(days=extra_days)
        dt_end = datetime(
            end_date.year, end_date.month, end_date.day,
            end_hour, 0, 0, tzinfo=timezone.utc,
        )
        episodes.append((int(dt_start.timestamp()), int(dt_end.timestamp())))
        current = date.fromordinal(current.toordinal() + 1)
    return episodes
