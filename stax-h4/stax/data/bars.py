"""Closed-bar access — the look-ahead guard at the root of the system.

Rule: index 0 from MT5 is always the bar currently forming (never use it).
copy_rates_from_pos(symbol, tf, start=1, n) skips it, so everything returned
here is a fully closed bar.
"""
from __future__ import annotations

import pandas as pd
import MetaTrader5 as mt5

from stax.data.mt5_client import TF_SECONDS, TIMEFRAME


def get_closed_bars(symbol: str, tf: str, n: int, asof_epoch: int | None = None) -> pd.DataFrame | None:
    """Return the n most recently closed bars (oldest first), optionally
    double-guarded against a wall-clock `asof_epoch` cutoff."""
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME[tf], 1, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["close_time"] = df["time"] + TF_SECONDS[tf]
    if asof_epoch is not None:
        df = df[df["close_time"] <= asof_epoch]
    return df.reset_index(drop=True)


def latest_closed_bar_time(symbol: str, tf: str) -> int | None:
    """Open-time epoch of the most recently closed bar, or None if unavailable."""
    r = mt5.copy_rates_from_pos(symbol, TIMEFRAME[tf], 1, 1)
    if r is None or len(r) == 0:
        return None
    return int(r[0]["time"])


def floor_to_h4(epoch: int, h4_anchor: int = 0) -> int:
    """Floor an epoch timestamp to the start of its H4 bucket.

    `h4_anchor` lets you correct for broker H4 boundary offset if IUX's H4
    bars don't align to 00:00 UTC — leave at 0 until verified against
    `latest_closed_bar_time(symbol, "H4")` % 14400.
    """
    period = TF_SECONDS["H4"]
    return epoch - ((epoch - h4_anchor) % period)


def m15_bars_in_current_h4(symbol: str, h4_open_epoch: int, asof_epoch: int) -> pd.DataFrame | None:
    """All closed M15 bars belonging to the H4 candle that opened at
    `h4_open_epoch`, as of `asof_epoch` (double look-ahead guard)."""
    h4_close_epoch = h4_open_epoch + TF_SECONDS["H4"]
    bars = get_closed_bars(symbol, "M15", 16, asof_epoch=asof_epoch)
    if bars is None:
        return None
    mask = (bars["time"] >= h4_open_epoch) & (bars["close_time"] <= h4_close_epoch)
    return bars[mask].reset_index(drop=True)


def market_open(symbol: str) -> bool:
    """Cheap liveness check: is there a fresh-ish tick for this symbol."""
    tick = mt5.symbol_info_tick(symbol)
    return tick is not None and tick.time > 0
