"""Offline data source for testing the pipeline without an MT5 terminal.

Reads the public OHLCV CSV mirrors at raw.githubusercontent.com (e.g.
ejtraderLabs/historical-data, komo135/forex-historical-data) and converts
them to the same bar schema MT5 produces (`time`, `open`, `high`, `low`,
`close`, `tick_volume`, `close_time`), so the rest of the pipeline
(stax.features.build, stax.backtest.walk_forward) doesn't know or care
where the bars came from.

This is for offline backtesting/validation only — it has nothing to do
with live execution, which always goes through stax.data.mt5_client on
the Windows box running the IUX terminal.
"""
from __future__ import annotations

import pandas as pd

from stax.data.timeframes import TF_SECONDS

# Public mirrors of the same dataset (10y EURUSD/XAUUSD/... at m15/m30/h1/h4/d1).
# Prices are stored scaled by PRICE_SCALE (5-decimal pips as an integer).
PUBLIC_CSV_BASE = "https://raw.githubusercontent.com/komo135/forex-historical-data/main"
PRICE_SCALE = 100_000.0

_TF_FILE_SUFFIX = {"M15": "m15", "H1": "h1", "H4": "h4"}


def public_csv_url(symbol: str, tf: str, base: str = PUBLIC_CSV_BASE) -> str:
    suf = _TF_FILE_SUFFIX[tf]
    return f"{base}/{symbol.upper()}/{symbol.upper()}{suf}.csv"


def load_public_csv(symbol: str, tf: str, base: str = PUBLIC_CSV_BASE,
                     price_scale: float = PRICE_SCALE) -> pd.DataFrame:
    url = public_csv_url(symbol, tf, base)
    raw = pd.read_csv(url)
    return _to_mt5_schema(raw, tf, price_scale)


def load_local_csv(path: str, tf: str, price_scale: float = 1.0) -> pd.DataFrame:
    """For your own downloaded CSVs (Dukascopy/HistData/etc). Expected
    columns: Date/time, open, high, low, close, [volume|tick_volume]."""
    raw = pd.read_csv(path)
    return _to_mt5_schema(raw, tf, price_scale)


def _to_mt5_schema(raw: pd.DataFrame, tf: str, price_scale: float) -> pd.DataFrame:
    cols = {c.lower(): c for c in raw.columns}
    date_col = cols.get("date") or cols.get("time") or cols.get("datetime")
    vol_col = cols.get("tick_volume") or cols.get("volume")

    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    dt = pd.to_datetime(raw[date_col], utc=True)
    df = pd.DataFrame({
        "time": (dt - epoch) // pd.Timedelta(seconds=1),
        "open": raw[cols["open"]] / price_scale,
        "high": raw[cols["high"]] / price_scale,
        "low": raw[cols["low"]] / price_scale,
        "close": raw[cols["close"]] / price_scale,
    })
    df["tick_volume"] = raw[vol_col] if vol_col else 0
    df = df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
    df["close_time"] = df["time"] + TF_SECONDS[tf]
    return df


def load_bars_for_symbol(symbol: str, timeframes: list[str] = ("M15", "H1", "H4"),
                          base: str = PUBLIC_CSV_BASE) -> dict[str, pd.DataFrame]:
    return {tf: load_public_csv(symbol, tf, base) for tf in timeframes}
