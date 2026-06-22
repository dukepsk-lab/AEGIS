"""Multi-TF indicator block. P1 starter set (see config/features.yaml),
implemented directly on pandas/numpy (no pandas-ta dependency) so the
feature pipeline has no exotic install requirements. All indicators here
operate on a DataFrame of already-closed bars — no look-ahead by
construction, since the caller never passes the forming bar in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def _rolling_z(s: pd.Series, window: int = 500) -> pd.Series:
    mean = s.rolling(window, min_periods=20).mean()
    std = s.rolling(window, min_periods=20).std()
    return (s - mean) / (std + 1e-9)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    return _true_range(high, low, close).ewm(span=length, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _true_range(high, low, close)
    atr = tr.ewm(span=length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(span=length, adjust=False).mean() / (atr + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(span=length, adjust=False).mean() / (atr + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(span=length, adjust=False).mean()


def _linreg_slope(s: pd.Series, length: int = 14) -> pd.Series:
    x = np.arange(length)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def slope(window: np.ndarray) -> float:
        return float(((window - window.mean()) * (x - x_mean)).sum() / denom)

    return s.rolling(length).apply(slope, raw=True)


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=length, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    lowest = low.rolling(length).min()
    highest = high.rolling(length).max()
    return 100 * (close - lowest) / (highest - lowest + 1e-9)


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line - signal_line


def _roc(close: pd.Series, length: int = 10) -> pd.Series:
    return 100 * close.pct_change(length)


def _bb_width(close: pd.Series, length: int = 20, n_std: float = 2.0) -> pd.Series:
    mid = close.rolling(length).mean()
    std = close.rolling(length).std()
    return (2 * n_std * std) / (mid.abs() + 1e-9)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    raw_flow = tp * volume
    pos_flow = raw_flow.where(tp.diff() > 0, 0.0).rolling(length).sum()
    neg_flow = raw_flow.where(tp.diff() < 0, 0.0).rolling(length).sum()
    ratio = pos_flow / (neg_flow + 1e-9)
    return 100 - 100 / (1 + ratio)


def indicator_block(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """df: closed OHLCV bars for one timeframe, ascending time order.
    Returns a DataFrame of indicator columns aligned to df's index, suffixed
    with the timeframe (e.g. rsi_m15)."""
    suf = tf.lower()
    out = pd.DataFrame(index=df.index)
    high, low, close = df["high"], df["low"], df["close"]

    out[f"ema_slope_{suf}"] = _ema(close, 21).diff()
    out[f"adx_{suf}"] = _adx(high, low, close)
    out[f"linreg_slope_{suf}"] = _linreg_slope(close)

    out[f"rsi_{suf}"] = _rsi(close)
    out[f"stoch_k_{suf}"] = _stoch_k(high, low, close)
    out[f"macd_hist_{suf}"] = _macd_hist(close)
    out[f"roc_{suf}"] = _roc(close)

    atr = _atr(high, low, close)
    out[f"atr_ratio_{suf}"] = atr / (close + 1e-9)
    out[f"bb_width_{suf}"] = _bb_width(close)
    out[f"realized_vol_{suf}"] = close.pct_change().rolling(20).std()

    vol_col = "tick_volume" if "tick_volume" in df.columns else ("volume" if "volume" in df.columns else None)
    if vol_col:
        vol = df[vol_col]
        out[f"obv_slope_{suf}"] = _obv(close, vol).diff()
        out[f"mfi_{suf}"] = _mfi(high, low, close, vol)
        out[f"volume_z_{suf}"] = _rolling_z(vol)

    return out
