"""Glue layer: turns closed multi-TF bars into the per-M15-step feature
table used by both backtest (vectorized, full history) and live (last row
of a recent window) code paths — same function, same column set, so there
is no train/serve skew.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stax.features.indicators import indicator_block


def _asof_merge_tf(m15: pd.DataFrame, higher: pd.DataFrame, ind: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Align a higher-TF indicator block onto the M15 index using as-of
    (backward) merge on close_time — only ever look at the most recent
    *closed* higher-TF bar relative to each M15 close."""
    higher_ind = higher[["close_time"]].join(ind)
    merged = pd.merge_asof(
        m15[["close_time"]].reset_index(),
        higher_ind,
        on="close_time",
        direction="backward",
    ).set_index("index")
    return merged.drop(columns=["close_time"])


def build_feature_table(
    symbol: str,
    bars: dict[str, pd.DataFrame],
    atr_window: int = 14,
) -> pd.DataFrame:
    """bars: {"M15": df, "H1": df, "H4": df, optionally "M5": df} of closed
    bars, ascending time order, each with an `open`/`high`/`low`/`close`/
    `time`/`close_time` column. Returns one row per closed M15 bar.

    M5 is optional — some data sources (e.g. the public CSV mirrors used
    for offline testing without MT5) don't carry it, and the P1 feature
    set (config/features.yaml) doesn't use M5 columns either.
    """
    m15 = bars["M15"].reset_index(drop=True)

    ind_m15 = indicator_block(m15, "M15")
    ind_h1 = indicator_block(bars["H1"], "H1")
    feat_h1 = _asof_merge_tf(m15, bars["H1"], ind_h1, "h1")
    parts = [m15[["time", "close_time"]], ind_m15, feat_h1]

    if "M5" in bars:
        ind_m5 = indicator_block(bars["M5"], "M5")
        feat_m5 = _asof_merge_tf(m15, bars["M5"], ind_m5, "m5")
        parts.append(feat_m5)

    feats = pd.concat(parts, axis=1)

    atr_h1 = ind_h1["atr_ratio_h1"] * bars["H1"]["close"]
    atr_ref_series = pd.merge_asof(
        m15[["close_time"]].reset_index(),
        bars["H1"][["close_time"]].assign(atr_ref=atr_h1.values),
        on="close_time",
        direction="backward",
    ).set_index("index")["atr_ref"]

    partial_df = _vectorized_partial_h4(m15, bars["H4"]["time"].values, atr_ref_series)
    feats = pd.concat([feats, partial_df], axis=1)
    return feats


def _vectorized_partial_h4(m15: pd.DataFrame, h4_open_times: np.ndarray, atr_ref_series: pd.Series,
                            eps: float = 1e-9) -> pd.DataFrame:
    """Vectorized equivalent of calling partial_h4.partial_h4_state() once
    per M15 row. A plain Python loop with a fresh boolean mask over the
    full M15 array per row is O(n^2) and unusable past a few thousand
    bars, so this uses searchsorted + groupby-cumulative ops instead —
    same definitions, same look-ahead guarantees (each row only ever sees
    bars up to and including itself within its own H4 bucket)."""
    n = len(m15)
    time_vals = m15["time"].values
    bucket = np.searchsorted(h4_open_times, time_vals, side="right") - 1
    valid = bucket >= 0

    g = pd.Series(np.where(valid, bucket, -1), index=m15.index)
    open_, high, low, close = m15["open"], m15["high"], m15["low"], m15["close"]

    k = g.groupby(g).cumcount() + 1
    o_first = open_.groupby(g).transform("first")
    run_high = high.groupby(g).cummax()
    run_low = low.groupby(g).cummin()

    body = close - o_first
    rng = run_high - run_low
    atr_ref = atr_ref_series.reindex(m15.index).fillna(1e-6)
    atr_ref = atr_ref.where(atr_ref != 0, 1e-6)

    pct_local = close.groupby(g).pct_change()
    m15_momentum = pct_local.groupby(g).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

    out = pd.DataFrame({
        "pos_in_h4": k / 16.0,
        "h4_body_atr": body / (atr_ref + eps),
        "h4_range_atr": rng / (atr_ref + eps),
        "h4_close_loc": (close - run_low) / (rng + eps),
        "h4_dir_sofar": np.sign(body),
        "h4_upper_wick": (run_high - np.maximum(o_first, close)) / (rng + eps),
        "h4_lower_wick": (np.minimum(o_first, close) - run_low) / (rng + eps),
        "m15_momentum": m15_momentum,
    }, index=m15.index).fillna(0.0)

    out.loc[~valid, :] = 0.0
    return out
