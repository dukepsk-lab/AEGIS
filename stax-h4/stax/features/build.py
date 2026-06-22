"""Glue layer: turns closed multi-TF bars into the per-M15-step feature
table used by both backtest (vectorized, full history) and live (last row
of a recent window) code paths — same function, same column set, so there
is no train/serve skew.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stax.features.indicators import indicator_block
from stax.features.partial_h4 import partial_h4_state


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
    """bars: {"M5": df, "M15": df, "H1": df, "H4": df} of closed bars,
    ascending time order, each with an `open`/`high`/`low`/`close`/`time`/
    `close_time` column. Returns one row per closed M15 bar.
    """
    m15 = bars["M15"].reset_index(drop=True)

    ind_m5 = indicator_block(bars["M5"], "M5")
    ind_m15 = indicator_block(m15, "M15")
    ind_h1 = indicator_block(bars["H1"], "H1")

    feat_m5 = _asof_merge_tf(m15, bars["M5"], ind_m5, "m5")
    feat_h1 = _asof_merge_tf(m15, bars["H1"], ind_h1, "h1")

    feats = pd.concat([m15[["time", "close_time"]], ind_m15, feat_m5, feat_h1], axis=1)

    atr_h1 = ind_h1["atr_ratio_h1"] * bars["H1"]["close"]
    atr_ref_series = pd.merge_asof(
        m15[["close_time"]].reset_index(),
        bars["H1"][["close_time"]].assign(atr_ref=atr_h1.values),
        on="close_time",
        direction="backward",
    ).set_index("index")["atr_ref"]

    h4_open_times = bars["H4"]["time"].values
    partial_rows = []
    for i in range(len(m15)):
        t = m15["time"].iloc[i]
        h4_open = h4_open_times[h4_open_times <= t]
        if len(h4_open) == 0:
            partial_rows.append({k: 0.0 for k in [
                "pos_in_h4", "h4_body_atr", "h4_range_atr", "h4_close_loc",
                "h4_dir_sofar", "h4_upper_wick", "h4_lower_wick", "m15_momentum",
            ]})
            continue
        h4_start = h4_open[-1]
        mask = (m15["time"] >= h4_start) & (m15["time"] <= t)
        atr_ref = atr_ref_series.iloc[i] if not np.isnan(atr_ref_series.iloc[i]) else 1e-6
        partial_rows.append(partial_h4_state(m15[mask], atr_ref))

    partial_df = pd.DataFrame(partial_rows, index=feats.index)
    feats = pd.concat([feats, partial_df], axis=1)
    return feats
