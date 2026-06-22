"""Partial-H4 state — features describing the still-forming H4 candle from
its closed M15 sub-bars. This is what makes the per-step trajectory mean
something: the model sees the candle unfold and predicts the *next* one."""
from __future__ import annotations

import numpy as np
import pandas as pd

PARTIAL_KEYS = [
    "pos_in_h4", "h4_body_atr", "h4_range_atr", "h4_close_loc",
    "h4_dir_sofar", "h4_upper_wick", "h4_lower_wick", "m15_momentum",
]


def partial_h4_state(m15_in_h4: pd.DataFrame, atr_ref: float, eps: float = 1e-9) -> dict:
    """m15_in_h4 = closed M15 bars (1..16) belonging to the current H4 candle."""
    if len(m15_in_h4) == 0:
        return {k: 0.0 for k in PARTIAL_KEYS}

    o = m15_in_h4["open"].iloc[0]
    h = m15_in_h4["high"].max()
    l = m15_in_h4["low"].min()
    c = m15_in_h4["close"].iloc[-1]
    k = len(m15_in_h4)
    rng, body = (h - l), (c - o)

    return {
        "pos_in_h4": k / 16.0,
        "h4_body_atr": body / (atr_ref + eps),
        "h4_range_atr": rng / (atr_ref + eps),
        "h4_close_loc": (c - l) / (rng + eps),
        "h4_dir_sofar": float(np.sign(body)),
        "h4_upper_wick": (h - max(o, c)) / (rng + eps),
        "h4_lower_wick": (min(o, c) - l) / (rng + eps),
        "m15_momentum": m15_in_h4["close"].pct_change().tail(3).mean() or 0.0,
    }
