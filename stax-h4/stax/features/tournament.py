"""Feature tournament — drop redundant indicators per feature group, then
RFE down to a small representative set. Per architecture doc §5.4: run
this separately per symbol (indicator usefulness on EURUSD doesn't
transfer to BTCUSD), then pin the winning columns into config/features.yaml.

Run this on a train-only chronological slice, never on the holdout/test
period you'll later walk-forward over — otherwise the feature selection
itself leaks future information into "test" performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE

# Groups indicator columns by the architecture doc's §5.1 taxonomy.
# Partial-H4 state columns are kept out of elimination entirely — they're
# the core mechanism the whole trajectory concept relies on, not
# candidates to be voted off.
GROUP_PREFIXES = {
    "trend": ("ema_slope_", "adx_", "linreg_slope_"),
    "momentum": ("rsi_", "stoch_k_", "macd_hist_", "roc_"),
    "volatility": ("atr_ratio_", "bb_width_", "realized_vol_"),
    "volume": ("obv_slope_", "mfi_", "volume_z_"),
}

PARTIAL_H4_COLS = [
    "pos_in_h4", "h4_body_atr", "h4_range_atr", "h4_close_loc",
    "h4_dir_sofar", "h4_upper_wick", "h4_lower_wick", "m15_momentum",
]


def group_columns(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {g: [] for g in GROUP_PREFIXES}
    for c in columns:
        for g, prefixes in GROUP_PREFIXES.items():
            if c.startswith(prefixes):
                groups[g].append(c)
                break
    return groups


def drop_correlated(X: pd.DataFrame, corr_thresh: float = 0.9) -> pd.DataFrame:
    """Greedy correlation pruning: walk columns in order, drop any column
    that correlates above corr_thresh with an already-kept column."""
    if X.shape[1] <= 1:
        return X
    corr = X.corr().abs()
    keep: list[str] = []
    for c in X.columns:
        if all(corr.loc[c, k] < corr_thresh for k in keep):
            keep.append(c)
    return X[keep]


def feature_tournament(X: pd.DataFrame, y: pd.Series, top_k_per_group: int = 3,
                        corr_thresh: float = 0.9) -> list[str]:
    """Per architecture doc §5.4. Returns the selected indicator columns
    (partial-H4 state columns are added back unconditionally by the caller)."""
    groups = group_columns(list(X.columns))
    selected: list[str] = []
    for g, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        Xg = X[cols].dropna()
        if Xg.empty:
            continue
        yg = y.loc[Xg.index]

        Xg_pruned = drop_correlated(Xg, corr_thresh)
        if Xg_pruned.shape[1] == 0:
            continue

        k = min(top_k_per_group, Xg_pruned.shape[1])
        rfe = RFE(LGBMClassifier(n_estimators=200, verbosity=-1), n_features_to_select=k)
        rfe.fit(Xg_pruned, yg)
        selected += list(Xg_pruned.columns[rfe.support_])

    return selected
