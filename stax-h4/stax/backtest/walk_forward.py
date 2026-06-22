"""P1 walk-forward harness — single base model (LightGBM), no meta-stacking
yet (that's P2). Goal here is strictly to prove the pipeline doesn't leak
and produce an honest, measurable expectancy baseline on EURUSD per the
architecture doc's P1 Definition of Done.

Decision at H4 close = simple edge threshold on calibration-free p_up
(min_edge from config/symbols.yaml). No Kelly sizing yet — fixed 1-unit
size so PnL is reported in price units, comparable across folds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stax.backtest.cost_model import CostParams, pnl_net
from stax.backtest.metrics import pr_auc_lift, summarize
from stax.models.base_lgbm import LGBMBasePredictor


@dataclass
class WFConfig:
    train_bars: int      # number of H4 samples per train window
    test_bars: int        # number of H4 samples per test window
    embargo_bars: int      # H4 samples purged between train/test to avoid leakage


def purged_walk_forward_splits(n_samples: int, cfg: WFConfig):
    """Forward-chaining splits over H4-sample index with an embargo gap.
    Never random K-fold — this is a time series."""
    splits = []
    start = 0
    while True:
        train_end = start + cfg.train_bars
        test_start = train_end + cfg.embargo_bars
        test_end = test_start + cfg.test_bars
        if test_end > n_samples:
            break
        splits.append((np.arange(start, train_end), np.arange(test_start, test_end)))
        start += cfg.test_bars
    return splits


def build_h4_samples(feat_table: pd.DataFrame, h4_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Reduce the per-M15-step feature table to one row per H4 close
    (pos_in_h4 == 1.0, i.e. the 16th/last M15 step seen before that H4
    candle closes), labeled with the *next* H4 candle's direction."""
    at_close = feat_table[np.isclose(feat_table["pos_in_h4"], 1.0)].copy()
    at_close = at_close.sort_values("close_time").reset_index(drop=True)

    h4 = h4_df.sort_values("close_time").reset_index(drop=True)
    h4_dir = np.sign(h4["close"] - h4["open"])
    h4_open_next = h4["open"].shift(-1)
    h4_close_next = h4["close"].shift(-1)
    label_map = pd.DataFrame({
        "close_time": h4["close_time"],
        "next_dir": np.sign(h4_close_next - h4_open_next),
        "next_open": h4_open_next,
        "next_close": h4_close_next,
    })

    merged = pd.merge_asof(at_close, label_map, on="close_time", direction="backward")
    merged = merged.dropna(subset=["next_dir", "next_open", "next_close"])
    merged["y"] = (merged["next_dir"] > 0).astype(int)
    return merged[["close_time", "next_open", "next_close", "y"] + feature_cols]


def simulate_trades(p_up: np.ndarray, samples: pd.DataFrame, min_edge: float, cost: CostParams) -> pd.DataFrame:
    rows = []
    for p, (_, row) in zip(p_up, samples.iterrows()):
        edge = abs(p - 0.5)
        if edge < min_edge:
            continue
        direction = 1 if p > 0.5 else -1
        pnl = pnl_net(direction, row["next_open"], row["next_close"], cost)
        rows.append({"close_time": row["close_time"], "direction": direction, "pnl": pnl})
    return pd.DataFrame(rows)


def walk_forward(feat_table: pd.DataFrame, h4_df: pd.DataFrame, feature_cols: list[str],
                  wf_cfg: WFConfig, min_edge: float, cost: CostParams) -> dict:
    samples = build_h4_samples(feat_table, h4_df, feature_cols)
    splits = purged_walk_forward_splits(len(samples), wf_cfg)
    if not splits:
        raise ValueError("Not enough H4 samples for the configured train/test/embargo window sizes")

    all_trades = []
    fold_metrics = []
    for fold_i, (tr, te) in enumerate(splits):
        train_df = samples.iloc[tr]
        test_df = samples.iloc[te]

        model = LGBMBasePredictor().fit(train_df[feature_cols], train_df["y"])
        p_up = model.predict_proba(test_df[feature_cols])

        trades = simulate_trades(p_up, test_df, min_edge, cost)
        all_trades.append(trades)
        fold_metrics.append({
            "fold": fold_i,
            **pr_auc_lift(test_df["y"].values, p_up),
            **summarize(trades, n_total_bars=len(test_df)),
        })

    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(columns=["pnl"])
    overall = summarize(trades_all, n_total_bars=len(samples))
    return {"folds": fold_metrics, "overall": overall, "trades": trades_all}
