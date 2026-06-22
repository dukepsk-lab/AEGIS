"""P1 entry point: pull EURUSD history from MT5, run purged walk-forward
with Model 1 (LightGBM) only, print honest expectancy/PR-AUC/selectivity
metrics, and save the final-fold model for paper trading.

Usage: python scripts/train_walk_forward.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from stax.backtest.cost_model import CostParams
from stax.backtest.walk_forward import WFConfig, walk_forward
from stax.data.bars import get_closed_bars
from stax.data.mt5_client import MT5Config, connect, disconnect, ensure_symbol, load_runtime_config
from stax.features.build import build_feature_table

SYMBOL = "EURUSD"
N_M5, N_M15, N_H1, N_H4 = 60_000, 20_000, 5_000, 1_300


def main() -> None:
    runtime_cfg = load_runtime_config()
    with open("config/symbols.yaml", encoding="utf-8") as f:
        symbol_cfg = yaml.safe_load(f)
    with open("config/features.yaml", encoding="utf-8") as f:
        feature_cfg = yaml.safe_load(f)

    feature_cols = feature_cfg[SYMBOL]["features"]
    min_edge = symbol_cfg[SYMBOL]["min_edge"]

    mt5_cfg = MT5Config(**runtime_cfg["mt5"])
    connect(mt5_cfg)
    try:
        ensure_symbol(SYMBOL)
        import MetaTrader5 as mt5

        info = mt5.symbol_info(SYMBOL)
        cost = CostParams(spread_pts=15, point=info.point, slippage_pts=10)

        bars = {
            "M5": get_closed_bars(SYMBOL, "M5", N_M5),
            "M15": get_closed_bars(SYMBOL, "M15", N_M15),
            "H1": get_closed_bars(SYMBOL, "H1", N_H1),
            "H4": get_closed_bars(SYMBOL, "H4", N_H4),
        }
    finally:
        disconnect()

    if any(v is None for v in bars.values()):
        raise RuntimeError("Failed to pull one or more timeframes from MT5 — check IUX connection/login")

    feat_table = build_feature_table(SYMBOL, bars)
    feat_table = feat_table.dropna(subset=feature_cols)

    wf_cfg = WFConfig(train_bars=600, test_bars=120, embargo_bars=16)  # ~100d/20d/2.7d at H4
    result = walk_forward(feat_table, bars["H4"], feature_cols, wf_cfg, min_edge, cost)

    print("\n=== Per-fold metrics ===")
    for fm in result["folds"]:
        print(fm)
    print("\n=== Overall (P1 Definition of Done: expectancy must be measurable) ===")
    print(result["overall"])


if __name__ == "__main__":
    main()
