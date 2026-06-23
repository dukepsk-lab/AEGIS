"""P1 walk-forward against public CSV mirrors (no MT5 / no IUX terminal
needed) — for validating the pipeline on a machine that can't run MT5.

Once this looks sane, the real test is scripts/train_walk_forward.py
against live IUX history, since broker-specific spread/slippage and exact
bar alignment matter for the honest expectancy number.

Usage: python scripts/train_walk_forward_csv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from stax.backtest.cost_model import CostParams
from stax.backtest.walk_forward import WFConfig, walk_forward
from stax.data.csv_source import load_bars_for_symbol
from stax.features.build import build_feature_table

SYMBOL = "EURUSD"


def main() -> None:
    with open("config/symbols.yaml", encoding="utf-8") as f:
        symbol_cfg = yaml.safe_load(f)
    with open("config/features.yaml", encoding="utf-8") as f:
        feature_cfg = yaml.safe_load(f)
    model_params_path = Path("config/model_params.yaml")
    model_params_cfg = yaml.safe_load(model_params_path.read_text(encoding="utf-8")) if model_params_path.exists() else {}

    feature_cols = feature_cfg[SYMBOL]["features"]
    min_edge = symbol_cfg[SYMBOL]["min_edge"]
    model_params = model_params_cfg.get(SYMBOL)

    print(f"Downloading {SYMBOL} M15/H1/H4 from public CSV mirror...")
    bars = load_bars_for_symbol(SYMBOL, ["M15", "H1", "H4"])
    for tf, df in bars.items():
        print(f"  {tf}: {len(df)} bars, {pd.to_datetime(df['time'].iloc[0], unit='s')} -> {pd.to_datetime(df['time'].iloc[-1], unit='s')}")

    feat_table = build_feature_table(SYMBOL, bars)
    feat_table = feat_table.dropna(subset=feature_cols)

    # EURUSD spread ~1pip @ 5-digit quotes = 10 points; IUX-specific value
    # should be recalibrated from real symbol_info() once on the live terminal.
    cost = CostParams(spread_pts=10, point=0.00001, slippage_pts=5)

    wf_cfg = WFConfig(train_bars=600, test_bars=120, embargo_bars=16)
    result = walk_forward(feat_table, bars["H4"], feature_cols, wf_cfg, min_edge, cost, model_params)

    print("\n=== Per-fold metrics ===")
    for fm in result["folds"]:
        print(fm)
    print("\n=== Overall (P1 baseline, public EURUSD data, IUX cost approximated) ===")
    print(result["overall"])


if __name__ == "__main__":
    main()
