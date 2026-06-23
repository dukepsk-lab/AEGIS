"""Feature tournament + Optuna hyperparameter search for the P1 EURUSD
baseline, run entirely on public CSV data (no MT5/IUX required).

Per the architecture doc's own sequencing (§16): don't expand to P2
stacking until the P1 baseline looks promising. This script answers that
question properly instead of guessing with placeholder features/defaults:

1. Chronological 70/30 split into tune/holdout — holdout is never touched
   by feature selection or hyperparameter search, so the final comparison
   isn't leaking.
2. Feature tournament (RFE + correlation pruning) on the tune slice.
3. Optuna search over LightGBM params + min_edge, scored by mean
   expectancy across a few forward-chaining folds within the tune slice.
4. Final apples-to-apples comparison on the untouched holdout: current
   placeholder config (config/features.yaml + config/symbols.yaml) vs.
   the tournament+tuned configuration, both via the same purged
   walk-forward harness.

Usage: python scripts/tune_and_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import optuna
import pandas as pd
import yaml

from stax.backtest.cost_model import CostParams
from stax.backtest.walk_forward import (
    WFConfig,
    build_h4_samples,
    purged_walk_forward_splits,
    simulate_trades,
    walk_forward,
)
from stax.data.csv_source import load_bars_for_symbol
from stax.features.build import build_feature_table
from stax.features.tournament import PARTIAL_H4_COLS, feature_tournament
from stax.models.base_lgbm import LGBMBasePredictor

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Cost approximations per symbol (real values should come from
# mt5.symbol_info() once on the live IUX terminal). FX majors are quoted
# in 5-decimal pips; XAUUSD in 2-decimal cents, hence the very different
# point size and point count below.
COST_BY_SYMBOL = {
    "EURUSD": CostParams(spread_pts=10, point=0.00001, slippage_pts=5),
    "XAUUSD": CostParams(spread_pts=175, point=0.01, slippage_pts=25),
}

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
YEARS = float(sys.argv[2]) if len(sys.argv) > 2 else None  # e.g. 3 to test only the most recent 3 years
COST = COST_BY_SYMBOL[SYMBOL]
TUNE_FRACTION = 0.7
N_TRIALS = 40
TUNE_WF = WFConfig(train_bars=800, test_bars=150, embargo_bars=16)  # used inside the tune slice for HPO scoring
HOLDOUT_WF = WFConfig(train_bars=600, test_bars=120, embargo_bars=16)  # final, matches the original baseline run


def all_numeric_feature_cols(feat_table: pd.DataFrame) -> list[str]:
    exclude = {"time", "close_time"}
    return [c for c in feat_table.columns if c not in exclude and pd.api.types.is_numeric_dtype(feat_table[c])]


def hpo_objective(trial, tune_samples: pd.DataFrame, feature_cols: list[str]) -> float:
    params = dict(
        num_leaves=trial.suggest_int("num_leaves", 7, 63),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
    )
    min_edge = trial.suggest_float("min_edge", 0.0, 0.15)

    splits = purged_walk_forward_splits(len(tune_samples), TUNE_WF)
    if not splits:
        return -1e9

    fold_expectancies = []
    for tr, te in splits:
        train_df = tune_samples.iloc[tr]
        test_df = tune_samples.iloc[te]
        model = LGBMBasePredictor(params).fit(train_df[feature_cols], train_df["y"])
        p_up = model.predict_proba(test_df[feature_cols])
        trades = simulate_trades(p_up, test_df, min_edge, COST)
        fold_expectancies.append(trades["pnl"].mean() if len(trades) else 0.0)

    return float(np.mean(fold_expectancies))


def run_walk_forward_with_config(feat_table, h4_df, feature_cols, params, min_edge) -> dict:
    samples = build_h4_samples(feat_table, h4_df, feature_cols)
    splits = purged_walk_forward_splits(len(samples), HOLDOUT_WF)
    all_trades = []
    for tr, te in splits:
        train_df = samples.iloc[tr]
        test_df = samples.iloc[te]
        model = LGBMBasePredictor(params).fit(train_df[feature_cols], train_df["y"])
        p_up = model.predict_proba(test_df[feature_cols])
        all_trades.append(simulate_trades(p_up, test_df, min_edge, COST))
    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(columns=["pnl"])
    from stax.backtest.metrics import summarize
    return summarize(trades_all, n_total_bars=len(samples))


def main() -> None:
    with open("config/symbols.yaml", encoding="utf-8") as f:
        symbol_cfg = yaml.safe_load(f)
    with open("config/features.yaml", encoding="utf-8") as f:
        feature_cfg = yaml.safe_load(f)

    baseline_features = feature_cfg[SYMBOL]["features"]
    baseline_min_edge = symbol_cfg[SYMBOL]["min_edge"]

    print(f"Downloading {SYMBOL} M15/H1/H4 from public CSV mirror..."
          + (f" (last {YEARS}y only)" if YEARS else ""))
    bars = load_bars_for_symbol(SYMBOL, ["M15", "H1", "H4"], years=YEARS)
    feat_table = build_feature_table(SYMBOL, bars)

    all_cols = all_numeric_feature_cols(feat_table)
    samples_all = build_h4_samples(feat_table, bars["H4"], all_cols).dropna(subset=all_cols)

    split_idx = int(len(samples_all) * TUNE_FRACTION)
    tune_samples = samples_all.iloc[:split_idx].reset_index(drop=True)
    holdout_samples_src = samples_all.iloc[split_idx:].reset_index(drop=True)
    print(f"tune={len(tune_samples)} samples, holdout={len(holdout_samples_src)} samples "
          f"(holdout never touched by tournament/HPO)")

    print("\n=== Step 1: feature tournament (on tune slice only) ===")
    indicator_cols = [c for c in all_cols if c not in PARTIAL_H4_COLS]
    X_tune = tune_samples[indicator_cols]
    y_tune = tune_samples["y"]
    selected_indicators = feature_tournament(X_tune, y_tune, top_k_per_group=3, corr_thresh=0.9)
    tournament_features = selected_indicators + PARTIAL_H4_COLS
    print(f"selected: {tournament_features}")

    print("\n=== Step 2: Optuna search ({} trials) on tune slice ===".format(N_TRIALS))
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: hpo_objective(t, tune_samples, tournament_features), n_trials=N_TRIALS, show_progress_bar=False)
    best_params = {k: v for k, v in study.best_params.items() if k != "min_edge"}
    best_min_edge = study.best_params["min_edge"]
    print(f"best tune-slice expectancy: {study.best_value:.6f}")
    print(f"best params: {best_params}")
    print(f"best min_edge: {best_min_edge:.4f}")

    print("\n=== Step 3: holdout comparison (untouched by tuning) ===")
    baseline_result = run_walk_forward_with_config(
        feat_table, bars["H4"], baseline_features, {}, baseline_min_edge
    )
    tuned_result = run_walk_forward_with_config(
        feat_table, bars["H4"], tournament_features, best_params, best_min_edge
    )

    print("\nBaseline (placeholder features, default params):")
    print(baseline_result)
    print("\nTournament + tuned:")
    print(tuned_result)

    improved = tuned_result["expectancy"] > baseline_result["expectancy"]
    print(f"\n=> Tuned config {'IMPROVES on' if improved else 'does NOT improve on'} baseline expectancy.")

    if improved:
        feature_cfg[SYMBOL]["features"] = tournament_features
        feature_cfg[SYMBOL]["tournament_run"] = True
        with open("config/features.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(feature_cfg, f, sort_keys=False, allow_unicode=True)

        symbol_cfg[SYMBOL]["min_edge"] = round(float(best_min_edge), 4)
        with open("config/symbols.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(symbol_cfg, f, sort_keys=False, allow_unicode=True)

        params_path = Path("config/model_params.yaml")
        all_params = yaml.safe_load(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
        all_params[SYMBOL] = best_params
        params_path.write_text(yaml.safe_dump(all_params, sort_keys=False), encoding="utf-8")

        print("\nconfig/features.yaml, config/symbols.yaml, and config/model_params.yaml updated.")
    else:
        print("\nLeft config/*.yaml untouched — tuning did not beat the untuned baseline on holdout.")


if __name__ == "__main__":
    main()
