"""Honest metrics — accuracy is misleading on a ~50-53% edge signal, so
report expectancy, PR-AUC/lift, payoff ratio, and selectivity (% HOLD)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def expectancy(pnl: pd.Series) -> float:
    return float(pnl.mean()) if len(pnl) else 0.0


def payoff_ratio(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    if len(wins) == 0 or len(losses) == 0:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


def hit_rate(pnl: pd.Series) -> float:
    return float((pnl > 0).mean()) if len(pnl) else float("nan")


def pr_auc_lift(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    base_rate = float(np.mean(y_true))
    pr_auc = float(average_precision_score(y_true, y_score))
    return {"pr_auc": pr_auc, "base_rate": base_rate, "lift": pr_auc / base_rate if base_rate > 0 else float("nan")}


def sharpe(pnl: pd.Series, periods_per_year: float) -> float:
    if pnl.std() == 0 or len(pnl) < 2:
        return float("nan")
    return float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max.replace(0, np.nan)
    return float(dd.min()) if len(dd) else float("nan")


def selectivity(n_hold: int, n_total: int) -> float:
    return n_hold / n_total if n_total else float("nan")


def summarize(trades: pd.DataFrame, n_total_bars: int, periods_per_year: float = 365 * 6) -> dict:
    """trades: DataFrame with column `pnl` (one row per executed trade, HOLD excluded)."""
    pnl = trades["pnl"] if "pnl" in trades else pd.Series(dtype=float)
    equity = pnl.cumsum() + 1.0
    return {
        "n_trades": len(trades),
        "n_hold": n_total_bars - len(trades),
        "selectivity": selectivity(n_total_bars - len(trades), n_total_bars),
        "expectancy": expectancy(pnl),
        "payoff_ratio": payoff_ratio(pnl),
        "hit_rate": hit_rate(pnl),
        "sharpe": sharpe(pnl, periods_per_year),
        "max_drawdown": max_drawdown(equity) if len(equity) else float("nan"),
    }
