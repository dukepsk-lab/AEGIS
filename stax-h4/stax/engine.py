"""Main event loop — bar-close driven, never wall-clock (broker H4
boundaries can be offset). P1 scope: EURUSD only, Model 1 (LightGBM)
directly gates the decision; no meta-model/Kelly sizing yet (P2/P3).
Paper mode by default — set runtime.yaml `loop.mode: live` for real orders,
and only after P3 risk/exec hardening per the architecture doc.
"""
from __future__ import annotations

import logging
import time

import MetaTrader5 as mt5

from stax.data import bars as bars_mod
from stax.data.mt5_client import MT5Config, connect, disconnect, ensure_symbol, load_runtime_config
from stax.features.build import build_feature_table
from stax.models.registry import load_latest

log = logging.getLogger("stax.engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def fetch_multi_tf(symbol: str, asof_epoch: int, lookback: dict[str, int]) -> dict:
    return {
        tf: bars_mod.get_closed_bars(symbol, tf, n, asof_epoch=asof_epoch)
        for tf, n in lookback.items()
    }


def on_m15_close(symbol: str, asof: int) -> None:
    log.info(f"[{symbol}] M15 close asof={asof} — feature refresh (P1: logged only, no trajectory buffer yet)")


def on_h4_close(symbol: str, asof: int, cfg: dict, model) -> None:
    log.info(f"[{symbol}] H4 close asof={asof} — running P1 decision")
    if model is None:
        log.warning(f"[{symbol}] no trained model found in model_store/ — skipping. Run scripts/train_walk_forward.py first.")
        return

    lookback = {"M5": 300, "M15": 100, "H1": 100, "H4": 30}
    multi = fetch_multi_tf(symbol, asof, lookback)
    if any(v is None or len(v) == 0 for v in multi.values()):
        log.warning(f"[{symbol}] insufficient bar history — HOLD")
        return

    feat_table = build_feature_table(symbol, multi)
    last_row = feat_table.iloc[[-1]][model.feature_names_]
    if last_row.isna().any(axis=1).iloc[0]:
        log.warning(f"[{symbol}] NaN in feature row — HOLD")
        return

    p_up = float(model.predict_proba(last_row)[0])
    edge = abs(p_up - 0.5)
    min_edge = cfg["symbols"][symbol]["min_edge"]

    if edge < min_edge:
        log.info(f"[{symbol}] p_up={p_up:.3f} edge={edge:.3f} < min_edge={min_edge} -> HOLD")
        return

    side = "BUY" if p_up > 0.5 else "SELL"
    log.info(f"[{symbol}] p_up={p_up:.3f} edge={edge:.3f} -> {side} (P1: paper-log only, execution wired in P3)")


def event_loop(symbols: list[str], runtime_cfg: dict, symbol_cfg: dict) -> None:
    last_m15 = {s: None for s in symbols}
    last_h4 = {s: None for s in symbols}
    models = {s: load_latest(s, "lgbm_base") for s in symbols}

    poll = runtime_cfg["loop"]["poll_interval_s"]
    log.info(f"event_loop starting for symbols={symbols} mode={runtime_cfg['loop']['mode']}")

    while True:
        for s in symbols:
            if not bars_mod.market_open(s):
                continue

            t15 = bars_mod.latest_closed_bar_time(s, "M15")
            if t15 is not None and t15 != last_m15[s]:
                last_m15[s] = t15
                on_m15_close(s, asof=t15 + 900)

            t4 = bars_mod.latest_closed_bar_time(s, "H4")
            if t4 is not None and t4 != last_h4[s]:
                last_h4[s] = t4
                on_h4_close(s, asof=t4 + 4 * 3600, cfg={"symbols": symbol_cfg}, model=models[s])

        time.sleep(poll)


def run(runtime_path: str = "config/runtime.yaml", symbols_path: str = "config/symbols.yaml") -> None:
    import yaml

    runtime_cfg = load_runtime_config(runtime_path)
    with open(symbols_path, encoding="utf-8") as f:
        symbol_cfg = yaml.safe_load(f)

    active_symbols = [s for s, c in symbol_cfg.items() if c.get("active")]
    if not active_symbols:
        raise RuntimeError("No active symbols in config/symbols.yaml")

    mt5_cfg = MT5Config(**runtime_cfg["mt5"])
    connect(mt5_cfg)
    try:
        for s in active_symbols:
            ensure_symbol(s)
        event_loop(active_symbols, runtime_cfg, symbol_cfg)
    finally:
        disconnect()


if __name__ == "__main__":
    run()
