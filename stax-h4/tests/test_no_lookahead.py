"""The most important test in this repo (per architecture doc §P0 DoD):
prove the feature/label pipeline never has access to information from the
future relative to the decision timestamp.

These tests run on synthetic OHLCV data — no live MT5 connection required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stax.backtest.walk_forward import build_h4_samples
from stax.features.build import build_feature_table
from stax.features.partial_h4 import partial_h4_state


def _make_synthetic_bars(n_h4: int = 60, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_m15 = n_h4 * 16

    t0 = 1_700_000_000  # arbitrary epoch, aligned to a clean H4 boundary
    m15_time = t0 + np.arange(n_m15) * 900
    price = 1.1000 + np.cumsum(rng.normal(0, 0.0003, n_m15))

    def ohlc_from_close(times, step_s, close):
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0001, len(close)))
        low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0001, len(close)))
        return pd.DataFrame({
            "time": times, "open": open_, "high": high, "low": low, "close": close,
            "tick_volume": rng.integers(10, 500, len(close)),
            "close_time": times + step_s,
        })

    m15 = ohlc_from_close(m15_time, 900, price)

    def resample(df, factor, step_s):
        idx = np.arange(0, len(df), factor)
        agg = []
        for i in idx:
            chunk = df.iloc[i:i + factor]
            if len(chunk) == 0:
                continue
            agg.append({
                "time": chunk["time"].iloc[0],
                "open": chunk["open"].iloc[0],
                "high": chunk["high"].max(),
                "low": chunk["low"].min(),
                "close": chunk["close"].iloc[-1],
                "tick_volume": chunk["tick_volume"].sum(),
                "close_time": chunk["time"].iloc[0] + step_s,
            })
        return pd.DataFrame(agg)

    m5 = ohlc_from_close(t0 + np.arange(n_m15 * 3) * 300, 300,
                          1.1000 + np.cumsum(rng.normal(0, 0.0001, n_m15 * 3)))
    h1 = resample(m15, 4, 3600)
    h4 = resample(m15, 16, 14400)
    return {"M5": m5, "M15": m15, "H1": h1, "H4": h4}


def test_partial_h4_state_uses_only_bars_up_to_k():
    bars = _make_synthetic_bars()
    m15 = bars["M15"]
    h4_start_time = bars["H4"]["time"].iloc[3]
    in_h4_full = m15[(m15["time"] >= h4_start_time) & (m15["time"] < h4_start_time + 14400)]

    state_k8 = partial_h4_state(in_h4_full.iloc[:8], atr_ref=0.001)
    state_k8_with_future = partial_h4_state(in_h4_full, atr_ref=0.001)  # full 16 bars

    assert state_k8["pos_in_h4"] == pytest.approx(8 / 16.0)
    # k=8 state must NOT equal the full-candle state — i.e. it can't see bars 9-16
    assert state_k8["h4_close_loc"] != state_k8_with_future["h4_close_loc"]


def test_feature_table_partial_h4_progression_is_monotonic_in_position():
    bars = _make_synthetic_bars()
    feat = build_feature_table("EURUSD", bars)
    h4_start = bars["H4"]["time"].iloc[5]
    in_h4 = feat[(feat["time"] >= h4_start) & (feat["time"] < h4_start + 14400)]
    assert list(in_h4["pos_in_h4"]) == sorted(in_h4["pos_in_h4"])  # strictly increasing 1/16..16/16
    assert in_h4["pos_in_h4"].iloc[-1] == pytest.approx(1.0)


def test_h4_samples_label_is_strictly_future_relative_to_decision_row():
    bars = _make_synthetic_bars()
    feat = build_feature_table("EURUSD", bars)
    feature_cols = [c for c in feat.columns if c not in ("time", "close_time")]
    samples = build_h4_samples(feat, bars["H4"], feature_cols)

    h4 = bars["H4"].reset_index(drop=True)
    for _, row in samples.iterrows():
        decision_close_time = row["close_time"]
        # the H4 bar whose direction is being predicted must open strictly
        # after the decision timestamp
        matching = h4[h4["close_time"] == decision_close_time]
        assert len(matching) >= 1
        h4_idx = matching.index[0]
        if h4_idx + 1 < len(h4):
            next_open_time = h4["time"].iloc[h4_idx + 1]
            assert next_open_time >= decision_close_time


def test_get_closed_bars_asof_guard_excludes_future_bars():
    from stax.data import bars as bars_mod

    class FakeRates:
        def __init__(self, n):
            self.n = n

        def __call__(self, symbol, tf, start, count):
            t0 = 1_700_000_000
            dtype = [("time", "i8"), ("open", "f8"), ("high", "f8"),
                     ("low", "f8"), ("close", "f8"), ("tick_volume", "i8")]
            arr = np.zeros(count, dtype=dtype)
            arr["time"] = t0 + np.arange(start, start + count) * 900
            arr["close"] = 1.1
            return arr

    import MetaTrader5 as mt5
    mt5.copy_rates_from_pos = FakeRates(20)

    asof = 1_700_000_000 + 5 * 900 + 900  # allow exactly bar index 5 to close
    df = bars_mod.get_closed_bars("EURUSD", "M15", 10, asof_epoch=asof)
    assert (df["close_time"] <= asof).all()
