"""MT5 connection management for broker IUX.

IUX does not suffix symbol names (no ".r" / ".cash" style decoration), so
symbol names in config/symbols.yaml are used as-is against the terminal.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import MetaTrader5 as mt5
import yaml

from stax.data.timeframes import TF_SECONDS

TIMEFRAME = {
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}


@dataclass
class MT5Config:
    login: int
    password: str
    server: str
    path: str | None = None


def load_runtime_config(path: str = "config/runtime.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def connect(cfg: MT5Config) -> None:
    """Initialize the MT5 terminal connection. Raises RuntimeError on failure."""
    init_kwargs = {}
    if cfg.path:
        init_kwargs["path"] = cfg.path
    if not mt5.initialize(**init_kwargs):
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

    password = cfg.password or os.environ.get("MT5_PASSWORD", "")
    if cfg.login:
        ok = mt5.login(cfg.login, password=password, server=cfg.server)
        if not ok:
            raise RuntimeError(f"mt5.login() failed: {mt5.last_error()}")


def disconnect() -> None:
    mt5.shutdown()


def ensure_symbol(symbol: str) -> None:
    """Make sure the symbol exists and is visible in Market Watch before use."""
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(
            f"Symbol '{symbol}' not found on IUX terminal — check exact name "
            f"in MT5 Market Watch (IUX uses no suffix, but confirm spelling)."
        )
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select symbol '{symbol}' in Market Watch")
