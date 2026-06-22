"""Feature/bar cache. P0/P1 uses Parquet on local disk; swap for TimescaleDB later."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path("data_cache")


def save_parquet(df: pd.DataFrame, symbol: str, name: str, base_dir: Path = DEFAULT_DIR) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{symbol}_{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_parquet(symbol: str, name: str, base_dir: Path = DEFAULT_DIR) -> pd.DataFrame | None:
    path = base_dir / f"{symbol}_{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)
