"""Save/load model weights per symbol per walk-forward fold."""
from __future__ import annotations

from pathlib import Path

import joblib

DEFAULT_DIR = Path("model_store")


def save_model(model, symbol: str, name: str, fold: int, base_dir: Path = DEFAULT_DIR) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{symbol}_{name}_fold{fold}.joblib"
    joblib.dump(model, path)
    return path


def load_model(symbol: str, name: str, fold: int, base_dir: Path = DEFAULT_DIR):
    path = base_dir / f"{symbol}_{name}_fold{fold}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def load_latest(symbol: str, name: str, base_dir: Path = DEFAULT_DIR):
    candidates = sorted(base_dir.glob(f"{symbol}_{name}_fold*.joblib"))
    if not candidates:
        return None
    return joblib.load(candidates[-1])
