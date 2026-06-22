"""Per-symbol normalization. Stats must be fit on the train window only —
never on test/future data (would leak)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class NormStats:
    mean: dict = field(default_factory=dict)
    std: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "NormStats":
        for c in columns:
            self.mean[c] = float(df[c].mean())
            self.std[c] = float(df[c].std() or 1.0)
        return self

    def transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        out = df.copy()
        for c in columns:
            mu = self.mean.get(c, 0.0)
            sd = self.std.get(c, 1.0) or 1.0
            out[c] = (df[c] - mu) / sd
        return out


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def rolling_z(series: pd.Series, window: int = 500, min_periods: int = 20) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / (std + 1e-9)
