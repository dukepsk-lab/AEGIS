"""Model 1 — LightGBM on multi-TF tabular features. P1 scope: this is the
only model trained; Models 2/3 + meta-stacking arrive in P2."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS = dict(
    objective="binary",
    n_estimators=300,
    num_leaves=15,
    min_child_samples=50,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    verbosity=-1,
)


class LGBMBasePredictor:
    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: lgb.LGBMClassifier | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMBasePredictor":
        self.feature_names_ = list(X.columns)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)
        return self

    def predict_proba(self, feats: pd.DataFrame | dict) -> np.ndarray:
        if isinstance(feats, dict):
            feats = pd.DataFrame([feats])
        feats = feats[self.feature_names_]
        return self.model.predict_proba(feats)[:, 1]

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.feature_names_).sort_values(ascending=False)
