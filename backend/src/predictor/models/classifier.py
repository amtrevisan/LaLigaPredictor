"""Gradient-boosted outcome classifier over the engineered pre-match features.

Histogram gradient boosting handles the missing shot statistics (Brazil, early seasons) natively,
so no imputation step can leak information or blur the signal. Hyperparameters are fixed up front
rather than searched on the evaluation seasons, keeping the backtest honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

HYPERPARAMETERS = {
    "learning_rate": 0.04,
    "max_iter": 500,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 80,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 30,
    "random_state": 7,
}


@dataclass
class OutcomeClassifier:
    columns: list[str]
    model: HistGradientBoostingClassifier = field(
        default_factory=lambda: HistGradientBoostingClassifier(**HYPERPARAMETERS)
    )

    def fit(self, features: pd.DataFrame, outcomes: np.ndarray) -> OutcomeClassifier:
        self.model.fit(features[self.columns], outcomes)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(features[self.columns])

    def importance(
        self, features: pd.DataFrame, outcomes: np.ndarray, repeats: int = 5
    ) -> list[dict[str, float]]:
        """Permutation importance (increase in log loss when a column is shuffled)."""
        result = permutation_importance(
            self.model,
            features[self.columns],
            outcomes,
            scoring="neg_log_loss",
            n_repeats=repeats,
            random_state=0,
        )
        ranked = sorted(
            zip(self.columns, result.importances_mean, strict=True), key=lambda kv: -kv[1]
        )
        return [{"feature": name, "importance": float(value)} for name, value in ranked]
