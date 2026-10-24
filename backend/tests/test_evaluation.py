from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictor.evaluation import backtest, metrics
from predictor.features.build import build_features, model_columns
from tests.factories import make_matches


def test_perfect_forecast_scores_zero() -> None:
    probs = np.eye(3)
    y = np.array([0, 1, 2])
    assert metrics.log_loss(probs, y) == pytest.approx(0.0, abs=1e-9)
    assert metrics.rps(probs, y) == 0.0
    assert metrics.brier(probs, y) == 0.0
    assert metrics.accuracy(probs, y) == 1.0


def test_rps_prefers_near_misses() -> None:
    # Home win happened: a draw-leaning forecast beats an away-leaning one under RPS.
    y = np.array([0])
    draw_lean = np.array([[0.2, 0.6, 0.2]])
    away_lean = np.array([[0.2, 0.2, 0.6]])
    assert metrics.rps(draw_lean, y) < metrics.rps(away_lean, y)
    assert metrics.brier(draw_lean, y) == pytest.approx(metrics.brier(away_lean, y))


def test_implied_probabilities_remove_margin() -> None:
    probs = metrics.implied_probabilities(np.array([[2.0, 3.4, 3.8]]))
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0, 0] > probs[0, 2]


def test_calibration_counts_every_probability() -> None:
    probs = np.array([[0.5, 0.3, 0.2], [0.6, 0.25, 0.15]])
    table = metrics.calibration(probs, np.array([0, 1]))
    assert sum(row["count"] for row in table) == probs.size


def test_retrain_boundaries_cover_the_target_window() -> None:
    start, end = pd.Timestamp("2023-08-10", tz="UTC"), pd.Timestamp("2024-05-20", tz="UTC")
    boundaries = backtest.retrain_boundaries(start, end)
    assert boundaries[0] <= start
    assert all(b.month in (1, 7) and b.day == 1 for b in boundaries)


def test_classifier_walk_forward_trains_only_on_the_past() -> None:
    matches = make_matches(n_seasons=5, seed=4)
    features, _ = build_features(matches)
    target = matches["season"] == matches["season"].max()
    preds, model = backtest.classifier_walk_forward(
        matches, features, model_columns(features), target
    )
    assert preds.notna().all().all()

    boundary = backtest.retrain_boundaries(
        matches.loc[target, "date"].min(), matches["date"].max()
    )[0]
    train = backtest.training_mask(matches, boundary)
    assert (matches.loc[train, "date"] < boundary).all()
    assert not (train & target).any()


def test_dixon_coles_walk_forward_only_fits_on_earlier_weeks() -> None:
    matches = make_matches(n_seasons=3, seed=5)
    target = matches["season"] == matches["season"].max()
    first_week = matches.loc[target, "date"].min()
    poisoned = matches.copy()
    # Rewriting results from the target season onward must not change the first week's forecast.
    later = target & (matches["date"] > first_week + pd.Timedelta(days=7))
    poisoned.loc[later, "home_goals"] = 9.0
    clean = backtest.dixon_coles_walk_forward(matches, target)
    dirty = backtest.dixon_coles_walk_forward(poisoned, target)
    first = matches.index[target & (matches["date"] < first_week + pd.Timedelta(days=1))]
    pd.testing.assert_frame_equal(clean.loc[first], dirty.loc[first])
