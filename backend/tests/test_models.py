from __future__ import annotations

import numpy as np
import pandas as pd

from predictor.features.build import build_features, model_columns
from predictor.models.classifier import OutcomeClassifier
from predictor.models.dixon_coles import DixonColes, outcome_probabilities, score_matrix
from tests.factories import make_matches


def _fit(matches: pd.DataFrame) -> DixonColes:
    return DixonColes.fit(
        matches["home"].to_numpy(),
        matches["away"].to_numpy(),
        matches["home_goals"].to_numpy(),
        matches["away_goals"].to_numpy(),
        np.ones(len(matches)),
    )


def test_score_matrix_is_a_distribution() -> None:
    grid = score_matrix(1.6, 1.1, -0.05)
    assert np.isclose(grid.sum(), 1.0)
    assert (grid >= 0).all()
    assert np.isclose(outcome_probabilities(grid).sum(), 1.0)


def test_dixon_coles_recovers_strength_order_and_home_advantage() -> None:
    model = _fit(make_matches(n_seasons=6, seed=3))
    net = {t: model.attack[i] - model.defence[i] for t, i in model.teams.items()}
    assert sorted(net, key=net.get, reverse=True)[:2] == ["T0", "T1"]
    assert 0.1 < model.home < 0.4

    probs, xg = model.predict(np.array(["T0", "T5"]), np.array(["T5", "T0"]))
    assert probs[0, 0] > 0.6 > probs[1, 0]
    assert xg[0, 0] > xg[0, 1]


def test_unknown_team_gets_newcomer_prior() -> None:
    model = _fit(make_matches(seed=1))
    lam, mu = model.rates("T0", "Promoted FC")
    assert lam > mu


def test_classifier_outputs_probabilities() -> None:
    matches = make_matches(n_seasons=4, seed=2)
    features, _ = build_features(matches)
    y = np.select(
        [matches.home_goals > matches.away_goals, matches.home_goals == matches.away_goals],
        [0, 1],
        2,
    )
    model = OutcomeClassifier(model_columns(features)).fit(features, y)
    probs = model.predict_proba(features.tail(5))
    assert probs.shape == (5, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_empty_model_gives_neutral_prediction() -> None:
    probs, xg = DixonColes.empty().predict(np.array(["A"]), np.array(["B"]))
    assert np.isclose(probs.sum(), 1.0)
    assert xg[0, 0] > xg[0, 1]


def test_score_matrix_stays_non_negative_for_extreme_rho() -> None:
    grid = score_matrix(3.5, 3.0, 0.25)
    assert (grid >= 0).all()
    assert np.isclose(grid.sum(), 1.0)
