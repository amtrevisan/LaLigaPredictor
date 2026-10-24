from __future__ import annotations

import numpy as np
import pandas as pd

from predictor.features import elo
from predictor.features.build import build_features, model_columns


def test_features_never_see_their_own_result(matches: pd.DataFrame) -> None:
    baseline, _ = build_features(matches)
    target = 40
    altered = matches.copy()
    altered.loc[target, ["home_goals", "away_goals"]] = [9.0, 0.0]
    changed, _ = build_features(altered)

    columns = model_columns(baseline)
    # Rows up to and including the altered match are identical...
    pd.testing.assert_frame_equal(baseline.loc[:target, columns], changed.loc[:target, columns])
    # ...and later rows involving those teams do react to it.
    assert not baseline.loc[target + 1 :, columns].equals(changed.loc[target + 1 :, columns])


def test_unplayed_fixtures_do_not_update_state(matches: pd.DataFrame) -> None:
    fixture = matches.iloc[[-1]].copy()
    fixture[["home_goals", "away_goals"]] = np.nan
    fixture["date"] = fixture["date"] + pd.Timedelta(days=7)
    extended = pd.concat(
        [matches, fixture, fixture.assign(date=fixture["date"] + pd.Timedelta(days=1))],
        ignore_index=True,
    )
    features, _ = build_features(extended)
    first, second = features.iloc[-2], features.iloc[-1]
    assert first["elo_diff"] == second["elo_diff"]
    assert first["home_gf_s"] == second["home_gf_s"]


def test_round_counts_matches_played_this_season(matches: pd.DataFrame) -> None:
    features, _ = build_features(matches)
    first_season = matches["season"] == matches["season"].min()
    assert features.loc[first_season, "round"].iloc[0] == 1
    assert features.loc[first_season, "round"].max() <= 2 * (matches["home"].nunique() - 1)


def test_strong_team_ends_with_highest_rating(matches: pd.DataFrame) -> None:
    _, ratings = build_features(matches)
    league = ratings["test"]
    assert max(league, key=league.get) == "T0"
    assert min(league, key=league.get) == "T5"


def test_elo_update_is_zero_sum_and_margin_scaled() -> None:
    h1, a1 = elo.update(1500, 1500, 1, 0)
    h3, a3 = elo.update(1500, 1500, 4, 0)
    assert h1 + a1 == 3000
    assert h3 + a3 == 3000
    assert h3 - 1500 > h1 - 1500 > 0


def test_elo_home_advantage_makes_equal_teams_home_favourites() -> None:
    assert elo.expected_home(1500, 1500) > 0.5
