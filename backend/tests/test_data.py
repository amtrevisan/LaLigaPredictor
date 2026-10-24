from __future__ import annotations

import json

import numpy as np
import pandas as pd

from predictor.config import LEAGUES_BY_ID
from predictor.data import normalize
from predictor.data.teams import team_info
from predictor.export import write_json
from predictor.models.dixon_coles import outcome_probabilities, score_matrix
from predictor.pipeline import _meetings, align_scoreline_grid, choose_blend_weight
from tests.factories import make_matches

LA_LIGA = LEAGUES_BY_ID["la-liga"]


def _raw(**overrides: object) -> pd.DataFrame:
    row = {
        "Div": "SP1",
        "Date": "15/08/2025",
        "Time": "20:30",
        "HomeTeam": "Ath Madrid",
        "AwayTeam": "Sociedad",
        "FTHG": 2,
        "FTAG": 1,
        "HS": 14,
        "AS": 9,
        "HST": 6,
        "AST": 3,
        "AvgH": 1.8,
        "AvgD": 3.6,
        "AvgA": 4.5,
        "AvgCH": 1.7,
        "AvgCD": 3.7,
        "AvgCA": 5.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_kickoff_is_converted_from_uk_time_to_utc() -> None:
    out = normalize.normalize_main_season(_raw(), LA_LIGA, 2025)
    assert out["date"].iloc[0] == pd.Timestamp("2025-08-15 19:30", tz="UTC")


def test_two_digit_years_and_missing_time_parse() -> None:
    raw = _raw(Date="05/08/05").drop(columns=["Time"])
    out = normalize.normalize_main_season(raw, LA_LIGA, 2005)
    assert out["date"].iloc[0] == pd.Timestamp("2005-08-05 14:00", tz="UTC")


def test_closing_odds_preferred_over_opening() -> None:
    out = normalize.normalize_main_season(_raw(), LA_LIGA, 2025)
    assert out["odds_home"].iloc[0] == 1.7
    assert out["odds_kind"].iloc[0] == "closing"
    opening_only = normalize.normalize_main_season(
        _raw(AvgCH=None, AvgCD=None, AvgCA=None), LA_LIGA, 2025
    )
    assert opening_only["odds_home"].iloc[0] == 1.8
    assert opening_only["odds_kind"].iloc[0] == "opening"


def test_combine_keeps_the_played_copy_of_a_match() -> None:
    played = normalize.normalize_main_season(_raw(), LA_LIGA, 2025)
    scheduled = normalize.normalize_main_season(_raw(FTHG=None, FTAG=None), LA_LIGA, 2025)
    combined = normalize.combine([played, scheduled])
    assert len(combined) == 1
    assert combined["home_goals"].iloc[0] == 2


def test_team_info_falls_back_to_feed_name() -> None:
    assert team_info("Ath Madrid").name == "Atlético Madrid"
    unknown = team_info("Real Zaragoza")
    assert unknown.name == "Real Zaragoza"
    assert unknown.short == "REA"


def test_scoreline_grid_alignment_matches_target_probabilities() -> None:
    target = np.array([0.5, 0.3, 0.2])
    aligned = align_scoreline_grid(score_matrix(1.4, 1.1, -0.05), target)
    assert np.allclose(outcome_probabilities(aligned), target)


def test_blend_weight_picks_the_better_model() -> None:
    y = np.array([0, 0, 2, 1])
    good = np.eye(3)[y] * 0.7 + 0.1
    bad = np.full((4, 3), 1 / 3)
    assert choose_blend_weight(good, bad, y) == 1.0


def test_write_json_ignores_timestamp_only_changes(tmp_path) -> None:
    path = tmp_path / "x.json"
    assert write_json(path, {"generatedAt": "a", "value": 1})
    assert not write_json(path, {"generatedAt": "b", "value": 1})
    assert json.loads(path.read_text())["generatedAt"] == "a"
    assert write_json(path, {"generatedAt": "c", "value": 2})


def test_write_json_refuses_nan_and_keeps_previous_file(tmp_path) -> None:
    import pytest

    from predictor.export import write_all

    path = tmp_path / "x.json"
    write_json(path, {"value": 1})
    with pytest.raises(ValueError):
        write_all({path: {"value": float("nan")}, tmp_path / "y.json": {"value": 2}})
    assert json.loads(path.read_text()) == {"value": 1}
    assert not (tmp_path / "y.json").exists()


def test_commit_hash_alone_does_not_rewrite(tmp_path) -> None:
    path = tmp_path / "model.json"
    write_json(path, {"commit": "abc1234", "value": 1})
    assert not write_json(path, {"commit": "def5678", "value": 1})


def test_fallback_crest_only_for_listed_clubs() -> None:
    assert (
        team_info("Barcelona").fallback_crest
        == "https://media.api-sports.io/football/teams/529.png"
    )
    assert team_info("Arsenal").fallback_crest is None


def test_meetings_come_from_earlier_results_between_the_pair():
    history = make_matches(n_seasons=3, teams=4)
    fixture = pd.Series({"home": "T0", "away": "T1", "date": pd.Timestamp("2017-01-01", tz="UTC")})
    meetings = _meetings(LEAGUES_BY_ID["la-liga"], history, fixture)

    # Both venues of 2015 and 2016 (2017 kicks off after the fixture), newest first.
    assert [m["date"][:4] for m in meetings] == ["2016", "2016", "2015", "2015"]
    assert all({m["home"], m["away"]} == {"T0", "T1"} for m in meetings)
    assert meetings[0]["competition"] == "La Liga"
