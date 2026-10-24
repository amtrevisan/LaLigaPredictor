"""Leak-free, pre-match feature construction.

One chronological pass over every match. For each row the features are read from team state
*before* the match is applied, then the state is updated with the result (unplayed fixtures only
read). That ordering is the whole leakage guarantee, and ``tests/test_features.py`` pins it.

Teams are tracked per league: a club relegated and later promoted keeps its history, while the
same feed name in another league (rare, but "Internacional" vs "Inter") never collides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from predictor.features import elo

# Exponential smoothing rates. ``SHORT`` tracks current form (half-life ~3 matches), ``LONG``
# underlying level (~14 matches), ``VENUE`` home-only / away-only scoring.
SHORT = 0.20
LONG = 0.05
VENUE = 0.12
LEAGUE = 1.0 / 400.0

# Rating given to a club entering a league mid-history, relative to the league mean. Promoted
# sides are weaker than average; -80 Elo is roughly the historical bottom-third gap.
NEWCOMER_OFFSET = -80.0
REST_CAP_DAYS = 21.0

TEAM_STATS = ("gf", "ga", "sf", "sa", "sotf", "sota", "pts")


@dataclass
class _Ewm:
    """Exponentially weighted mean with a running-mean warm-up (no zero-initialisation bias)."""

    alpha: float
    value: float = math.nan
    count: int = 0

    def update(self, x: float) -> None:
        if math.isnan(x):
            return
        self.count += 1
        rate = max(self.alpha, 1.0 / self.count)
        self.value = x if math.isnan(self.value) else self.value + rate * (x - self.value)


@dataclass
class _Team:
    rating: float
    short: dict[str, _Ewm] = field(default_factory=lambda: {k: _Ewm(SHORT) for k in TEAM_STATS})
    long: dict[str, _Ewm] = field(
        default_factory=lambda: {k: _Ewm(LONG) for k in ("gf", "ga", "pts")}
    )
    home_gf: _Ewm = field(default_factory=lambda: _Ewm(VENUE))
    home_ga: _Ewm = field(default_factory=lambda: _Ewm(VENUE))
    away_gf: _Ewm = field(default_factory=lambda: _Ewm(VENUE))
    away_ga: _Ewm = field(default_factory=lambda: _Ewm(VENUE))
    last_match: pd.Timestamp | None = None
    season: int | None = None
    season_played: int = 0


@dataclass
class _League:
    teams: dict[str, _Team] = field(default_factory=dict)
    season: int | None = None
    home_rate: _Ewm = field(default_factory=lambda: _Ewm(LEAGUE))
    draw_rate: _Ewm = field(default_factory=lambda: _Ewm(LEAGUE))
    goals: _Ewm = field(default_factory=lambda: _Ewm(LEAGUE))

    def mean_rating(self) -> float:
        if not self.teams:
            return elo.BASE_RATING
        return float(np.mean([t.rating for t in self.teams.values()]))

    def team(self, name: str) -> _Team:
        if name not in self.teams:
            start = elo.BASE_RATING if not self.teams else self.mean_rating() + NEWCOMER_OFFSET
            self.teams[name] = _Team(rating=start)
        return self.teams[name]

    def start_season(self, season: int) -> None:
        """Re-centre the league on ``BASE_RATING`` and regress every club toward it.

        Promotion and relegation are not zero-sum (newcomers enter below average, relegated
        clubs leave with their points), so without re-centring the league mean drifts down
        season after season.
        """
        if self.season is not None and season > self.season:
            active = [t.rating for t in self.teams.values() if t.season == self.season]
            shift = elo.BASE_RATING - float(np.mean(active)) if active else 0.0
            for team in self.teams.values():
                team.rating = elo.regress(team.rating + shift, elo.BASE_RATING)
        self.season = season


def _points(goals_for: float, goals_against: float) -> float:
    return 3.0 if goals_for > goals_against else 1.0 if goals_for == goals_against else 0.0


def _side_features(team: _Team, prefix: str, venue: str, date: pd.Timestamp, season: int) -> dict:
    out = {f"{prefix}_elo": team.rating}
    for stat, ewm in team.short.items():
        out[f"{prefix}_{stat}_s"] = ewm.value
    for stat, ewm in team.long.items():
        out[f"{prefix}_{stat}_l"] = ewm.value
    gf, ga = (team.home_gf, team.home_ga) if venue == "home" else (team.away_gf, team.away_ga)
    out[f"{prefix}_venue_gf"] = gf.value
    out[f"{prefix}_venue_ga"] = ga.value
    rest = math.nan
    if team.last_match is not None:
        rest = min((date - team.last_match).total_seconds() / 86400.0, REST_CAP_DAYS)
    out[f"{prefix}_rest"] = rest
    out[f"{prefix}_season_played"] = team.season_played if team.season == season else 0
    return out


def _apply_result(
    team: _Team,
    venue: str,
    gf: float,
    ga: float,
    sf: float,
    sa: float,
    sotf: float,
    sota: float,
    date: pd.Timestamp,
    season: int,
) -> None:
    pts = _points(gf, ga)
    for stat, value in zip(TEAM_STATS, (gf, ga, sf, sa, sotf, sota, pts), strict=True):
        team.short[stat].update(value)
    for stat, value in (("gf", gf), ("ga", ga), ("pts", pts)):
        team.long[stat].update(value)
    if venue == "home":
        team.home_gf.update(gf)
        team.home_ga.update(ga)
    else:
        team.away_gf.update(gf)
        team.away_ga.update(ga)
    if team.season != season:
        team.season, team.season_played = season, 0
    team.season_played += 1
    team.last_match = date


def build_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Pre-match features for every row of ``matches`` (index-aligned), plus a ``round`` column,
    and the Elo rating of every team after the last played match (``{league: {team: elo}}``).

    ``matches`` must be sorted chronologically (``data.loader`` guarantees it).
    """
    leagues: dict[str, _League] = {}
    rows: list[dict] = []
    for m in matches.itertuples(index=False):
        league = leagues.setdefault(m.league, _League())
        league.start_season(m.season)
        home, away = league.team(m.home), league.team(m.away)

        row = _side_features(home, "home", "home", m.date, m.season)
        row.update(_side_features(away, "away", "away", m.date, m.season))
        row["elo_diff"] = home.rating - away.rating
        row["elo_expected"] = elo.expected_home(home.rating, away.rating)
        row["league_home_rate"] = league.home_rate.value
        row["league_draw_rate"] = league.draw_rate.value
        row["league_goals"] = league.goals.value
        row["round"] = max(row["home_season_played"], row["away_season_played"]) + 1
        rows.append(row)

        if math.isnan(m.home_goals) or math.isnan(m.away_goals):
            continue
        hg, ag = float(m.home_goals), float(m.away_goals)
        _apply_result(
            home,
            "home",
            hg,
            ag,
            m.home_shots,
            m.away_shots,
            m.home_sot,
            m.away_sot,
            m.date,
            m.season,
        )
        _apply_result(
            away,
            "away",
            ag,
            hg,
            m.away_shots,
            m.home_shots,
            m.away_sot,
            m.home_sot,
            m.date,
            m.season,
        )
        home.rating, away.rating = elo.update(home.rating, away.rating, int(hg), int(ag))
        league.home_rate.update(1.0 if hg > ag else 0.0)
        league.draw_rate.update(1.0 if hg == ag else 0.0)
        league.goals.update(hg + ag)

    features = pd.DataFrame(rows, index=matches.index)
    for stat in ("gf_s", "ga_s", "sotf_s", "sota_s", "pts_s", "gf_l", "ga_l", "pts_l"):
        features[f"diff_{stat}"] = features[f"home_{stat}"] - features[f"away_{stat}"]
    features["diff_venue"] = (features["home_venue_gf"] - features["home_venue_ga"]) - (
        features["away_venue_gf"] - features["away_venue_ga"]
    )
    features["diff_rest"] = features["home_rest"] - features["away_rest"]
    ratings = {
        league_id: {name: team.rating for name, team in league.teams.items()}
        for league_id, league in leagues.items()
    }
    return features, ratings


def model_columns(features: pd.DataFrame) -> list[str]:
    """Columns fed to the classifier (everything except bookkeeping)."""
    return [c for c in features.columns if c != "round"]
