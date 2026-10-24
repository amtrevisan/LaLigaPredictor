"""Elo ratings with home advantage and a goal-margin multiplier (World Football Elo style)."""

from __future__ import annotations

BASE_RATING = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 65.0
# Fraction of the distance to the league mean recovered over a summer break.
SEASON_REGRESSION = 0.25


def expected_home(
    home_rating: float, away_rating: float, home_advantage: float = HOME_ADVANTAGE
) -> float:
    """Expected score (win = 1, draw = 0.5) for the home side."""
    return 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating - home_advantage) / 400.0))


def margin_multiplier(goal_difference: int) -> float:
    margin = abs(goal_difference)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def update(
    home_rating: float, away_rating: float, home_goals: int, away_goals: int
) -> tuple[float, float]:
    """Ratings after a result. The exchange is zero-sum."""
    actual = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
    expected = expected_home(home_rating, away_rating)
    delta = K_FACTOR * margin_multiplier(home_goals - away_goals) * (actual - expected)
    return home_rating + delta, away_rating - delta


def regress(rating: float, mean: float, fraction: float = SEASON_REGRESSION) -> float:
    return rating + fraction * (mean - rating)
