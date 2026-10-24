"""Assemble the full match table (history plus upcoming fixtures) for the configured leagues."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from predictor.config import LEAGUES, League
from predictor.data import normalize, sources

log = logging.getLogger(__name__)

STALE_FIXTURE_GRACE = timedelta(days=2)


def current_main_season(now: datetime) -> int:
    """European seasons start in July/August and are keyed by their starting year."""
    return now.year if now.month >= 7 else now.year - 1


def load_league(
    league: League, now: datetime, fixtures: dict[str, pd.DataFrame], max_age: timedelta
) -> pd.DataFrame:
    if league.source == "main":
        current = current_main_season(now)
        frames = []
        for season in range(league.first_season, current + 1):
            raw = sources.fetch_main_season(
                league, season, current=season == current, max_age=max_age
            )
            if raw is None:
                log.warning("%s: no archive file for season %s", league.id, season)
                continue
            frames.append(normalize.normalize_main_season(raw, league, season))
        frames.append(normalize.normalize_main_fixtures(fixtures["main"], league, current))
    else:
        history = normalize.normalize_extra(
            sources.fetch_extra_league(league, max_age=max_age), league
        )
        current = max(int(history["season"].max()), now.year)
        frames = [history, normalize.normalize_extra_fixtures(fixtures["extra"], league, current)]
    matches = normalize.combine(frames)
    # Unplayed rows in the past are abandoned or postponed-without-a-date matches; a postponed
    # match reappears in the fixtures feed once it is rescheduled.
    stale = matches["home_goals"].isna() & (matches["date"] < now - STALE_FIXTURE_GRACE)
    matches = matches[~stale].reset_index(drop=True)
    log.info(
        "%s: %d matches (%d scheduled) across %d seasons",
        league.id,
        len(matches),
        int(matches["home_goals"].isna().sum()),
        matches["season"].nunique(),
    )
    return matches


def load_matches(
    now: datetime,
    leagues: tuple[League, ...] = LEAGUES,
    max_age: timedelta = timedelta(hours=1),
) -> pd.DataFrame:
    fixtures = {
        "main": sources.fetch_fixtures("main", max_age=max_age),
        "extra": sources.fetch_fixtures("extra", max_age=max_age),
    }
    frames = [load_league(league, now, fixtures, max_age) for league in leagues]
    matches = pd.concat(frames, ignore_index=True)
    return matches.sort_values(["date", "league", "home"], kind="stable").reset_index(drop=True)
