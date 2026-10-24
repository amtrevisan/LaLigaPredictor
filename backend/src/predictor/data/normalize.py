"""Normalise the heterogeneous football-data.co.uk layouts into one match table.

Output schema (one row per match, played or scheduled)::

    league, season, date (UTC), home, away,
    home_goals, away_goals,            # NaN when not yet played
    home_shots, away_shots, home_sot, away_sot,   # NaN where the feed has no statistics
    odds_home, odds_draw, odds_away,   # decimal odds, closing line when available
    odds_kind                          # "closing" | "opening" | None
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from predictor.config import League

COLUMNS = [
    "league",
    "season",
    "date",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "home_shots",
    "away_shots",
    "home_sot",
    "away_sot",
    "odds_home",
    "odds_draw",
    "odds_away",
    "odds_kind",
]

# Odds columns in order of preference. Closing prices are the sharpest benchmark, so they win
# whenever the feed has them; older seasons and unplayed fixtures only carry opening prices.
CLOSING_ODDS = (
    ("AvgCH", "AvgCD", "AvgCA"),
    ("PSCH", "PSCD", "PSCA"),
    ("B365CH", "B365CD", "B365CA"),
)
OPENING_ODDS = (
    ("AvgH", "AvgD", "AvgA"),
    ("BbAvH", "BbAvD", "BbAvA"),
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
)

DEFAULT_KICKOFF = "15:00"
FEED_TIMEZONE = "Europe/London"


def _parse_kickoff(dates: pd.Series, times: pd.Series | None) -> pd.Series:
    clock = times.fillna(DEFAULT_KICKOFF) if times is not None else DEFAULT_KICKOFF
    stamp = dates.astype(str).str.strip() + " " + pd.Series(clock, index=dates.index).astype(str)
    parsed = pd.to_datetime(stamp, format="%d/%m/%Y %H:%M", errors="coerce")
    # Seasons before 2017/18 use two-digit years.
    short = parsed.isna()
    if short.any():
        parsed[short] = pd.to_datetime(stamp[short], format="%d/%m/%y %H:%M", errors="coerce")
    # Kick-offs in the repeated autumn hour are read as the later (GMT) occurrence rather than
    # dropped.
    ambiguous = np.zeros(len(parsed), dtype=bool)
    localized = parsed.dt.tz_localize(
        FEED_TIMEZONE, ambiguous=ambiguous, nonexistent="shift_forward"
    )
    return localized.dt.tz_convert("UTC")


def _odds(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    result = pd.DataFrame(
        np.nan, index=frame.index, columns=["odds_home", "odds_draw", "odds_away"]
    )
    kind = pd.Series(None, index=frame.index, dtype=object)
    for label, candidates in (("closing", CLOSING_ODDS), ("opening", OPENING_ODDS)):
        for columns in candidates:
            if not set(columns) <= set(frame.columns):
                continue
            values = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
            usable = values.notna().all(axis=1) & (values > 1.0).all(axis=1) & kind.isna()
            result.loc[usable] = values.loc[usable].to_numpy()
            kind.loc[usable] = label
    return result, kind


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _build(
    frame: pd.DataFrame,
    league: League,
    season: pd.Series | int,
    home_col: str,
    away_col: str,
    goal_cols: tuple[str, str],
) -> pd.DataFrame:
    odds, odds_kind = _odds(frame)
    out = pd.DataFrame(
        {
            "league": league.id,
            "season": season,
            "date": _parse_kickoff(frame["Date"], frame.get("Time")),
            "home": frame[home_col].astype(str).str.strip(),
            "away": frame[away_col].astype(str).str.strip(),
            "home_goals": _numeric(frame, goal_cols[0]),
            "away_goals": _numeric(frame, goal_cols[1]),
            "home_shots": _numeric(frame, "HS"),
            "away_shots": _numeric(frame, "AS"),
            "home_sot": _numeric(frame, "HST"),
            "away_sot": _numeric(frame, "AST"),
            "odds_kind": odds_kind,
        }
    )
    out = pd.concat([out, odds], axis=1)
    out = out.dropna(subset=["date"])
    out = out[(out["home"] != "nan") & (out["away"] != "nan")]
    return out[COLUMNS]


def normalize_main_season(raw: pd.DataFrame, league: League, season: int) -> pd.DataFrame:
    return _build(raw, league, season, "HomeTeam", "AwayTeam", ("FTHG", "FTAG"))


def normalize_extra(raw: pd.DataFrame, league: League) -> pd.DataFrame:
    seasons = pd.to_numeric(raw["Season"], errors="coerce")
    raw = raw[seasons.notna() & (seasons >= league.first_season)]
    return _build(raw, league, seasons[raw.index].astype(int), "Home", "Away", ("HG", "AG"))


def normalize_main_fixtures(raw: pd.DataFrame, league: League, season: int) -> pd.DataFrame:
    rows = raw[raw["Div"].astype(str).str.strip() == league.code]
    rows = rows.drop(columns=[c for c in ("FTHG", "FTAG") if c in rows.columns])
    return _build(rows, league, season, "HomeTeam", "AwayTeam", ("FTHG", "FTAG"))


def normalize_extra_fixtures(raw: pd.DataFrame, league: League, season: int) -> pd.DataFrame:
    country = league.country
    rows = raw[(raw["Country"].astype(str).str.strip() == country) & (raw["League"] == "Serie A")]
    return _build(rows, league, season, "Home", "Away", ("HG", "AG"))


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate, drop duplicate listings of the same match and sort chronologically.

    A scheduled match can appear both in the season file and in the fixtures feed; the copy with
    a result (or, failing that, the most recent listing) wins.
    """
    matches = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    matches["_played"] = matches["home_goals"].notna()
    matches["_day"] = matches["date"].dt.floor("D")
    matches = matches.sort_values(["_played"], kind="stable")
    matches = matches.drop_duplicates(["league", "home", "away", "season"], keep="last")
    matches = matches.drop(columns=["_played", "_day"])
    matches = matches.sort_values(["date", "league", "home"], kind="stable").reset_index(drop=True)
    return matches
