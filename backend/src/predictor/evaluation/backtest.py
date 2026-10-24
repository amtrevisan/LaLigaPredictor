"""Walk-forward prediction: every forecast uses only matches completed before it.

Two cadences:

* Dixon-Coles is refitted per league at the start of every week, on the trailing
  ``DIXON_COLES_WINDOW_DAYS`` of results.
* The classifier is retrained, pooled across leagues, at every 1 January and 1 July, on all
  completed matches before that boundary.

The same functions produce the published track record for the current season and the historical
backtest, so the numbers on the site are exactly what the model would have said at the time.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from predictor.config import DIXON_COLES_WINDOW_DAYS, DIXON_COLES_XI
from predictor.models.classifier import OutcomeClassifier
from predictor.models.dixon_coles import DixonColes

log = logging.getLogger(__name__)

# Seasons per league ignored for classifier training while Elo and form states warm up.
BURN_IN_SEASONS = 2


def outcomes(matches: pd.DataFrame) -> np.ndarray:
    """Integer outcome per row (0 home, 1 draw, 2 away); meaningless for unplayed rows."""
    return np.select(
        [
            matches["home_goals"] > matches["away_goals"],
            matches["home_goals"] == matches["away_goals"],
        ],
        [0, 1],
        2,
    )


def fit_dixon_coles(history: pd.DataFrame, as_of: pd.Timestamp) -> DixonColes:
    window = history[
        history["home_goals"].notna()
        & (history["date"] < as_of)
        & (history["date"] >= as_of - pd.Timedelta(days=DIXON_COLES_WINDOW_DAYS))
    ]
    age_days = (as_of - window["date"]).dt.total_seconds().to_numpy() / 86400.0
    return DixonColes.fit(
        window["home"].to_numpy(),
        window["away"].to_numpy(),
        window["home_goals"].to_numpy(),
        window["away_goals"].to_numpy(),
        np.exp(-DIXON_COLES_XI * age_days),
    )


def dixon_coles_walk_forward(matches: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Probabilities and expected goals for ``matches[target]`` from weekly league refits."""
    out = pd.DataFrame(
        np.nan, index=matches.index[target], columns=["dc_h", "dc_d", "dc_a", "xg_h", "xg_a"]
    )
    for league_id, league_matches in matches.groupby("league", sort=False):
        rows = league_matches[target.loc[league_matches.index]]
        if rows.empty:
            continue
        week_start = rows["date"].dt.tz_convert(None).dt.to_period("W-SUN").dt.start_time
        for start, block in rows.groupby(week_start.to_numpy()):
            as_of = pd.Timestamp(start).tz_localize("UTC")
            model = fit_dixon_coles(league_matches, as_of)
            probs, xg = model.predict(block["home"].to_numpy(), block["away"].to_numpy())
            out.loc[block.index, ["dc_h", "dc_d", "dc_a"]] = probs
            out.loc[block.index, ["xg_h", "xg_a"]] = xg
        log.debug("dixon-coles walk-forward done for %s", league_id)
    return out


def training_mask(matches: pd.DataFrame, before: pd.Timestamp) -> pd.Series:
    first_season = matches.groupby("league")["season"].transform("min")
    return (
        matches["home_goals"].notna()
        & (matches["date"] < before)
        & (matches["season"] >= first_season + BURN_IN_SEASONS)
    )


def retrain_boundaries(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Every 1 January / 1 July in ``(start - 6 months, end]``, ascending."""
    first = pd.Timestamp(year=start.year - 1, month=7, day=1, tz="UTC")
    boundaries = pd.date_range(first, end, freq="6MS", tz="UTC")
    boundaries = boundaries[(boundaries.month == 1) | (boundaries.month == 7)]
    return [b for b in boundaries if b > start - pd.DateOffset(months=6)]


def classifier_walk_forward(
    matches: pd.DataFrame, features: pd.DataFrame, columns: list[str], target: pd.Series
) -> tuple[pd.DataFrame, OutcomeClassifier]:
    """Probabilities for ``matches[target]`` from semi-annual retrains; returns the last model."""
    y = outcomes(matches)
    out = pd.DataFrame(np.nan, index=matches.index[target], columns=["gb_h", "gb_d", "gb_a"])
    dates = matches.loc[target, "date"]
    boundaries = retrain_boundaries(dates.min(), dates.max())
    model: OutcomeClassifier | None = None
    for i, boundary in enumerate(boundaries):
        until = boundaries[i + 1] if i + 1 < len(boundaries) else None
        block = target & (matches["date"] >= boundary)
        if until is not None:
            block &= matches["date"] < until
        if not block.any():
            continue
        train = training_mask(matches, boundary)
        model = OutcomeClassifier(columns).fit(features[train], y[train.to_numpy()])
        out.loc[block[block].index] = model.predict_proba(features[block])
        log.info("classifier retrained at %s on %d matches", boundary.date(), int(train.sum()))
    assert model is not None, "walk-forward target is empty"
    return out, model
