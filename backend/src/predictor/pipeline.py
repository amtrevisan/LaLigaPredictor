"""End-to-end run: ingest, build features, walk-forward evaluation, final predictions, export.

Model
-----
The published probability is a convex blend of two models,
``p = w * classifier + (1 - w) * dixon_coles``, with ``w`` chosen on a validation season that
precedes every evaluated season. Dixon-Coles also supplies expected goals and the scoreline grid,
rescaled so its home/draw/away mass agrees with the blended probabilities.

Seasons, per league
-------------------
``current`` (in progress) is published as the live track record; the ``BACKTEST_SEASONS``
before it form the historical backtest; the one before those is the validation season.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from predictor import media
from predictor.config import BACKTEST_SEASONS, LEAGUES, LEAGUES_BY_ID, OUTCOMES, League
from predictor.data.loader import load_matches
from predictor.data.teams import team_info
from predictor.evaluation import backtest, metrics
from predictor.export import SCHEMA_VERSION, iso, r, triple, write_all
from predictor.features.build import build_features, model_columns
from predictor.models.classifier import HYPERPARAMETERS, OutcomeClassifier
from predictor.models.dixon_coles import DixonColes, outcome_probabilities

log = logging.getLogger(__name__)

MODEL_VERSION = "2.0"
BLEND_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)
TOP_SCORELINES = 6
MEETINGS_PER_FIXTURE = 4

DC_COLS = ["dc_h", "dc_d", "dc_a"]
GB_COLS = ["gb_h", "gb_d", "gb_a"]
ODDS_COLS = ["odds_home", "odds_draw", "odds_away"]


@dataclass(frozen=True)
class RunConfig:
    now: datetime
    export_dir: Path
    fetch_media: bool = True
    leagues: tuple[League, ...] = LEAGUES


# ---------------------------------------------------------------------------------------------
# Modelling helpers
# ---------------------------------------------------------------------------------------------


def choose_blend_weight(gb: np.ndarray, dc: np.ndarray, y: np.ndarray) -> float:
    losses = {float(w): metrics.log_loss(w * gb + (1 - w) * dc, y) for w in BLEND_GRID}
    return min(losses, key=losses.get)


def align_scoreline_grid(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rescale the home-win, draw and away-win regions of a score grid to ``target`` mass."""
    current = outcome_probabilities(matrix)
    scale = np.divide(target, current, out=np.ones(3), where=current > 0)
    i, j = np.indices(matrix.shape)
    region = np.where(i > j, 0, np.where(i == j, 1, 2))
    return matrix * scale[region]


def base_rates(matches: pd.DataFrame, before_season: pd.Series) -> dict[str, np.ndarray]:
    """Per-league home/draw/away frequencies from seasons before the validation season."""
    rates = {}
    history = matches[matches["home_goals"].notna() & (matches["season"] < before_season)]
    for league_id, rows in history.groupby("league"):
        counts = np.bincount(backtest.outcomes(rows), minlength=3).astype(float)
        rates[league_id] = counts / counts.sum()
    return rates


# ---------------------------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------------------------


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _team(league: League, feed_name: str, registry: dict, photos: bool = True) -> dict:
    """Club as exported. Photography is only needed for upcoming fixtures; a full season of
    results would otherwise repeat every club's image list hundreds of times."""
    info = team_info(feed_name)
    entry = registry.get(f"{league.id}/{feed_name}", {})
    fanart = (entry.get("fanart") or []) if photos else []
    return {
        "name": info.name,
        "short": info.short,
        "crest": media.sized(entry.get("badge"), "small"),
        "fanart": [media.sized(url, "medium") for url in fanart],
        "stadium": entry.get("stadium") if photos else None,
    }


def _outcome(home_goals: float, away_goals: float) -> str:
    return "H" if home_goals > away_goals else "D" if home_goals == away_goals else "A"


def _fixture_base(
    league: League, row: pd.Series, feature_row: pd.Series, registry: dict, photos: bool = True
) -> dict:
    probs = row[["p_h", "p_d", "p_a"]].to_numpy(dtype=float)
    odds = row[ODDS_COLS].to_numpy(dtype=float)
    market = metrics.implied_probabilities(odds[None, :])[0] if np.isfinite(odds).all() else None
    return {
        "id": f"{league.id}-{row['date']:%Y%m%d}-{_slug(row['home'])}-{_slug(row['away'])}",
        "round": int(feature_row["round"]),
        "kickoff": iso(row["date"]),
        "home": _team(league, row["home"], registry, photos),
        "away": _team(league, row["away"], registry, photos),
        "probs": triple(probs),
        "pick": OUTCOMES[int(np.argmax(probs))],
        "market": triple(market),
        "xg": triple([row["xg_h"], row["xg_a"]], 2),
        "components": {"classifier": triple(row[GB_COLS]), "goals": triple(row[DC_COLS])},
        "elo": triple([feature_row["home_elo"], feature_row["away_elo"]], 0),
    }


def _meetings(events: list[dict], kickoff: pd.Timestamp) -> tuple[str | None, list[dict]]:
    """Split TheSportsDB events into this fixture's own artwork and earlier meetings."""
    day = kickoff.strftime("%Y-%m-%d")
    poster = next((e["thumb"] for e in events if e["date"] == day and e.get("thumb")), None)
    earlier = [
        {
            "date": e["date"],
            "home": e["home"],
            "away": e["away"],
            "score": [e["home_score"], e["away_score"]] if e["home_score"] is not None else None,
            "competition": e["competition"],
            "image": media.sized(e["thumb"], "medium"),
            "video": e["video"],
        }
        for e in events
        if e["date"] < day and e["home_score"] is not None
    ]
    return media.sized(poster, "medium"), earlier[:MEETINGS_PER_FIXTURE]


def _standings(
    league: League,
    season_rows: pd.DataFrame,
    ratings: dict[str, float],
    dc_model: DixonColes,
    registry: dict,
) -> list[dict]:
    teams = sorted(set(season_rows["home"]) | set(season_rows["away"]))
    played = season_rows[season_rows["home_goals"].notna()]
    table = []
    for name in teams:
        home = played[played["home"] == name]
        away = played[played["away"] == name]
        gf = home["home_goals"].sum() + away["away_goals"].sum()
        ga = home["away_goals"].sum() + away["home_goals"].sum()
        wins = int(
            (home["home_goals"] > home["away_goals"]).sum()
            + (away["away_goals"] > away["home_goals"]).sum()
        )
        draws = int(
            (home["home_goals"] == home["away_goals"]).sum()
            + (away["away_goals"] == away["home_goals"]).sum()
        )
        n = len(home) + len(away)
        attack, defence = dc_model._params(name)
        table.append(
            {
                "team": _team(league, name, registry),
                "played": n,
                "won": wins,
                "drawn": draws,
                "lost": n - wins - draws,
                "goalsFor": int(gf),
                "goalsAgainst": int(ga),
                "points": 3 * wins + draws,
                "elo": r(ratings.get(name), 0),
                "attack": r(np.exp(attack), 3),
                "defence": r(np.exp(defence), 3),
            }
        )
    table.sort(key=lambda t: (-t["points"], -(t["goalsFor"] - t["goalsAgainst"]), -t["goalsFor"]))
    return table


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------


def run(config: RunConfig) -> dict:
    started = time.monotonic()
    now = (
        pd.Timestamp(config.now).tz_convert("UTC")
        if pd.Timestamp(config.now).tzinfo
        else pd.Timestamp(config.now, tz="UTC")
    )

    matches = load_matches(now.to_pydatetime(), config.leagues)
    features, ratings = build_features(matches)
    columns = model_columns(features)
    played = matches["home_goals"].notna()
    current_season = matches.groupby("league")["season"].transform("max")
    validation_season = current_season - BACKTEST_SEASONS - 1
    evaluated = played & (matches["season"] >= validation_season)
    upcoming = ~played

    # Walk-forward predictions for every played match from the validation season onward.
    log.info("walk-forward: %d matches", int(evaluated.sum()))
    walk = matches[evaluated].join(backtest.dixon_coles_walk_forward(matches, evaluated))
    gb_walk, _ = backtest.classifier_walk_forward(matches, features, columns, evaluated)
    walk = walk.join(gb_walk)

    y_walk = backtest.outcomes(walk)
    is_validation = (walk["season"] == validation_season[walk.index]).to_numpy()
    weight = choose_blend_weight(
        walk.loc[is_validation, GB_COLS].to_numpy(),
        walk.loc[is_validation, DC_COLS].to_numpy(),
        y_walk[is_validation],
    )
    log.info("blend weight (classifier share) = %.2f", weight)
    walk[["p_h", "p_d", "p_a"]] = (
        weight * walk[GB_COLS].to_numpy() + (1 - weight) * walk[DC_COLS].to_numpy()
    )

    # Production models for fixtures that have not been played yet.
    final_classifier = OutcomeClassifier(columns).fit(
        features[backtest.training_mask(matches, now)],
        backtest.outcomes(matches[backtest.training_mask(matches, now)]),
    )
    dc_models = {
        league.id: backtest.fit_dixon_coles(matches[matches["league"] == league.id], now)
        for league in config.leagues
    }
    future = matches[upcoming].copy()
    # Created up front so an off-season run (no fixtures at all) still has every column.
    future[DC_COLS + ["xg_h", "xg_a"]] = np.nan
    future[GB_COLS] = (
        final_classifier.predict_proba(features[upcoming]) if len(future) else np.empty((0, 3))
    )
    grids: dict[int, np.ndarray] = {}
    for idx, row in future.iterrows():
        model = dc_models[row["league"]]
        grid = model.score_matrix(row["home"], row["away"])
        future.loc[idx, DC_COLS] = outcome_probabilities(grid)
        future.loc[idx, ["xg_h", "xg_a"]] = model.rates(row["home"], row["away"])
        grids[idx] = grid
    future[["p_h", "p_d", "p_a"]] = (
        weight * future[GB_COLS].to_numpy() + (1 - weight) * future[DC_COLS].to_numpy()
    )

    # Evaluation tables.
    rates = base_rates(matches, validation_season)
    # Score against closing prices only: opening odds are a softer benchmark and would flatter
    # the model. Every evaluated season currently has closing odds for every match.
    has_closing = (walk["odds_kind"] == "closing") & walk[ODDS_COLS].notna().all(axis=1)
    scored = walk[~is_validation & has_closing.to_numpy()]
    y_scored = backtest.outcomes(scored)

    def evaluate(rows: pd.DataFrame, y: np.ndarray) -> dict:
        market = metrics.implied_probabilities(rows[ODDS_COLS].to_numpy())
        base = np.vstack([rates[lg] for lg in rows["league"]])
        return {
            "matches": len(rows),
            "model": metrics.summary(rows[["p_h", "p_d", "p_a"]].to_numpy(), y),
            "market": metrics.summary(market, y),
            "baseline": metrics.summary(base, y),
            "classifier": metrics.summary(rows[GB_COLS].to_numpy(), y),
            "goals": metrics.summary(rows[DC_COLS].to_numpy(), y),
        }

    by_league = []
    for (league_id, season), rows in scored.groupby(["league", "season"], sort=True):
        league = LEAGUES_BY_ID[league_id]
        y = backtest.outcomes(rows)
        by_league.append(
            {
                "league": league_id,
                "season": league.season_label(int(season)),
                "current": bool(season == current_season[rows.index[0]]),
                **evaluate(rows, y),
            }
        )
    completed = (scored["season"] < current_season[scored.index]).to_numpy()
    current_rows = ~completed
    model_probs = scored[["p_h", "p_d", "p_a"]].to_numpy()
    market_probs = metrics.implied_probabilities(scored[ODDS_COLS].to_numpy())

    # Out-of-sample feature importance: train on everything up to a year ago, score the last year.
    cutoff = now - pd.Timedelta(days=365)
    importance_train = backtest.training_mask(matches, cutoff)
    importance_eval = played & (matches["date"] >= cutoff)
    importance_model = OutcomeClassifier(columns).fit(
        features[importance_train], backtest.outcomes(matches[importance_train])
    )
    importance = importance_model.importance(
        features[importance_eval], backtest.outcomes(matches[importance_eval])
    )

    # Media (best effort).
    registry = media._load(media.TEAM_MEDIA_PATH)
    meetings_cache = media.load_meetings()
    client = media.SportsDb() if config.fetch_media else None

    export_dir = config.export_dir
    outputs: dict[Path, dict] = {}
    manifest_leagues = []
    for league in config.leagues:
        season = int(current_season[matches["league"] == league.id].iloc[0])
        season_rows = matches[(matches["league"] == league.id) & (matches["season"] == season)]
        if client is not None:
            teams = sorted(set(season_rows["home"]) | set(season_rows["away"]))
            try:
                registry = media.resolve_teams(client, league, teams)
            except Exception:  # noqa: BLE001 - enrichment must never fail a run
                log.exception("team media resolution failed for %s", league.id)

        fixtures = []
        for idx, row in future[future["league"] == league.id].sort_values("date").iterrows():
            item = _fixture_base(league, row, features.loc[idx], registry)
            grid = align_scoreline_grid(
                grids[idx], row[["p_h", "p_d", "p_a"]].to_numpy(dtype=float)
            )
            flat = np.argsort(grid, axis=None)[::-1][:TOP_SCORELINES]
            item["scorelines"] = [
                {"score": [int(i), int(j)], "p": r(grid[i, j])}
                for i, j in zip(*np.unravel_index(flat, grid.shape), strict=True)
            ]
            item["poster"], item["meetings"] = None, []
            home_media = registry.get(f"{league.id}/{row['home']}")
            away_media = registry.get(f"{league.id}/{row['away']}")
            if home_media and away_media:
                try:
                    events = (
                        media.meetings_for(
                            client,
                            meetings_cache,
                            league,
                            home_media,
                            away_media,
                            season,
                            now.to_pydatetime(),
                        )
                        if client is not None and client.available
                        else _cached_meetings(meetings_cache, home_media, away_media, season)
                    )
                    item["poster"], item["meetings"] = _meetings(events, row["date"])
                except Exception:  # noqa: BLE001
                    log.exception("meeting lookup failed for %s", item["id"])
            fixtures.append(item)
        if client is not None:
            # Persist per league so an interrupted run keeps the lookups it already paid for.
            media.save_meetings(meetings_cache)

        results = []
        league_walk = walk[(walk["league"] == league.id) & (walk["season"] == season)]
        for idx, row in league_walk.sort_values("date", ascending=False).iterrows():
            item = _fixture_base(league, row, features.loc[idx], registry, photos=False)
            outcome = _outcome(row["home_goals"], row["away_goals"])
            item["result"] = {
                "score": [int(row["home_goals"]), int(row["away_goals"])],
                "outcome": outcome,
            }
            item["correct"] = item["pick"] == outcome
            results.append(item)

        record_rows = scored[(scored["league"] == league.id) & (scored["season"] == season)]
        record = None
        if len(record_rows):
            summary = evaluate(record_rows, backtest.outcomes(record_rows))
            record = {
                "matches": summary["matches"],
                "model": summary["model"],
                "market": summary["market"],
                "baseline": summary["baseline"],
            }

        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "generatedAt": iso(pd.Timestamp.now(tz="UTC")),
            "league": {"id": league.id, "name": league.name, "country": league.country},
            "season": league.season_label(season),
            "record": record,
            "upcoming": fixtures,
            "results": results,
            "standings": _standings(
                league, season_rows, ratings.get(league.id, {}), dc_models[league.id], registry
            ),
        }
        outputs[export_dir / "leagues" / f"{league.id}.json"] = payload
        next_kickoff = future.loc[future["league"] == league.id, "date"].min()
        manifest_leagues.append(
            {
                "id": league.id,
                "name": league.name,
                "country": league.country,
                "locale": league.locale,
                "season": league.season_label(season),
                "upcoming": len(fixtures),
                "nextKickoff": iso(next_kickoff) if pd.notna(next_kickoff) else None,
                "nextRound": fixtures[0]["round"] if fixtures else None,
                "record": record,
            }
        )

    model_payload = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": iso(pd.Timestamp.now(tz="UTC")),
        "version": MODEL_VERSION,
        "commit": _git_commit(),
        "runtimeSeconds": round(time.monotonic() - started, 1),
        "blend": {"classifierWeight": weight, "goalsWeight": round(1 - weight, 2)},
        "validation": {"matches": int(is_validation.sum()), "seasonsBeforeBacktest": 1},
        "classifier": {
            "algorithm": "HistGradientBoostingClassifier",
            "hyperparameters": HYPERPARAMETERS,
            "features": columns,
            "trainingMatches": int(backtest.training_mask(matches, now).sum()),
        },
        "data": {
            "matches": int(played.sum()),
            "leagues": len(config.leagues),
            "firstMatch": iso(matches["date"].min()),
            "lastResult": iso(matches.loc[played, "date"].max()),
        },
        "backtest": {
            "seasons": BACKTEST_SEASONS,
            "overall": evaluate(scored[completed], y_scored[completed]),
            "currentSeason": evaluate(scored[current_rows], y_scored[current_rows])
            if current_rows.any()
            else None,
            "byLeague": by_league,
            "calibration": {
                "model": metrics.calibration(model_probs[completed], y_scored[completed]),
                "market": metrics.calibration(market_probs[completed], y_scored[completed]),
            },
        },
        "importance": importance[:20],
    }
    outputs[export_dir / "model.json"] = model_payload

    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": iso(pd.Timestamp.now(tz="UTC")),
        "dataAsOf": iso(matches.loc[played, "date"].max()),
        "leagues": manifest_leagues,
    }
    outputs[export_dir / "manifest.json"] = manifest
    changed = [str(path.relative_to(export_dir)) for path in write_all(outputs)]

    log.info(
        "export complete in %.1fs; changed: %s",
        time.monotonic() - started,
        ", ".join(changed) or "nothing",
    )
    return {
        "changed": changed,
        "blendWeight": weight,
        "overall": model_payload["backtest"]["overall"],
    }


def _cached_meetings(cache: dict, home: dict, away: dict, season: int) -> list[dict]:
    events: list[dict] = []
    for first, second in ((home, away), (away, home)):
        for year in (season, season - 1):
            events.extend(cache.get(f"{first['id']}-{second['id']}-{year}", {}).get("events", []))
    unique = {(e["date"], e["home"], e["away"]): e for e in events}
    return sorted(unique.values(), key=lambda e: e["date"], reverse=True)


def default_now() -> datetime:
    return datetime.now(UTC)
