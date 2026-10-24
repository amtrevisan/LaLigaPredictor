"""Club imagery from TheSportsDB: crests, fan-art and pictures of recent meetings.

Everything here is best-effort enrichment. A failed lookup leaves a fixture without images and
never fails the pipeline. Results are cached in two committed JSON files so the free API tier
(roughly 30 requests a minute) is only hit for new teams and new pairings:

* ``data/team_media.json``  ``{"<league>/<feed name>": TeamMedia}``
* ``data/meetings.json``    ``{"<home id>-<away id>-<season>": {"fetched": iso, "events": [...]}}``

Images are hot-linked from TheSportsDB's CDN rather than copied into the repository.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import requests

from predictor.config import DATA_DIR, League
from predictor.data.teams import team_info

log = logging.getLogger(__name__)

API = "https://www.thesportsdb.com/api/v1/json/123"
REQUEST_INTERVAL = 2.1
# A fixture's own event often gets its artwork close to kick-off; re-check after this long.
EVENT_REFRESH = timedelta(hours=20)

TEAM_MEDIA_PATH = DATA_DIR / "team_media.json"
MEETINGS_PATH = DATA_DIR / "meetings.json"


@dataclass
class TeamMedia:
    id: str
    name: str
    badge: str | None
    fanart: list[str] = field(default_factory=list)
    stadium: str | None = None


@dataclass
class Meeting:
    date: str
    home: str
    away: str
    home_score: int | None
    away_score: int | None
    competition: str | None
    thumb: str | None
    video: str | None


class SportsDb:
    def __init__(self) -> None:
        self._session = requests.Session()
        self._last = 0.0

    def get(self, endpoint: str, **params: str) -> dict:
        wait = REQUEST_INTERVAL - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        response = self._session.get(f"{API}/{endpoint}", params=params, timeout=20)
        if response.status_code == 429:
            time.sleep(30)
            return self.get(endpoint, **params)
        response.raise_for_status()
        return response.json() or {}


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def _pick_team(candidates: list[dict], league: League) -> dict | None:
    soccer = [t for t in candidates if t.get("strSport") == "Soccer"]
    for team in soccer:
        leagues = {team.get(f"strLeague{suffix}") for suffix in ("", "2", "3", "4", "5", "6", "7")}
        if league.sportsdb_league in leagues:
            return team
    same_country = [t for t in soccer if t.get("strCountry") == league.country]
    return (same_country or soccer or [None])[0]


def resolve_teams(client: SportsDb, league: League, feed_names: list[str]) -> dict[str, dict]:
    """Ensure every team in ``feed_names`` has a registry entry; returns the full registry."""
    registry = _load(TEAM_MEDIA_PATH)
    changed = False
    for feed_name in feed_names:
        key = f"{league.id}/{feed_name}"
        if key in registry:
            continue
        search = team_info(feed_name).search
        try:
            found = _pick_team(client.get("searchteams.php", t=search).get("teams") or [], league)
        except requests.RequestException as exc:
            log.warning("team lookup failed for %s: %s", search, exc)
            continue
        if found is None:
            log.warning("no TheSportsDB match for %s (%s)", feed_name, search)
            continue
        fanart = [
            found[k]
            for k in ("strFanart1", "strFanart2", "strFanart3", "strFanart4")
            if found.get(k)
        ]
        registry[key] = asdict(
            TeamMedia(
                id=found["idTeam"],
                name=found["strTeam"],
                badge=found.get("strBadge"),
                fanart=fanart,
                stadium=found.get("strStadium"),
            )
        )
        changed = True
        log.info("resolved %s -> %s (%s)", feed_name, found["strTeam"], found["idTeam"])
    if changed:
        _save(TEAM_MEDIA_PATH, registry)
    return registry


def sportsdb_season(league: League, season: int) -> str:
    return str(season) if league.source == "extra" else f"{season}-{season + 1}"


def _meeting(event: dict) -> Meeting:
    def score(value: str | None) -> int | None:
        return int(value) if value not in (None, "") else None

    return Meeting(
        date=event.get("dateEvent") or "",
        home=event.get("strHomeTeam") or "",
        away=event.get("strAwayTeam") or "",
        home_score=score(event.get("intHomeScore")),
        away_score=score(event.get("intAwayScore")),
        competition=event.get("strLeague"),
        thumb=event.get("strThumb"),
        video=event.get("strVideo") or None,
    )


def meetings_for(
    client: SportsDb,
    cache: dict,
    league: League,
    home: dict,
    away: dict,
    season: int,
    now: datetime,
) -> list[dict]:
    """Events between two clubs in the given and previous season (both venues), newest first."""
    events: list[dict] = []
    lookups = [
        (home, away, season),
        (away, home, season),
        (home, away, season - 1),
        (away, home, season - 1),
    ]
    for first, second, year in lookups:
        key = f"{first['id']}-{second['id']}-{year}"
        entry = cache.get(key)
        is_current = year == season
        stale = entry is None or (
            is_current and now - datetime.fromisoformat(entry["fetched"]) > EVENT_REFRESH
        )
        if stale:
            name = f"{first['name']}_vs_{second['name']}".replace(" ", "_")
            try:
                found = client.get("searchevents.php", e=name, s=sportsdb_season(league, year))
            except requests.RequestException as exc:
                log.warning("event lookup failed for %s: %s", name, exc)
                continue
            entry = {
                "fetched": now.isoformat(),
                "events": [asdict(_meeting(e)) for e in found.get("event") or []],
            }
            cache[key] = entry
        events.extend(entry["events"])
    unique = {(e["date"], e["home"], e["away"]): e for e in events}
    return sorted(unique.values(), key=lambda e: e["date"], reverse=True)


def load_meetings() -> dict:
    return _load(MEETINGS_PATH)


def save_meetings(cache: dict) -> None:
    _save(MEETINGS_PATH, cache)


def sized(url: str | None, size: str) -> str | None:
    """TheSportsDB serves resized variants at ``<url>/<tiny|small|medium>``."""
    return f"{url}/{size}" if url else None
