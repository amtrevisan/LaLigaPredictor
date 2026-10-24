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
import unicodedata
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
# A lookup that found nothing is retried after this long, in case the free tier dropped it.
EMPTY_REFRESH = timedelta(days=7)
# Wall-clock budget for all lookups in one run; the scheduled job must stay well inside its
# timeout even when the API is slow rather than refusing.
RUN_BUDGET_SECONDS = 15 * 60

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


class MediaUnavailable(requests.RequestException):
    """TheSportsDB keeps refusing requests; no further lookups are attempted this run."""


class SportsDb:
    """Throttled client for the shared free API key.

    The free key is shared by every user of TheSportsDB, so 429s are routine. Each request gets
    a bounded exponential backoff. Any failed request (rate limit, server error, timeout) is a
    strike; after ``MAX_STRIKES`` in a row, or once the run's time budget is spent, the client
    gives up and every later call fails fast. Cached media is still exported, so a bad run
    degrades to slightly stale images, never to a hang.
    """

    BACKOFF = (15.0, 45.0, 90.0)
    MAX_STRIKES = 3

    def __init__(self, budget_seconds: float = RUN_BUDGET_SECONDS) -> None:
        self._session = requests.Session()
        self._last = 0.0
        self._strikes = 0
        self._deadline = time.monotonic() + budget_seconds

    @property
    def available(self) -> bool:
        return self._strikes < self.MAX_STRIKES and time.monotonic() < self._deadline

    def _strike(self, reason: str) -> MediaUnavailable:
        self._strikes += 1
        if not self.available:
            log.warning("TheSportsDB unavailable (%s); skipping remaining media lookups", reason)
        return MediaUnavailable(reason)

    def get(self, endpoint: str, **params: str) -> dict:
        if not self.available:
            raise MediaUnavailable("TheSportsDB lookups disabled for this run")
        for delay in (*self.BACKOFF, None):
            wait = REQUEST_INTERVAL - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                response = self._session.get(f"{API}/{endpoint}", params=params, timeout=20)
                if response.status_code != 429:
                    response.raise_for_status()
                    self._strikes = 0
                    return response.json() or {}
            except (requests.RequestException, ValueError) as exc:
                raise self._strike(f"{endpoint}: {exc}") from exc
            if delay is None or time.monotonic() + delay > self._deadline:
                break
            log.info("TheSportsDB rate limit, retrying in %.0fs", delay)
            time.sleep(delay)
        raise self._strike(f"{endpoint} still rate limited after retries")


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


# The free tier returns a single search hit, which is often a women's, youth or other-sport
# namesake ("Nottingham Forest" is also a netball club). Anything else is rejected rather than
# shown with the wrong crest.
_NOT_FIRST_TEAM = (
    "women",
    "femen",
    "femin",
    "frauen",
    "ladies",
    "u19",
    "u21",
    "u23",
    " ii",
    " b",
    "fabril",
    "castilla",
    "atletic",
    "promesas",
    "reserves",
)


def _fold(text: str) -> str:
    """Lower-case and strip accents, so "Coruña" matches "Coruna"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _names_match(search: str, team: dict) -> bool:
    """Every significant word of the search term appears in the club's name or alternates."""
    haystack = _fold(f"{team.get('strTeam', '')} {team.get('strTeamAlternate') or ''}")
    words = [w for w in _fold(search).replace("-", " ").split() if len(w) >= 4]
    return all(word in haystack for word in words)


def _is_first_team(team: dict, league: League) -> bool:
    name = f" {_fold(team.get('strTeam', ''))}"
    return (
        team.get("strSport") == "Soccer"
        and team.get("strGender") != "Female"
        and team.get("strCountry") == league.country
        and not any(marker in name for marker in _NOT_FIRST_TEAM)
    )


def _pick_team(candidates: list[dict], league: League, search: str = "") -> dict | None:
    valid = [
        t
        for t in candidates
        if _is_first_team(t, league) and (not search or _names_match(search, t))
    ]
    for team in valid:
        leagues = {team.get(f"strLeague{suffix}") for suffix in ("", "2", "3", "4", "5", "6", "7")}
        if league.sportsdb_league in leagues:
            return team
    return valid[0] if valid else None


def _lookup(client: SportsDb, feed_name: str, league: League) -> dict | None:
    info = team_info(feed_name)
    if info.sportsdb_id:
        teams = client.get("lookupteam.php", id=info.sportsdb_id).get("teams") or []
        return teams[0] if teams else None
    teams = client.get("searchteams.php", t=info.search).get("teams") or []
    return _pick_team(teams, league, info.search)


def resolve_teams(client: SportsDb, league: League, feed_names: list[str]) -> dict[str, dict]:
    """Ensure every team in ``feed_names`` has a registry entry; returns the full registry."""
    registry = _load(TEAM_MEDIA_PATH)
    changed = False
    for feed_name in feed_names:
        key = f"{league.id}/{feed_name}"
        if key in registry:
            continue
        try:
            found = _lookup(client, feed_name, league)
        except MediaUnavailable:
            break
        except requests.RequestException as exc:
            log.warning("team lookup failed for %s: %s", feed_name, exc)
            continue
        if found is None:
            log.warning("no TheSportsDB first team found for %s; add an id in teams.py", feed_name)
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
        age = now - datetime.fromisoformat(entry["fetched"]) if entry else None
        stale = (
            entry is None
            or (is_current and age > EVENT_REFRESH)
            or (not entry["events"] and age > EMPTY_REFRESH)
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
