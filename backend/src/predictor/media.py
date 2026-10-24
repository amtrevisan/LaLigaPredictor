"""Club imagery from TheSportsDB: crests and club photographs.

Everything here is best-effort enrichment. A failed lookup leaves a fixture without images and
never fails the pipeline. Results are cached in ``data/team_media.json``
(``{"<league>/<feed name>": TeamMedia}``) so the free API tier (roughly 30 requests a minute) is
only hit for new teams.

Images are hot-linked from TheSportsDB's CDN rather than copied into the repository.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image

from predictor.config import DATA_DIR, League
from predictor.data.teams import team_info

log = logging.getLogger(__name__)

API = "https://www.thesportsdb.com/api/v1/json/123"
REQUEST_INTERVAL = 2.1
# Wall-clock budget for all lookups in one run; the scheduled job must stay well inside its
# timeout even when the API is slow rather than refusing.
RUN_BUDGET_SECONDS = 15 * 60

TEAM_MEDIA_PATH = DATA_DIR / "team_media.json"


@dataclass
class TeamMedia:
    id: str
    name: str
    badge: str | None
    fanart: list[str] = field(default_factory=list)
    stadium: str | None = None
    # Subset of ``fanart`` verified as photographs; ``None`` until classified.
    photos: list[str] | None = None


# --- Photograph detection -----------------------------------------------------------------
#
# TheSportsDB fan-art mixes real photographs (stadiums, players, supporters) with designed
# wallpapers (crests on flags, branded renders). Nothing in the API says which is which, so
# each image is classified once and the result cached in team_media.json:
#
# * flat graphics put most of their pixels into a handful of colours; photographs don't;
# * the designed, league-branded wallpapers were all uploaded from 2024 onward, and the
#   upload time is the Unix timestamp embedded in every file name.
#
# Both checks have to pass. Misses err towards showing fewer images, never a crest as a photo.
FLAT_SHARE_LIMIT = 0.4
WALLPAPER_ERA = datetime(2024, 1, 1).timestamp()
_UPLOADED = re.compile(r"(\d{10})\.(?:jpe?g|png)$")


def _uploaded_at(url: str) -> float | None:
    match = _UPLOADED.search(url)
    return float(match.group(1)) if match else None


def _flat_share(raw: bytes) -> float:
    """Share of pixels held by the three most common of 32 quantised colours."""
    image = Image.open(io.BytesIO(raw)).convert("RGB").resize((160, 90))
    counts = sorted(image.quantize(colors=32).histogram(), reverse=True)
    return sum(counts[:3]) / (160 * 90)


def is_photograph(url: str, session: requests.Session) -> bool:
    uploaded = _uploaded_at(url)
    if uploaded is None or uploaded >= WALLPAPER_ERA:
        return False
    try:
        response = session.get(f"{url}/small", timeout=20)
        response.raise_for_status()
        return _flat_share(response.content) < FLAT_SHARE_LIMIT
    except (requests.RequestException, OSError) as exc:
        log.debug("could not classify %s: %s", url, exc)
        return False


def classify_photos(registry: dict) -> dict:
    """Fill in ``photos`` for clubs not classified yet. Images come from the CDN, not the
    rate-limited API, so this is cheap after the first run."""
    pending = [entry for entry in registry.values() if entry.get("photos") is None]
    if not pending:
        return registry
    session = requests.Session()
    for entry in pending:
        entry["photos"] = [url for url in entry.get("fanart") or [] if is_photograph(url, session)]
    _save(TEAM_MEDIA_PATH, registry)
    log.info("classified fan-art for %d clubs", len(pending))
    return registry


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


def sized(url: str | None, size: str) -> str | None:
    """TheSportsDB serves resized variants at ``<url>/<tiny|small|medium>``."""
    return f"{url}/{size}" if url else None
