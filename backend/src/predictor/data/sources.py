"""Download and cache raw CSV feeds from football-data.co.uk.

Completed seasons never change, so they are cached indefinitely. The in-progress season and the
fixture lists are refreshed whenever the cached copy is older than ``max_age``.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

from predictor.config import CACHE_DIR, League

log = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk"
MAIN_FIXTURES_URL = f"{BASE_URL}/fixtures.csv"
EXTRA_FIXTURES_URL = f"{BASE_URL}/new_league_fixtures.csv"

_session = requests.Session()
_session.headers["User-Agent"] = "predictor/2.0 (+https://github.com/amtrevisan/LaLigaPredictor)"


def season_code(season: int) -> str:
    """``2025`` -> ``"2526"``, the path segment used by the main-league archive."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def _download(url: str, attempts: int = 4) -> bytes:
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            response = _session.get(url, timeout=30)
            if response.status_code == 404:
                raise FileNotFoundError(url)
            response.raise_for_status()
            return response.content
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if attempt == attempts:
                raise
            log.warning("GET %s failed (%s), retrying in %.0fs", url, exc, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _cached(url: str, path: Path, max_age: timedelta | None) -> bytes:
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if max_age is None or age < max_age.total_seconds():
            return path.read_bytes()
    try:
        content = _download(url)
    except (requests.RequestException, OSError) as exc:
        if isinstance(exc, FileNotFoundError) or not path.exists():
            raise
        # A stale copy beats no copy: the run publishes slightly older data instead of dropping
        # a whole season.
        log.warning("refresh of %s failed (%s); using cached copy", url, exc)
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)
    return content


def _read_csv(content: bytes) -> pd.DataFrame:
    # Files carry a UTF-8 BOM, occasional latin-1 bytes and trailing empty columns/rows.
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    frame = pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    return frame.dropna(how="all")


def fetch_main_season(
    league: League, season: int, *, current: bool, max_age: timedelta
) -> pd.DataFrame | None:
    """One season of a main league, or ``None`` when the archive has no file for it."""
    code = season_code(season)
    url = f"{BASE_URL}/mmz4281/{code}/{league.code}.csv"
    path = CACHE_DIR / "raw" / league.code / f"{code}.csv"
    try:
        content = _cached(url, path, max_age if current else None)
    except FileNotFoundError:
        return None
    return _read_csv(content)


def fetch_extra_league(league: League, *, max_age: timedelta) -> pd.DataFrame:
    """Every season of an extra league in a single frame."""
    url = f"{BASE_URL}/new/{league.code}.csv"
    path = CACHE_DIR / "raw" / league.code / "all.csv"
    return _read_csv(_cached(url, path, max_age))


def fetch_fixtures(source: str, *, max_age: timedelta) -> pd.DataFrame:
    """Upcoming fixtures for every main (``source="main"``) or extra league."""
    url = MAIN_FIXTURES_URL if source == "main" else EXTRA_FIXTURES_URL
    path = CACHE_DIR / "raw" / f"fixtures_{source}.csv"
    return _read_csv(_cached(url, path, max_age))
