"""Static configuration: covered leagues, filesystem layout and model constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
CACHE_DIR = BACKEND_ROOT / ".cache"
DATA_DIR = BACKEND_ROOT / "data"
DEFAULT_EXPORT_DIR = REPO_ROOT / "frontend" / "public" / "data"

SourceKind = Literal["main", "extra"]


@dataclass(frozen=True)
class League:
    """A competition covered by the pipeline.

    ``source`` selects the football-data.co.uk feed layout: ``main`` leagues publish one CSV
    per season (European calendar, ``2526``-style codes), ``extra`` leagues publish a single
    file with every season (calendar-year seasons, fewer statistics).
    """

    id: str
    name: str
    country: str
    code: str
    source: SourceKind
    first_season: int
    locale: str
    sportsdb_league: str

    def season_label(self, season: int) -> str:
        """Human label for a season keyed by its starting year."""
        if self.source == "extra":
            return str(season)
        return f"{season}/{(season + 1) % 100:02d}"


LEAGUES: tuple[League, ...] = (
    League(
        "premier-league",
        "Premier League",
        "England",
        "E0",
        "main",
        2005,
        "en",
        "English Premier League",
    ),
    League("la-liga", "La Liga", "Spain", "SP1", "main", 2005, "es", "Spanish La Liga"),
    League("serie-a", "Serie A", "Italy", "I1", "main", 2005, "it", "Italian Serie A"),
    League("bundesliga", "Bundesliga", "Germany", "D1", "main", 2005, "de", "German Bundesliga"),
    League("ligue-1", "Ligue 1", "France", "F1", "main", 2005, "fr", "French Ligue 1"),
    League(
        "brasileirao",
        "Brasileirão Série A",
        "Brazil",
        "BRA",
        "extra",
        2012,
        "pt",
        "Brazilian Serie A",
    ),
)

LEAGUES_BY_ID = {league.id: league for league in LEAGUES}

# Outcome encoding shared by every model and metric: index 0 home win, 1 draw, 2 away win.
OUTCOMES = ("H", "D", "A")

# Number of completed seasons, per league, evaluated by the walk-forward backtest.
BACKTEST_SEASONS = 3

# Exponential time decay for the Dixon-Coles likelihood, per day (half-life of roughly a year).
DIXON_COLES_XI = 0.0019
# Training window for Dixon-Coles refits, in days.
DIXON_COLES_WINDOW_DAYS = 3 * 365
# Maximum goals per side in the scoreline grid.
MAX_GOALS = 10
