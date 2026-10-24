from __future__ import annotations

import numpy as np
import pandas as pd


def make_matches(
    n_seasons: int = 3, teams: int = 6, seed: int = 0, league: str = "test"
) -> pd.DataFrame:
    """A synthetic double round-robin league with known team strengths.

    Team ``T0`` is strongest, ``T{teams-1}`` weakest; goals are Poisson so every model in the
    package has a recoverable signal.
    """
    rng = np.random.default_rng(seed)
    names = [f"T{i}" for i in range(teams)]
    strength = np.linspace(0.4, -0.4, teams)
    rows = []
    for season in range(2015, 2015 + n_seasons):
        date = pd.Timestamp(f"{season}-08-01", tz="UTC")
        for home in range(teams):
            for away in range(teams):
                if home == away:
                    continue
                lam = np.exp(0.25 + strength[home] - strength[away])
                mu = np.exp(strength[away] - strength[home])
                rows.append(
                    {
                        "league": league,
                        "season": season,
                        "date": date,
                        "home": names[home],
                        "away": names[away],
                        "home_goals": float(rng.poisson(lam)),
                        "away_goals": float(rng.poisson(mu)),
                        "home_shots": float(rng.poisson(12)),
                        "away_shots": float(rng.poisson(10)),
                        "home_sot": float(rng.poisson(4)),
                        "away_sot": float(rng.poisson(3)),
                        "odds_home": 2.2,
                        "odds_draw": 3.3,
                        "odds_away": 3.4,
                        "odds_kind": "closing",
                    }
                )
                date += pd.Timedelta(days=3)
    return pd.DataFrame(rows)
