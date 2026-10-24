"""Time-weighted Dixon-Coles goals model.

Goals are Poisson with ``log λ_home = home + attack[h] + defence[a]`` and
``log λ_away = attack[a] + defence[h]`` (``defence`` is defensive *weakness*: higher concedes more).
The Dixon-Coles ``τ`` term corrects the independent-Poisson probabilities of 0-0, 1-0, 0-1 and 1-1.

Fitting is two-stage for speed: the Poisson parameters by L-BFGS with an analytic gradient, then
``ρ`` by a bounded scalar search with the rates held fixed. A small ridge penalty pins the
attack/defence identifiability and shrinks clubs with little data toward average.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.stats import poisson

from predictor.config import MAX_GOALS

RIDGE = 0.01
# Prior for a club absent from the training window (newly promoted): a below-average attack and
# above-average concession, roughly the bottom quartile of fitted top-flight clubs.
NEWCOMER_ATTACK = -0.20
NEWCOMER_DEFENCE = 0.20


@dataclass
class DixonColes:
    teams: dict[str, int]
    attack: np.ndarray
    defence: np.ndarray
    home: float
    rho: float

    @classmethod
    def fit(
        cls,
        home: np.ndarray,
        away: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weights: np.ndarray,
    ) -> DixonColes:
        names = np.unique(np.concatenate([home, away]))
        teams = {name: i for i, name in enumerate(names)}
        h = np.array([teams[t] for t in home])
        a = np.array([teams[t] for t in away])
        n = len(names)
        hg = home_goals.astype(float)
        ag = away_goals.astype(float)
        w = weights.astype(float) / weights.sum() * len(weights)

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            att, dfn, home_adv = theta[:n], theta[n : 2 * n], theta[-1]
            log_lam = home_adv + att[h] + dfn[a]
            log_mu = att[a] + dfn[h]
            lam, mu = np.exp(log_lam), np.exp(log_mu)
            nll = np.sum(w * (lam - hg * log_lam + mu - ag * log_mu))
            nll += RIDGE * (att @ att + dfn @ dfn)
            g_lam = w * (lam - hg)
            g_mu = w * (mu - ag)
            grad = np.empty_like(theta)
            grad[:n] = np.bincount(h, g_lam, n) + np.bincount(a, g_mu, n) + 2 * RIDGE * att
            grad[n : 2 * n] = np.bincount(a, g_lam, n) + np.bincount(h, g_mu, n) + 2 * RIDGE * dfn
            grad[-1] = g_lam.sum()
            return nll, grad

        theta0 = np.zeros(2 * n + 1)
        theta0[-1] = 0.25
        theta = optimize.minimize(objective, theta0, jac=True, method="L-BFGS-B").x
        att, dfn, home_adv = theta[:n], theta[n : 2 * n], theta[-1]
        lam = np.exp(home_adv + att[h] + dfn[a])
        mu = np.exp(att[a] + dfn[h])

        def rho_nll(rho: float) -> float:
            tau = _tau(hg, ag, lam, mu, rho)
            return -np.sum(w * np.log(np.clip(tau, 1e-10, None)))

        rho = optimize.minimize_scalar(rho_nll, bounds=(-0.25, 0.25), method="bounded").x
        return cls(teams=teams, attack=att, defence=dfn, home=float(home_adv), rho=float(rho))

    @classmethod
    def empty(cls) -> DixonColes:
        """A model with no fitted clubs: average league home advantage, no low-score correction."""
        return cls(teams={}, attack=np.zeros(0), defence=np.zeros(0), home=0.25, rho=0.0)

    def _params(self, team: str) -> tuple[float, float]:
        index = self.teams.get(team)
        if index is None:
            return NEWCOMER_ATTACK, NEWCOMER_DEFENCE
        return float(self.attack[index]), float(self.defence[index])

    def rates(self, home: str, away: str) -> tuple[float, float]:
        att_h, def_h = self._params(home)
        att_a, def_a = self._params(away)
        return math.exp(self.home + att_h + def_a), math.exp(att_a + def_h)

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """``P[i, j]`` = probability the home side scores ``i`` and the away side ``j``."""
        lam, mu = self.rates(home, away)
        return score_matrix(lam, mu, self.rho)

    def predict(self, home: np.ndarray, away: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Outcome probabilities ``(n, 3)`` and expected goals ``(n, 2)``."""
        probs = np.empty((len(home), 3))
        xg = np.empty((len(home), 2))
        for i, (h, a) in enumerate(zip(home, away, strict=True)):
            lam, mu = self.rates(h, a)
            probs[i] = outcome_probabilities(score_matrix(lam, mu, self.rho))
            xg[i] = lam, mu
        return probs, xg


def _tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    tau = np.ones_like(lam)
    tau = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1 + lam * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1 + mu * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
    return tau


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    matrix[0, 0] *= 1 - lam * mu * rho
    matrix[0, 1] *= 1 + lam * rho
    matrix[1, 0] *= 1 + mu * rho
    matrix[1, 1] *= 1 - rho
    # With rho > 0 and large rates the 0-0 correction can dip below zero; probabilities can't.
    np.clip(matrix, 0.0, None, out=matrix)
    return matrix / matrix.sum()


def outcome_probabilities(matrix: np.ndarray) -> np.ndarray:
    return np.array([np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()])
