"""Proper scoring rules for three-way (home / draw / away) probabilistic forecasts.

``probs`` is always ``(n, 3)`` in ``config.OUTCOMES`` order and ``outcomes`` the integer index of
what happened. Lower is better for every loss; accuracy is reported for readability only, since
it ignores how confident a forecast was.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _onehot(outcomes: np.ndarray) -> np.ndarray:
    return np.eye(3)[outcomes]


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    picked = probs[np.arange(len(outcomes)), outcomes]
    return float(-np.mean(np.log(np.clip(picked, EPS, 1.0))))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(np.sum((probs - _onehot(outcomes)) ** 2, axis=1)))


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked probability score: respects the ordering home > draw > away, the standard
    football-forecasting metric (a draw is a nearer miss for a home pick than an away win)."""
    cum_p = np.cumsum(probs, axis=1)[:, :-1]
    cum_o = np.cumsum(_onehot(outcomes), axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / 2.0))


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == outcomes))


def summary(probs: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    return {
        "logLoss": log_loss(probs, outcomes),
        "rps": rps(probs, outcomes),
        "brier": brier(probs, outcomes),
        "accuracy": accuracy(probs, outcomes),
    }


def calibration(probs: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Reliability table over all three outcome probabilities pooled together."""
    p = probs.ravel()
    hit = _onehot(outcomes).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    table = []
    for b in range(bins):
        mask = index == b
        if mask.sum() == 0:
            continue
        table.append(
            {
                "predicted": float(p[mask].mean()),
                "observed": float(hit[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return table


def implied_probabilities(odds: np.ndarray) -> np.ndarray:
    """Bookmaker decimal odds ``(n, 3)`` to probabilities, removing the margin proportionally."""
    raw = 1.0 / odds
    return raw / raw.sum(axis=1, keepdims=True)
