# How the model works

This document covers the modelling and evaluation choices behind the predictions. The code lives
in [`backend/src/predictor`](../backend/src/predictor); live numbers are in
[`frontend/public/data/model.json`](../frontend/public/data/model.json) and on the site's Model page.

## Problem

For every upcoming league match, estimate the probability of a home win, a draw and an away win,
plus expected goals and the most likely scorelines. Forecasts are judged as probabilities, not
picks: the main metrics are **log loss** and the **ranked probability score** (RPS).

Bookmaker odds are deliberately **not** a model input. They encode almost everything public about
a match, so a model fed odds mostly learns to repeat the market. Keeping them out means the
comparison with the market is the honest measure of what the football data alone can do.

## Data

[football-data.co.uk](https://www.football-data.co.uk) publishes free CSVs per league and season:
results, shots, shots on target and odds from many bookmakers, including closing prices.

| League | Seasons | Shot statistics |
| --- | --- | --- |
| Premier League, La Liga, Serie A, Bundesliga, Ligue 1 | 2005/06 onward | yes (no shots on target in 2005/06) |
| Brasileirão Série A | 2012 onward | no |

`data/normalize.py` maps both feed layouts onto one match table, converts kick-off times from UK
time to UTC, and keeps one set of odds per match: the market-average closing line when published,
otherwise the best available pre-match average. Unplayed rows older than two days (abandoned or
postponed matches) are dropped; rescheduled matches come back through the fixtures feed.

## Features

`features/build.py` makes one chronological pass over ~44,000 matches. For each row it records
each team's state *before* the match and only then applies the result. That ordering is the
leakage guarantee, and `tests/test_features.py` checks it directly: rewriting a match's score
never changes its own features or anything earlier.

State is tracked per team within its league:

- **Elo** with a 65-point home advantage, a goal-margin multiplier (World Football Elo style),
  25% regression to the mean over the summer, and a re-centre on 1500 each season so that
  promotion and relegation don't drag the scale down. Newly promoted clubs enter 80 points below
  the league average.
- **Form**: exponentially weighted means of goals, shots, shots on target and points, at a short
  (half-life ≈ 3 matches) and a long (≈ 14 matches) horizon, plus home-only and away-only scoring.
  The averages warm up as running means so early-season values are not biased toward zero.
- **Context**: rest days, matches played this season, and the league's rolling home-win rate,
  draw rate and goals per game, so a single pooled model can serve leagues with different
  base rates.

Missing shot data (Brazil, the earliest season) stays missing; the classifier handles it natively.

## Models

### Gradient-boosted classifier

`models/classifier.py`: scikit-learn's `HistGradientBoostingClassifier` on the 45 engineered features,
pooled across all six leagues. Hyperparameters are fixed up front (learning rate 0.04, 15-leaf
trees, 80 samples per leaf, L2 = 1, early stopping on an internal 10% split) rather than tuned on
the evaluation seasons. The first two seasons of each league are excluded from training while
Elo and form warm up.

### Dixon-Coles goals model

`models/dixon_coles.py`: per-league Poisson attack and defence ratings with a home term and the
Dixon-Coles correction for low scores (0-0, 1-0, 0-1, 1-1). Matches are weighted by
`exp(-0.0019 · age in days)` (half-life about a year) over a three-year window. The Poisson part
is fitted with L-BFGS using an analytic gradient and a small ridge penalty, then ρ by a bounded
one-dimensional search; one league refit takes around 10 ms.

This model supplies expected goals and the scoreline grid.

### Blend

The published probability is `w · classifier + (1 − w) · goals model`. The weight is chosen on a
**validation season** that comes before every evaluated season, and has settled at 0.7 / 0.3.
The scoreline grid is then rescaled so that its home-win, draw and away-win regions sum to the
blended probabilities, so "likeliest score" and the 1X2 numbers never disagree.

Two alternatives were tried and dropped:

- **Stacking** (Dixon-Coles outputs as classifier features) improved log loss by about 0.001,
  not enough to justify the extra coupling.
- **Temperature scaling** fitted on the validation season made test log loss slightly *worse*;
  the blend is already well calibrated (see the reliability diagram on the Model page).

## Evaluation

`evaluation/backtest.py` produces every evaluated prediction walk-forward:

- Dixon-Coles is refitted per league at the start of every week, on results before that week.
- The classifier is retrained every 1 January and 1 July, on all results before that date.

Per league, the in-progress season is the live track record, the three seasons before it are the
backtest, and the season before those is the validation season. The same code path produces the
current-season record shown on the site, so a past prediction is exactly what the model would
have said before kick-off.

Every forecast is scored against three references:

| Reference | What it is |
| --- | --- |
| Base rate | Each league's historical home/draw/away frequencies |
| Component models | The classifier and Dixon-Coles on their own |
| Market | Closing odds with the bookmaker margin removed proportionally |

The model beats the base rate and both components comfortably and trails the closing line by
roughly 0.02 in log loss, which is the expected gap for a model that sees no odds, team news or
lineups. Closing odds are the sharpest public forecast there is; matching them would require
information the model deliberately does not use.

## Known limitations

- No player-level information: injuries, suspensions and rotation are invisible until they show
  up in results.
- Promoted teams start from a fixed prior; the first few rounds of a season are the weakest part
  of the record.
- Brazil has no shot data and a congested calendar; its numbers are the worst of the six leagues,
  and the market's are too.
- Rounds are inferred from matches played, so a postponed match is grouped with the round it was
  originally scheduled in.
