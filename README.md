# Oddsless

**Football match predictions for six leagues, from a model that never sees betting odds, and is
scored against them in public.**

[![Live site](https://img.shields.io/badge/live-oddsless.netlify.app-000?style=flat-square)](https://oddsless.netlify.app)
[![CI](https://img.shields.io/github/actions/workflow/status/amtrevisan/oddsless/ci.yml?branch=main&label=ci&style=flat-square&color=000)](https://github.com/amtrevisan/oddsless/actions/workflows/ci.yml)
[![Predictions](https://img.shields.io/github/actions/workflow/status/amtrevisan/oddsless/predictions.yml?branch=main&label=data%20refresh&style=flat-square&color=000)](https://github.com/amtrevisan/oddsless/actions/workflows/predictions.yml)
![Python](https://img.shields.io/badge/python-3.12-000?style=flat-square)
![React](https://img.shields.io/badge/react-19-000?style=flat-square)

[![La Liga round 10 on the live site](docs/img/fixtures.webp)](https://oddsless.netlify.app/la-liga)

|  44k matches  |  6 leagues  |  6,396 backtested predictions  |  refreshed 4× a day  |  6 languages  |
| :---: | :---: | :---: | :---: | :---: |

Premier League · La Liga · Serie A · Bundesliga · Ligue 1 · Brasileirão. The UI is available in
English, Spanish, Italian, German, French and Portuguese.

## How well it predicts

Every number below comes from a **walk-forward backtest**: each match was predicted using only
data available before kick-off. The bookmakers' closing line is the benchmark, and it is the
hardest public one there is.

<p align="center"><img src="docs/img/skill.svg" alt="Log-loss skill over the base rate: goals model 7.2%, classifier 7.8%, published blend 8.0%, closing odds 9.6%" width="760"></p>

|  | Log loss ↓ | RPS ↓ | Accuracy |
| --- | :---: | :---: | :---: |
| Base rate | 1.073 | 0.229 | 43.9% |
| **Published blend** | **0.987** | **0.200** | **52.3%** |
| Closing odds | 0.969 | 0.194 | 54.1% |

<p align="center"><img src="docs/img/leagues.svg" alt="Log loss by league for the model, closing odds and base rate" width="760"></p>

<table>
<tr>
<td width="50%"><img src="docs/img/calibration.webp" alt="Calibration chart: predicted probability against observed frequency"></td>
<td width="50%"><img src="docs/img/importance.webp" alt="Permutation importance: Elo difference dominates"></td>
</tr>
<tr>
<td>When it says 40%, it happens about 40% of the time, about as well calibrated as the market.</td>
<td>Elo difference does most of the work; shot and form features refine it.</td>
</tr>
</table>

## How it works

```mermaid
flowchart LR
  A[football-data.co.uk<br/>results, shots, fixtures] --> B[features<br/>Elo + weighted form,<br/>one leak-free pass]
  B --> C[gradient-boosted<br/>classifier]
  A --> D[Dixon-Coles<br/>goals model]
  C --> E[blend 0.7 / 0.3]
  D --> E
  E --> F[walk-forward backtest<br/>vs closing odds]
  E --> G[static JSON]
  G --> H[React site]
```

- **Features without leakage.** One chronological pass records each team's Elo and
  exponentially weighted form *before* every match, and only then applies the result.
- **Two models, blended.** A scikit-learn `HistGradientBoostingClassifier` pooled across leagues,
  plus a per-league, time-weighted Dixon-Coles goals model fitted with an analytic gradient.
  The blend weight is chosen on a held-out season.
- **Honest evaluation.** Dixon-Coles refits weekly and the classifier retrains every six months,
  scored with log loss, RPS and Brier against the market and a base rate.
- **No servers.** A scheduled GitHub Actions job runs the pipeline, commits versioned JSON when
  it changes, and Netlify redeploys. No database, no API keys.

The full write-up is in [docs/model.md](docs/model.md).

## Up close

Each fixture opens into the model's 1X2 probabilities next to the market's, the two component
models, expected goals and the likeliest scorelines.

[![Barcelona v Real Madrid detail](docs/img/match.webp)](https://oddsless.netlify.app/la-liga)

## Stack

| | |
| --- | --- |
| **ML / data** | Python 3.12, pandas, scikit-learn, SciPy, uv, pytest, ruff |
| **Frontend** | React 19, TypeScript (strict), Vite, TanStack Query, React Router, CSS modules, Vitest |
| **Infra** | GitHub Actions (CI + scheduled refresh), Netlify |

```
backend/src/predictor/   data · features · models · evaluation · export · pipeline
frontend/src/            views · components · i18n (6 locales) · typed data contract
docs/model.md            modelling and evaluation write-up
```

<details>
<summary><b>Run it locally</b></summary>

```bash
# backend: full refresh in about 20 s, writes ../frontend/public/data
cd backend && uv sync
uv run predictor run              # add --no-media to skip image lookups
uv run pytest

# frontend
cd frontend && npm ci
npm run dev                       # http://localhost:5173
npm test
```

</details>

## Credits

Match data from [football-data.co.uk](https://www.football-data.co.uk). Crests and photography
from [TheSportsDB](https://www.thesportsdb.com); club crests are trademarks of their clubs.
Probabilities are model estimates, not betting advice.

---

Designed and built by **Alex Morales Trevisan** ·
[LinkedIn](https://www.linkedin.com/in/alex-morales-dev/) ·
[GitHub](https://github.com/amtrevisan) ·
[YouTube](https://www.youtube.com/@alexmoralesdev)
