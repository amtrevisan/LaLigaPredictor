const en = {
  meta: {
    title: "Oddsless: football predictions for six leagues",
    description:
      "Win, draw and loss probabilities for the Premier League, La Liga, Serie A, Bundesliga, Ligue 1 and Brasileirão, scored against the betting market.",
  },
  nav: {
    skip: "Skip to content",
    fixtures: "Fixtures",
    results: "Results",
    table: "Table",
    model: "Model",
    language: "Language",
    sections: "Sections",
    leagues: "Leagues",
    updated: "Data as of {time}",
  },
  outcome: { H: "Home", D: "Draw", A: "Away", short: { H: "1", D: "X", A: "2" } },
  fixtures: {
    round: "Round {n}",
    count: "{n} fixtures",
    kickoff: "Kick-off",
    xg: "xG",
    likeliest: "Likeliest scores",
    model: "Model",
    market: "Market",
    classifier: "Classifier",
    goals: "Goals model",
    elo: "Elo",
    venue: "Venue",
    meetings: "Previous meetings",
    noMeetings: "No earlier league meetings on record.",
    highlights: "Highlights",
    preview: "Match artwork",
    select: "Select a fixture for the full breakdown.",
    gap: "vs market",
    empty: "No fixtures scheduled yet. The next round appears here once it is published.",
    legend: "Bars show the model's probability of a home win, draw and away win.",
  },
  results: {
    title: "Season so far",
    barcode:
      "One bar per match, newest on the right. Height is the probability the model gave to what actually happened; solid bars were called correctly.",
    hit: "Hit",
    miss: "Miss",
    prediction: "Prediction",
    result: "Result",
    gave: "Model gave the result {p}",
    allRounds: "All rounds",
    filter: "Round",
    search: "Team",
    noMatch: "No matches for “{q}”.",
    empty: "No matches played this season yet.",
  },
  table: {
    title: "Standings",
    caption:
      "League table with the model's ratings. Attack and defence are relative to an average side (1.00).",
    pos: "Pos",
    team: "Team",
    played: "P",
    won: "W",
    drawn: "D",
    lost: "L",
    gd: "GD",
    points: "Pts",
    elo: "Elo",
    attack: "Att",
    defence: "Def",
  },
  record: {
    title: "Season record",
    matches: "{n} matches",
    accuracy: "Accuracy",
    logLoss: "Log loss",
    rps: "RPS",
    model: "Model",
    market: "Market",
    lowerBetter: "Lower is better for log loss and RPS.",
    none: "The season has not started yet.",
  },
  model: {
    title: "How the model works",
    intro:
      "Every probability on this site comes from a pipeline that never sees betting odds. Odds are used only afterwards, as the benchmark the model is scored against.",
    performance: "Performance",
    performanceNote:
      "Walk-forward backtest over the last {seasons} completed seasons in every league: each match was predicted using only results available before kick-off.",
    metric: "Metric",
    baseline: "Base rate",
    classifier: "Classifier only",
    goals: "Goals model only",
    blend: "Published blend",
    market: "Closing odds",
    matches: "{n} matches",
    calibration: "Calibration",
    calibrationNote:
      "When the model says 40%, does it happen 40% of the time? Points on the diagonal are perfectly calibrated.",
    predicted: "Predicted probability",
    observed: "Observed frequency",
    byLeague: "By league and season",
    season: "Season",
    league: "League",
    current: "in progress",
    drivers: "What drives a prediction",
    driversNote:
      "Permutation importance on the last twelve months, out of sample: how much log loss worsens when one feature is shuffled.",
    pipeline: "Pipeline",
    steps: {
      ingest: [
        "Ingest",
        "Results, shots and odds for {matches} matches since {first}, from football-data.co.uk.",
      ],
      features: [
        "Features",
        "Elo ratings and exponentially weighted form, built in one chronological pass so no feature sees its own result.",
      ],
      models: [
        "Models",
        "A gradient-boosted classifier and a time-weighted Dixon-Coles goals model, blended {w} / {g}.",
      ],
      evaluate: [
        "Evaluate",
        "Weekly and semi-annual walk-forward refits, scored with log loss and ranked probability score against closing odds.",
      ],
      publish: [
        "Publish",
        "A scheduled GitHub Actions job commits static JSON; the site rebuilds on every data change.",
      ],
    },
    glossary: "Glossary",
    terms: {
      logLoss: [
        "Log loss",
        "Average negative log of the probability given to the actual result. Punishes confident mistakes.",
      ],
      rps: [
        "RPS",
        "Ranked probability score. Respects order: predicting a draw when the home side wins is a nearer miss than predicting an away win.",
      ],
      market: [
        "Market",
        "Bookmaker closing odds with the margin removed. The hardest public benchmark there is.",
      ],
    },
    run: "Model {version} · run took {seconds}s · commit {commit}",
  },
  state: {
    loading: "Loading",
    error: "Could not load the data. Check your connection and try again.",
    retry: "Try again",
    notFound: "That page does not exist.",
    home: "Back to fixtures",
  },
  footer: {
    builtBy: "Designed and built by",
    data: "Data: football-data.co.uk",
    images: "Images: TheSportsDB",
    source: "Source code",
    disclaimer: "Probabilities are model estimates, not betting advice.",
  },
} as const;

type Widen<T> = T extends string
  ? string
  : T extends readonly [string, string]
    ? readonly [string, string]
    : { [K in keyof T]: Widen<T[K]> };

export type Messages = Widen<typeof en>;
export default en as Messages;
