// Static JSON contract written by `predictor run` (backend/src/predictor/export.py).
// Probabilities are always [home, draw, away].

export type Triple = [number, number, number];
export type Outcome = "H" | "D" | "A";

export interface Scores {
  logLoss: number;
  rps: number;
  brier: number;
  accuracy: number;
}

export interface SeasonRecord {
  matches: number;
  model: Scores;
  market: Scores;
  baseline: Scores;
}

export interface LeagueSummary {
  id: string;
  name: string;
  country: string;
  locale: string;
  season: string;
  upcoming: number;
  nextKickoff: string | null;
  nextRound: number | null;
  record: SeasonRecord | null;
}

export interface Manifest {
  schemaVersion: number;
  generatedAt: string;
  dataAsOf: string;
  leagues: LeagueSummary[];
}

export interface Team {
  name: string;
  short: string;
  crest: string | null;
  /** Club photography (stadium, squad, fans), possibly empty. */
  fanart: string[];
  stadium: string | null;
}

export interface Meeting {
  date: string;
  home: string;
  away: string;
  score: [number, number] | null;
  competition: string | null;
  image: string | null;
  video: string | null;
}

export interface Fixture {
  id: string;
  round: number;
  kickoff: string;
  home: Team;
  away: Team;
  probs: Triple;
  pick: Outcome;
  market: Triple | null;
  xg: [number, number] | null;
  components: { classifier: Triple | null; goals: Triple | null };
  elo: [number, number] | null;
}

export interface UpcomingFixture extends Fixture {
  scorelines: { score: [number, number]; p: number }[];
  poster: string | null;
  meetings: Meeting[];
}

export interface PlayedFixture extends Fixture {
  result: { score: [number, number]; outcome: Outcome };
  correct: boolean;
}

export interface Standing {
  team: Team;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  goalsFor: number;
  goalsAgainst: number;
  points: number;
  elo: number | null;
  attack: number | null;
  defence: number | null;
}

export interface LeagueData {
  schemaVersion: number;
  generatedAt: string;
  league: { id: string; name: string; country: string };
  season: string;
  record: SeasonRecord | null;
  upcoming: UpcomingFixture[];
  results: PlayedFixture[];
  standings: Standing[];
}

export interface Evaluation {
  matches: number;
  model: Scores;
  market: Scores;
  baseline: Scores;
  classifier: Scores;
  goals: Scores;
}

export interface CalibrationBin {
  predicted: number;
  observed: number;
  count: number;
}

export interface ModelReport {
  schemaVersion: number;
  generatedAt: string;
  version: string;
  commit: string | null;
  runtimeSeconds: number;
  blend: { classifierWeight: number; goalsWeight: number };
  validation: { matches: number; seasonsBeforeBacktest: number };
  classifier: {
    algorithm: string;
    hyperparameters: Record<string, number | boolean>;
    features: string[];
    trainingMatches: number;
  };
  data: { matches: number; leagues: number; firstMatch: string; lastResult: string };
  backtest: {
    seasons: number;
    overall: Evaluation;
    currentSeason: Evaluation | null;
    byLeague: (Evaluation & { league: string; season: string; current: boolean })[];
    calibration: { model: CalibrationBin[]; market: CalibrationBin[] };
  };
  importance: { feature: string; importance: number }[];
}
