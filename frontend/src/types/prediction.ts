export interface Prediction {
  id: string;
  gameweek: number;
  homeTeam: string;
  awayTeam: string;
  predictedHomeScore: number;
  predictedAwayScore: number;
  confidence: number;
  actualHomeScore?: number;
  actualAwayScore?: number;
  date: string;
  homeWinProbability?: number;
  drawProbability?: number;
  awayWinProbability?: number;
  isCorrect?: boolean;
}

export interface GameweekStats {
  gameweek: number;
  totalMatches: number;
  correctPredictions: number;
  accuracy: number;
}
