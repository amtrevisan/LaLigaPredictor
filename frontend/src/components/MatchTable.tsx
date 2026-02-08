import { useState } from "react";
import { Prediction } from "@/types/prediction";
import { getTeamLogo } from "@/lib/teamLogos";
import { ChevronLeft, ChevronRight, CheckCircle2, XCircle } from "lucide-react";

interface MatchTableProps {
  predictions: Prediction[];
  showActual?: boolean;
  onGameweekChange?: (gameweek: number) => void;
  currentGameweek?: number;
  hasNextGameweek?: boolean;
  hasPrevGameweek?: boolean;
}

const MatchTable = ({ predictions, showActual = false, onGameweekChange, currentGameweek, hasNextGameweek, hasPrevGameweek }: MatchTableProps) => {
  const [imgErrors, setImgErrors] = useState<Set<string>>(new Set());

  console.log("MatchTable rendered with predictions:", predictions);
  console.log("showActual:", showActual);

  // Check if predictions is empty
  if (!predictions || predictions.length === 0) {
    console.log("MatchTable: No predictions to render");
    return <div>No predictions available</div>;
  }

  const generateFallback = (name: string) => {
    const initials = name.split(" ").map(w => w[0]).join("").substring(0, 3).toUpperCase();
    return `https://ui-avatars.com/api/?name=${encodeURIComponent(initials)}&background=1a1a3e&color=fff&size=64&bold=true&font-size=0.4`;
  };

  const handleImgError = (teamName: string, e: React.SyntheticEvent<HTMLImageElement>) => {
    setImgErrors(prev => new Set(prev).add(teamName));
    e.currentTarget.src = generateFallback(teamName);
  };

  const formatDate = (dateStr: string) => {
    const d = new Date(dateStr);
    const days = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"];
    const day = days[d.getDay()];
    const dd = String(d.getDate()).padStart(2, "0");
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const yyyy = d.getFullYear();
    return `${day} ${dd}.${mm}.${yyyy}`;
  };

  const formatTime = (dateStr: string) => {
    const d = new Date(dateStr);
    return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  };

  const getProbabilityColors = (p: Prediction) => {
    const probs = [
      { key: "home", val: p.homeWinProbability },
      { key: "draw", val: p.drawProbability },
      { key: "away", val: p.awayWinProbability },
    ].sort((a, b) => b.val - a.val);

    const colors: Record<string, string> = {};
    colors[probs[0].key] = "bg-success";
    colors[probs[1].key] = "bg-warning";
    colors[probs[2].key] = "bg-destructive";
    return colors;
  };

  const gameweek = predictions[0]?.gameweek || 0;

  const handlePrevGameweek = () => {
    if (onGameweekChange && gameweek > 1) {
      onGameweekChange(gameweek - 1);
    }
  };

  const handleNextGameweek = () => {
    if (onGameweekChange) {
      onGameweekChange(gameweek + 1);
    }
  };

  return (
    <div className="w-full">
      {/* Matchday Header */}
      <div className="bg-laliga-navy text-primary-foreground flex items-center justify-center gap-4 py-3 rounded-t-md">
        <button 
          onClick={handlePrevGameweek}
          disabled={!hasPrevGameweek}
          className="p-1 hover:bg-laliga-navy-light rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ChevronLeft size={20} />
        </button>
        <span className="text-sm font-semibold tracking-widest uppercase">
          Matchday {gameweek} | 2025/2026
        </span>
        <button 
          onClick={handleNextGameweek}
          disabled={!hasNextGameweek}
          className="p-1 hover:bg-laliga-navy-light rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ChevronRight size={20} />
        </button>
      </div>

      {/* Table Header */}
      <div className="hidden md:grid grid-cols-[140px_80px_1fr_200px] bg-secondary/80 border-x border-border text-xs font-semibold text-muted-foreground uppercase tracking-widest py-3 px-4">
        <span>Date</span>
        <span>Time</span>
        <span className="text-center">Match</span>
        <span className="text-center">Prediction</span>
      </div>

      {/* Match Rows */}
      <div className="border-x border-b border-border rounded-b-md overflow-hidden">
        {predictions.map((prediction, idx) => {
          const colors = getProbabilityColors(prediction);
          const hasResult = prediction.actualHomeScore !== null && prediction.actualAwayScore !== null;

          return (
            <div
              key={prediction.id}
              className={`${idx % 2 === 0 ? "bg-background" : "bg-secondary/30"} 
                border-b border-border last:border-b-0 
                hover:bg-secondary/60 transition-colors duration-200
                md:grid md:grid-cols-[140px_80px_1fr_200px] 
                flex flex-col gap-3 md:gap-0 md:items-center 
                py-4 px-4`}
            >
              {/* Date */}
              <div className="text-xs font-medium text-muted-foreground uppercase">
                {formatDate(prediction.date)}
              </div>

              {/* Time */}
              <div className="text-sm font-semibold text-foreground">
                {formatTime(prediction.date)}
              </div>

              {/* Match */}
              <div className="flex items-center justify-center gap-2 sm:gap-4">
                {/* Home Team */}
                <div className="flex items-center gap-2 flex-1 justify-end min-w-0">
                  <span className="text-sm font-semibold text-foreground text-right truncate uppercase">
                    {prediction.homeTeam}
                  </span>
                  <img
                    src={getTeamLogo(prediction.homeTeam)}
                    alt={prediction.homeTeam}
                    className="w-7 h-7 object-contain flex-shrink-0"
                    loading="lazy"
                    onError={(e) => handleImgError(prediction.homeTeam, e)}
                  />
                </div>

                {/* Score / VS */}
                <div className="flex-shrink-0 w-20 text-center">
                  {hasResult ? (
                    <div className="flex items-center justify-center gap-1">
                      <span className="text-lg font-bold text-foreground">{prediction.actualHomeScore}</span>
                      <span className="text-muted-foreground">-</span>
                      <span className="text-lg font-bold text-foreground">{prediction.actualAwayScore}</span>
                      {showActual && prediction.isCorrect !== undefined && (
                        prediction.isCorrect ? (
                          <CheckCircle2 className="w-4 h-4 text-success ml-1" />
                        ) : (
                          <XCircle className="w-4 h-4 text-destructive ml-1" />
                        )
                      )}
                    </div>
                  ) : (
                    <span className="text-sm font-bold text-primary">VS</span>
                  )}
                </div>

                {/* Away Team */}
                <div className="flex items-center gap-2 flex-1 justify-start min-w-0">
                  <img
                    src={getTeamLogo(prediction.awayTeam)}
                    alt={prediction.awayTeam}
                    className="w-7 h-7 object-contain flex-shrink-0"
                    loading="lazy"
                    onError={(e) => handleImgError(prediction.awayTeam, e)}
                  />
                  <span className="text-sm font-semibold text-foreground truncate uppercase">
                    {prediction.awayTeam}
                  </span>
                </div>
              </div>

              {/* Prediction Bar */}
              <div className="flex flex-col items-center gap-2">
                {/* Clear labels for Home/Draw/Away */}
                <div className="flex items-center justify-between w-full max-w-[160px] text-[10px] font-semibold text-muted-foreground">
                  <span className="text-foreground">{prediction.homeTeam}</span>
                  <span>Draw</span>
                  <span className="text-foreground">{prediction.awayTeam}</span>
                </div>
                
                {/* Probability percentages */}
                <div className="flex items-center justify-between w-full max-w-[160px] text-[10px] text-muted-foreground">
                  <span className="font-semibold text-foreground">{prediction.homeWinProbability}%</span>
                  <span className="font-semibold text-foreground">{prediction.drawProbability}%</span>
                  <span className="font-semibold text-foreground">{prediction.awayWinProbability}%</span>
                </div>
                
                {/* Visual probability bar */}
                <div className="w-full max-w-[160px] bg-border rounded-full h-3 overflow-hidden flex">
                  <div
                    className={`${colors.home} h-full transition-all duration-500 relative group`}
                    style={{ width: `${prediction.homeWinProbability}%` }}
                  >
                    <span className="absolute -top-6 left-1/2 transform -translate-x-1/2 bg-black text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                      {prediction.homeTeam}: {prediction.homeWinProbability}%
                    </span>
                  </div>
                  <div
                    className={`${colors.draw} h-full transition-all duration-500 relative group`}
                    style={{ width: `${prediction.drawProbability}%` }}
                  >
                    <span className="absolute -top-6 left-1/2 transform -translate-x-1/2 bg-black text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                      Draw: {prediction.drawProbability}%
                    </span>
                  </div>
                  <div
                    className={`${colors.away} h-full transition-all duration-500 relative group`}
                    style={{ width: `${prediction.awayWinProbability}%` }}
                  >
                    <span className="absolute -top-6 left-1/2 transform -translate-x-1/2 bg-black text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                      {prediction.awayTeam}: {prediction.awayWinProbability}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default MatchTable;