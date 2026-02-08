import { useState, useMemo, useEffect } from "react";
import LaLigaHeader from "@/components/LaLigaHeader";
import HeroSection from "@/components/HeroSection";
import MatchTable from "@/components/MatchTable";
import StatsCard from "@/components/StatsCard";
import InfoFooter from "@/components/InfoFooter";
import { fetchAllPredictions, fetchModelStats } from "@/lib/api";
import { Prediction } from "@/types/prediction";

const Index = () => {
  const [activeTab, setActiveTab] = useState<"current" | "history">("current");
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [modelStats, setModelStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Determine current gameweek from model stats
  const currentGameweek = modelStats?.currentGameweek || 1;
  const [selectedGameweek, setSelectedGameweek] = useState<number>(currentGameweek);

  // Fetch data from Supabase
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const [predictionsData, statsData] = await Promise.all([
          fetchAllPredictions(),
          fetchModelStats()
        ]);
        
        setPredictions(predictionsData);
        setModelStats(statsData);
        setSelectedGameweek(statsData?.currentGameweek || 1);
      } catch (err) {
        console.error("Error loading data:", err);
        setError("Failed to load predictions. Please try again later.");
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  // Helper functions for stats calculation
  const calculateCurrentStreak = (predictions: Prediction[]) => {
    let streak = 0;
    for (let i = predictions.length - 1; i >= 0; i--) {
      if (predictions[i].isCorrect) {
        streak++;
      } else {
        break;
      }
    }
    return streak;
  };

  const calculateBestAccuracy = (predictions: Prediction[]) => {
    const gameweekStats = new Map<number, { correct: number; total: number }>();
    
    predictions.forEach(p => {
      if (p.isCorrect !== undefined && p.actualHomeScore !== null) {
        const stats = gameweekStats.get(p.gameweek) || { correct: 0, total: 0 };
        stats.total++;
        if (p.isCorrect) stats.correct++;
        gameweekStats.set(p.gameweek, stats);
      }
    });

    let bestAccuracy = 0;
    gameweekStats.forEach(stats => {
      const accuracy = stats.total > 0 ? Math.round((stats.correct / stats.total) * 100) : 0;
      if (accuracy > bestAccuracy) bestAccuracy = accuracy;
    });

    return bestAccuracy;
  };

  // Calculate stats from fetched data
  const stats = useMemo(() => {
    const withActuals = predictions.filter(
      (p) => p.actualHomeScore !== null && p.actualHomeScore !== undefined
    );
    const correct = withActuals.filter((p) => p.isCorrect).length;
    const accuracy = withActuals.length > 0 
      ? Math.round((correct / withActuals.length) * 100) 
      : 0;
    
    const currentStreak = calculateCurrentStreak(predictions);
    const bestAccuracy = calculateBestAccuracy(predictions);

    return {
      totalPredictions: withActuals.length,
      accuracy,
      currentStreak,
      bestAccuracy,
    };
  }, [predictions]);

  // Get all available gameweeks
  const availableGameweeks = useMemo(() => {
    const gameweeks = [...new Set(predictions.map(p => p.gameweek))].sort((a, b) => a - b);
    return gameweeks;
  }, [predictions]);

  // Check if next gameweek has predictions
  const hasNextGameweek = useMemo(() => {
    const nextGameweek = selectedGameweek + 1;
    return availableGameweeks.includes(nextGameweek);
  }, [selectedGameweek, availableGameweeks]);

  // Check if previous gameweek has predictions
  const hasPrevGameweek = useMemo(() => {
    const prevGameweek = selectedGameweek - 1;
    return availableGameweeks.includes(prevGameweek);
  }, [selectedGameweek, availableGameweeks]);

  // Basic error handling
  if (loading) {
    return <div className="flex justify-center items-center h-screen">Loading predictions...</div>;
  }

  if (error) {
    return <div className="flex justify-center items-center h-screen text-red-500">{error}</div>;
  }

  if (!predictions || predictions.length === 0) {
    return <div className="flex justify-center items-center h-screen">No predictions available</div>;
  }

  console.log("Rendering Index component with data:", {
    totalPredictions: predictions.length,
    currentGameweek,
    selectedGameweek,
    stats
  });

  const handleGameweekChange = (gameweek: number) => {
    setSelectedGameweek(gameweek);
  };

  // Get predictions for the selected gameweek
  const currentGameweekPredictions = predictions.filter(p => p.gameweek === selectedGameweek && p.actualHomeScore === null);
  const historicalGameweekPredictions = predictions.filter(p => p.gameweek === selectedGameweek && p.actualHomeScore !== null);

  return (
    <div className="min-h-screen bg-background">
      <LaLigaHeader />
      <HeroSection />

      {/* Main Content */}
      <main className="max-w-[1400px] mx-auto px-4 py-8">
        {/* Stats Section */}
        <section id="stats" className="mb-10 animate-fade-in">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatsCard
              title="Total Predictions"
              value={stats.totalPredictions}
              subtitle="All-time"
              trend="up"
            />
            <StatsCard
              title="Accuracy"
              value={`${stats.accuracy}%`}
              subtitle="Overall rate"
              trend="up"
            />
            <StatsCard
              title="Current Streak"
              value={stats.currentStreak}
              subtitle="Correct in a row"
              trend="up"
            />
            <StatsCard
              title="Best Accuracy"
              value={`${stats.bestAccuracy}%`}
              subtitle="Gameweek 18"
              trend="neutral"
            />
          </div>
        </section>

        {/* Match Table */}
        <section id="predictions" className="mb-8">
          <div className="animate-fade-in">
            <MatchTable 
              predictions={currentGameweekPredictions.length > 0 ? currentGameweekPredictions : historicalGameweekPredictions}
              showActual={historicalGameweekPredictions.length > 0}
              onGameweekChange={handleGameweekChange}
              currentGameweek={currentGameweek}
              hasNextGameweek={hasNextGameweek}
              hasPrevGameweek={hasPrevGameweek}
            />
          </div>
        </section>
      </main>

      <InfoFooter />
    </div>
  );
};

export default Index;
