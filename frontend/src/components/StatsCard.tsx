import { Card } from "@/components/ui/card";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface StatsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: "up" | "down" | "neutral";
}

const StatsCard = ({ title, value, subtitle, trend }: StatsCardProps) => {
  console.log("StatsCard rendered:", { title, value, subtitle, trend });
  return (
    <div className="bg-gradient-card border border-border rounded-none shadow-lg">
      {/* Top border accent */}
      <div className="h-1 bg-gradient-to-r from-primary to-laliga-coral rounded-t-none" />
      
      <div className="p-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">{title}</p>
            <p className="text-4xl font-heading text-foreground mb-1 font-bold">{value}</p>
            {subtitle && (
              <p className="text-xs text-muted-foreground font-medium">{subtitle}</p>
            )}
          </div>
          <div className="flex items-center justify-center">
            {trend === "up" && (
              <div className="bg-success/10 border border-success/20 p-3 rounded-none">
                <TrendingUp className="w-6 h-6 text-success font-bold" />
              </div>
            )}
            {trend === "down" && (
              <div className="bg-destructive/10 border border-destructive/20 p-3 rounded-none">
                <TrendingDown className="w-6 h-6 text-destructive font-bold" />
              </div>
            )}
            {trend === "neutral" && (
              <div className="bg-warning/10 border border-warning/20 p-3 rounded-none">
                <Minus className="w-6 h-6 text-warning font-bold" />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatsCard;
