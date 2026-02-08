import { useState, useMemo } from "react";
import { Menu, X } from "lucide-react";
import { getTeamLogo, teamLogos } from "@/lib/teamLogos";

const LaLigaHeader = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Get unique team logos (no duplicates)
  const uniqueTeams = useMemo(() => {
    const logoMap = new Map<string, string>();
    
    // Iterate through all teams and keep only the first occurrence of each unique logo URL
    Object.entries(teamLogos).forEach(([teamName, logoUrl]) => {
      if (!logoMap.has(logoUrl)) {
        logoMap.set(logoUrl, teamName);
      }
    });
    
    // Return array of team names with unique logos
    return Array.from(logoMap.values());
  }, []);

  return (
    <header className="w-full sticky top-0 z-50">
      {/* Club Logos Ticker */}
      <div className="bg-background border-b border-border overflow-hidden">
        <div className="relative w-full">
          {/* Two identical sets for seamless infinite scroll */}
          <div className="flex animate-ticker-scroll">
            {/* First set */}
            <div className="flex whitespace-nowrap py-2">
              {uniqueTeams.map((team) => (
                <img
                  key={`set1-${team}`}
                  src={getTeamLogo(team)}
                  alt={team}
                  className="mx-4 w-8 h-8 object-contain opacity-60 hover:opacity-100 transition-all duration-300 cursor-pointer flex-shrink-0 hover:scale-110"
                  loading="lazy"
                />
              ))}
            </div>
            {/* Second set (duplicate for seamless loop) */}
            <div className="flex whitespace-nowrap py-2">
              {uniqueTeams.map((team) => (
                <img
                  key={`set2-${team}`}
                  src={getTeamLogo(team)}
                  alt={team}
                  className="mx-4 w-8 h-8 object-contain opacity-60 hover:opacity-100 transition-all duration-300 cursor-pointer flex-shrink-0 hover:scale-110"
                  loading="lazy"
                />
              ))}
            </div>
          </div>
        </div>
      </div>

    </header>
  );
};

export default LaLigaHeader;