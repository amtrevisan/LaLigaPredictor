"""Team naming: football-data.co.uk feed names to display names, short codes and image lookups.

The feeds use compact, stable names ("Ath Madrid", "Nott'm Forest"). The site needs a readable
name, a three-letter code for dense layouts and a search term TheSportsDB resolves to the right
club. Teams missing from the table fall back to their feed name, so a newly promoted club never
breaks the pipeline; it only renders with its feed name until an entry is added here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TeamInfo:
    name: str
    short: str
    search: str
    sportsdb_id: str | None = None


# (feed name) -> (display name, short code, TheSportsDB search term or None to reuse the name)
_TEAMS: dict[str, tuple[str, str, str | None]] = {
    # Premier League
    "Arsenal": ("Arsenal", "ARS", None),
    "Aston Villa": ("Aston Villa", "AVL", None),
    "Bournemouth": ("Bournemouth", "BOU", None),
    "Brentford": ("Brentford", "BRE", None),
    "Brighton": ("Brighton & Hove Albion", "BHA", "Brighton and Hove Albion"),
    "Burnley": ("Burnley", "BUR", None),
    "Chelsea": ("Chelsea", "CHE", None),
    "Coventry": ("Coventry City", "COV", None),
    "Crystal Palace": ("Crystal Palace", "CRY", None),
    "Everton": ("Everton", "EVE", None),
    "Fulham": ("Fulham", "FUL", None),
    "Hull": ("Hull City", "HUL", None),
    "Ipswich": ("Ipswich Town", "IPS", None),
    "Leeds": ("Leeds United", "LEE", None),
    "Leicester": ("Leicester City", "LEI", None),
    "Liverpool": ("Liverpool", "LIV", None),
    "Man City": ("Manchester City", "MCI", None),
    "Man United": ("Manchester United", "MUN", None),
    "Newcastle": ("Newcastle United", "NEW", None),
    "Nott'm Forest": ("Nottingham Forest", "NFO", None),
    "Southampton": ("Southampton", "SOU", None),
    "Sunderland": ("Sunderland", "SUN", None),
    "Tottenham": ("Tottenham Hotspur", "TOT", None),
    "West Ham": ("West Ham United", "WHU", None),
    "Wolves": ("Wolverhampton Wanderers", "WOL", None),
    # La Liga
    "Alaves": ("Deportivo Alavés", "ALA", "Alaves"),
    "Ath Bilbao": ("Athletic Club", "ATH", "Athletic Bilbao"),
    "Ath Madrid": ("Atlético Madrid", "ATM", "Atletico Madrid"),
    "Barcelona": ("Barcelona", "BAR", None),
    "Betis": ("Real Betis", "BET", "Real Betis"),
    "Celta": ("Celta Vigo", "CEL", "Celta Vigo"),
    "Elche": ("Elche", "ELC", None),
    "Espanol": ("Espanyol", "ESP", "Espanyol"),
    "Getafe": ("Getafe", "GET", None),
    "Girona": ("Girona", "GIR", None),
    "La Coruna": ("Deportivo La Coruña", "DEP", "Deportivo La Coruna"),
    "Las Palmas": ("Las Palmas", "LPA", None),
    "Leganes": ("Leganés", "LEG", "Leganes"),
    "Levante": ("Levante", "LEV", None),
    "Malaga": ("Málaga", "MAL", "Malaga"),
    "Mallorca": ("Mallorca", "MLL", None),
    "Osasuna": ("Osasuna", "OSA", None),
    "Oviedo": ("Real Oviedo", "OVI", "Real Oviedo"),
    "Real Madrid": ("Real Madrid", "RMA", None),
    "Santander": ("Racing Santander", "RAC", "Racing Santander"),
    "Sevilla": ("Sevilla", "SEV", None),
    "Sociedad": ("Real Sociedad", "RSO", "Real Sociedad"),
    "Valencia": ("Valencia", "VAL", None),
    "Valladolid": ("Real Valladolid", "VLL", "Real Valladolid"),
    "Vallecano": ("Rayo Vallecano", "RAY", "Rayo Vallecano"),
    "Villarreal": ("Villarreal", "VIL", None),
    # Serie A
    "Atalanta": ("Atalanta", "ATA", None),
    "Bologna": ("Bologna", "BOL", None),
    "Cagliari": ("Cagliari", "CAG", None),
    "Como": ("Como", "COM", "Como 1907"),
    "Cremonese": ("Cremonese", "CRE", None),
    "Empoli": ("Empoli", "EMP", None),
    "Fiorentina": ("Fiorentina", "FIO", None),
    "Frosinone": ("Frosinone", "FRO", None),
    "Genoa": ("Genoa", "GEN", None),
    "Inter": ("Inter Milan", "INT", "Inter Milan"),
    "Juventus": ("Juventus", "JUV", None),
    "Lazio": ("Lazio", "LAZ", None),
    "Lecce": ("Lecce", "LEC", None),
    "Milan": ("AC Milan", "MIL", "AC Milan"),
    "Monza": ("Monza", "MON", None),
    "Napoli": ("Napoli", "NAP", None),
    "Parma": ("Parma", "PAR", None),
    "Pisa": ("Pisa", "PIS", None),
    "Roma": ("Roma", "ROM", "AS Roma"),
    "Sassuolo": ("Sassuolo", "SAS", None),
    "Torino": ("Torino", "TOR", None),
    "Udinese": ("Udinese", "UDI", None),
    "Venezia": ("Venezia", "VEN", None),
    "Verona": ("Hellas Verona", "VER", "Hellas Verona"),
    # Bundesliga
    "Augsburg": ("FC Augsburg", "FCA", "FC Augsburg"),
    "Bayern Munich": ("Bayern Munich", "FCB", "Bayern Munich"),
    "Bochum": ("VfL Bochum", "BOC", "VfL Bochum"),
    "Dortmund": ("Borussia Dortmund", "BVB", "Borussia Dortmund"),
    "Ein Frankfurt": ("Eintracht Frankfurt", "SGE", "Eintracht Frankfurt"),
    "Elversberg": ("SV Elversberg", "SVE", "SV Elversberg"),
    "FC Koln": ("1. FC Köln", "KOE", "FC Koln"),
    "Freiburg": ("SC Freiburg", "SCF", "SC Freiburg"),
    "Hamburg": ("Hamburger SV", "HSV", "Hamburger SV"),
    "Heidenheim": ("1. FC Heidenheim", "HDH", "Heidenheim"),
    "Hoffenheim": ("TSG Hoffenheim", "TSG", "Hoffenheim"),
    "Holstein Kiel": ("Holstein Kiel", "KIE", None),
    "Leverkusen": ("Bayer Leverkusen", "B04", "Bayer Leverkusen"),
    "M'gladbach": ("Borussia Mönchengladbach", "BMG", "Borussia Monchengladbach"),
    "Mainz": ("Mainz 05", "M05", "Mainz"),
    "Paderborn": ("SC Paderborn", "SCP", "Paderborn"),
    "RB Leipzig": ("RB Leipzig", "RBL", None),
    "Schalke 04": ("Schalke 04", "S04", "Schalke 04"),
    "St Pauli": ("FC St. Pauli", "STP", "St Pauli"),
    "Stuttgart": ("VfB Stuttgart", "VFB", "VfB Stuttgart"),
    "Union Berlin": ("Union Berlin", "FCU", "Union Berlin"),
    "Werder Bremen": ("Werder Bremen", "SVW", "Werder Bremen"),
    "Wolfsburg": ("VfL Wolfsburg", "WOB", "Wolfsburg"),
    # Ligue 1
    "Angers": ("Angers", "ANG", None),
    "Auxerre": ("Auxerre", "AUX", None),
    "Brest": ("Stade Brestois", "BRE", "Brest"),
    "Le Havre": ("Le Havre", "HAC", None),
    "Le Mans": ("Le Mans", "LMA", None),
    "Lens": ("Lens", "RCL", None),
    "Lille": ("Lille", "LIL", None),
    "Lorient": ("Lorient", "LOR", None),
    "Lyon": ("Lyon", "OL", None),
    "Marseille": ("Marseille", "OM", None),
    "Metz": ("Metz", "MET", None),
    "Monaco": ("Monaco", "MON", None),
    "Montpellier": ("Montpellier", "MTP", None),
    "Nantes": ("Nantes", "NAN", None),
    "Nice": ("Nice", "NIC", None),
    "Paris FC": ("Paris FC", "PFC", None),
    "Paris SG": ("Paris Saint-Germain", "PSG", "Paris SG"),
    "Reims": ("Reims", "REI", None),
    "Rennes": ("Rennes", "REN", None),
    "St Etienne": ("Saint-Étienne", "ASSE", "Saint-Etienne"),
    "Strasbourg": ("Strasbourg", "RCS", None),
    "Toulouse": ("Toulouse", "TFC", None),
    "Troyes": ("Troyes", "ESTAC", None),
    # Brasileirão
    "Athletico-PR": ("Athletico Paranaense", "CAP", "Athletico Paranaense"),
    "Atletico-MG": ("Atlético Mineiro", "CAM", "Atletico Mineiro"),
    "Bahia": ("Bahia", "BAH", None),
    "Botafogo RJ": ("Botafogo", "BOT", "Botafogo"),
    "Bragantino": ("Red Bull Bragantino", "RBB", "Bragantino"),
    "Ceara": ("Ceará", "CEA", "Ceara"),
    "Chapecoense-SC": ("Chapecoense", "CHA", "Chapecoense"),
    "Corinthians": ("Corinthians", "COR", None),
    "Coritiba": ("Coritiba", "CFC", None),
    "Cruzeiro": ("Cruzeiro", "CRU", None),
    "Flamengo RJ": ("Flamengo", "FLA", "Flamengo"),
    "Fluminense": ("Fluminense", "FLU", None),
    "Fortaleza": ("Fortaleza", "FOR", None),
    "Gremio": ("Grêmio", "GRE", "Gremio"),
    "Internacional": ("Internacional", "INT", None),
    "Juventude": ("Juventude", "JUV", None),
    "Mirassol": ("Mirassol", "MIR", None),
    "Palmeiras": ("Palmeiras", "PAL", None),
    "Remo": ("Remo", "REM", "Clube do Remo"),
    "Santos": ("Santos", "SAN", None),
    "Sao Paulo": ("São Paulo", "SAO", "Sao Paulo"),
    "Sport Recife": ("Sport Recife", "SPT", None),
    "Vasco": ("Vasco da Gama", "VAS", "Vasco da Gama"),
    "Vitoria": ("Vitória", "VIT", "Vitoria"),
}


# Clubs whose name search on TheSportsDB's free tier lands on a namesake (another sport, the
# women's side). Resolved by id instead.
_SPORTSDB_IDS: dict[str, str] = {
    "Nott'm Forest": "133720",
}


def team_info(feed_name: str) -> TeamInfo:
    entry = _TEAMS.get(feed_name)
    if entry is None:
        letters = "".join(ch for ch in feed_name.upper() if ch.isalpha())
        return TeamInfo(feed_name, letters[:3] or "???", feed_name)
    name, short, search = entry
    return TeamInfo(name, short, search or name, _SPORTSDB_IDS.get(feed_name))
