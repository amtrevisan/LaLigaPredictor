import type { Messages } from "./en";

const de: Messages = {
  meta: {
    title: "Matchfield: Fußballprognosen für sechs Ligen",
    description:
      "Sieg-, Remis- und Niederlagenwahrscheinlichkeiten für Premier League, La Liga, Serie A, Bundesliga, Ligue 1 und Brasileirão, gemessen am Wettmarkt.",
  },
  nav: {
    skip: "Zum Inhalt springen",
    fixtures: "Spiele",
    results: "Ergebnisse",
    table: "Tabelle",
    model: "Modell",
    language: "Sprache",
    leagues: "Ligen",
    updated: "Datenstand {time}",
  },
  outcome: { H: "Heim", D: "Remis", A: "Auswärts", short: { H: "1", D: "X", A: "2" } },
  fixtures: {
    round: "{n}. Spieltag",
    count: "{n} Spiele",
    kickoff: "Anstoß",
    xg: "xG",
    likeliest: "Wahrscheinlichste Ergebnisse",
    model: "Modell",
    market: "Markt",
    classifier: "Klassifikator",
    goals: "Tormodell",
    elo: "Elo",
    venue: "Stadion",
    meetings: "Letzte Duelle",
    noMeetings: "Keine Duelle in den letzten zwei Spielzeiten.",
    highlights: "Highlights",
    preview: "Spielplakat",
    select: "Wähle ein Spiel für die vollständige Aufschlüsselung.",
    gap: "Modell minus Markt",
    empty: "Noch keine Spiele angesetzt. Der nächste Spieltag erscheint hier, sobald er veröffentlicht ist.",
    legend: "Die Balken zeigen die Modellwahrscheinlichkeit für Heimsieg, Remis und Auswärtssieg.",
  },
  results: {
    title: "Die Saison bisher",
    barcode:
      "Ein Balken pro Spiel, das neueste rechts. Die Höhe ist die Wahrscheinlichkeit, die das Modell dem tatsächlichen Ausgang gab; volle Balken waren Treffer.",
    hit: "Treffer",
    miss: "Daneben",
    prediction: "Prognose",
    result: "Ergebnis",
    gave: "Das Modell gab dem Ergebnis {p}",
    allRounds: "Alle Spieltage",
    filter: "Spieltag",
    empty: "In dieser Saison wurde noch kein Spiel ausgetragen.",
  },
  table: {
    title: "Tabelle",
    caption:
      "Ligatabelle mit den Ratings des Modells. Angriff und Abwehr sind relativ zu einem durchschnittlichen Team (1,00).",
    pos: "Pl.",
    team: "Team",
    played: "Sp",
    won: "S",
    drawn: "U",
    lost: "N",
    gd: "TD",
    points: "Pkt",
    elo: "Elo",
    attack: "Ang",
    defence: "Abw",
  },
  record: {
    title: "Saisonbilanz",
    matches: "{n} Spiele",
    accuracy: "Trefferquote",
    logLoss: "Log-Loss",
    rps: "RPS",
    model: "Modell",
    market: "Markt",
    lowerBetter: "Bei Log-Loss und RPS ist niedriger besser.",
    none: "Die Saison hat noch nicht begonnen.",
  },
  model: {
    title: "So funktioniert das Modell",
    intro:
      "Jede Wahrscheinlichkeit auf dieser Seite stammt aus einer Pipeline, die nie Wettquoten sieht. Quoten dienen erst danach als Maßstab, an dem das Modell gemessen wird.",
    performance: "Leistung",
    performanceNote:
      "Walk-forward-Backtest über die letzten {seasons} abgeschlossenen Spielzeiten jeder Liga: Jedes Spiel wurde nur mit Ergebnissen vor dem Anstoß vorhergesagt.",
    metric: "Kennzahl",
    baseline: "Basisrate",
    classifier: "Nur Klassifikator",
    goals: "Nur Tormodell",
    blend: "Veröffentlichte Mischung",
    market: "Schlussquoten",
    matches: "{n} Spiele",
    calibration: "Kalibrierung",
    calibrationNote:
      "Wenn das Modell 40 % sagt, tritt es dann in 40 % der Fälle ein? Punkte auf der Diagonale sind perfekt kalibriert.",
    predicted: "Vorhergesagte Wahrscheinlichkeit",
    observed: "Beobachtete Häufigkeit",
    byLeague: "Nach Liga und Saison",
    season: "Saison",
    league: "Liga",
    current: "laufend",
    drivers: "Was eine Prognose bestimmt",
    driversNote:
      "Permutationswichtigkeit auf den letzten zwölf Monaten, out of sample: wie stark sich der Log-Loss verschlechtert, wenn ein Merkmal gemischt wird.",
    pipeline: "Pipeline",
    steps: {
      ingest: [
        "Import",
        "Ergebnisse, Schüsse und Quoten von {matches} Spielen seit {first}, von football-data.co.uk.",
      ],
      features: [
        "Merkmale",
        "Elo-Ratings und exponentiell gewichtete Form, in einem chronologischen Durchlauf berechnet, damit kein Merkmal sein eigenes Ergebnis sieht.",
      ],
      models: [
        "Modelle",
        "Ein Gradient-Boosting-Klassifikator und ein zeitgewichtetes Dixon-Coles-Tormodell, gemischt {w} / {g}.",
      ],
      evaluate: [
        "Bewertung",
        "Wöchentliche und halbjährliche Walk-forward-Neuanpassungen, bewertet mit Log-Loss und RPS gegen die Schlussquoten.",
      ],
      publish: [
        "Veröffentlichung",
        "Ein geplanter GitHub-Actions-Job committet statisches JSON; die Seite wird bei jeder Datenänderung neu gebaut.",
      ],
    },
    glossary: "Glossar",
    terms: {
      logLoss: [
        "Log-Loss",
        "Mittlerer negativer Logarithmus der Wahrscheinlichkeit für das tatsächliche Ergebnis. Bestraft selbstsichere Fehler.",
      ],
      rps: [
        "RPS",
        "Ranked Probability Score. Berücksichtigt die Reihenfolge: Ein Remis zu tippen, wenn das Heimteam gewinnt, ist knapper daneben als ein Auswärtssieg.",
      ],
      market: [
        "Markt",
        "Schlussquoten der Buchmacher ohne Marge. Der härteste öffentliche Maßstab überhaupt.",
      ],
    },
    run: "Modell {version} · Laufzeit {seconds}s · Commit {commit}",
  },
  state: {
    loading: "Lädt",
    error: "Die Daten konnten nicht geladen werden. Prüfe deine Verbindung und versuche es erneut.",
    retry: "Erneut versuchen",
    notFound: "Diese Seite gibt es nicht.",
    home: "Zurück zu den Spielen",
  },
  footer: {
    data: "Daten: football-data.co.uk",
    images: "Bilder: TheSportsDB",
    source: "Quellcode",
    disclaimer: "Die Wahrscheinlichkeiten sind Modellschätzungen, keine Wettempfehlungen.",
  },
};

export default de;
