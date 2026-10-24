import type { Messages } from "./en";

const it: Messages = {
  meta: {
    title: "Matchfield: pronostici di calcio per sei campionati",
    description:
      "Probabilità di vittoria, pareggio e sconfitta per Premier League, Liga, Serie A, Bundesliga, Ligue 1 e Brasileirão, confrontate con il mercato delle scommesse.",
  },
  nav: {
    skip: "Vai al contenuto",
    fixtures: "Partite",
    results: "Risultati",
    table: "Classifica",
    model: "Modello",
    language: "Lingua",
    sections: "Sezioni",
    leagues: "Campionati",
    updated: "Dati aggiornati al {time}",
  },
  outcome: { H: "Casa", D: "Pareggio", A: "Trasferta", short: { H: "1", D: "X", A: "2" } },
  fixtures: {
    round: "Giornata {n}",
    count: "{n} partite",
    kickoff: "Calcio d'inizio",
    xg: "xG",
    likeliest: "Risultati più probabili",
    model: "Modello",
    market: "Mercato",
    classifier: "Classificatore",
    goals: "Modello dei gol",
    elo: "Elo",
    venue: "Stadio",
    meetings: "Precedenti",
    noMeetings: "Nessun precedente nelle ultime due stagioni.",
    highlights: "Highlights",
    preview: "Locandina della partita",
    select: "Seleziona una partita per il dettaglio completo.",
    gap: "Modello meno mercato",
    empty: "Nessuna partita in programma. La prossima giornata comparirà qui appena pubblicata.",
    legend: "Le barre mostrano la probabilità del modello di vittoria in casa, pareggio e vittoria in trasferta.",
  },
  results: {
    title: "La stagione finora",
    barcode:
      "Una barra per partita, la più recente a destra. L'altezza è la probabilità che il modello aveva dato a ciò che è successo; le barre piene sono pronostici azzeccati.",
    hit: "Preso",
    miss: "Sbagliato",
    prediction: "Pronostico",
    result: "Risultato",
    gave: "Il modello dava al risultato il {p}",
    allRounds: "Tutte le giornate",
    filter: "Giornata",
    empty: "Nessuna partita giocata in questa stagione.",
  },
  table: {
    title: "Classifica",
    caption:
      "Classifica con i rating del modello. Attacco e difesa sono relativi a una squadra media (1,00).",
    pos: "Pos",
    team: "Squadra",
    played: "G",
    won: "V",
    drawn: "N",
    lost: "P",
    gd: "DR",
    points: "Pt",
    elo: "Elo",
    attack: "Att",
    defence: "Dif",
  },
  record: {
    title: "Bilancio stagionale",
    matches: "{n} partite",
    accuracy: "Precisione",
    logLoss: "Log loss",
    rps: "RPS",
    model: "Modello",
    market: "Mercato",
    lowerBetter: "Per log loss e RPS, più basso è meglio.",
    none: "La stagione non è ancora iniziata.",
  },
  model: {
    title: "Come funziona il modello",
    intro:
      "Ogni probabilità di questo sito viene da una pipeline che non vede mai le quote delle scommesse. Le quote servono solo dopo, come termine di paragone.",
    performance: "Prestazioni",
    performanceNote:
      "Backtest walk-forward sulle ultime {seasons} stagioni concluse di ogni campionato: ogni partita è stata prevista usando solo i risultati disponibili prima del fischio d'inizio.",
    metric: "Metrica",
    baseline: "Frequenza di base",
    classifier: "Solo classificatore",
    goals: "Solo modello dei gol",
    blend: "Combinazione pubblicata",
    market: "Quote di chiusura",
    matches: "{n} partite",
    calibration: "Calibrazione",
    calibrationNote:
      "Quando il modello dice 40%, succede il 40% delle volte? I punti sulla diagonale sono perfettamente calibrati.",
    predicted: "Probabilità prevista",
    observed: "Frequenza osservata",
    byLeague: "Per campionato e stagione",
    season: "Stagione",
    league: "Campionato",
    current: "in corso",
    drivers: "Cosa determina un pronostico",
    driversNote:
      "Importanza per permutazione sugli ultimi dodici mesi, fuori campione: quanto peggiora il log loss mescolando una variabile.",
    pipeline: "Pipeline",
    steps: {
      ingest: [
        "Acquisizione",
        "Risultati, tiri e quote di {matches} partite dal {first}, da football-data.co.uk.",
      ],
      features: [
        "Variabili",
        "Rating Elo e forma con media esponenziale, calcolati in un solo passaggio cronologico così che nessuna variabile veda il proprio risultato.",
      ],
      models: [
        "Modelli",
        "Un classificatore gradient boosting e un modello dei gol Dixon-Coles pesato nel tempo, combinati {w} / {g}.",
      ],
      evaluate: [
        "Valutazione",
        "Riaddestramenti walk-forward settimanali e semestrali, valutati con log loss e RPS rispetto alle quote di chiusura.",
      ],
      publish: [
        "Pubblicazione",
        "Un job programmato di GitHub Actions salva JSON statici; il sito si ricostruisce a ogni cambio dei dati.",
      ],
    },
    glossary: "Glossario",
    terms: {
      logLoss: [
        "Log loss",
        "Media del logaritmo negativo della probabilità data al risultato reale. Punisce gli errori fatti con troppa sicurezza.",
      ],
      rps: [
        "RPS",
        "Ranked probability score. Rispetta l'ordine: prevedere il pareggio quando vince la squadra di casa è un errore minore che prevedere la vittoria in trasferta.",
      ],
      market: [
        "Mercato",
        "Quote di chiusura dei bookmaker senza margine. Il termine di paragone pubblico più difficile.",
      ],
    },
    run: "Modello {version} · esecuzione di {seconds}s · commit {commit}",
  },
  state: {
    loading: "Caricamento",
    error: "Impossibile caricare i dati. Controlla la connessione e riprova.",
    retry: "Riprova",
    notFound: "Questa pagina non esiste.",
    home: "Torna alle partite",
  },
  footer: {
    data: "Dati: football-data.co.uk",
    images: "Immagini: TheSportsDB",
    source: "Codice sorgente",
    disclaimer: "Le probabilità sono stime del modello, non consigli di scommessa.",
  },
};

export default it;
