import type { Messages } from "./en";

const fr: Messages = {
  meta: {
    title: "Matchfield : pronostics de football pour six championnats",
    description:
      "Probabilités de victoire, nul et défaite pour la Premier League, la Liga, la Serie A, la Bundesliga, la Ligue 1 et le Brasileirão, comparées au marché des paris.",
  },
  nav: {
    skip: "Aller au contenu",
    fixtures: "Matchs",
    results: "Résultats",
    table: "Classement",
    model: "Modèle",
    language: "Langue",
    leagues: "Championnats",
    updated: "Données au {time}",
  },
  outcome: { H: "Domicile", D: "Nul", A: "Extérieur", short: { H: "1", D: "N", A: "2" } },
  fixtures: {
    round: "{n}e journée",
    count: "{n} matchs",
    kickoff: "Coup d'envoi",
    xg: "xG",
    likeliest: "Scores les plus probables",
    model: "Modèle",
    market: "Marché",
    classifier: "Classifieur",
    goals: "Modèle de buts",
    elo: "Elo",
    venue: "Stade",
    meetings: "Confrontations précédentes",
    noMeetings: "Aucune confrontation lors des deux dernières saisons.",
    highlights: "Résumé",
    preview: "Affiche du match",
    select: "Sélectionnez un match pour le détail complet.",
    gap: "Modèle moins marché",
    empty: "Aucun match programmé pour l'instant. La prochaine journée apparaîtra ici dès sa publication.",
    legend: "Les barres montrent la probabilité du modèle pour une victoire à domicile, un nul et une victoire à l'extérieur.",
  },
  results: {
    title: "La saison jusqu'ici",
    barcode:
      "Une barre par match, la plus récente à droite. La hauteur est la probabilité que le modèle donnait à ce qui s'est réellement passé ; les barres pleines sont des pronostics justes.",
    hit: "Juste",
    miss: "Raté",
    prediction: "Pronostic",
    result: "Résultat",
    gave: "Le modèle donnait {p} à ce résultat",
    allRounds: "Toutes les journées",
    filter: "Journée",
    empty: "Aucun match joué cette saison pour l'instant.",
  },
  table: {
    title: "Classement",
    caption:
      "Classement avec les notes du modèle. Attaque et défense sont relatives à une équipe moyenne (1,00).",
    pos: "Pos",
    team: "Équipe",
    played: "J",
    won: "G",
    drawn: "N",
    lost: "P",
    gd: "Diff",
    points: "Pts",
    elo: "Elo",
    attack: "Att",
    defence: "Déf",
  },
  record: {
    title: "Bilan de la saison",
    matches: "{n} matchs",
    accuracy: "Réussite",
    logLoss: "Log loss",
    rps: "RPS",
    model: "Modèle",
    market: "Marché",
    lowerBetter: "Pour le log loss et le RPS, plus bas est meilleur.",
    none: "La saison n'a pas encore commencé.",
  },
  model: {
    title: "Comment fonctionne le modèle",
    intro:
      "Chaque probabilité de ce site provient d'une chaîne de traitement qui ne voit jamais les cotes. Les cotes ne servent qu'après coup, comme référence à laquelle le modèle se mesure.",
    performance: "Performance",
    performanceNote:
      "Backtest walk-forward sur les {seasons} dernières saisons terminées de chaque championnat : chaque match a été prédit uniquement avec les résultats connus avant le coup d'envoi.",
    metric: "Mesure",
    baseline: "Fréquence de base",
    classifier: "Classifieur seul",
    goals: "Modèle de buts seul",
    blend: "Mélange publié",
    market: "Cotes de clôture",
    matches: "{n} matchs",
    calibration: "Calibration",
    calibrationNote:
      "Quand le modèle annonce 40 %, cela arrive-t-il 40 % du temps ? Les points sur la diagonale sont parfaitement calibrés.",
    predicted: "Probabilité prédite",
    observed: "Fréquence observée",
    byLeague: "Par championnat et saison",
    season: "Saison",
    league: "Championnat",
    current: "en cours",
    drivers: "Ce qui fait un pronostic",
    driversNote:
      "Importance par permutation sur les douze derniers mois, hors échantillon : de combien le log loss se dégrade quand une variable est mélangée.",
    pipeline: "Chaîne de traitement",
    steps: {
      ingest: [
        "Collecte",
        "Résultats, tirs et cotes de {matches} matchs depuis {first}, issus de football-data.co.uk.",
      ],
      features: [
        "Variables",
        "Classements Elo et forme pondérée exponentiellement, calculés en un seul passage chronologique pour qu'aucune variable ne voie son propre résultat.",
      ],
      models: [
        "Modèles",
        "Un classifieur à gradient boosting et un modèle de buts Dixon-Coles pondéré dans le temps, mélangés {w} / {g}.",
      ],
      evaluate: [
        "Évaluation",
        "Réentraînements walk-forward hebdomadaires et semestriels, évalués au log loss et au RPS face aux cotes de clôture.",
      ],
      publish: [
        "Publication",
        "Une tâche GitHub Actions planifiée enregistre du JSON statique ; le site se reconstruit à chaque changement de données.",
      ],
    },
    glossary: "Glossaire",
    terms: {
      logLoss: [
        "Log loss",
        "Moyenne du logarithme négatif de la probabilité donnée au résultat réel. Pénalise les erreurs commises avec assurance.",
      ],
      rps: [
        "RPS",
        "Ranked probability score. Respecte l'ordre : prédire un nul quand l'équipe à domicile gagne est une erreur plus légère que prédire une victoire à l'extérieur.",
      ],
      market: [
        "Marché",
        "Cotes de clôture des bookmakers, marge retirée. La référence publique la plus exigeante.",
      ],
    },
    run: "Modèle {version} · exécution de {seconds} s · commit {commit}",
  },
  state: {
    loading: "Chargement",
    error: "Impossible de charger les données. Vérifiez votre connexion et réessayez.",
    retry: "Réessayer",
    notFound: "Cette page n'existe pas.",
    home: "Retour aux matchs",
  },
  footer: {
    data: "Données : football-data.co.uk",
    images: "Images : TheSportsDB",
    source: "Code source",
    disclaimer: "Les probabilités sont des estimations du modèle, pas des conseils de paris.",
  },
};

export default fr;
