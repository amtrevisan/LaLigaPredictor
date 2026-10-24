import type { Messages } from "./en";

const es: Messages = {
  meta: {
    title: "Matchfield: predicciones de fútbol para seis ligas",
    description:
      "Probabilidades de victoria, empate y derrota para la Premier League, LaLiga, Serie A, Bundesliga, Ligue 1 y Brasileirão, comparadas con el mercado de apuestas.",
  },
  nav: {
    skip: "Ir al contenido",
    fixtures: "Partidos",
    results: "Resultados",
    table: "Clasificación",
    model: "Modelo",
    language: "Idioma",
    sections: "Secciones",
    leagues: "Ligas",
    updated: "Datos a {time}",
  },
  outcome: { H: "Local", D: "Empate", A: "Visitante", short: { H: "1", D: "X", A: "2" } },
  fixtures: {
    round: "Jornada {n}",
    count: "{n} partidos",
    kickoff: "Inicio",
    xg: "xG",
    likeliest: "Marcadores más probables",
    model: "Modelo",
    market: "Mercado",
    classifier: "Clasificador",
    goals: "Modelo de goles",
    elo: "Elo",
    venue: "Estadio",
    meetings: "Enfrentamientos anteriores",
    noMeetings: "Sin enfrentamientos en las dos últimas temporadas.",
    highlights: "Resumen",
    preview: "Cartel del partido",
    select: "Selecciona un partido para ver el desglose completo.",
    gap: "Modelo menos mercado",
    empty: "Aún no hay partidos programados. La próxima jornada aparecerá aquí en cuanto se publique.",
    legend: "Las barras muestran la probabilidad del modelo de victoria local, empate y victoria visitante.",
  },
  results: {
    title: "La temporada hasta ahora",
    barcode:
      "Una barra por partido, la más reciente a la derecha. La altura es la probabilidad que el modelo dio a lo que finalmente ocurrió; las barras sólidas fueron aciertos.",
    hit: "Acierto",
    miss: "Fallo",
    prediction: "Predicción",
    result: "Resultado",
    gave: "El modelo dio al resultado un {p}",
    allRounds: "Todas las jornadas",
    filter: "Jornada",
    empty: "Todavía no se ha jugado ningún partido esta temporada.",
  },
  table: {
    title: "Clasificación",
    caption:
      "Tabla de la liga con las valoraciones del modelo. Ataque y defensa son relativos a un equipo medio (1,00).",
    pos: "Pos",
    team: "Equipo",
    played: "PJ",
    won: "G",
    drawn: "E",
    lost: "P",
    gd: "DG",
    points: "Pts",
    elo: "Elo",
    attack: "Atq",
    defence: "Def",
  },
  record: {
    title: "Balance de la temporada",
    matches: "{n} partidos",
    accuracy: "Acierto",
    logLoss: "Log loss",
    rps: "RPS",
    model: "Modelo",
    market: "Mercado",
    lowerBetter: "En log loss y RPS, cuanto más bajo, mejor.",
    none: "La temporada aún no ha empezado.",
  },
  model: {
    title: "Cómo funciona el modelo",
    intro:
      "Cada probabilidad de esta web sale de un sistema que nunca ve las cuotas de apuestas. Las cuotas solo se usan después, como referencia con la que se mide el modelo.",
    performance: "Rendimiento",
    performanceNote:
      "Backtest walk-forward sobre las últimas {seasons} temporadas completas de cada liga: cada partido se predijo usando solo resultados anteriores al inicio.",
    metric: "Métrica",
    baseline: "Frecuencia base",
    classifier: "Solo clasificador",
    goals: "Solo modelo de goles",
    blend: "Mezcla publicada",
    market: "Cuotas de cierre",
    matches: "{n} partidos",
    calibration: "Calibración",
    calibrationNote:
      "Cuando el modelo dice 40 %, ¿ocurre el 40 % de las veces? Los puntos sobre la diagonal están perfectamente calibrados.",
    predicted: "Probabilidad prevista",
    observed: "Frecuencia observada",
    byLeague: "Por liga y temporada",
    season: "Temporada",
    league: "Liga",
    current: "en curso",
    drivers: "Qué mueve una predicción",
    driversNote:
      "Importancia por permutación en los últimos doce meses, fuera de muestra: cuánto empeora el log loss al barajar una variable.",
    pipeline: "Proceso",
    steps: {
      ingest: [
        "Ingesta",
        "Resultados, tiros y cuotas de {matches} partidos desde {first}, de football-data.co.uk.",
      ],
      features: [
        "Variables",
        "Ratings Elo y forma con media exponencial, construidos en un único recorrido cronológico para que ninguna variable vea su propio resultado.",
      ],
      models: [
        "Modelos",
        "Un clasificador de gradient boosting y un modelo de goles Dixon-Coles ponderado en el tiempo, mezclados {w} / {g}.",
      ],
      evaluate: [
        "Evaluación",
        "Reentrenos walk-forward semanales y semestrales, medidos con log loss y RPS frente a las cuotas de cierre.",
      ],
      publish: [
        "Publicación",
        "Un job programado de GitHub Actions guarda JSON estático; la web se reconstruye con cada cambio de datos.",
      ],
    },
    glossary: "Glosario",
    terms: {
      logLoss: [
        "Log loss",
        "Media del logaritmo negativo de la probabilidad asignada al resultado real. Castiga los errores con mucha confianza.",
      ],
      rps: [
        "RPS",
        "Ranked probability score. Respeta el orden: pronosticar empate cuando gana el local es un error menor que pronosticar victoria visitante.",
      ],
      market: [
        "Mercado",
        "Cuotas de cierre sin el margen de la casa. La referencia pública más exigente que existe.",
      ],
    },
    run: "Modelo {version} · ejecución de {seconds}s · commit {commit}",
  },
  state: {
    loading: "Cargando",
    error: "No se pudieron cargar los datos. Revisa tu conexión e inténtalo de nuevo.",
    retry: "Reintentar",
    notFound: "Esa página no existe.",
    home: "Volver a los partidos",
  },
  footer: {
    data: "Datos: football-data.co.uk",
    images: "Imágenes: TheSportsDB",
    source: "Código fuente",
    disclaimer: "Las probabilidades son estimaciones del modelo, no consejos de apuestas.",
  },
};

export default es;
