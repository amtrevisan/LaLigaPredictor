import type { Messages } from "./en";

const pt: Messages = {
  meta: {
    title: "Matchfield: previsões de futebol para seis ligas",
    description:
      "Probabilidades de vitória, empate e derrota para Premier League, La Liga, Serie A italiana, Bundesliga, Ligue 1 e Brasileirão, comparadas com o mercado de apostas.",
  },
  nav: {
    skip: "Pular para o conteúdo",
    fixtures: "Jogos",
    results: "Resultados",
    table: "Classificação",
    model: "Modelo",
    language: "Idioma",
    sections: "Seções",
    leagues: "Ligas",
    updated: "Dados de {time}",
  },
  outcome: { H: "Mandante", D: "Empate", A: "Visitante", short: { H: "1", D: "X", A: "2" } },
  fixtures: {
    round: "Rodada {n}",
    count: "{n} jogos",
    kickoff: "Início",
    xg: "xG",
    likeliest: "Placares mais prováveis",
    model: "Modelo",
    market: "Mercado",
    classifier: "Classificador",
    goals: "Modelo de gols",
    elo: "Elo",
    venue: "Estádio",
    meetings: "Confrontos anteriores",
    noMeetings: "Nenhum confronto nas duas últimas temporadas.",
    highlights: "Melhores momentos",
    preview: "Arte do jogo",
    select: "Selecione um jogo para ver o detalhamento completo.",
    gap: "vs mercado",
    empty: "Nenhum jogo marcado ainda. A próxima rodada aparece aqui assim que for divulgada.",
    legend: "As barras mostram a probabilidade do modelo de vitória do mandante, empate e vitória do visitante.",
  },
  results: {
    title: "A temporada até agora",
    barcode:
      "Uma barra por jogo, a mais recente à direita. A altura é a probabilidade que o modelo deu ao que de fato aconteceu; barras cheias foram acertos.",
    hit: "Acerto",
    miss: "Erro",
    prediction: "Previsão",
    result: "Resultado",
    gave: "O modelo deu {p} a esse resultado",
    allRounds: "Todas as rodadas",
    filter: "Rodada",
    empty: "Nenhum jogo disputado nesta temporada ainda.",
  },
  table: {
    title: "Classificação",
    caption:
      "Tabela do campeonato com as notas do modelo. Ataque e defesa são relativos a um time médio (1,00).",
    pos: "Pos",
    team: "Time",
    played: "J",
    won: "V",
    drawn: "E",
    lost: "D",
    gd: "SG",
    points: "Pts",
    elo: "Elo",
    attack: "Ata",
    defence: "Def",
  },
  record: {
    title: "Desempenho na temporada",
    matches: "{n} jogos",
    accuracy: "Acerto",
    logLoss: "Log loss",
    rps: "RPS",
    model: "Modelo",
    market: "Mercado",
    lowerBetter: "Em log loss e RPS, quanto menor, melhor.",
    none: "A temporada ainda não começou.",
  },
  model: {
    title: "Como o modelo funciona",
    intro:
      "Toda probabilidade deste site vem de um pipeline que nunca vê as odds das casas de apostas. As odds só entram depois, como referência para avaliar o modelo.",
    performance: "Desempenho",
    performanceNote:
      "Backtest walk-forward nas últimas {seasons} temporadas completas de cada liga: cada jogo foi previsto usando apenas resultados anteriores ao apito inicial.",
    metric: "Métrica",
    baseline: "Frequência base",
    classifier: "Só classificador",
    goals: "Só modelo de gols",
    blend: "Combinação publicada",
    market: "Odds de fechamento",
    matches: "{n} jogos",
    calibration: "Calibração",
    calibrationNote:
      "Quando o modelo diz 40%, isso acontece em 40% das vezes? Pontos sobre a diagonal estão perfeitamente calibrados.",
    predicted: "Probabilidade prevista",
    observed: "Frequência observada",
    byLeague: "Por liga e temporada",
    season: "Temporada",
    league: "Liga",
    current: "em andamento",
    drivers: "O que move uma previsão",
    driversNote:
      "Importância por permutação nos últimos doze meses, fora da amostra: quanto o log loss piora quando uma variável é embaralhada.",
    pipeline: "Pipeline",
    steps: {
      ingest: [
        "Coleta",
        "Resultados, finalizações e odds de {matches} jogos desde {first}, do football-data.co.uk.",
      ],
      features: [
        "Variáveis",
        "Ratings Elo e forma com média exponencial, calculados em uma única passagem cronológica para que nenhuma variável veja o próprio resultado.",
      ],
      models: [
        "Modelos",
        "Um classificador de gradient boosting e um modelo de gols Dixon-Coles ponderado no tempo, combinados {w} / {g}.",
      ],
      evaluate: [
        "Avaliação",
        "Retreinos walk-forward semanais e semestrais, avaliados com log loss e RPS contra as odds de fechamento.",
      ],
      publish: [
        "Publicação",
        "Um job agendado do GitHub Actions grava JSON estático; o site é reconstruído a cada mudança nos dados.",
      ],
    },
    glossary: "Glossário",
    terms: {
      logLoss: [
        "Log loss",
        "Média do logaritmo negativo da probabilidade dada ao resultado real. Pune erros cometidos com muita confiança.",
      ],
      rps: [
        "RPS",
        "Ranked probability score. Respeita a ordem: prever empate quando o mandante vence é um erro menor do que prever vitória do visitante.",
      ],
      market: [
        "Mercado",
        "Odds de fechamento sem a margem da casa. A referência pública mais difícil de bater.",
      ],
    },
    run: "Modelo {version} · execução de {seconds}s · commit {commit}",
  },
  state: {
    loading: "Carregando",
    error: "Não foi possível carregar os dados. Verifique sua conexão e tente novamente.",
    retry: "Tentar de novo",
    notFound: "Essa página não existe.",
    home: "Voltar aos jogos",
  },
  footer: {
    data: "Dados: football-data.co.uk",
    images: "Imagens: TheSportsDB",
    source: "Código-fonte",
    disclaimer: "As probabilidades são estimativas do modelo, não dicas de aposta.",
  },
};

export default pt;
