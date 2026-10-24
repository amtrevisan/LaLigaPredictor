import { CalibrationChart } from "@/components/CalibrationChart";
import { ArrowRight } from "@/components/icons";
import { SOURCE_URL } from "@/components/Shell";
import { ErrorState, FieldSkeleton } from "@/components/States";
import { useI18n } from "@/i18n";
import { useManifest, useModelReport } from "@/lib/api";
import { decimal, fill, integer, longDate, percent, points } from "@/lib/format";
import type { Evaluation, Scores } from "@/types/data";
import styles from "./ModelView.module.css";

type Column = { key: keyof Omit<Evaluation, "matches">; label: string };

const METRICS: { key: keyof Scores; lowerIsBetter: boolean }[] = [
  { key: "logLoss", lowerIsBetter: true },
  { key: "rps", lowerIsBetter: true },
  { key: "brier", lowerIsBetter: true },
  { key: "accuracy", lowerIsBetter: false },
];

export function ModelView() {
  const { t, intl } = useI18n();
  const report = useModelReport();
  const manifest = useManifest();

  if (!report.data) {
    return report.isError ? <ErrorState onRetry={() => void report.refetch()} /> : <FieldSkeleton />;
  }

  const data = report.data;
  const overall = data.backtest.overall;
  const leagueName = (id: string) => manifest.data?.leagues.find((l) => l.id === id)?.name ?? id;
  const metricLabel: Record<keyof Scores, string> = {
    logLoss: t.record.logLoss,
    rps: t.record.rps,
    brier: "Brier",
    accuracy: t.record.accuracy,
  };
  const columns: Column[] = [
    { key: "baseline", label: t.model.baseline },
    { key: "goals", label: t.model.goals },
    { key: "classifier", label: t.model.classifier },
    { key: "model", label: t.model.blend },
    { key: "market", label: t.model.market },
  ];
  const format = (key: keyof Scores, value: number) =>
    key === "accuracy" ? percent(value, intl, 1) : decimal(value, intl);
  const topImportance = data.importance.slice(0, 12);
  const maxImportance = Math.max(...topImportance.map((i) => i.importance), 1e-9);
  const steps = t.model.steps;

  return (
    <article className={styles.page}>
      <header className={styles.head}>
        <h1 className={styles.title}>{t.model.title}</h1>
        <p className={styles.lede}>{t.model.intro}</p>
      </header>

      <section className={styles.section} aria-labelledby="performance">
        <h2 id="performance" className={styles.heading}>
          {t.model.performance}
        </h2>
        <p className={styles.note}>
          {fill(t.model.performanceNote, { seasons: data.backtest.seasons })}{" "}
          {fill(t.model.matches, { n: integer(overall.matches, intl) })}.
        </p>
        <div className={styles.scroll}>
          <table className={styles.table} aria-labelledby="performance">
            <thead>
              <tr>
                <th scope="col">{t.model.metric}</th>
                {columns.map((c) => (
                  <th key={c.key} scope="col" className={c.key === "model" ? styles.ours : undefined}>
                    {c.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {METRICS.map(({ key }) => (
                <tr key={key}>
                  <th scope="row">{metricLabel[key]}</th>
                  {columns.map((c) => (
                    <td key={c.key} className={c.key === "model" ? styles.ours : undefined}>
                      {format(key, overall[c.key][key])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className={styles.split}>
        <section className={styles.section} aria-labelledby="calibration">
          <h2 id="calibration" className={styles.heading}>
            {t.model.calibration}
          </h2>
          <p className={styles.note}>{t.model.calibrationNote}</p>
          <CalibrationChart model={data.backtest.calibration.model} market={data.backtest.calibration.market} />
        </section>

        <section className={styles.section} aria-labelledby="drivers">
          <h2 id="drivers" className={styles.heading}>
            {t.model.drivers}
          </h2>
          <p className={styles.note}>{t.model.driversNote}</p>
          <ol className={styles.importance}>
            {topImportance.map((item) => (
              <li key={item.feature}>
                <code>{item.feature}</code>
                <span className={styles.importanceBar} style={{ width: `${(item.importance / maxImportance) * 100}%` }} />
                <span className={styles.importanceValue}>{decimal(item.importance, intl, 4)}</span>
              </li>
            ))}
          </ol>
        </section>
      </div>

      <section className={styles.section} aria-labelledby="by-league">
        <h2 id="by-league" className={styles.heading}>
          {t.model.byLeague}
        </h2>
        <div className={styles.scroll}>
          <table className={styles.table} aria-labelledby="by-league">
            <thead>
              <tr>
                <th scope="col">{t.model.league}</th>
                <th scope="col">{t.model.season}</th>
                <th scope="col">n</th>
                <th scope="col">
                  {t.record.logLoss} · {t.record.model}
                </th>
                <th scope="col">
                  {t.record.logLoss} · {t.record.market}
                </th>
                <th scope="col">
                  {t.record.accuracy} · {t.record.model}
                </th>
                <th scope="col">
                  {t.record.accuracy} · {t.record.market}
                </th>
                <th scope="col">Δ {t.record.accuracy}</th>
              </tr>
            </thead>
            <tbody>
              {data.backtest.byLeague.map((row) => (
                <tr key={`${row.league}-${row.season}`}>
                  <th scope="row">{leagueName(row.league)}</th>
                  <td>
                    {row.season}
                    {row.current && <span className={styles.current}> · {t.model.current}</span>}
                  </td>
                  <td>{row.matches}</td>
                  <td className={styles.ours}>{decimal(row.model.logLoss, intl)}</td>
                  <td>{decimal(row.market.logLoss, intl)}</td>
                  <td className={styles.ours}>{percent(row.model.accuracy, intl, 1)}</td>
                  <td>{percent(row.market.accuracy, intl, 1)}</td>
                  <td>{points(row.model.accuracy - row.market.accuracy, intl)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className={styles.section} aria-labelledby="pipeline">
        <h2 id="pipeline" className={styles.heading}>
          {t.model.pipeline}
        </h2>
        <ol className={styles.pipeline}>
          {(
            [
              [steps.ingest, { matches: integer(data.data.matches, intl), first: longDate(data.data.firstMatch, intl) }],
              [steps.features, {}],
              [
                steps.models,
                {
                  w: percent(data.blend.classifierWeight, intl),
                  g: percent(data.blend.goalsWeight, intl),
                },
              ],
              [steps.evaluate, {}],
              [steps.publish, {}],
            ] as const
          ).map(([[name, text], params], i, all) => (
            <li key={name}>
              <h3 className={styles.stepName}>
                {name}
                {i < all.length - 1 && <ArrowRight />}
              </h3>
              <p>{fill(text, params)}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className={styles.section} aria-labelledby="glossary">
        <h2 id="glossary" className={styles.heading}>
          {t.model.glossary}
        </h2>
        <dl className={styles.glossary}>
          {Object.values(t.model.terms).map(([term, definition]) => (
            <div key={term}>
              <dt>{term}</dt>
              <dd>{definition}</dd>
            </div>
          ))}
        </dl>
      </section>

      <p className={styles.run}>
        {fill(t.model.run, {
          version: data.version,
          seconds: decimal(data.runtimeSeconds, intl, 1),
          commit: data.commit ?? "local",
        })}{" "}
        · <a href={SOURCE_URL}>{t.footer.source}</a>
      </p>
    </article>
  );
}
