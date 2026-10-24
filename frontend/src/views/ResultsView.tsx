import { useState } from "react";
import { Crest } from "@/components/Crest";
import { OutcomeBar } from "@/components/OutcomeBar";
import { SeasonBarcode } from "@/components/SeasonBarcode";
import { EmptyState } from "@/components/States";
import { useI18n } from "@/i18n";
import { fill, kickoffDay, percent, probabilityOf } from "@/lib/format";
import type { LeagueData } from "@/types/data";
import styles from "./views.module.css";

export function ResultsView({ data }: { data: LeagueData }) {
  const { t, intl } = useI18n();
  const [round, setRound] = useState<number | "all">("all");
  const rounds = [...new Set(data.results.map((r) => r.round))].sort((a, b) => b - a);
  const rows = round === "all" ? data.results : data.results.filter((r) => r.round === round);
  const hits = data.results.filter((r) => r.correct).length;

  return (
    <div className={styles.stack}>
      <header className={styles.viewHead}>
        <h1 className={styles.viewTitle}>
          {t.results.title} <span>{data.league.name} · {data.season}</span>
        </h1>
        {data.results.length > 0 && (
          <div className={styles.viewMeta}>
            <span>
              {hits} {t.results.hit} · {data.results.length - hits} {t.results.miss}
            </span>
          </div>
        )}
      </header>

      {data.results.length === 0 ? (
        <EmptyState>{t.results.empty}</EmptyState>
      ) : (
        <>
          <section className={styles.barcode} aria-labelledby="barcode-note">
            <SeasonBarcode results={data.results} label={t.results.title} />
            <p id="barcode-note" className={styles.note}>
              {t.results.barcode}
            </p>
          </section>

          <div className={styles.toolbar}>
            <label className={styles.select}>
              <span>{t.results.filter}</span>
              <select
                value={round}
                onChange={(e) => setRound(e.target.value === "all" ? "all" : Number(e.target.value))}
              >
                <option value="all">{t.results.allRounds}</option>
                {rounds.map((r) => (
                  <option key={r} value={r}>
                    {fill(t.fixtures.round, { n: r })}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <table className={styles.table}>
            <thead>
              <tr>
                <th scope="col">{t.fixtures.kickoff}</th>
                <th scope="col">{t.results.result}</th>
                <th scope="col" className={styles.wide}>
                  {t.results.prediction}
                </th>
                <th scope="col" className={styles.right}>
                  <span className="visually-hidden">{t.results.hit}</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id} className={r.correct ? undefined : styles.missRow}>
                  <td className={styles.dim}>{kickoffDay(r.kickoff, intl)}</td>
                  <td>
                    <span className={styles.match}>
                      <span className={styles.club}>
                        <Crest team={r.home} size={20} />
                        {r.home.name}
                      </span>
                      <b className={styles.scoreCell}>
                        {r.result.score[0]}–{r.result.score[1]}
                      </b>
                      <span className={styles.club}>
                        <Crest team={r.away} size={20} />
                        {r.away.name}
                      </span>
                    </span>
                  </td>
                  <td className={styles.wide}>
                    <span className={styles.predCell}>
                      <OutcomeBar probs={r.probs} orientation="horizontal" compact />
                      <span className={styles.dim}>
                        {t.outcome.short[r.pick]} ·{" "}
                        {fill(t.results.gave, { p: percent(probabilityOf(r.probs, r.result.outcome), intl) })}
                      </span>
                    </span>
                  </td>
                  <td className={styles.right}>
                    <span className={r.correct ? styles.hitTag : styles.missTag}>
                      {r.correct ? t.results.hit : t.results.miss}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
