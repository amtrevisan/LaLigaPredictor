import { Crest } from "@/components/Crest";
import { useI18n } from "@/i18n";
import { decimal, integer } from "@/lib/format";
import type { LeagueData } from "@/types/data";
import styles from "./views.module.css";

/** Ratings relative to an average side (1.00), drawn as a hairline tick on a fixed scale. */
function RatingCell({ value, invert = false }: { value: number | null; invert?: boolean }) {
  const { intl } = useI18n();
  if (value == null) return <td className={styles.right}>–</td>;
  // 0.5–1.5 covers every top-flight side; clamp the rest.
  const position = Math.min(Math.max((value - 0.5) / 1.0, 0), 1);
  const good = invert ? value < 1 : value > 1;
  return (
    <td className={styles.right}>
      <span className={styles.rating}>
        <span className={styles.ratingScale} aria-hidden="true">
          <span className={styles.ratingMid} />
          <span className={good ? styles.ratingTickSolid : styles.ratingTick} style={{ left: `${position * 100}%` }} />
        </span>
        {decimal(value, intl, 2)}
      </span>
    </td>
  );
}

export function TableView({ data }: { data: LeagueData }) {
  const { t, intl } = useI18n();
  return (
    <div className={styles.stack}>
      <header className={styles.viewHead}>
        <h1 className={styles.viewTitle}>
          {t.table.title} <span>{data.league.name} · {data.season}</span>
        </h1>
      </header>
      <div className={styles.scroll}>
        <table className={`${styles.table} ${styles.standings}`}>
          <caption className={styles.caption}>{t.table.caption}</caption>
          <thead>
            <tr>
              <th scope="col" className={styles.right}>
                {t.table.pos}
              </th>
              <th scope="col">{t.table.team}</th>
              <th scope="col" className={styles.right}>{t.table.played}</th>
              <th scope="col" className={styles.right}>{t.table.won}</th>
              <th scope="col" className={styles.right}>{t.table.drawn}</th>
              <th scope="col" className={styles.right}>{t.table.lost}</th>
              <th scope="col" className={styles.right}>{t.table.gd}</th>
              <th scope="col" className={styles.right}>{t.table.points}</th>
              <th scope="col" className={styles.right}>{t.table.elo}</th>
              <th scope="col" className={styles.right}>{t.table.attack}</th>
              <th scope="col" className={styles.right}>{t.table.defence}</th>
            </tr>
          </thead>
          <tbody>
            {data.standings.map((row, i) => {
              const gd = row.goalsFor - row.goalsAgainst;
              return (
                <tr key={row.team.name}>
                  <td className={`${styles.right} ${styles.dim}`}>{i + 1}</td>
                  <th scope="row">
                    <span className={styles.club}>
                      <Crest team={row.team} size={20} />
                      {row.team.name}
                    </span>
                  </th>
                  <td className={styles.right}>{row.played}</td>
                  <td className={styles.right}>{row.won}</td>
                  <td className={styles.right}>{row.drawn}</td>
                  <td className={styles.right}>{row.lost}</td>
                  <td className={styles.right}>{gd > 0 ? `+${gd}` : gd < 0 ? `−${-gd}` : "0"}</td>
                  <td className={`${styles.right} ${styles.strong}`}>{row.points}</td>
                  <td className={styles.right}>{row.elo == null ? "–" : integer(row.elo, intl)}</td>
                  <RatingCell value={row.attack} />
                  <RatingCell value={row.defence} invert />
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
