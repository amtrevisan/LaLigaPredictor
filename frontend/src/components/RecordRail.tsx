import { Link } from "react-router";
import { useI18n } from "@/i18n";
import { decimal, fill, percent } from "@/lib/format";
import type { PlayedFixture, SeasonRecord } from "@/types/data";
import { ArrowRight } from "./icons";
import { SeasonBarcode } from "./SeasonBarcode";
import styles from "./RecordRail.module.css";

interface Props {
  leagueId: string;
  season: string;
  record: SeasonRecord | null;
  results: PlayedFixture[];
}

/** Season-to-date scoring against the market, plus the most recent stretch as a barcode. */
export function RecordRail({ leagueId, season, record, results }: Props) {
  const { t, intl } = useI18n();
  return (
    <aside className={styles.rail} aria-labelledby="record-title">
      <h2 id="record-title" className={styles.title}>
        {t.record.title} <span>{season}</span>
      </h2>
      {record ? (
        <>
          <table className={styles.table}>
            <caption className="visually-hidden">{fill(t.record.matches, { n: record.matches })}</caption>
            <thead>
              <tr>
                <th scope="col" />
                <th scope="col">{t.record.model}</th>
                <th scope="col">{t.record.market}</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">{t.record.accuracy}</th>
                <td>{percent(record.model.accuracy, intl, 1)}</td>
                <td>{percent(record.market.accuracy, intl, 1)}</td>
              </tr>
              <tr>
                <th scope="row">{t.record.logLoss}</th>
                <td>{decimal(record.model.logLoss, intl)}</td>
                <td>{decimal(record.market.logLoss, intl)}</td>
              </tr>
              <tr>
                <th scope="row">{t.record.rps}</th>
                <td>{decimal(record.model.rps, intl)}</td>
                <td>{decimal(record.market.rps, intl)}</td>
              </tr>
            </tbody>
          </table>
          <p className={styles.note}>
            {fill(t.record.matches, { n: record.matches })}. {t.record.lowerBetter}
          </p>
          <SeasonBarcode results={results} limit={60} label={t.results.title} />
          <Link to={`/${leagueId}/results`} className={styles.more}>
            {t.nav.results} <ArrowRight />
          </Link>
        </>
      ) : (
        <p className={styles.note}>{t.record.none}</p>
      )}
    </aside>
  );
}
